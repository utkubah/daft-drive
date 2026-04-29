"""
train.py
========
Fine-tune EfficientViT-SAM l0 on medical image segmentation.

Covers BOTH stages of the DAFT paper (Ma et al., CVPR 2024):
  Stage 1 -- Global fine-tune : train on ALL modalities at once
  Stage 2 -- DAFT specialists : train one model per modality subset

The loss is identical in both stages:
  Dice  +  Binary Cross-Entropy  +  IoU-regression MSE
  (same three-loss formulation as finetune.py / daft.py in the original repo)

Checkpoints are saved under:
  checkpoints/<name>/best.pth     <- lowest validation loss
  checkpoints/<name>/latest.pth   <- most recent epoch

Usage examples
--------------
  # Global fine-tune starting from the merged/distilled weights:
  python train.py --train_csv data/datasplit/train.csv \\
                  --val_csv   data/datasplit/val.csv   \\
                  --weights   checkpoints/global.pth   \\
                  --name      global

  # DAFT: fine-tune a US specialist from the global checkpoint:
  python train.py --train_csv data/datasplit/modalities/US.train.csv \\
                  --val_csv   data/datasplit/modalities/US.val.csv    \\
                  --weights   checkpoints/global/best.pth             \\
                  --name      US

  # Resume an interrupted run:
  python train.py --train_csv data/datasplit/train.csv \\
                  --val_csv   data/datasplit/val.csv   \\
                  --resume    checkpoints/global/latest.pth \\
                  --name      global
"""

import os
import random
import argparse
from os.path import join, isfile
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import monai
import matplotlib
matplotlib.use("Agg")           # no display needed
import matplotlib.pyplot as plt

from model   import MedSAM
from dataset import MedSegDataset


# --- helpers ---
def cal_iou(pred_mask, gt_mask):
    """
    Per-sample IoU used to supervise the model's IoU prediction head.

    Args:
        pred_mask  (B, 1, H, W)  bool  -- thresholded predicted mask
        gt_mask    (B, 1, H, W)  bool  -- ground-truth binary mask
    Returns:
        iou        (B, 1)        float
    """
    dims  = list(range(1, pred_mask.ndim))
    inter = torch.count_nonzero(pred_mask & gt_mask, dim=dims).float()
    union = torch.count_nonzero(pred_mask | gt_mask, dim=dims).float()
    return (inter / union.clamp(min=1e-6)).unsqueeze(1)


def set_seeds(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def save_rng():
    state = {"py": random.getstate(), "np": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def load_rng(state):
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])


# --- argument parsing ---
def get_args():
    p = argparse.ArgumentParser(
        description="Fine-tune EfficientViT-SAM (global or per-modality DAFT)."
    )
    # --- data ---
    p.add_argument("--train_csv",    required=True,
                   help="CSV with a 'file' column listing training .npz paths")
    p.add_argument("--val_csv",      required=True,
                   help="CSV with a 'file' column listing validation .npz paths")
    # --- weights ---
    p.add_argument("--weights",      default=None,
                   help="Pretrained model weights to start from (ignored if --resume)")
    p.add_argument("--resume",       default=None,
                   help="Training checkpoint to resume (restores optimizer & epoch too)")
    # --- run identity ---
    p.add_argument("--name",         default="run",
                   help="Subdirectory name for checkpoints  (e.g. 'global', 'US')")
    # --- hyperparameters ---
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--lr",           type=float, default=5e-5,
                   help="AdamW learning rate (paper uses 5e-5)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--patience",     type=int,   default=7,
                   help="Early-stopping: epochs without val improvement before stopping")
    p.add_argument("--bbox_shift",   type=int,   default=5,
                   help="Max pixel jitter applied to bounding boxes during training")
    # --- loss weights (all 1.0 as in the paper) ---
    p.add_argument("--seg_weight",   type=float, default=1.0, help="Dice loss weight")
    p.add_argument("--ce_weight",    type=float, default=1.0, help="BCE  loss weight")
    p.add_argument("--iou_weight",   type=float, default=1.0, help="IoU  loss weight")
    # --- device ---
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# --- one epoch of training or validation ---
def run_epoch(model, loader, optimizer,
              seg_loss, ce_loss, iou_loss,
              sw, cw, iw, device, train=True):
    """
    Runs one full pass over the dataloader.
    Returns the mean combined loss.
    """
    model.train() if train else model.eval()
    total = 0.0

    with (torch.enable_grad() if train else torch.no_grad()):
        for batch in loader:
            images = batch["image"].to(device)   # (B, 3, 256, 256)
            masks  = batch["mask"].to(device)    # (B, 1, 256, 256)  long
            bboxes = batch["bbox"].to(device)    # (B, 1, 1, 4)

            logits, iou_pred = model(images, bboxes)   # (B,1,256,256), (B,1)

            l_seg = seg_loss(logits, masks.float())
            l_ce  = ce_loss(logits, masks.float())
            iou_gt = cal_iou(
                (torch.sigmoid(logits) > 0.5),
                masks.bool()
            )
            l_iou = iou_loss(iou_pred, iou_gt)

            loss = sw * l_seg + cw * l_ce + iw * l_iou

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total += loss.item()

    return total / max(len(loader), 1)


# --- main ---
def main():
    args = get_args()

    ckpt_dir = join("checkpoints", args.name)
    os.makedirs(ckpt_dir, exist_ok=True)

    SEED = 2024
    set_seeds(SEED)

    # --- build model ---
    # If --resume is given, we load model weights from the resume checkpoint below.
    # If not resuming, MedSAM() loads --weights (or starts random if None).
    init_weights = None if args.resume else args.weights
    model = MedSAM(init_weights).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model ready. Trainable params: {n_params:,}")

    # --- optimizer + scheduler + losses ---
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=5, cooldown=5
    )
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss  = nn.BCEWithLogitsLoss(reduction="mean")
    iou_loss = nn.MSELoss(reduction="mean")

    # --- resume state ---
    start_epoch       = 0
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    train_losses      = []

    if args.resume and isfile(args.resume):
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(args.device)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch       = ckpt["epoch"]
        best_val_loss     = ckpt["best_val_loss"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        train_losses      = ckpt.get("train_losses", [])
        if "rng" in ckpt:
            load_rng(ckpt["rng"])
        print(f"    Resumed at epoch {start_epoch}, best val {best_val_loss:.4f}")

    # --- datasets ---
    train_ds = MedSegDataset(args.train_csv, bbox_shift=args.bbox_shift, augment=True)
    val_ds   = MedSegDataset(args.val_csv,   bbox_shift=0,               augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # --- training loop ---
    for epoch in range(start_epoch + 1, args.epochs + 1):

        if epochs_no_improve > args.patience:
            print("  Early stopping triggered.")
            break

        t0 = time()

        train_loss = run_epoch(
            model, train_loader, optimizer,
            seg_loss, ce_loss, iou_loss,
            args.seg_weight, args.ce_weight, args.iou_weight,
            args.device, train=True,
        )
        train_losses.append(train_loss)

        # Validation is deterministic -- freeze RNG, run val, restore for training
        train_rng = save_rng()
        set_seeds(SEED)
        val_loss = run_epoch(
            model, val_loader, optimizer,
            seg_loss, ce_loss, iou_loss,
            args.seg_weight, args.ce_weight, args.iou_weight,
            args.device, train=False,
        )
        load_rng(train_rng)

        scheduler.step(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss     = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"  Epoch {epoch:3d}/{args.epochs}  "
            f"train {train_loss:.4f}  val {val_loss:.4f}  "
            f"best {best_val_loss:.4f}  ({time()-t0:.1f}s)"
        )

        # Save best checkpoint only (skip latest.pth to conserve disk space)
        if improved:
            ckpt = {
                "model":             model.state_dict(),
                "optimizer":         optimizer.state_dict(),
                "epoch":             epoch,
                "best_val_loss":     best_val_loss,
                "epochs_no_improve": epochs_no_improve,
                "train_losses":      train_losses,
                "rng":               save_rng(),
            }
            torch.save(ckpt, join(ckpt_dir, "best.pth"))
            print(f"    -> New best saved: {join(ckpt_dir, 'best.pth')}")

    # --- loss curve ---
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss  (Dice + BCE + IoU)")
    plt.title(f"Training loss -- {args.name}")
    plt.tight_layout()
    plt.savefig(join(ckpt_dir, "loss_curve.png"))
    plt.close()
    print(f"  Loss curve -> {join(ckpt_dir, 'loss_curve.png')}")


if __name__ == "__main__":
    main()
