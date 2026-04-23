"""
distill.py
==========
Knowledge distillation: train the EfficientViT-l0 image encoder to match
the feature maps produced by the TinyViT image encoder from LiteMedSAM.

Why distillation?
-----------------
EfficientViT-l0 is a faster architecture than TinyViT, but starting it from
random weights gives poor segmentation quality.  We pre-train it by minimising
the MSE between its feature maps and those of the already-trained TinyViT
teacher.  This gives the EfficientViT encoder a strong initialisation before
fine-tuning on the segmentation task.

  Teacher  :  TinyViT image encoder from lite_medsam.pth  (frozen)
  Student  :  EfficientViT-l0 image encoder               (trained)
  Loss     :  MSE( student_features, teacher_features )
  Shapes   :  (B, 256, 64, 64)  -- same for both encoders

Output
------
  checkpoints/distilled_encoder.pth   -- EfficientViT encoder state dict only
  Feed this into merge_weights.py to build a full inference-ready model.

Dependencies
------------
  tiny_vit_sam.py must be importable.  It lives in the sibling directory
  CVPR24-MedSAM-on-Laptop/ and is added to sys.path automatically below.
  Alternatively, copy tiny_vit_sam.py into the daft-clean/ directory.

Usage
-----
  python distill.py \\
    --train_csv  data/datasplit/train.csv \\
    --val_csv    data/datasplit/val.csv   \\
    --lite_medsam ../lite_medsam.pth
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- import TinyViT ---
# Searches for tiny_vit_sam.py in several locations so the project works
# regardless of how the folders are arranged:
#   1. Same directory as this script  (best: just copy tiny_vit_sam.py here)
#   2. A subfolder called CVPR24-MedSAM-on-Laptop/  inside the project
#   3. One level up  (../CVPR24-MedSAM-on-Laptop/)
#   4. Two levels up (../../CVPR24-MedSAM-on-Laptop/) -- nested repo layouts
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TINYVIT_CANDIDATES = [
    _SCRIPT_DIR,
    os.path.join(_SCRIPT_DIR, "CVPR24-MedSAM-on-Laptop"),
    os.path.join(_SCRIPT_DIR, "..", "CVPR24-MedSAM-on-Laptop"),
    os.path.join(_SCRIPT_DIR, "..", "..", "CVPR24-MedSAM-on-Laptop"),
]
for _candidate in _TINYVIT_CANDIDATES:
    _candidate = os.path.abspath(_candidate)
    if os.path.isfile(os.path.join(_candidate, "tiny_vit_sam.py")):
        sys.path.insert(0, _candidate)
        break

try:
    from tiny_vit_sam import TinyViT
except ImportError as e:
    raise ImportError(
        "\nCannot import TinyViT (tiny_vit_sam.py not found).\n"
        "Easiest fix: copy tiny_vit_sam.py into the same folder as distill.py.\n"
        "  cp ../CVPR24-MedSAM-on-Laptop/tiny_vit_sam.py .\n"
        f"Original error: {e}"
    )

from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from dataset import MedSegDataset


# --- build teacher (TinyViT architecture from LiteMedSAM) ---
def build_tinvit_encoder():
    """
    Construct the TinyViT image encoder used inside LiteMedSAM.
    Architecture is identical to the one in the original repo.
    """
    return TinyViT(
        img_size=256,
        in_chans=3,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8,
    )


def load_teacher(lite_medsam_path, device):
    """
    Load the TinyViT image encoder weights from a LiteMedSAM checkpoint.

    lite_medsam.pth stores the full LiteMedSAM state dict with keys:
      image_encoder.*  /  prompt_encoder.*  /  mask_decoder.*
    We only need the image_encoder part.
    """
    teacher = build_tinvit_encoder()

    state      = torch.load(lite_medsam_path, map_location="cpu", weights_only=False)
    enc_state  = {
        k[len("image_encoder."):]: v
        for k, v in state.items()
        if k.startswith("image_encoder.")
    }
    if not enc_state:
        raise ValueError(
            f"No 'image_encoder.*' keys found in {lite_medsam_path}.\n"
            "Is this really a LiteMedSAM checkpoint?"
        )

    teacher.load_state_dict(enc_state, strict=True)
    teacher.to(device).eval()

    for p in teacher.parameters():
        p.requires_grad = False

    print(f"  Teacher (TinyViT) loaded from {lite_medsam_path}")
    return teacher


# --- argument parsing ---
def get_args():
    p = argparse.ArgumentParser(
        description="Distil EfficientViT-l0 encoder from TinyViT teacher."
    )
    p.add_argument("--train_csv",   required=True)
    p.add_argument("--val_csv",     required=True)
    p.add_argument("--lite_medsam", default="lite_medsam.pth",
                   help="LiteMedSAM checkpoint (teacher weights)")
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="checkpoints/distilled_encoder.pth",
                   help="Output path for the distilled EfficientViT encoder state dict")
    return p.parse_args()


# --- main ---
def main():
    args = get_args()

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # --- teacher ---
    teacher = load_teacher(args.lite_medsam, args.device)

    # --- student: EfficientViT-l0 image encoder only ---
    base    = create_efficientvit_sam_model("efficientvit-sam-l0", pretrained=False)
    student = base.image_encoder.to(args.device)
    student.train()
    print(f"  Student (EfficientViT-l0) params: "
          f"{sum(p.numel() for p in student.parameters()):,}")

    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    mse_loss  = nn.MSELoss()

    # --- data ---
    train_ds = MedSegDataset(args.train_csv, augment=True)
    val_ds   = MedSegDataset(args.val_csv,   augment=False)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers, pin_memory=True)

    best_val  = float("inf")
    train_log = []

    for epoch in range(1, args.epochs + 1):

        # --- train ---
        student.train()
        tr_loss = 0.0
        for batch in train_ld:
            images = batch["image"].to(args.device)      # (B, 3, 256, 256)
            with torch.no_grad():
                t_feat = teacher(images)                 # (B, 256, 64, 64)
            s_feat = student(images)
            loss   = mse_loss(s_feat, t_feat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= max(len(train_ld), 1)

        # --- validation ---
        student.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for batch in val_ld:
                images = batch["image"].to(args.device)
                t_feat = teacher(images)
                s_feat = student(images)
                vl_loss += mse_loss(s_feat, t_feat).item()
        vl_loss /= max(len(val_ld), 1)

        train_log.append(tr_loss)
        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"train MSE {tr_loss:.5f}  val MSE {vl_loss:.5f}")

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(student.state_dict(), args.out)
            print(f"    -> Saved distilled encoder: {args.out}")

    # --- loss curve ---
    plt.figure()
    plt.plot(range(1, len(train_log) + 1), train_log)
    plt.xlabel("Epoch")
    plt.ylabel("MSE  (student vs teacher features)")
    plt.title("Distillation loss  (EfficientViT <- TinyViT)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "distill_loss.png"))
    plt.close()
    print(f"  Loss curve -> {os.path.join(out_dir, 'distill_loss.png')}")


if __name__ == "__main__":
    main()
