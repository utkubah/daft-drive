"""
train_router.py
===============
Fine-tune MobileNetV3-small to classify which of the 5 DAFT conditions
an image belongs to. Saved checkpoint is loaded by ImageRouter in router.py.

Dataset
-------
Reads directly from the manifests produced by prepare_data.py.
Each row has image_path + condition — no extra preparation needed.
Rows with condition=None (tunnel, parking lot) are skipped automatically.

Class imbalance is handled with weighted cross-entropy loss.
Basic augmentation (horizontal flip, color jitter) is applied during training.

Usage
-----
  python train_router.py --device cuda
  python train_router.py --epochs 20 --batch 64
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

from router import CONDITIONS

MANIFEST_DIR = Path("data/bdd100k/manifests")
OUT_DIR      = Path("checkpoints/router")
IMG_SIZE     = 224


# ── Dataset ───────────────────────────────────────────────────────────────────

class ConditionDataset(Dataset):
    """
    Reads image_path + condition from a manifest CSV.
    Skips rows where condition is not one of the 5 valid CONDITIONS.
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, manifest_csv: Path, augment: bool = False):
        self.augment  = augment
        self._cls_idx = {c: i for i, c in enumerate(CONDITIONS)}
        self.samples: list[tuple[str, int]] = []

        with open(manifest_csv, newline="") as f:
            for row in csv.DictReader(f):
                cond = row.get("condition", "")
                if cond in self._cls_idx:
                    self.samples.append((row["image_path"], self._cls_idx[cond]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[i]

        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if self.augment:
            if torch.rand(1).item() > 0.5:
                tensor = TF.hflip(tensor)
            tensor = TF.adjust_brightness(tensor, 0.8 + 0.4 * torch.rand(1).item())
            tensor = TF.adjust_contrast(tensor,  0.8 + 0.4 * torch.rand(1).item())

        tensor = TF.normalize(tensor, mean=self.MEAN, std=self.STD)
        return tensor, label

    def class_distribution(self) -> dict[str, int]:
        counts = Counter(label for _, label in self.samples)
        return {CONDITIONS[i]: counts[i] for i in range(len(CONDITIONS))}


def compute_class_weights(dataset: ConditionDataset, device: torch.device) -> torch.Tensor:
    dist   = dataset.class_distribution()
    counts = torch.tensor([dist[c] for c in CONDITIONS], dtype=torch.float)
    weights = counts.sum() / (len(CONDITIONS) * counts.clamp(min=1))
    return weights.to(device)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(n_classes: int = len(CONDITIONS)) -> nn.Module:
    """
    MobileNetV3-small pretrained on ImageNet.
    Only the last classifier layer is replaced — fine-tune all layers.
    """
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features   # 1024
    model.classifier[-1] = nn.Linear(in_features, n_classes)
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict]:
    model.eval()
    correct = total = 0
    per_correct = torch.zeros(len(CONDITIONS))
    per_total   = torch.zeros(len(CONDITIONS))
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += len(labels)
            for c in range(len(CONDITIONS)):
                mask = labels == c
                per_correct[c] += (preds[mask] == c).sum().item()
                per_total[c]   += mask.sum().item()
    acc = correct / total if total else 0.0
    per_class = {
        CONDITIONS[i]: per_correct[i].item() / max(per_total[i].item(), 1)
        for i in range(len(CONDITIONS))
    }
    return acc, per_class


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir",  default=str(OUT_DIR))
    p.add_argument("--epochs",   type=int,   default=20)
    p.add_argument("--batch",    type=int,   default=64)
    p.add_argument("--lr",       type=float, default=1e-4,
                   help="Lower LR since we fine-tune a pretrained model")
    p.add_argument("--workers",  type=int,   default=2)
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args    = get_args()
    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = ConditionDataset(MANIFEST_DIR / "train.csv", augment=True)
    val_ds   = ConditionDataset(MANIFEST_DIR / "val.csv",   augment=False)

    print(f"Train: {len(train_ds)} samples")
    print(f"Val:   {len(val_ds)} samples")
    print(f"Class distribution (train):")
    for cond, n in train_ds.class_distribution().items():
        pct = 100 * n / len(train_ds)
        print(f"  {cond:<20}  {n:>5}  ({pct:.1f}%)")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers,
                              pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers,
                              pin_memory=device.type == "cuda")

    # ── Model + loss ──────────────────────────────────────────────────────────
    model     = build_model().to(device)
    cw        = compute_class_weights(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nModel: MobileNetV3-small  ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    print(f"Device: {device} | Epochs: {args.epochs} | Batch: {args.batch} | LR: {args.lr}")

    # ── Train loop ────────────────────────────────────────────────────────────
    best_acc  = 0.0
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        acc, per_class = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  val_acc={acc:.3f}", end="")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)
            print("  ✓ best", end="")
        print()

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\nBest val accuracy: {best_acc:.3f}")
    print("Per-condition accuracy (last epoch):")
    for cond, a in per_class.items():
        n = val_ds.class_distribution().get(cond, 0)
        print(f"  {cond:<20}  {a:.3f}  ({n} val samples)")

    print(f"\nSaved → {best_path}")
    print("Use in inference:")
    print("  python inference.py --source <imgs> \\")
    print("    --router_ckpt checkpoints/router/best.pt --top_k auto")


if __name__ == "__main__":
    main()
