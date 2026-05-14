"""
distill.py
==========
Backbone feature distillation: YOLOv8m (teacher) -> YOLOv8s (student).

The SPPF layer at the end of the backbone is hooked in both models.
A learned 1x1 projection aligns the student's channel dimension to
the teacher's, then MSE loss trains the student backbone to produce
similar feature maps.

Only the backbone is updated; detection heads remain at their
pretrained values and are fine-tuned separately in train.py.

Saved weights can be loaded in train.py via --weights.

Usage
-----
  python distill.py --img_dir data/bdd100k/yolo/images/train
  python distill.py --teacher yolov8m.pt --student yolov8s.pt --epochs 20
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO
from ultralytics.nn.modules import SPPF


class ImageFolder(Dataset):
    """Minimal image loader — no labels needed for distillation."""

    def __init__(self, img_dir: str, img_size: int = 640):
        self.paths    = sorted(Path(img_dir).rglob("*.jpg"))
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        img = cv2.imread(str(self.paths[i]))
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def find_sppf_idx(model: nn.Module) -> int:
    """Return the index of the SPPF layer in model.model (the backbone end)."""
    for i, m in enumerate(model.model):
        if isinstance(m, SPPF):
            return i
    raise RuntimeError("SPPF layer not found — is this a YOLOv8 model?")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher",  default="yolov8m.pt",
                   help="Teacher weights (larger model)")
    p.add_argument("--student",  default="yolov8s.pt",
                   help="Student weights (smaller model)")
    p.add_argument("--img_dir",  default="data/bdd100k/yolo/images/train")
    p.add_argument("--out_dir",  default="checkpoints/distilled")
    p.add_argument("--epochs",   type=int,   default=20)
    p.add_argument("--batch",    type=int,   default=16)
    p.add_argument("--lr",       type=float, default=1e-4)
    p.add_argument("--workers",  type=int,   default=2)
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = get_args()
    dev  = args.device
    if dev.isdigit():
        dev = f"cuda:{dev}"
    device  = torch.device(dev)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load models (nn.Module level) ---
    teacher_nn = YOLO(args.teacher).model.to(device).eval()
    student_nn = YOLO(args.student).model.to(device).train()

    for p in teacher_nn.parameters():
        p.requires_grad_(False)

    # freeze student detection heads — only train backbone
    sppf_idx = find_sppf_idx(student_nn)
    for i, m in enumerate(student_nn.model):
        if i > sppf_idx:
            for p in m.parameters():
                p.requires_grad_(False)

    # We use forward hooks to capture the SPPF output tensors without modifying
    # the model's forward() method. The hook writes into a one-element list so
    # the inner closure can update it across calls (plain variables can't be
    # rebound inside a closure in Python 2-style, but a mutable container works).
    teacher_feat: list[torch.Tensor | None] = [None]
    student_feat: list[torch.Tensor | None] = [None]

    def make_hook(store: list):
        def hook(_, __, output):
            store[0] = output
        return hook

    t_handle = teacher_nn.model[sppf_idx].register_forward_hook(make_hook(teacher_feat))
    s_handle = student_nn.model[sppf_idx].register_forward_hook(make_hook(student_feat))

    # --- projection: student channels -> teacher channels ---
    t_ch = teacher_nn.model[sppf_idx].cv2.conv.out_channels
    s_ch = student_nn.model[sppf_idx].cv2.conv.out_channels
    proj  = nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False).to(device)

    print(f"  Teacher SPPF channels: {t_ch}")
    print(f"  Student SPPF channels: {s_ch}  (projected -> {t_ch})")

    trainable = (
        list(filter(lambda p: p.requires_grad, student_nn.parameters()))
        + list(proj.parameters())
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    # --- data ---
    dataset = ImageFolder(args.img_dir)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                         num_workers=args.workers,
                         pin_memory=device.type == "cuda")
    print(f"  {len(dataset)} images | device={args.device}")

    # --- distillation loop ---
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for imgs in loader:
            imgs = imgs.to(device)

            with torch.no_grad():
                teacher_nn(imgs)

            student_nn(imgs)

            loss = F.mse_loss(proj(student_feat[0]), teacher_feat[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        print(f"  Epoch {epoch:3d}/{args.epochs}  distill_loss={total/len(loader):.5f}")

    # --- save distilled student ---
    # Remove hooks before saving — hooks can't be pickled
    t_handle.remove()
    s_handle.remove()
    out_path = out_dir / "distilled.pt"
    torch.save({"model": deepcopy(student_nn).half()}, str(out_path))
    print(f"\nSaved distilled student -> {out_path}")
    print("Next: python train.py --weights checkpoints/distilled/distilled.pt "
          "--data data/bdd100k/yolo/dataset.yaml --name global")


if __name__ == "__main__":
    main()
