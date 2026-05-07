"""
train.py
========
Fine-tune YOLOv8 on BDD100K object detection.

Mirrors the two-stage DAFT pipeline:
  Stage 1 -- Global:   train on all conditions -> checkpoints/global/
  Stage 2 -- DAFT:     train one specialist per condition -> checkpoints/{condition}/

Starting weights can be:
  - pretrained ultralytics weights (e.g. yolov8s.pt)  [default]
  - distilled backbone from distill.py                 [recommended]
  - a previous checkpoint for resuming

Checkpoints are saved by ultralytics under:
  checkpoints/<name>/weights/best.pt    <- best validation mAP
  checkpoints/<name>/weights/last.pt    <- most recent epoch

Usage
-----
  # Stage 1: global fine-tune (from distilled backbone)
  python train.py --data data/bdd100k/yolo/dataset.yaml \\
                  --weights checkpoints/distilled/distilled.pt --name global

  # Stage 2: DAFT specialists (one per condition, mosaic=0 to avoid mixing conditions)
  python train.py --data data/bdd100k/yolo/city_day.yaml \\
                  --weights checkpoints/global/weights/best.pt --name city_day --mosaic 0
  python train.py --data data/bdd100k/yolo/city_night.yaml \\
                  --weights checkpoints/global/weights/best.pt --name city_night --mosaic 0
  python train.py --data data/bdd100k/yolo/highway_day.yaml \\
                  --weights checkpoints/global/weights/best.pt --name highway_day --mosaic 0
  python train.py --data data/bdd100k/yolo/highway_night.yaml \\
                  --weights checkpoints/global/weights/best.pt --name highway_night --mosaic 0
  python train.py --data data/bdd100k/yolo/residential.yaml \\
                  --weights checkpoints/global/weights/best.pt --name residential --mosaic 0
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",     required=True,
                   help="YOLO dataset.yaml (use condition yaml for specialists)")
    p.add_argument("--weights",  default="yolov8s.pt",
                   help="Starting weights: pretrained, distilled, or checkpoint")
    p.add_argument("--name",     required=True,
                   help="Run name — saved to checkpoints/<name>/")
    p.add_argument("--epochs",   type=int,   default=30)
    p.add_argument("--batch",    type=int,   default=16)
    p.add_argument("--imgsz",    type=int,   default=640)
    p.add_argument("--lr",       type=float, default=5e-5,
                   help="Initial LR (paper: 5e-5)")
    p.add_argument("--patience", type=int,   default=10,
                   help="Early stopping patience (epochs without mAP improvement)")
    p.add_argument("--mosaic",   type=float, default=1.0,
                   help="Mosaic augmentation probability (set 0.0 for specialists "
                        "to avoid mixing images from different driving conditions)")
    p.add_argument("--workers",  type=int,   default=4,
                   help="DataLoader worker threads")
    p.add_argument("--cos_lr",   action="store_true",
                   help="Use cosine LR schedule instead of linear warmup+decay")
    p.add_argument("--device",   default="",
                   help="Device: '' = auto, 'cpu', '0', '0,1', ...")
    return p.parse_args()


def main():
    args  = get_args()
    model = YOLO(args.weights)

    model.train(
        data      = args.data,
        epochs    = args.epochs,
        batch     = args.batch,
        imgsz     = args.imgsz,
        lr0       = args.lr,
        project   = str(Path("checkpoints").resolve()),
        name      = args.name,
        device    = args.device,
        patience  = args.patience,
        mosaic    = args.mosaic,
        workers   = args.workers,
        cos_lr    = args.cos_lr,
        val       = True,
        save      = True,
        exist_ok  = True,
    )


if __name__ == "__main__":
    main()
