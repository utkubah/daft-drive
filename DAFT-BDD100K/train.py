"""
train.py
========
Fine-tune YOLOv8 on BDD100K object detection.

Mirrors the two-stage DAFT pipeline:
  Stage 1 -- Global:   train on all conditions -> checkpoints/global/
  Stage 2 -- DAFT:     train one specialist per condition -> checkpoints/{condition}/

Starting weights can be:
  - pretrained ultralytics weights (e.g. yolov8n.pt)  [default]
  - distilled backbone from distill.py                 [recommended]
  - a previous checkpoint for resuming

Checkpoints are saved by ultralytics under:
  checkpoints/<name>/weights/best.pt    <- best validation mAP
  checkpoints/<name>/weights/last.pt    <- most recent epoch

Usage
-----
  # Stage 1: global fine-tune
  python train.py --data data/bdd100k/yolo/dataset.yaml --name global

  # Stage 1 from distilled backbone:
  python train.py --data data/bdd100k/yolo/dataset.yaml \\
                  --weights checkpoints/distilled/distilled.pt --name global

  # Stage 2: DAFT specialist (one per condition)
  python train.py --data data/bdd100k/yolo/day.yaml \\
                  --weights checkpoints/global/weights/best.pt --name day
  python train.py --data data/bdd100k/yolo/night.yaml \\
                  --weights checkpoints/global/weights/best.pt --name night
  python train.py --data data/bdd100k/yolo/rain.yaml \\
                  --weights checkpoints/global/weights/best.pt --name rain
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",     required=True,
                   help="YOLO dataset.yaml (use condition yaml for specialists)")
    p.add_argument("--weights",  default="yolov8n.pt",
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
        val       = True,
        save      = True,
        exist_ok  = True,
    )


if __name__ == "__main__":
    main()
