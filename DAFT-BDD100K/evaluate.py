"""
evaluate.py
===========
Evaluate a YOLO checkpoint and report mAP50 / mAP50-95.

Usage
-----
  # Global model on all data:
  python evaluate.py --weights checkpoints/global/weights/best.pt \\
                     --data data/bdd100k/yolo/dataset.yaml

  # Specialist on its own condition:
  python evaluate.py --weights checkpoints/day/weights/best.pt \\
                     --data data/bdd100k/yolo/day.yaml

  # Compare global vs specialist side-by-side:
  python evaluate.py --weights checkpoints/global/weights/best.pt \\
                     --data data/bdd100k/yolo/night.yaml   # global on night data
"""

import argparse
from ultralytics import YOLO


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data",    required=True)
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--device",  default="")
    return p.parse_args()


def main():
    args    = get_args()
    model   = YOLO(args.weights)
    metrics = model.val(
        data   = args.data,
        imgsz  = args.imgsz,
        batch  = args.batch,
        device = args.device,
    )

    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
