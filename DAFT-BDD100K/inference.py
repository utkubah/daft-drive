"""
inference.py
============
DAFT routing for BDD100K object detection.

For each image, loads the specialist model for its condition
(day / night / rain). Falls back to the global model if no
specialist checkpoint exists for that condition.

Condition can be:
  - forced via --condition (e.g. when you know it's a night scene)
  - read per-image from a manifest CSV produced by prepare_data.py
  - defaulted to "global" if unknown

Usage
-----
  # Force a single condition for all images:
  python inference.py --source data/test_imgs/ --condition night

  # Route per-image using manifest metadata:
  python inference.py --source data/test_imgs/ \\
                      --manifest data/bdd100k/manifests/val.csv

  # Force global (no DAFT routing):
  python inference.py --source data/test_imgs/ --condition global
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

CONDITIONS = ["day", "night", "rain"]
CKPT_DIR   = Path("checkpoints")

_MODEL_CACHE: dict[str, YOLO] = {}


def load_model(condition: str) -> YOLO:
    """Load specialist checkpoint, falling back to global if not found."""
    if condition in _MODEL_CACHE:
        return _MODEL_CACHE[condition]

    candidates = [
        CKPT_DIR / condition / "weights" / "best.pt",
        CKPT_DIR / "global"  / "weights" / "best.pt",
        CKPT_DIR / "global.pt",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"No checkpoint found for '{condition}'. Searched:\n"
            + "\n".join(f"  {p}" for p in candidates)
        )

    used = "specialist" if path.parent.parent.name == condition else "global fallback"
    print(f"  [{condition}] loading {used}: {path}")
    model = YOLO(str(path))
    _MODEL_CACHE[condition] = model
    return model


def build_manifest_index(manifest_csv: str) -> dict[str, str]:
    """Return {image_name: condition} from a manifest CSV."""
    index: dict[str, str] = {}
    with open(manifest_csv, newline="") as f:
        for row in csv.DictReader(f):
            cond = row.get("condition") or "global"
            index[row["image_name"]] = cond
    return index


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",    required=True,
                   help="Image file or directory of .jpg files")
    p.add_argument("--condition", default=None,
                   help="Force condition: day | night | rain | global")
    p.add_argument("--manifest",  default=None,
                   help="Manifest CSV for per-image condition routing")
    p.add_argument("--pred_dir",  default="data/predictions")
    p.add_argument("--conf",      type=float, default=0.25)
    p.add_argument("--device",    default="")
    return p.parse_args()


def main():
    args     = get_args()
    source   = Path(args.source)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    imgs     = sorted(source.rglob("*.jpg")) if source.is_dir() else [source]
    manifest = build_manifest_index(args.manifest) if args.manifest else {}

    print(f"Found {len(imgs)} images.")
    for img_path in imgs:
        condition = args.condition or manifest.get(img_path.name, "global")
        model     = load_model(condition)

        results = model.predict(
            source  = str(img_path),
            conf    = args.conf,
            device  = args.device,
            save    = False,
            verbose = False,
        )

        for r in results:
            r.save(filename=str(pred_dir / img_path.name))

        n = len(results[0].boxes) if results[0].boxes else 0
        print(f"  {img_path.name}  [{condition}]  -> {n} detections")

    print("\nDone.")


if __name__ == "__main__":
    main()
