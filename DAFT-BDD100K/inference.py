"""
inference.py
============
DAFT routing for BDD100K object detection.

For each image, loads the specialist model for its condition.
Conditions: city_day / city_night / highway_day / highway_night / residential
Falls back to the global model if no specialist checkpoint exists.

Condition is read per-image from the manifest CSV (scene + timeofday columns)
produced by prepare_data.py, or forced via --condition.

Usage
-----
  # Route per-image using manifest metadata:
  python inference.py --source data/bdd100k/yolo/images/val/ \\
                      --manifest data/bdd100k/manifests/val.csv

  # Force a single condition for all images:
  python inference.py --source data/test_imgs/ --condition city_night

  # Force global (no DAFT routing):
  python inference.py --source data/test_imgs/ --condition global
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO
from router import MetadataRouter, CONDITIONS

CKPT_DIR         = Path("checkpoints")
ALT_CKPT_DIR     = Path("runs/detect/checkpoints")

_MODEL_CACHE: dict[str, YOLO] = {}
_router = MetadataRouter()


def load_model(condition: str) -> YOLO:
    """Load specialist checkpoint, falling back to global if not found."""
    if condition in _MODEL_CACHE:
        return _MODEL_CACHE[condition]

    candidates = [
        CKPT_DIR     / condition / "weights" / "best.pt",
        ALT_CKPT_DIR / condition / "weights" / "best.pt",
        CKPT_DIR     / "global"  / "weights" / "best.pt",
        ALT_CKPT_DIR / "global"  / "weights" / "best.pt",
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


def build_manifest_index(manifest_csv: str) -> dict[str, dict]:
    """Return {image_name: {scene, timeofday, condition}} from manifest CSV."""
    index: dict[str, dict] = {}
    with open(manifest_csv, newline="") as f:
        for row in csv.DictReader(f):
            index[row["image_name"]] = {
                "scene":     row.get("scene", ""),
                "timeofday": row.get("timeofday", ""),
                "condition": row.get("condition", ""),
            }
    return index


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",    required=True,
                   help="Image file or directory of .jpg files")
    p.add_argument("--condition", default=None,
                   help="Force condition for all images (skips routing)")
    p.add_argument("--manifest",  default=None,
                   help="Manifest CSV for per-image metadata routing")
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
        # ── Routing ──────────────────────────────────────────────────────
        if args.condition:
            # Hard override: user forced a single condition
            condition = args.condition
            label     = condition
        elif img_path.name in manifest:
            meta = manifest[img_path.name]
            # Use stored condition if clean; otherwise re-derive via router
            if meta["condition"] and meta["condition"] in CONDITIONS:
                condition = meta["condition"]
            else:
                weights   = _router.weights(meta["scene"], meta["timeofday"])
                selected  = _router.select(weights)
                condition = selected[0][0]   # hard routing: top-1
            label = condition
        else:
            condition = "global"
            label     = "global (no metadata)"

        model   = load_model(condition)
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
        print(f"  {img_path.name}  [{label}]  -> {n} detections")

    print("\nDone.")


if __name__ == "__main__":
    main()
