"""
patch_dawn_dusk.py
==================
Patch the per-condition image-list .txt files so that city_night and
highway_night training sets also include dawn/dusk images from their
day counterparts.

This fixes poor dawn/dusk detection without re-downloading the dataset.
Run this once on the cluster after which you only need to retrain the
two night specialists:

  python patch_dawn_dusk.py
  python train.py --data data/bdd100k/yolo/city_night.yaml \\
                  --weights checkpoints/global/weights/best.pt \\
                  --name city_night --mosaic 0 --cos_lr --epochs 40 --patience 15
  python train.py --data data/bdd100k/yolo/highway_night.yaml \\
                  --weights checkpoints/global/weights/best.pt \\
                  --name highway_night --mosaic 0 --cos_lr --epochs 40 --patience 15
"""

import csv
from pathlib import Path

MANIFEST_DIR = Path("data/bdd100k/manifests")
YOLO_DIR     = Path("data/bdd100k/yolo")

NIGHT_GETS_DAWN_DUSK = {
    "city_night":    "city_day",
    "highway_night": "highway_day",
}

for split in ["train", "val"]:
    manifest = MANIFEST_DIR / f"{split}.csv"
    if not manifest.exists():
        print(f"  SKIP: {manifest} not found")
        continue

    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    for night_cond, day_cond in NIGHT_GETS_DAWN_DUSK.items():
        txt_path = YOLO_DIR / f"{night_cond}.{split}.txt"
        if not txt_path.exists():
            print(f"  SKIP: {txt_path} not found")
            continue

        # Existing paths in the night list
        existing = set(txt_path.read_text().splitlines())

        # Dawn/dusk images from the day counterpart
        dawn_dusk = [
            r["image_path"] for r in rows
            if r.get("condition") == day_cond
            and r.get("is_ambiguous") == "1"
            and r["image_path"] not in existing
        ]

        if not dawn_dusk:
            print(f"  {night_cond}.{split}: no new dawn/dusk images to add")
            continue

        # Append and rewrite
        updated = sorted(existing | set(dawn_dusk))
        txt_path.write_text("\n".join(updated) + "\n")
        print(f"  {night_cond}.{split}: +{len(dawn_dusk)} dawn/dusk images "
              f"({len(existing)} → {len(updated)} total)")

print("\nDone. Now retrain city_night and highway_night specialists only.")
