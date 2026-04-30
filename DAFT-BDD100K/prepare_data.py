"""
prepare_data.py
===============
Load BDD100K via FiftyOne (HuggingFace), export to YOLO format, and
build per-condition splits for DAFT specialist training.

Data source: FiftyOne loads dgural/bdd100k from HuggingFace.
Set --hf_token (or HF_TOKEN env var) to avoid anonymous rate limits.

Conditions (from BDD100K metadata):
  day   -- timeofday == "daytime"
  night -- timeofday == "night"
  rain  -- weather   == "rainy"

Outputs
-------
  data/bdd100k/
    classes.txt
    yolo/
      images/{train,val}/       symlinked images
      labels/{train,val}/       YOLO .txt label files
      dataset.yaml              full dataset (all conditions)
      {condition}.train.txt     image-list files for specialist training
      {condition}.val.txt
      {condition}.yaml          per-condition dataset configs
    manifests/
      {split}.csv               all usable samples per split
      {condition}.{split}.csv   per-condition subsets

Usage
-----
  python prepare_data.py
  python prepare_data.py --max_samples 5000        # limit per split for testing
  python prepare_data.py --hf_token hf_xxx...      # avoid HF rate limits
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.utils.huggingface as fouh

HUB_NAME   = "dgural/bdd100k"
HF_SPLITS  = ["train", "validation"]
CONDITIONS = ["city_day", "city_night", "highway_day", "highway_night", "residential"]
OUT_DIR    = Path("data/bdd100k")

MANIFEST_FIELDS = [
    "image_name", "image_path", "yolo_label",
    "timeofday", "weather", "scene", "condition", "num_boxes",
]


def get_label(sample, field: str) -> str | None:
    val = sample.get_field(field)
    return getattr(val, "label", None) if val else None


def map_condition(scene: str | None, timeofday: str | None) -> str | None:
    scene     = (scene     or "").lower().strip()
    timeofday = (timeofday or "").lower().strip()

    if "city" in scene:
        return "city_night" if timeofday == "night" else "city_day"
    if "highway" in scene:
        return "highway_night" if timeofday == "night" else "highway_day"
    if "residential" in scene:
        return "residential"
    return None   # tunnel, parking lot, gas station, undefined


def build_class_index(dataset) -> dict[str, int]:
    labels: set[str] = set()
    for sample in dataset:
        dets = sample.get_field("detections")
        if dets:
            for d in dets.detections:
                labels.add(d.label)
    return {cls: i for i, cls in enumerate(sorted(labels))}


def yolo_line(det, class_to_idx: dict) -> str | None:
    if det.label not in class_to_idx:
        return None
    x, y, w, h = det.bounding_box   # FiftyOne: normalized top-left [x, y, w, h]
    return f"{class_to_idx[det.label]} {x+w/2:.6f} {y+h/2:.6f} {w:.6f} {h:.6f}"


def process_split(dataset, split: str, yolo_dir: Path, class_to_idx: dict) -> list[dict]:
    """Export one dataset split to YOLO format. Returns manifest rows."""
    yolo_split = "val" if split == "validation" else split
    img_out = yolo_dir / "images" / yolo_split
    lbl_out = yolo_dir / "labels" / yolo_split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for sample in dataset:
        img_path = Path(sample.filepath)
        dets     = sample.get_field("detections")
        if not dets:
            continue

        lines = [yolo_line(d, class_to_idx) for d in dets.detections]
        lines = [l for l in lines if l]
        if not lines:
            continue

        dst = img_out / img_path.name
        if not dst.exists():
            try:
                dst.symlink_to(img_path.resolve())
            except OSError:
                shutil.copy2(img_path, dst)

        lbl_path = lbl_out / f"{img_path.stem}.txt"
        lbl_path.write_text("\n".join(lines))

        timeofday = get_label(sample, "timeofday")
        weather   = get_label(sample, "weather")
        scene     = get_label(sample, "scene")
        rows.append({
            "image_name": img_path.name,
            "image_path": str(dst),
            "yolo_label":  str(lbl_path),
            "timeofday":  timeofday,
            "weather":    weather,
            "scene":      scene,
            "condition":  map_condition(scene, timeofday),
            "num_boxes":  len(lines),
        })

    print(f"  {split}: {len(rows)} usable samples")
    return rows


def write_yaml(path: Path, train_key: str, val_key: str, yolo_dir: Path, classes: list[str]):
    lines = [
        f"path: {yolo_dir.resolve()}",
        f"train: {train_key}",
        f"val:   {val_key}",
        f"nc: {len(classes)}",
        "names:",
    ] + [f"  {i}: {json.dumps(c)}" for i, c in enumerate(classes)]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples", type=int, default=10000,
                    help="Max samples per split (0 = all)")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"),
                    help="HuggingFace token to avoid rate limits (or set HF_TOKEN env var)")
    args = ap.parse_args()

    if not args.hf_token:
        print("WARNING: no --hf_token set. Anonymous HF downloads may hit rate limits.")
        print("  Get a free token at https://huggingface.co/settings/tokens")

    yolo_dir = Path(args.out_dir) / "yolo"
    man_dir  = Path(args.out_dir) / "manifests"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    max_s = args.max_samples or None
    class_to_idx: dict[str, int] | None = None
    all_rows: dict[str, list[dict]] = {}

    load_kwargs = {}
    if args.hf_token:
        load_kwargs["token"] = args.hf_token

    for split in HF_SPLITS:
        print(f"\nLoading {split} from FiftyOne / HuggingFace...")
        dataset = fouh.load_from_hub(
            HUB_NAME,
            split=split,
            max_samples=max_s,
            name=f"bdd100k_{split}",
            overwrite=True,
            **load_kwargs,
        )

        if class_to_idx is None:
            print("  Building class index...")
            class_to_idx = build_class_index(dataset)
            print(f"  {len(class_to_idx)} classes found")
            (Path(args.out_dir) / "classes.txt").write_text(
                "\n".join(sorted(class_to_idx, key=class_to_idx.get)) + "\n"
            )

        all_rows[split] = process_split(dataset, split, yolo_dir, class_to_idx)
        fo.delete_dataset(f"bdd100k_{split}")

    classes = sorted(class_to_idx, key=class_to_idx.get)

    # full dataset.yaml (all conditions)
    write_yaml(yolo_dir / "dataset.yaml", "images/train", "images/val", yolo_dir, classes)

    # per-condition: image-list .txt files + condition-specific yamls
    for cond in CONDITIONS:
        for split in HF_SPLITS:
            yolo_split = "val" if split == "validation" else split
            sub = [r for r in all_rows[split] if r["condition"] == cond]
            txt = yolo_dir / f"{cond}.{yolo_split}.txt"
            txt.write_text("\n".join(r["image_path"] for r in sub) + "\n")
            print(f"  {cond}.{yolo_split}: {len(sub)} samples")
        write_yaml(
            yolo_dir / f"{cond}.yaml",
            f"{cond}.train.txt", f"{cond}.val.txt",
            yolo_dir, classes,
        )

    # manifests (all + per-condition CSVs)
    for split, rows in all_rows.items():
        yolo_split = "val" if split == "validation" else split
        for cond in [None] + CONDITIONS:
            sub  = rows if cond is None else [r for r in rows if r["condition"] == cond]
            name = f"{yolo_split}.csv" if cond is None else f"{cond}.{yolo_split}.csv"
            with open(man_dir / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
                w.writeheader()
                w.writerows(sub)

    print(f"\nDataset ready in {Path(args.out_dir).resolve()}")
    print("Next steps:")
    print("  python distill.py --img_dir data/bdd100k/yolo/images/train")
    print("  python train.py --data data/bdd100k/yolo/dataset.yaml --name global")


if __name__ == "__main__":
    main()
