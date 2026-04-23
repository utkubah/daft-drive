from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Merge BDD100K segmentation files with image-level metadata JSONs")
    p.add_argument("--seg_root", type=str, required=True, help="Path to bdd100k_seg/bdd100k/seg")
    p.add_argument("--train_json", type=str, required=True, help="Path to bdd100k_labels_images_train.json")
    p.add_argument("--val_json", type=str, required=True, help="Path to bdd100k_labels_images_val.json")
    p.add_argument("--out", type=str, default="data/bdd100k_npz_200/splits/manifest_with_metadata.csv")
    return p.parse_args()


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def build_meta_index(records):
    out = {}
    for r in records:
        name = r.get("name")
        attrs = r.get("attributes", {}) or {}
        out[name] = {
            "weather": attrs.get("weather"),
            "timeofday": attrs.get("timeofday"),
            "scene": attrs.get("scene"),
        }
    return out


def map_condition(timeofday, weather):
    if weather == "rainy":
        return "rain"
    if timeofday == "night":
        return "night"
    if timeofday == "daytime":
        return "day"
    return "skip"


def main():
    args = parse_args()

    seg_root = Path(args.seg_root).expanduser().resolve()
    train_json = Path(args.train_json).expanduser().resolve()
    val_json = Path(args.val_json).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    train_meta = build_meta_index(load_json(train_json))
    val_meta = build_meta_index(load_json(val_json))

    rows = []

    for split, meta_index in [("train", train_meta), ("val", val_meta)]:
        img_dir = seg_root / "images" / split
        lbl_dir = seg_root / "labels" / split

        for img_path in sorted(img_dir.glob("*.jpg")):
            name = img_path.name
            lbl_path = lbl_dir / f"{img_path.stem}_train_id.png"
            meta = meta_index.get(name)

            rows.append({
                "split": split,
                "image_name": name,
                "image_path": str(img_path),
                "mask_path": str(lbl_path),
                "has_mask": lbl_path.exists(),
                "has_metadata": meta is not None,
                "weather": None if meta is None else meta.get("weather"),
                "timeofday": None if meta is None else meta.get("timeofday"),
                "scene": None if meta is None else meta.get("scene"),
                "condition": None if meta is None else map_condition(meta.get("timeofday"), meta.get("weather")),
            })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("saved:", out_path)
    print("rows:", len(df))
    print("\nmask coverage:")
    print(df["has_mask"].value_counts(dropna=False).to_string())

    print("\nmetadata coverage:")
    print(df["has_metadata"].value_counts(dropna=False).to_string())

    print("\ncondition counts:")
    print(df["condition"].fillna("MISSING").value_counts().to_string())


if __name__ == "__main__":
    main()