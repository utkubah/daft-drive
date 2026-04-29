from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


SEG_ROOT = Path("/Users/berkay/datasets/bdd100k_seg/bdd100k/seg")
TRAIN_JSON = Path("/Users/berkay/Downloads/archive (2) 2/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json")
VAL_JSON = Path("/Users/berkay/Downloads/archive (2) 2/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json")
OUT_DIR = Path("data/conditions_v1")


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = {}
    meta.update(build_meta_index(load_json(TRAIN_JSON)))
    meta.update(build_meta_index(load_json(VAL_JSON)))

    rows = []

    for split in ["train", "val"]:
        img_dir = SEG_ROOT / "images" / split
        lbl_dir = SEG_ROOT / "labels" / split

        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / f"{img_path.stem}_train_id.png"
            m = meta.get(img_path.name)

            rows.append({
                "split": split,
                "image_name": img_path.name,
                "image_path": str(img_path),
                "mask_path": str(lbl_path),
                "has_mask": lbl_path.exists(),
                "has_metadata": m is not None,
                "weather": None if m is None else m.get("weather"),
                "timeofday": None if m is None else m.get("timeofday"),
                "scene": None if m is None else m.get("scene"),
                "condition": None if m is None else map_condition(m.get("timeofday"), m.get("weather")),
            })

    df = pd.DataFrame(rows)

    all_manifest = OUT_DIR / "manifest_all_v1.csv"
    df.to_csv(all_manifest, index=False)

    usable = df[
        (df["has_mask"] == True) &
        (df["has_metadata"] == True) &
        (df["condition"].isin(["day", "night", "rain"]))
    ].copy()

    usable_manifest = OUT_DIR / "manifest_usable_v1.csv"
    usable.to_csv(usable_manifest, index=False)

    print("saved:", all_manifest)
    print("saved:", usable_manifest)
    print("\nusable counts by split + condition:")
    print(usable.groupby(["split", "condition"]).size().to_string())

    for split in ["train", "val"]:
        for cond in ["day", "night", "rain"]:
            sub = usable[(usable["split"] == split) & (usable["condition"] == cond)].copy()
            out = OUT_DIR / f"{cond}.{split}.v1.csv"
            sub.to_csv(out, index=False)
            print("saved:", out, "| rows:", len(sub))


if __name__ == "__main__":
    main()