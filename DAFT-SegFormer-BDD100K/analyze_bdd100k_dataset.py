from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


CLASS_NAMES = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic_light",
    7: "traffic_sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
    255: "ignore",
}

PALETTE = {
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
    6: (250, 170, 30),
    7: (220, 220, 0),
    8: (107, 142, 35),
    9: (152, 251, 152),
    10: (70, 130, 180),
    11: (220, 20, 60),
    12: (255, 0, 0),
    13: (0, 0, 142),
    14: (0, 0, 70),
    15: (0, 60, 100),
    16: (0, 80, 100),
    17: (0, 0, 230),
    18: (119, 11, 32),
    255: (0, 0, 0),
}


def parse_args():
    p = argparse.ArgumentParser(description="Deep analysis for BDD100K segmentation + metadata overlap.")
    p.add_argument("--seg-root", type=str, required=True,
                   help="Path like /.../bdd100k_seg/bdd100k/seg")
    p.add_argument("--out-dir", type=str, default="analysis/bdd100k_deep_dive",
                   help="Where outputs will be written")
    p.add_argument("--datasetninja-root", type=str, default="",
                   help="Optional: DatasetNinja root containing train/ann, val/ann, test/ann")
    p.add_argument("--labels-train-json", type=str, default="",
                   help="Optional: bdd100k_labels_images_train.json")
    p.add_argument("--labels-val-json", type=str, default="",
                   help="Optional: bdd100k_labels_images_val.json")
    p.add_argument("--max-viz-per-split", type=int, default=12,
                   help="How many random overlay samples per split")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label_id, color in PALETTE.items():
        out[mask == label_id] = color
    return out


def image_metrics(img: np.ndarray) -> dict:
    img_f = img.astype(np.float32) / 255.0
    rgb_mean = img_f.mean(axis=(0, 1))
    gray = 0.299 * img_f[..., 0] + 0.587 * img_f[..., 1] + 0.114 * img_f[..., 2]
    brightness_mean = float(gray.mean())
    brightness_std = float(gray.std())
    dark_ratio = float((gray < 0.20).mean())
    bright_ratio = float((gray > 0.80).mean())

    rgb_max = img_f.max(axis=2)
    rgb_min = img_f.min(axis=2)
    saturation = rgb_max - rgb_min
    saturation_mean = float(saturation.mean())

    gx = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
    edge_strength = float((gx + gy) / 2.0)

    return {
        "brightness_mean": brightness_mean,
        "brightness_std": brightness_std,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "saturation_mean": saturation_mean,
        "edge_strength": edge_strength,
        "r_mean": float(rgb_mean[0]),
        "g_mean": float(rgb_mean[1]),
        "b_mean": float(rgb_mean[2]),
    }


def safe_json_load(path: Path):
    try:
        with open(path, "r") as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)


def parse_dn_tags(obj: dict) -> dict:
    tags = obj.get("tags", []) or []
    out = {"weather": None, "timeofday": None, "scene": None}
    for tag in tags:
        name = tag.get("name")
        value = tag.get("value")
        if name in out:
            out[name] = value
    return out


def condition_from_meta(timeofday, weather):
    if weather == "rainy":
        return "rain"
    if timeofday == "night":
        return "night"
    if timeofday == "daytime":
        return "day"
    return "skip"


def write_json(path: Path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_label_distribution(df: pd.DataFrame, split: str, out_path: Path):
    sub = df[df["split"] == split].copy()
    sub = sub[sub["label_id"] != 255].sort_values("pixel_count", ascending=False)
    if len(sub) == 0:
        return

    labels = [f"{int(x)}:{CLASS_NAMES.get(int(x), str(int(x)))}" for x in sub["label_id"]]
    values = sub["pixel_fraction"].values

    plt.figure(figsize=(12, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("pixel fraction")
    plt.title(f"Label distribution - {split}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_metric_hist(manifest: pd.DataFrame, metric: str, out_path: Path):
    plt.figure(figsize=(8, 5))
    for split in sorted(manifest["split"].unique()):
        vals = manifest[manifest["split"] == split][metric].dropna().values
        if len(vals) > 0:
            plt.hist(vals, bins=40, alpha=0.55, label=split)
    plt.title(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_random_overlays(manifest: pd.DataFrame, split: str, out_dir: Path, n: int, seed: int):
    sub = manifest[(manifest["split"] == split) & (manifest["is_valid_pair"] == True)].copy()
    if len(sub) == 0:
        return

    rng = random.Random(seed)
    idxs = list(range(len(sub)))
    rng.shuffle(idxs)
    chosen = idxs[:min(n, len(sub))]

    for rank, idx in enumerate(chosen):
        row = sub.iloc[idx]
        img = load_rgb(Path(row["image_path"]))
        mask = load_mask(Path(row["mask_path"]))
        mask_color = colorize_mask(mask)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].imshow(img)
        axes[0].set_title(f"{split} image")

        axes[1].imshow(mask, interpolation="nearest")
        axes[1].set_title("mask raw")

        axes[2].imshow(img)
        axes[2].imshow(mask_color, alpha=0.50)
        axes[2].set_title("mask overlay")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        save_path = out_dir / f"{split}_sample_{rank:02d}_{row['image_name'].replace('.jpg', '')}.png"
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close()


def analyze_segmentation(seg_root: Path, out_dir: Path, max_viz_per_split: int, seed: int):
    summary = {}
    manifest_rows = []
    label_pixel_counter = defaultdict(Counter)
    label_image_counter = defaultdict(Counter)
    shape_counter = defaultdict(Counter)
    bad_pairs = []

    for split in ["train", "val"]:
        img_dir = seg_root / "images" / split
        mask_dir = seg_root / "labels" / split

        images = sorted(img_dir.glob("*.jpg"))
        masks = sorted(mask_dir.glob("*.png"))
        mask_names = {p.name for p in masks}

        matched = 0
        missing_mask = 0
        mismatched_shape = 0

        for img_path in images:
            mask_path = mask_dir / f"{img_path.stem}_train_id.png"
            is_valid_pair = mask_path.exists()

            row = {
                "split": split,
                "image_name": img_path.name,
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "has_mask": bool(mask_path.exists()),
                "is_valid_pair": False,
                "height": None,
                "width": None,
                "mask_height": None,
                "mask_width": None,
                "num_unique_labels": None,
                "labels_present": None,
                "ignore_ratio": None,
                "dominant_label": None,
                "dominant_label_name": None,
            }

            if not mask_path.exists():
                missing_mask += 1
                manifest_rows.append(row)
                continue

            try:
                img = load_rgb(img_path)
                mask = load_mask(mask_path)
            except Exception as e:
                bad_pairs.append({"split": split, "image_name": img_path.name, "error": str(e)})
                manifest_rows.append(row)
                continue

            h, w = img.shape[:2]
            mh, mw = mask.shape[:2]
            row["height"] = h
            row["width"] = w
            row["mask_height"] = mh
            row["mask_width"] = mw

            if (h, w) != (mh, mw):
                mismatched_shape += 1

            labels, counts = np.unique(mask, return_counts=True)
            label_list = [int(x) for x in labels.tolist()]
            label_count_map = {int(k): int(v) for k, v in zip(labels.tolist(), counts.tolist())}

            total_pixels = int(mask.size)
            ignore_pixels = int(label_count_map.get(255, 0))
            ignore_ratio = ignore_pixels / max(total_pixels, 1)

            valid_labels = {k: v for k, v in label_count_map.items() if k != 255}
            if len(valid_labels) > 0:
                dominant_label = max(valid_labels.items(), key=lambda x: x[1])[0]
            else:
                dominant_label = 255

            row["is_valid_pair"] = True
            row["num_unique_labels"] = len(label_list)
            row["labels_present"] = ",".join(map(str, label_list))
            row["ignore_ratio"] = ignore_ratio
            row["dominant_label"] = dominant_label
            row["dominant_label_name"] = CLASS_NAMES.get(dominant_label, str(dominant_label))
            row.update(image_metrics(img))

            manifest_rows.append(row)
            matched += 1
            shape_counter[split][(h, w)] += 1

            present_set = set(label_list)
            for label_id, pixel_count in label_count_map.items():
                label_pixel_counter[split][int(label_id)] += int(pixel_count)
            for label_id in present_set:
                label_image_counter[split][int(label_id)] += 1

        summary[split] = {
            "images_total": len(images),
            "masks_total": len(masks),
            "matched_pairs": matched,
            "missing_mask": missing_mask,
            "mismatched_shape": mismatched_shape,
            "top_shapes": [
                {"shape": f"{k[0]}x{k[1]}", "count": v}
                for k, v in shape_counter[split].most_common(10)
            ],
            "first_mask_names": sorted(list(mask_names))[:5],
        }

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "seg_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    bad_pairs_path = out_dir / "bad_pairs.csv"
    pd.DataFrame(bad_pairs).to_csv(bad_pairs_path, index=False)

    label_rows = []
    for split in ["train", "val"]:
        total_pixels = sum(label_pixel_counter[split].values())
        for label_id, pixel_count in sorted(label_pixel_counter[split].items()):
            label_rows.append({
                "split": split,
                "label_id": label_id,
                "label_name": CLASS_NAMES.get(label_id, str(label_id)),
                "pixel_count": pixel_count,
                "pixel_fraction": pixel_count / max(total_pixels, 1),
                "image_count": label_image_counter[split][label_id],
            })

    label_df = pd.DataFrame(label_rows)
    label_df.to_csv(out_dir / "label_summary.csv", index=False)

    write_json(out_dir / "seg_summary.json", summary)

    plot_label_distribution(label_df, "train", out_dir / "label_distribution_train.png")
    plot_label_distribution(label_df, "val", out_dir / "label_distribution_val.png")

    for metric in ["brightness_mean", "brightness_std", "saturation_mean", "dark_ratio", "edge_strength"]:
        if metric in manifest.columns:
            plot_metric_hist(manifest, metric, out_dir / f"hist_{metric}.png")

    overlay_dir = out_dir / "sample_overlays"
    ensure_dir(overlay_dir)
    save_random_overlays(manifest, "train", overlay_dir, max_viz_per_split, seed)
    save_random_overlays(manifest, "val", overlay_dir, max_viz_per_split, seed + 1)

    return manifest, label_df, summary


def load_datasetninja_metadata(root: Path):
    rows = []
    bad_json = []

    for split in ["train", "val", "test"]:
        ann_dir = root / split / "ann"
        if not ann_dir.exists():
            continue

        for p in sorted(ann_dir.glob("*.json")):
            obj, err = safe_json_load(p)
            if err is not None:
                bad_json.append({"split": split, "file": p.name, "error": err})
                continue

            tags = parse_dn_tags(obj)
            objects = obj.get("objects", []) or []
            class_titles = sorted({x.get("classTitle") for x in objects if x.get("classTitle") is not None})
            geom_types = sorted({x.get("geometryType") for x in objects if x.get("geometryType") is not None})

            rows.append({
                "image_name": p.name.replace(".json", ""),
                "dn_split": split,
                "weather": tags.get("weather"),
                "timeofday": tags.get("timeofday"),
                "scene": tags.get("scene"),
                "condition": condition_from_meta(tags.get("timeofday"), tags.get("weather")),
                "object_count": len(objects),
                "object_classes": ",".join(class_titles),
                "geometry_types": ",".join(geom_types),
                "source": "datasetninja",
            })

    meta_df = pd.DataFrame(rows)
    bad_df = pd.DataFrame(bad_json)
    return meta_df, bad_df


def load_labels_release_metadata(train_json: Path, val_json: Path):
    rows = []

    for src_name, path in [("train_json", train_json), ("val_json", val_json)]:
        if path is None or not path.exists():
            continue
        with open(path, "r") as f:
            records = json.load(f)

        for r in records:
            attrs = r.get("attributes", {}) or {}
            rows.append({
                "image_name": r.get("name"),
                "lr_source": src_name,
                "weather": attrs.get("weather"),
                "timeofday": attrs.get("timeofday"),
                "scene": attrs.get("scene"),
                "condition": condition_from_meta(attrs.get("timeofday"), attrs.get("weather")),
                "source": "labels_release",
            })

    return pd.DataFrame(rows)


def overlap_report(manifest: pd.DataFrame, meta_df: pd.DataFrame, out_dir: Path, prefix: str):
    if len(meta_df) == 0:
        return None

    joined = manifest.merge(meta_df, on="image_name", how="left", suffixes=("", "_meta"))
    joined["has_metadata"] = joined["source"].notna()

    joined.to_csv(out_dir / f"{prefix}_joined.csv", index=False)

    rows = []
    for split in ["train", "val"]:
        sub = joined[joined["split"] == split]
        rows.append({
            "split": split,
            "seg_total": len(sub),
            "match_count": int(sub["has_metadata"].sum()),
            "match_fraction": float(sub["has_metadata"].mean()) if len(sub) > 0 else 0.0,
        })
    overlap_df = pd.DataFrame(rows)
    overlap_df.to_csv(out_dir / f"{prefix}_overlap_summary.csv", index=False)

    usable = joined[(joined["has_metadata"] == True) & (joined["condition"].isin(["day", "night", "rain"]))].copy()
    usable.to_csv(out_dir / f"{prefix}_usable_conditions.csv", index=False)

    if len(usable) > 0:
        counts = (
            usable.groupby(["split", "condition"])
            .size()
            .reset_index(name="count")
            .sort_values(["split", "condition"])
        )
        counts.to_csv(out_dir / f"{prefix}_condition_counts.csv", index=False)
    else:
        counts = pd.DataFrame(columns=["split", "condition", "count"])
        counts.to_csv(out_dir / f"{prefix}_condition_counts.csv", index=False)

    return {
        "joined": joined,
        "usable": usable,
        "overlap_df": overlap_df,
        "condition_counts": counts,
    }


def write_candidate_lists(manifest: pd.DataFrame, out_dir: Path):
    valid = manifest[manifest["is_valid_pair"] == True].copy()
    if len(valid) == 0:
        return

    # These are only ranking heuristics, not labels.
    darkest = valid.sort_values("brightness_mean", ascending=True).head(500)
    lowest_contrast = valid.sort_values("brightness_std", ascending=True).head(500)
    lowest_saturation = valid.sort_values("saturation_mean", ascending=True).head(500)
    edge_low = valid.sort_values("edge_strength", ascending=True).head(500)

    darkest.to_csv(out_dir / "candidates_darkest_500.csv", index=False)
    lowest_contrast.to_csv(out_dir / "candidates_low_contrast_500.csv", index=False)
    lowest_saturation.to_csv(out_dir / "candidates_low_saturation_500.csv", index=False)
    edge_low.to_csv(out_dir / "candidates_low_edge_500.csv", index=False)

    # Simple heuristic scores to help manual review, not automatic truth.
    score_df = valid.copy()

    def minmax(col):
        x = score_df[col].values.astype(np.float32)
        lo, hi = float(np.min(x)), float(np.max(x))
        if math.isclose(lo, hi):
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    b = minmax("brightness_mean")
    c = minmax("brightness_std")
    s = minmax("saturation_mean")
    d = minmax("dark_ratio")

    score_df["night_score"] = 0.45 * (1 - b) + 0.20 * (1 - c) + 0.20 * (1 - s) + 0.15 * d
    score_df["rain_like_score"] = 0.35 * (1 - c) + 0.25 * (1 - s) + 0.20 * d + 0.20 * (1 - b)
    score_df["day_like_score"] = 0.45 * b + 0.25 * c + 0.15 * s + 0.15 * (1 - d)

    score_df.sort_values("night_score", ascending=False).head(500).to_csv(
        out_dir / "candidates_night_score_top500.csv", index=False
    )
    score_df.sort_values("rain_like_score", ascending=False).head(500).to_csv(
        out_dir / "candidates_rain_score_top500.csv", index=False
    )
    score_df.sort_values("day_like_score", ascending=False).head(500).to_csv(
        out_dir / "candidates_day_score_top500.csv", index=False
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    seg_root = Path(args.seg_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)

    print("SEG_ROOT:", seg_root)
    print("OUT_DIR :", out_dir)

    manifest, label_df, seg_summary = analyze_segmentation(
        seg_root=seg_root,
        out_dir=out_dir,
        max_viz_per_split=args.max_viz_per_split,
        seed=args.seed,
    )

    write_candidate_lists(manifest, out_dir)

    final_summary = {
        "segmentation": seg_summary,
        "metadata_sources": {},
    }

    if args.datasetninja_root:
        dn_root = Path(args.datasetninja_root).expanduser().resolve()
        dn_meta, dn_bad = load_datasetninja_metadata(dn_root)
        dn_meta.to_csv(out_dir / "datasetninja_metadata.csv", index=False)
        dn_bad.to_csv(out_dir / "datasetninja_bad_json.csv", index=False)

        report = overlap_report(manifest, dn_meta, out_dir, prefix="datasetninja")
        if report is not None:
            final_summary["metadata_sources"]["datasetninja"] = {
                "rows": len(dn_meta),
                "bad_json_files": len(dn_bad),
                "overlap_summary": report["overlap_df"].to_dict(orient="records"),
                "usable_condition_counts": report["condition_counts"].to_dict(orient="records"),
            }
        print("DatasetNinja metadata rows:", len(dn_meta))
        print("DatasetNinja bad json files:", len(dn_bad))

    if args.labels_train_json or args.labels_val_json:
        train_json = Path(args.labels_train_json).expanduser().resolve() if args.labels_train_json else None
        val_json = Path(args.labels_val_json).expanduser().resolve() if args.labels_val_json else None
        lr_meta = load_labels_release_metadata(train_json, val_json)
        lr_meta.to_csv(out_dir / "labels_release_metadata.csv", index=False)

        report = overlap_report(manifest, lr_meta, out_dir, prefix="labels_release")
        if report is not None:
            final_summary["metadata_sources"]["labels_release"] = {
                "rows": len(lr_meta),
                "overlap_summary": report["overlap_df"].to_dict(orient="records"),
                "usable_condition_counts": report["condition_counts"].to_dict(orient="records"),
            }
        print("Labels-release metadata rows:", len(lr_meta))

    write_json(out_dir / "final_summary.json", final_summary)

    print("\nDone.")
    print("Main files written:")
    for name in [
        "seg_summary.json",
        "seg_manifest.csv",
        "label_summary.csv",
        "final_summary.json",
        "sample_overlays/",
        "candidates_darkest_500.csv",
        "candidates_low_contrast_500.csv",
        "candidates_low_saturation_500.csv",
        "candidates_night_score_top500.csv",
        "candidates_rain_score_top500.csv",
        "candidates_day_score_top500.csv",
    ]:
        print("-", out_dir / name)


if __name__ == "__main__":
    main()