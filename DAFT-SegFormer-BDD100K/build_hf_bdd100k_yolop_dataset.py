from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path
from PIL import Image


PROJECT_ROOT = Path.cwd()

SRC_ROOT = PROJECT_ROOT / "data" / "hf_bdd100k_od"
MANIFESTS_DIR = SRC_ROOT / "manifests"
YOLO_DIR = SRC_ROOT / "yolo"

HF_REPO_ROOT = Path.home() / "datasets" / "hf_bdd100k_full_repo"

OUT_ROOT = PROJECT_ROOT / "data" / "hf_bdd100k_yolop_global"
IMAGES_OUT = OUT_ROOT / "images"
LABELS_OUT = OUT_ROOT / "labels"
MASKS_OUT = OUT_ROOT / "masks"
LANES_OUT = OUT_ROOT / "lanes"
META_OUT = OUT_ROOT / "metadata"

SPLITS = ["train", "val", "test"]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def load_rows(csv_path: Path):
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def make_dummy_lane(mask_path: Path, out_path: Path):
    if out_path.exists():
        return

    if mask_path.exists():
        with Image.open(mask_path) as im:
            w, h = im.size
    else:
        w, h = 1280, 720

    dummy = Image.new("L", (w, h), 0)
    dummy.save(out_path)


def main():
    if not MANIFESTS_DIR.exists():
        raise FileNotFoundError(f"Missing manifests dir: {MANIFESTS_DIR}")
    if not YOLO_DIR.exists():
        raise FileNotFoundError(f"Missing YOLO dir: {YOLO_DIR}")
    if not HF_REPO_ROOT.exists():
        raise FileNotFoundError(f"Missing HF repo root: {HF_REPO_ROOT}")

    for base in [IMAGES_OUT, LABELS_OUT, MASKS_OUT, LANES_OUT, META_OUT]:
        ensure_dir(base)

    summary = []

    for split in SPLITS:
        rows = load_rows(MANIFESTS_DIR / f"{split}.csv")

        img_dir = IMAGES_OUT / split
        lbl_dir = LABELS_OUT / split
        mask_dir = MASKS_OUT / split
        lane_dir = LANES_OUT / split

        for d in [img_dir, lbl_dir, mask_dir, lane_dir]:
            ensure_dir(d)

        meta_csv = META_OUT / f"{split}.csv"
        with open(meta_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_name",
                    "image_path",
                    "scene",
                    "scene_group",
                    "timeofday",
                    "weather",
                    "num_detections",
                    "has_drivable",
                ],
            )
            writer.writeheader()

            n = 0
            n_mask = 0
            n_lane = 0

            for row in rows:
                image_name = row["image_name"]
                stem = Path(image_name).stem

                src_img = Path(row["image_path"])
                src_lbl = YOLO_DIR / "labels" / split / f"{stem}.txt"
                src_mask = HF_REPO_ROOT / "fields" / "drivable" / f"{stem}.png"

                dst_img = img_dir / image_name
                dst_lbl = lbl_dir / f"{stem}.txt"
                dst_mask = mask_dir / f"{stem}.png"
                dst_lane = lane_dir / f"{stem}.png"

                if not src_img.exists():
                    raise FileNotFoundError(f"Missing image: {src_img}")
                if not src_lbl.exists():
                    raise FileNotFoundError(f"Missing label: {src_lbl}")
                if not src_mask.exists():
                    raise FileNotFoundError(f"Missing drivable mask: {src_mask}")

                link_or_copy(src_img, dst_img)
                link_or_copy(src_lbl, dst_lbl)
                link_or_copy(src_mask, dst_mask)
                make_dummy_lane(src_mask, dst_lane)

                writer.writerow(
                    {
                        "image_name": row["image_name"],
                        "image_path": str(dst_img),
                        "scene": row["scene"],
                        "scene_group": row["scene_group"],
                        "timeofday": row["timeofday"],
                        "weather": row["weather"],
                        "num_detections": row["num_detections"],
                        "has_drivable": row["has_drivable"],
                    }
                )

                n += 1
                n_mask += int(dst_mask.exists())
                n_lane += int(dst_lane.exists())

        summary.append(
            {
                "split": split,
                "rows": len(rows),
                "images": len(list(img_dir.glob("*.jpg"))),
                "labels": len(list(lbl_dir.glob("*.txt"))),
                "masks": len(list(mask_dir.glob("*.png"))),
                "lanes": len(list(lane_dir.glob("*.png"))),
            }
        )

    summary_csv = OUT_ROOT / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "rows", "images", "labels", "masks", "lanes"]
        )
        writer.writeheader()
        writer.writerows(summary)

    print("Saved YOLOP dataset to:")
    print(OUT_ROOT)
    print("\nSummary:")
    for row in summary:
        print(row)
    print("\nWrote:")
    print(summary_csv)


if __name__ == "__main__":
    main()
