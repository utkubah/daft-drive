from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Convert BDD100K segmentation data to .npz format")
    p.add_argument("--root", type=str, required=True, help="BDD100K seg root")
    p.add_argument("--out", type=str, default="data/bdd100k_npz", help="Output root")
    p.add_argument("--max_per_split", type=int, default=0, help="0 = all files")
    return p.parse_args()


def remap_mask(mask: np.ndarray) -> np.ndarray:
    """
    BDD100K train_id masks use 255 as ignore.
    Our current dataset loader assumes:
      0 = background / ignore
      1..N = foreground classes

    So we map:
      255 -> 0
      k   -> k + 1   for all valid class ids
    """
    mask = mask.astype(np.int16)
    out = np.zeros_like(mask, dtype=np.uint8)
    valid = mask != 255
    out[valid] = (mask[valid] + 1).astype(np.uint8)
    return out


def convert_split(root: Path, out_root: Path, split: str, max_per_split: int = 0):
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    npz_dir = out_root / "train_npz" / "BDD100K" / split
    npz_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(img_dir.glob("*.jpg"))
    if max_per_split > 0:
        img_paths = img_paths[:max_per_split]

    rows = []
    missing = 0
    failed = 0

    for i, img_path in enumerate(img_paths, start=1):
        lbl_path = lbl_dir / f"{img_path.stem}_train_id.png"
        if not lbl_path.exists():
            missing += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(lbl_path), cv2.IMREAD_UNCHANGED)

        if img is None or mask is None:
            failed += 1
            continue

        if mask.ndim == 3:
            mask = mask[..., 0]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = remap_mask(mask)

        out_path = npz_dir / f"{img_path.stem}.npz"
        np.savez_compressed(out_path, imgs=img, gts=mask)

        rows.append({"file": str(out_path)})

        if i % 100 == 0 or i == len(img_paths):
            print(f"[{split}] {i}/{len(img_paths)}")

    print(f"\n{split}:")
    print(f"  converted: {len(rows)}")
    print(f"  missing labels: {missing}")
    print(f"  failed reads: {failed}")

    return rows


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    train_rows = convert_split(root, out_root, "train", args.max_per_split)
    val_rows = convert_split(root, out_root, "val", args.max_per_split)

    split_dir = out_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"

    pd.DataFrame(train_rows).to_csv(train_csv, index=False)
    pd.DataFrame(val_rows).to_csv(val_csv, index=False)

    print("\nSaved CSVs:")
    print(f"  {train_csv}")
    print(f"  {val_csv}")


if __name__ == "__main__":
    main()