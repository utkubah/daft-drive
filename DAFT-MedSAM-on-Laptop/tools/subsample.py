"""
subsample.py
============
Reduce a downloaded MedSAM training folder so it fits in limited disk space.

Walks  <input>/train_npz/<Modality>/<SubCategory>/*.npz
       <input>/validation-box/<Modality>/<SubCategory>/*.npz   (if present)
and copies a random subset of files to <output>/ preserving structure.

Per-modality caps are tunable.  Defaults aim for ~5 GB total:
  3D modalities (CT, MR, PET):  20 .npz files per sub-category
  2D modalities:                300 .npz files per sub-category

Usage
-----
  python tools/subsample.py --input ./medsam-raw --output ./medsam-data

  # Custom caps:
  python tools/subsample.py --input ./medsam-raw --output ./medsam-data \\
      --cap_3d 30 --cap_2d 500
"""

import argparse
import os
import random
import shutil
from pathlib import Path

# 3D modalities are huge -- we cap them tighter
THREED_MODALITIES = {"CT", "MR", "PET"}


def subsample_dir(in_dir: Path, out_dir: Path, cap_3d: int, cap_2d: int):
    """Walk modality/subcategory tree and copy <= cap files from each leaf."""
    if not in_dir.exists():
        print(f"  Skipping (not found): {in_dir}")
        return

    total_in, total_out = 0, 0
    for modality_dir in sorted(in_dir.iterdir()):
        if not modality_dir.is_dir():
            continue
        cap = cap_3d if modality_dir.name in THREED_MODALITIES else cap_2d

        for subcat_dir in sorted(modality_dir.iterdir()):
            if not subcat_dir.is_dir():
                continue

            files = sorted(subcat_dir.glob("*.npz"))
            total_in += len(files)
            picked = random.sample(files, cap) if len(files) > cap else files
            total_out += len(picked)

            target = out_dir / modality_dir.name / subcat_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for src in picked:
                dst = target / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)

            print(f"    {modality_dir.name:12s} / {subcat_dir.name:30s}  "
                  f"{len(picked):4d} / {len(files):5d}")

    print(f"  -> {in_dir.name}: kept {total_out} / {total_in} files")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",  required=True, help="Folder containing train_npz/")
    ap.add_argument("--output", required=True, help="Where to write the reduced copy")
    ap.add_argument("--cap_3d", type=int, default=20,
                    help="Max files per sub-category for CT/MR/PET")
    ap.add_argument("--cap_2d", type=int, default=300,
                    help="Max files per sub-category for 2D modalities")
    ap.add_argument("--seed",   type=int, default=2024)
    args = ap.parse_args()

    random.seed(args.seed)

    in_root  = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Subsampling train_npz/ ...")
    subsample_dir(in_root / "train_npz", out_root / "train_npz",
                  args.cap_3d, args.cap_2d)

    print("\nSubsampling validation-box/ ...")
    subsample_dir(in_root / "validation-box", out_root / "validation-box",
                  args.cap_3d, args.cap_2d)

    # Final disk usage
    total_bytes = sum(p.stat().st_size for p in out_root.rglob("*") if p.is_file())
    print(f"\nDone.  Output: {out_root}  ({total_bytes / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
