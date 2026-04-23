"""
prepare_data.py
===============
Create train / validation CSV file lists for fine-tuning and DAFT.

Expects the official MedSAM-on-Laptop dataset layout:
  <data_dir>/
    train_npz/
      <Modality>/
        <SubCategory>/
          *.npz

The 80 / 20 split is applied per sub-category folder, seeded at 2024
(same as split_dataset.py in the original repo).

Outputs written to --output_dir (default: data/datasplit/):
  train.csv                     all training files (all modalities)
  val.csv                       all validation files
  modalities/
    US.train.csv                per-modality subsets
    US.val.csv
    XRay.train.csv
    XRay.val.csv
    3D.train.csv
    ...  (one pair per entry in MODALITY_PATTERNS below)

Usage
-----
  python prepare_data.py --data_dir /path/to/CVPR24-MedSAMLaptopData

  # Custom output directory:
  python prepare_data.py --data_dir /path/to/data --output_dir data/datasplit
"""

import os
import glob
import random
import argparse
import pandas as pd

random.seed(2024)

# ---------------------------------------------------------------------------
# Modality -> filename-prefix regex mapping.
# Must stay in sync with inference.py:filename_to_modelname().
# Source: datasplit/modalities3D.py in the original repo.
# ---------------------------------------------------------------------------
MODALITY_PATTERNS = {
    "Dermoscopy":  r"^Dermoscopy",
    "Endoscopy":   r"^Endoscopy",
    "Fundus":      r"^Fundus",
    "Mammography": r"^Mamm",
    "Microscopy":  r"^Microscopy",
    "OCT":         r"^OCT",
    "US":          r"^US",
    "XRay":        r"^(XRay|X-Ray|CXR|XR)",
    "3D":          r"^(CT|MR|PET)",
}


def get_args():
    p = argparse.ArgumentParser(
        description="Build 80/20 train-val CSV splits for DAFT fine-tuning."
    )
    p.add_argument("--data_dir",   required=True,
                   help="Root directory that contains the 'train_npz' subfolder")
    p.add_argument("--output_dir", default="data/datasplit",
                   help="Where to write the CSV files  (default: data/datasplit)")
    return p.parse_args()


def main():
    args    = get_args()
    root    = os.path.join(args.data_dir, "train_npz")
    out     = args.output_dir
    mod_out = os.path.join(out, "modalities")

    os.makedirs(out,     exist_ok=True)
    os.makedirs(mod_out, exist_ok=True)

    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"'train_npz' not found inside {args.data_dir}.\n"
            f"Expected: {root}"
        )

    # --- collect all .npz files, split 80/20 per sub-category ---
    train_files, val_files = [], []

    for modality_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(modality_dir):
            continue
        for subcat_dir in sorted(glob.glob(os.path.join(modality_dir, "*"))):
            if not os.path.isdir(subcat_dir):
                continue
            npzs = sorted(glob.glob(os.path.join(subcat_dir, "*.npz")))
            if not npzs:
                continue
            random.shuffle(npzs)
            split = int(len(npzs) * 0.8)
            train_files.extend(npzs[:split])
            val_files.extend(npzs[split:])

    print(f"  Total split: {len(train_files)} train | {len(val_files)} val")

    # --- global CSVs ---
    pd.DataFrame({"file": train_files}).to_csv(os.path.join(out, "train.csv"), index=False)
    pd.DataFrame({"file": val_files  }).to_csv(os.path.join(out, "val.csv"),   index=False)
    print(f"  Saved: {out}/train.csv  and  {out}/val.csv")

    # --- per-modality CSVs ---
    all_train = pd.DataFrame({"file": train_files})
    all_val   = pd.DataFrame({"file": val_files  })

    for modality, pattern in MODALITY_PATTERNS.items():
        # Match on the bare filename, not the full path
        def _match(df, pat=pattern):
            names = df["file"].str.extract(r"([^\\/]+)$")[0]
            return df[names.str.match(pat)].reset_index(drop=True)

        m_train = _match(all_train)
        m_val   = _match(all_val)

        m_train.to_csv(os.path.join(mod_out, f"{modality}.train.csv"), index=False)
        m_val.to_csv(  os.path.join(mod_out, f"{modality}.val.csv"),   index=False)
        print(f"  {modality:12s}:  {len(m_train):6d} train  |  {len(m_val):5d} val")

    print("\nDone. Use train.csv / val.csv for global fine-tuning.")
    print("Use modalities/<name>.train.csv for DAFT specialist fine-tuning.")


if __name__ == "__main__":
    main()
