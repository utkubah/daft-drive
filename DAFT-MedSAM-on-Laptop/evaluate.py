"""
evaluate.py
===========
Compute Dice Similarity Coefficient (DSC) for segmentation predictions.

Matches prediction files against ground-truth files by filename.
DSC is computed per foreground label class, then averaged across classes.

Input
-----
  pred_dir/name.npz  ->  key 'segs'  uint16  (H,W) or (D,H,W)
  gt_dir/name.npz    ->  key 'gts'   uint8   (H,W) or (D,H,W)

Output
------
  CSV with columns: case, dsc
  Summary printed to stdout

Usage
-----
  # Evaluate our predictions against the ground truth:
  python evaluate.py \\
    --pred_dir data/test_preds \\
    --gt_dir   data/test_gts   \\
    --out_csv  results.csv

  # Evaluate the reference paper predictions (sanity check):
  python evaluate.py \\
    --pred_dir data/test_demo_segs \\
    --gt_dir   data/test_gts       \\
    --out_csv  results_reference.csv
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def dice(pred_bool, gt_bool):
    """
    Binary Dice Similarity Coefficient.
    Both inputs are boolean arrays of the same shape.
    Returns 1.0 when both arrays are empty (perfect agreement on background).
    """
    inter = int((pred_bool & gt_bool).sum())
    total = int(pred_bool.sum() + gt_bool.sum())
    if total == 0:
        return 1.0
    return 2.0 * inter / total


def file_dsc(pred_path, gt_path):
    """
    Load one prediction / ground-truth pair and return mean DSC
    averaged over all foreground label classes (background = 0 excluded).
    Returns NaN if the ground truth has no foreground.
    """
    segs = np.load(pred_path, allow_pickle=True)["segs"]
    gts  = np.load(gt_path,  allow_pickle=True)["gts"]

    labels = np.unique(gts)
    labels = labels[labels > 0]      # skip background

    if len(labels) == 0:
        return float("nan")

    scores = [dice(segs == lb, gts == lb) for lb in labels]
    return float(np.mean(scores))


def get_args():
    p = argparse.ArgumentParser(
        description="Evaluate segmentation predictions using Dice Score."
    )
    p.add_argument("--pred_dir", required=True,
                   help="Directory with prediction .npz files (key: 'segs')")
    p.add_argument("--gt_dir",   required=True,
                   help="Directory with ground-truth .npz files (key: 'gts')")
    p.add_argument("--out_csv",  default="results.csv",
                   help="Where to write the per-case DSC table")
    return p.parse_args()


def main():
    args = get_args()

    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.npz")))
    if not gt_files:
        print(f"No ground-truth .npz files found in {args.gt_dir}")
        return

    rows = []
    for gt_path in tqdm(gt_files, desc="Evaluating"):
        name      = os.path.basename(gt_path)
        pred_path = os.path.join(args.pred_dir, name)

        if not os.path.isfile(pred_path):
            print(f"  WARNING: no prediction for {name} -- skipping")
            continue

        dsc = file_dsc(pred_path, gt_path)
        rows.append({"case": name, "dsc": round(dsc, 4)})

    if not rows:
        print("No matching prediction / ground-truth pairs found.")
        return

    df       = pd.DataFrame(rows).sort_values("case").reset_index(drop=True)
    mean_dsc = df["dsc"].mean()

    df.to_csv(args.out_csv, index=False)

    print(f"\n  Cases evaluated : {len(df)}")
    print(f"  Mean DSC        : {mean_dsc:.4f}")
    print(f"  Results saved   -> {args.out_csv}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
