"""
compare.py
==========
Three-way comparison on the 10 demo test files:

    Paper specialist  vs  Our DAFT specialist  vs  Our global (no DAFT)

Steps
-----
  1. Run inference with DAFT routing      -> data/preds_daft/
  2. Run inference forced to global.pth   -> data/preds_global/
  3. Compute DSC for both prediction sets
  4. Plot a grouped bar chart vs paper numbers (hard-coded below)

Paper numbers come from Table 6 of the DAFT paper / our README results table.

Usage
-----
  python compare.py
  python compare.py --device cpu
"""

import os
import argparse
import subprocess
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import inference  # we monkey-patch its router for the global-only pass


# Paper specialist DSC scores (from Table 6 / README)
PAPER_DSC = {
    "CXR":         0.925,
    "Dermoscopy":  0.971,
    "Endoscopy":   0.987,
    "Fundus":      0.982,
    "Mammography": 0.748,
    "Microscopy":  0.957,
    "OCT":         0.875,
    "Ultrasound":  0.875,
    "CT":          0.720,
    "MR":          0.899,
}

# Map demo filename prefix -> display label used in the chart
FILE_TO_LABEL = {
    "2DBox_CXR":         "CXR",
    "2DBox_Dermoscopy":  "Dermoscopy",
    "2DBox_Endoscopy":   "Endoscopy",
    "2DBox_Fundus":      "Fundus",
    "2DBox_Mammography": "Mammography",
    "2DBox_Microscope":  "Microscopy",
    "2DBox_OCT":         "OCT",
    "2DBox_US":          "Ultrasound",
    "3DBox_CT":          "CT",
    "3DBox_MR":          "MR",
}


def label_from_filename(fname):
    for prefix, label in FILE_TO_LABEL.items():
        if fname.startswith(prefix):
            return label
    return fname


def run_inference(img_dir, pred_dir, ckpt_dir, device, force_global=False):
    """Run inference.py main flow programmatically."""
    os.makedirs(pred_dir, exist_ok=True)
    inference._MODEL_CACHE.clear()  # force re-load between passes

    if force_global:
        original_router = inference.filename_to_modelname
        inference.filename_to_modelname = lambda _name: "global"

    try:
        import sys
        sys.argv = [
            "inference.py",
            "--img_dir",  img_dir,
            "--pred_dir", pred_dir,
            "--ckpt_dir", ckpt_dir,
            "--device",   device,
        ]
        inference.main()
    finally:
        if force_global:
            inference.filename_to_modelname = original_router


def evaluate(pred_dir, gt_dir, out_csv):
    """Call evaluate.py as a subprocess (it's already a clean CLI)."""
    subprocess.run(
        ["python", "evaluate.py",
         "--pred_dir", pred_dir,
         "--gt_dir",   gt_dir,
         "--out_csv",  out_csv],
        check=True,
    )
    return pd.read_csv(out_csv)


def plot(daft_df, global_df, out_png):
    """Grouped bar chart: paper / our DAFT / our global, per file."""
    daft_df["label"]   = daft_df["file"].apply(label_from_filename)
    global_df["label"] = global_df["file"].apply(label_from_filename)

    # Average DSC per label (one number per file in our case)
    daft_by   = daft_df.groupby("label")["dsc"].mean()
    global_by = global_df.groupby("label")["dsc"].mean()

    labels = list(FILE_TO_LABEL.values())
    paper  = [PAPER_DSC.get(l,    float("nan")) for l in labels]
    daft   = [daft_by.get(l,      float("nan")) for l in labels]
    glob   = [global_by.get(l,    float("nan")) for l in labels]

    import numpy as np
    x = np.arange(len(labels))
    w = 0.27

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, paper, w, label="Paper specialist",   color="#666666")
    ax.bar(x,     daft,  w, label="Our DAFT specialist", color="#2a8a4a")
    ax.bar(x + w, glob,  w, label="Our global (no DAFT)", color="#c44")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Dice score")
    ax.set_title("DAFT vs. global vs. paper -- 10 demo files")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  Chart saved to {out_png}")


def get_args():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir",  default="data/test_imgs")
    p.add_argument("--gt_dir",   default="data/test_gts")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir",  default="data")
    return p.parse_args()


def main():
    args = get_args()

    daft_dir   = os.path.join(args.out_dir, "preds_daft")
    global_dir = os.path.join(args.out_dir, "preds_global")

    print("\n=== Pass 1: DAFT routing ===")
    run_inference(args.img_dir, daft_dir,   args.ckpt_dir, args.device, force_global=False)

    print("\n=== Pass 2: forced global.pth (no DAFT) ===")
    run_inference(args.img_dir, global_dir, args.ckpt_dir, args.device, force_global=True)

    print("\n=== Evaluation ===")
    daft_csv   = os.path.join(args.out_dir, "results_daft.csv")
    global_csv = os.path.join(args.out_dir, "results_global.csv")
    daft_df    = evaluate(daft_dir,   args.gt_dir, daft_csv)
    global_df  = evaluate(global_dir, args.gt_dir, global_csv)

    print("\n=== Chart ===")
    plot(daft_df, global_df, os.path.join(args.out_dir, "compare.png"))

    # Summary table
    daft_df["label"]   = daft_df["file"].apply(label_from_filename)
    global_df["label"] = global_df["file"].apply(label_from_filename)
    summary = pd.DataFrame({
        "label":  list(FILE_TO_LABEL.values()),
    })
    summary["paper"]   = summary["label"].map(PAPER_DSC)
    summary["daft"]    = summary["label"].map(daft_df.groupby("label")["dsc"].mean())
    summary["global"]  = summary["label"].map(global_df.groupby("label")["dsc"].mean())
    summary["gain"]    = summary["daft"] - summary["global"]
    summary.to_csv(os.path.join(args.out_dir, "compare.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
