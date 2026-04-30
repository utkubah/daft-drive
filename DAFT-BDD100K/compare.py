"""
compare.py
==========
Evaluate global vs DAFT specialist on each condition and plot results.

For each condition (day / night / rain):
  - Global model on that condition's val data
  - Specialist model on that condition's val data

Outputs
-------
  results/compare.csv     mAP50 and mAP50-95 for all combinations
  results/compare.png     grouped bar chart

Usage
-----
  python compare.py
  python compare.py --device cpu --batch 4
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

CONDITIONS       = ["city_day", "city_night", "highway_day", "highway_night", "residential"]
GLOBAL_CKPT_BASE = Path("runs/detect/checkpoints")   # ultralytics default output
SPEC_CKPT_BASE   = Path("checkpoints")               # absolute-path output from train.py
DATA_BASE        = Path("data/bdd100k/yolo")
OUT_DIR     = Path("results")


def evaluate(weights: Path, data: Path, device: str, batch: int) -> dict:
    model   = YOLO(str(weights))
    metrics = model.val(data=str(data), device=device, batch=batch, verbose=False)
    return {
        "map50":    round(metrics.box.map50, 4),
        "map50_95": round(metrics.box.map,   4),
    }


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch",  type=int, default=4)
    return p.parse_args()


def main():
    args = get_args()
    OUT_DIR.mkdir(exist_ok=True)

    # global checkpoint may be in either location depending on when train.py was fixed
    global_ckpt = next(
        (p for p in [
            GLOBAL_CKPT_BASE / "global" / "weights" / "best.pt",
            SPEC_CKPT_BASE   / "global" / "weights" / "best.pt",
        ] if p.exists()),
        None,
    )
    if global_ckpt is None:
        raise FileNotFoundError("Global checkpoint not found. Run pretrain_bdd100k.sh first.")

    rows = []
    for cond in CONDITIONS:
        specialist_ckpt = next(
            (p for p in [
                SPEC_CKPT_BASE   / cond / "weights" / "best.pt",
                GLOBAL_CKPT_BASE / cond / "weights" / "best.pt",
            ] if p.exists()),
            None,
        )
        data_yaml       = DATA_BASE / f"{cond}.yaml"

        if not data_yaml.exists():
            print(f"  Skipping {cond} — {data_yaml} not found")
            continue

        print(f"\n--- {cond.upper()} ---")

        print(f"  Evaluating global on {cond}...")
        g = evaluate(global_ckpt, data_yaml, args.device, args.batch)
        print(f"    mAP50={g['map50']}  mAP50-95={g['map50_95']}")

        if specialist_ckpt is not None:
            print(f"  Evaluating specialist on {cond}...")
            s = evaluate(specialist_ckpt, data_yaml, args.device, args.batch)
            print(f"    mAP50={s['map50']}  mAP50-95={s['map50_95']}")
        else:
            print(f"  No specialist found for {cond}, skipping.")
            s = {"map50": None, "map50_95": None}

        rows.append({
            "condition":         cond,
            "global_map50":      g["map50"],
            "global_map50_95":   g["map50_95"],
            "specialist_map50":  s["map50"],
            "specialist_map50_95": s["map50_95"],
            "gain_map50":        round(s["map50"] - g["map50"], 4) if s["map50"] else None,
        })

    # --- save CSV ---
    csv_path = OUT_DIR / "compare.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # --- print summary table ---
    print(f"\n{'Condition':<10} {'Global mAP50':>13} {'Specialist mAP50':>17} {'Gain':>7}")
    print("-" * 52)
    for r in rows:
        gain = f"{r['gain_map50']:+.4f}" if r["gain_map50"] is not None else "   N/A"
        spec = f"{r['specialist_map50']:.4f}" if r["specialist_map50"] else "   N/A"
        print(f"{r['condition']:<10} {r['global_map50']:>13.4f} {spec:>17} {gain:>7}")

    # --- bar chart ---
    valid = [r for r in rows if r["specialist_map50"] is not None]
    if valid:
        labels  = [r["condition"] for r in valid]
        g_vals  = [r["global_map50"]     for r in valid]
        s_vals  = [r["specialist_map50"] for r in valid]
        x = np.arange(len(labels))
        w = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - w/2, g_vals, w, label="Global",     color="#5577aa")
        ax.bar(x + w/2, s_vals, w, label="Specialist",  color="#44aa66")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("mAP50")
        ax.set_title("DAFT: Global vs Specialist per Condition")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        png_path = OUT_DIR / "compare.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
