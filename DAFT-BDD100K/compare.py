"""
compare.py
==========
Three-part evaluation:

  1. mAP comparison  — global vs DAFT specialist per condition
  2. Speed benchmark — ms/image across 5 strategies
  3. Routing distribution — how the adaptive router splits images across conditions

Strategies benchmarked
----------------------
  large         YOLOv8m  (upper-bound accuracy baseline)
  global        YOLOv8s global fine-tune
  hard          metadata → top-1 specialist  (no blending)
  adaptive-1    ImageRouter → top-1 specialist  (auto confident path)
  adaptive-2    ImageRouter → top-2 blend       (forced blend)

Outputs
-------
  results/compare.csv      mAP table
  results/compare.png      mAP bar chart
  results/speed.csv        ms/image per strategy
  results/speed.png        speed bar chart

Usage
-----
  python compare.py --device cpu --batch 4
  python compare.py --device cpu --n_bench 30 --router_ckpt checkpoints/router/best.pt
"""

import argparse
import csv
import random
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


def _set_pub_rc() -> None:
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "axes.linewidth":    0.8,
        "grid.linewidth":    0.4,
        "grid.color":        "#CCCCCC",
        "grid.alpha":        1.0,
        "legend.fontsize":   9,
        "legend.framealpha": 0.93,
        "legend.edgecolor":  "#CCCCCC",
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })

from router import CONDITIONS, MetadataRouter, ImageRouter, blend_detections, select_top_k

GLOBAL_CKPT_BASE = Path("runs/detect/checkpoints")
SPEC_CKPT_BASE   = Path("checkpoints")
DATA_BASE        = Path("data/bdd100k/yolo")
MANIFEST_DIR     = Path("data/bdd100k/manifests")
OUT_DIR          = Path("results")


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def find_global_ckpt() -> Path:
    for p in [SPEC_CKPT_BASE / "global" / "weights" / "best.pt",
              GLOBAL_CKPT_BASE / "global" / "weights" / "best.pt"]:
        if p.exists():
            return p
    raise FileNotFoundError("Global checkpoint not found. Run pretrain_bdd100k.sh first.")


def find_specialist_ckpt(cond: str) -> Path | None:
    for p in [SPEC_CKPT_BASE / cond / "weights" / "best.pt",
              GLOBAL_CKPT_BASE / cond / "weights" / "best.pt"]:
        if p.exists():
            return p
    return None


# ── mAP evaluation ────────────────────────────────────────────────────────────

def evaluate(weights: Path, data: Path, device: str, batch: int) -> dict:
    model   = YOLO(str(weights))
    metrics = model.val(data=str(data), device=device, batch=batch, verbose=False)
    # per-class AP50: list aligned with metrics.box.ap_class_index
    per_class_ap50 = {}
    if hasattr(metrics.box, "ap_class_index") and hasattr(metrics.box, "ap50"):
        for idx, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
            per_class_ap50[int(idx)] = round(float(ap), 4)
    return {
        "map50":         round(float(metrics.box.map50), 4),
        "map50_95":      round(float(metrics.box.map),   4),
        "precision":     round(float(metrics.box.mp),    4),
        "recall":        round(float(metrics.box.mr),    4),
        "per_class_ap50": per_class_ap50,
    }


# ── Speed benchmark ───────────────────────────────────────────────────────────

def load_bench_images(n: int, seed: int = 42) -> list[str]:
    """Pick N random val images from manifest."""
    import csv as csv_mod
    paths = []
    manifest = MANIFEST_DIR / "val.csv"
    if not manifest.exists():
        # fallback: glob val images directly
        paths = [str(p) for p in sorted((DATA_BASE / "images" / "val").rglob("*.jpg"))]
    else:
        with open(manifest, newline="") as f:
            paths = [row["image_path"] for row in csv_mod.DictReader(f)]

    random.seed(seed)
    return random.sample(paths, min(n, len(paths)))


def time_strategy(fn, imgs: list[str], n_warmup: int = 3) -> tuple[float, float]:
    """Returns (mean_ms, std_ms) per image."""
    # warmup
    for img in imgs[:n_warmup]:
        fn(img)
    # timed
    times = []
    for img in imgs:
        t0 = time.perf_counter()
        fn(img)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def run_speed_benchmark(imgs: list[str], device: str, global_ckpt: Path,
                        router_ckpt: Path | None) -> list[dict]:
    results = []

    # 1. YOLOv8m — large model baseline
    print("  [speed] YOLOv8m (large)...")
    large = YOLO("yolov8m.pt")
    mean, std = time_strategy(
        lambda p: large.predict(p, device=device, verbose=False, conf=0.25), imgs
    )
    results.append({"strategy": "large (YOLOv8m)", "mean_ms": round(mean, 2), "std_ms": round(std, 2)})

    # 2. YOLOv8s global
    print("  [speed] YOLOv8s global...")
    global_m = YOLO(str(global_ckpt))
    mean, std = time_strategy(
        lambda p: global_m.predict(p, device=device, verbose=False, conf=0.25), imgs
    )
    results.append({"strategy": "global (YOLOv8s)", "mean_ms": round(mean, 2), "std_ms": round(std, 2)})

    # 3. Hard routing — metadata top-1 (use city_day specialist as representative)
    spec_ckpt = find_specialist_ckpt("city_day") or global_ckpt
    print(f"  [speed] Hard routing (city_day specialist)...")
    spec_m = YOLO(str(spec_ckpt))
    mean, std = time_strategy(
        lambda p: spec_m.predict(p, device=device, verbose=False, conf=0.25), imgs
    )
    results.append({"strategy": "hard routing (top-1)", "mean_ms": round(mean, 2), "std_ms": round(std, 2)})

    # 4 & 5. Adaptive routing — needs router checkpoint
    if router_ckpt and router_ckpt.exists():
        print("  [speed] Adaptive routing (top-1 auto)...")
        image_router = ImageRouter(str(router_ckpt), device=device if device != "" else "cpu")

        # preload all specialists to avoid disk I/O in timing
        specialists: dict[str, YOLO] = {}
        for cond in CONDITIONS:
            ckpt = find_specialist_ckpt(cond) or global_ckpt
            specialists[cond] = YOLO(str(ckpt))

        def run_adaptive(img_path: str, top_k: str) -> None:
            weights  = image_router.weights(img_path)
            selected = select_top_k(weights, top_k=top_k)
            rw = []
            for cond, w in selected:
                m    = specialists.get(cond, global_m)
                preds = m.predict(img_path, device=device, verbose=False, conf=0.25)
                boxes = _result_to_boxes(preds[0])
                rw.append((boxes, w))
            blend_detections(rw)

        mean, std = time_strategy(lambda p: run_adaptive(p, "auto"), imgs)
        results.append({"strategy": "adaptive top-1 (auto)", "mean_ms": round(mean, 2), "std_ms": round(std, 2)})

        print("  [speed] Adaptive routing (top-2 blend)...")
        mean, std = time_strategy(lambda p: run_adaptive(p, "2"), imgs)
        results.append({"strategy": "adaptive top-2 (blend)", "mean_ms": round(mean, 2), "std_ms": round(std, 2)})

        # Routing distribution
        print("  [speed] Computing routing distribution...")
        dist = {c: 0 for c in CONDITIONS}
        for img in imgs:
            w = image_router.weights(img)
            selected = select_top_k(w, top_k="auto")
            dist[selected[0][0]] += 1
        results.append({"strategy": "--- routing distribution ---",
                        "mean_ms": None, "std_ms": None,
                        **{f"dist_{c}": dist[c] for c in CONDITIONS}})
    else:
        print("  [speed] Skipping adaptive routing — no router checkpoint found.")
        print("          Train with: python train_router.py --device cuda")

    return results


def _result_to_boxes(result) -> np.ndarray:
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy().reshape(-1, 1)
    cls  = result.boxes.cls.cpu().numpy().reshape(-1, 1)
    return np.concatenate([xyxy, conf, cls], axis=1).astype(np.float32)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_map(rows: list[dict]):
    _set_pub_rc()
    valid = [r for r in rows if r["specialist_map50"] is not None]
    if not valid:
        return
    labels = [r["condition"] for r in valid]
    g_vals = [r["global_map50"]     for r in valid]
    s_vals = [r["specialist_map50"] for r in valid]
    x, w   = np.arange(len(labels)), 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, g_vals, w, label="Global (YOLOv8s)",  color="#6C757D",
           alpha=0.88, edgecolor="white", linewidth=1.2)
    ax.bar(x + w/2, s_vals, w, label="Specialist (DAFT)", color="#52B788",
           alpha=0.88, edgecolor="white", linewidth=1.2)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.0); ax.set_ylabel("mAP50")
    ax.set_title("DAFT: Global vs Specialist mAP50 per Condition")
    ax.legend()
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = OUT_DIR / "compare.png"
    plt.savefig(path); plt.close()
    print(f"Saved: {path}")


def plot_speed(speed_rows: list[dict]):
    _set_pub_rc()
    rows = [r for r in speed_rows if r["mean_ms"] is not None]
    if not rows:
        return
    labels = [r["strategy"] for r in rows]
    means  = [r["mean_ms"]  for r in rows]
    stds   = [r["std_ms"]   for r in rows]
    colors = ["#9B2226", "#6C757D", "#005F73", "#52B788", "#F4A261"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=4,
                  color=colors[:len(labels)], alpha=0.88,
                  edgecolor="white", linewidth=1.2)
    ax.bar_label(bars, fmt="%.1f ms", padding=4, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Inference time (ms/image)")
    ax.set_title("Inference Speed Comparison")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = OUT_DIR / "speed.png"
    plt.savefig(path); plt.close()
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device",      default="cpu")
    p.add_argument("--batch",       type=int, default=4)
    p.add_argument("--n_bench",     type=int, default=20,
                   help="Number of images for speed benchmark")
    p.add_argument("--router_ckpt", default="checkpoints/router/best.pt",
                   help="ImageRouter checkpoint for adaptive routing benchmark")
    return p.parse_args()


def main():
    args = get_args()
    OUT_DIR.mkdir(exist_ok=True)

    global_ckpt  = find_global_ckpt()
    router_ckpt  = Path(args.router_ckpt) if args.router_ckpt else None

    # ── 1. mAP comparison ────────────────────────────────────────────────────
    print("\n===== mAP Evaluation =====")
    map_rows     = []
    perclass_rows = []
    for cond in CONDITIONS:
        spec_ckpt = find_specialist_ckpt(cond)
        data_yaml = DATA_BASE / f"{cond}.yaml"
        if not data_yaml.exists():
            print(f"  Skipping {cond} — {data_yaml} not found")
            continue

        print(f"\n--- {cond.upper()} ---")
        g = evaluate(global_ckpt, data_yaml, args.device, args.batch)
        print(f"  global:     mAP50={g['map50']:.4f}  mAP50-95={g['map50_95']:.4f}")

        if spec_ckpt:
            s = evaluate(spec_ckpt, data_yaml, args.device, args.batch)
            print(f"  specialist: mAP50={s['map50']:.4f}  mAP50-95={s['map50_95']:.4f}")
        else:
            print(f"  specialist: not found")
            s = {"map50": None, "map50_95": None}

        gain = round(s["map50"] - g["map50"], 4) if s["map50"] is not None else None
        map_rows.append({
            "condition":          cond,
            "global_map50":       g["map50"],
            "global_map50_95":    g["map50_95"],
            "global_precision":   g["precision"],
            "global_recall":      g["recall"],
            "spec_map50":         s["map50"],
            "spec_map50_95":      s["map50_95"],
            "gain_map50":         gain,
            # keep old name too for backwards compatibility
            "specialist_map50":   s["map50"],
            "specialist_map50_95": s["map50_95"],
        })
        # per-class rows
        all_classes = set(g["per_class_ap50"]) | set(s["per_class_ap50"])
        for cls_idx in sorted(all_classes):
            g_ap = g["per_class_ap50"].get(cls_idx)
            s_ap = s["per_class_ap50"].get(cls_idx)
            cls_gain = round(s_ap - g_ap, 4) if (s_ap is not None and g_ap is not None) else None
            perclass_rows.append({
                "condition":  cond,
                "class":      cls_idx,
                "global_ap50": g_ap,
                "spec_ap50":   s_ap,
                "gain":        cls_gain,
            })

    # ── 2. Speed benchmark ────────────────────────────────────────────────────
    print(f"\n===== Speed Benchmark ({args.n_bench} images) =====")
    bench_imgs  = load_bench_images(args.n_bench)
    speed_rows  = run_speed_benchmark(bench_imgs, args.device, global_ckpt, router_ckpt)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    if map_rows:
        # compare.csv — original output (backwards compat)
        with open(OUT_DIR / "compare.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(map_rows[0].keys()))
            w.writeheader(); w.writerows(map_rows)
        print(f"\nSaved: {OUT_DIR / 'compare.csv'}")

        # map_comparison.csv — what eval_full.py expects
        with open(OUT_DIR / "map_comparison.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(map_rows[0].keys()))
            w.writeheader(); w.writerows(map_rows)
        print(f"Saved: {OUT_DIR / 'map_comparison.csv'}")

    if perclass_rows:
        with open(OUT_DIR / "perclass_map.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["condition", "class", "global_ap50", "spec_ap50", "gain"])
            w.writeheader(); w.writerows(perclass_rows)
        print(f"Saved: {OUT_DIR / 'perclass_map.csv'}")

    with open(OUT_DIR / "speed.csv", "w", newline="") as f:
        all_keys = list({k for r in speed_rows for k in r})
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader(); w.writerows(speed_rows)
    print(f"Saved: {OUT_DIR / 'speed.csv'}")

    # ── Print tables ──────────────────────────────────────────────────────────
    print(f"\n{'Condition':<20} {'Global mAP50':>13} {'Specialist':>12} {'Gain':>7}")
    print("-" * 56)
    for r in map_rows:
        gain = f"{r['gain_map50']:+.4f}" if r["gain_map50"] is not None else "   N/A"
        spec = f"{r['specialist_map50']:.4f}" if r["specialist_map50"] else "   N/A"
        print(f"{r['condition']:<20} {r['global_map50']:>13.4f} {spec:>12} {gain:>7}")

    print(f"\n{'Strategy':<30} {'ms/image':>10} {'±std':>8}")
    print("-" * 52)
    for r in speed_rows:
        if r["mean_ms"] is not None:
            print(f"{r['strategy']:<30} {r['mean_ms']:>10.1f} {r['std_ms']:>7.1f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_map(map_rows)
    plot_speed(speed_rows)


if __name__ == "__main__":
    main()
