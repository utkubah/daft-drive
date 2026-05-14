"""
eval_topk.py
============
Sweep the adaptive router over top-K = 1, 2, 3, 4, 5 and measure
both end-to-end mAP50 and inference speed for each setting.
Also includes the global model and hard-routing (top-1 no blend) as baselines.

This produces the accuracy–speed tradeoff data for the presentation.

How mAP is computed
-------------------
For each K setting we run the full routing + blend pipeline on every val
image, collect (predicted boxes, ground-truth boxes) pairs, then compute
mAP50 using torchmetrics.detection.MeanAveragePrecision.
Ground truth is read from the YOLO label files referenced in val.csv.

Outputs (results/)
------------------
  topk_sweep.csv          — mAP50, ms/image, FPS per K setting
  topk_tradeoff.png       — accuracy–speed scatter plot (the money slide)
  topk_map_bar.png        — mAP50 bar chart across K settings
  topk_speed_bar.png      — latency bar chart across K settings

Usage
-----
  python eval_topk.py --device cpu --n_bench 30
  python eval_topk.py --device cuda --n_bench 100 --full_map
  python eval_topk.py --device cpu --router_ckpt checkpoints/router/best.pt
"""

from __future__ import annotations

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
import torch


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
from ultralytics import YOLO

from router import CONDITIONS, ImageRouter, blend_detections, select_top_k

SPEC_CKPT_BASE   = Path("checkpoints")
GLOBAL_CKPT_BASE = Path("runs/detect/checkpoints")
MANIFEST_DIR     = Path("data/bdd100k/manifests")
OUT_DIR          = Path("results")

# K settings to sweep + two non-router baselines
K_SETTINGS = [1, 2, 3, 4, 5]


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def find_global_ckpt() -> Path:
    for p in [SPEC_CKPT_BASE / "global" / "weights" / "best.pt",
              GLOBAL_CKPT_BASE / "global" / "weights" / "best.pt"]:
        if p.exists():
            return p
    raise FileNotFoundError("Global checkpoint not found.")


def find_specialist_ckpt(cond: str) -> Path | None:
    for p in [SPEC_CKPT_BASE / cond / "weights" / "best.pt",
              GLOBAL_CKPT_BASE / cond / "weights" / "best.pt"]:
        if p.exists():
            return p
    return None


# ── Val manifest loading ──────────────────────────────────────────────────────

def load_val_samples(max_samples: int = 0, seed: int = 42) -> list[dict]:
    """
    Load (image_path, yolo_label) pairs from val.csv.
    max_samples=0 → use all.
    """
    manifest = MANIFEST_DIR / "val.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"val.csv not found at {manifest}")

    with open(manifest, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if Path(r["image_path"]).exists()]

    random.seed(seed)
    if max_samples and max_samples < len(rows):
        rows = random.sample(rows, max_samples)

    print(f"  Loaded {len(rows)} val samples")
    return rows


# ── Ground truth & prediction helpers ────────────────────────────────────────

def load_gt_boxes(label_path: str, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a YOLO label file (cls xc yc bw bh normalised).
    Returns boxes_xyxy (N,4) float32 and labels (N,) int64 in pixel coords.
    """
    p = Path(label_path)
    if not p.exists() or p.stat().st_size == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

    boxes, labels = [], []
    for line in p.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = (xc - bw / 2) * img_w
        y1 = (yc - bh / 2) * img_h
        x2 = (xc + bw / 2) * img_w
        y2 = (yc + bh / 2) * img_h
        boxes.append([x1, y1, x2, y2])
        labels.append(cls)

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def result_to_boxes(result) -> np.ndarray:
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy().reshape(-1, 1)
    cls  = result.boxes.cls.cpu().numpy().reshape(-1, 1)
    return np.concatenate([xyxy, conf, cls], axis=1).astype(np.float32)


# ── mAP via torchmetrics ──────────────────────────────────────────────────────

def compute_map50(all_preds: list[np.ndarray],
                  all_gt_boxes: list[np.ndarray],
                  all_gt_labels: list[np.ndarray]) -> float:
    """
    Compute mAP50 over a list of images using torchmetrics.
    Falls back to a simple numpy implementation if torchmetrics unavailable.
    """
    try:
        from torchmetrics.detection import MeanAveragePrecision
        metric = MeanAveragePrecision(iou_thresholds=[0.5], box_format="xyxy")
        metric.update(
            [{"boxes":  torch.tensor(p[:, :4]),
              "scores": torch.tensor(p[:, 4]),
              "labels": torch.tensor(p[:, 5].astype(np.int64))} for p in all_preds],
            [{"boxes":  torch.tensor(b),
              "labels": torch.tensor(l)} for b, l in zip(all_gt_boxes, all_gt_labels)]
        )
        result = metric.compute()
        return float(result["map_50"].item())
    except ImportError:
        return _numpy_map50(all_preds, all_gt_boxes, all_gt_labels)


def _numpy_map50(all_preds: list[np.ndarray],
                 all_gt_boxes: list[np.ndarray],
                 all_gt_labels: list[np.ndarray],
                 iou_thr: float = 0.5) -> float:
    """Lightweight per-class AP50 → mean."""
    from collections import defaultdict

    # gather per-class detections and ground truths
    cls_dets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    cls_ngt:  dict[int, int] = defaultdict(int)

    for preds, gt_b, gt_l in zip(all_preds, all_gt_boxes, all_gt_labels):
        for lbl in gt_l:
            cls_ngt[int(lbl)] += 1

        # sort preds by confidence desc
        if len(preds) == 0:
            continue
        order = np.argsort(-preds[:, 4])
        preds_s = preds[order]

        matched = np.zeros(len(gt_b), dtype=bool)
        for det in preds_s:
            cls = int(det[5])
            gt_mask = gt_l == cls
            gt_idx  = np.where(gt_mask)[0]
            if len(gt_idx) == 0:
                cls_dets[cls].append((float(det[4]), False))
                continue
            gt_sel = gt_b[gt_idx]
            ious   = _iou(det[:4], gt_sel)
            best   = int(np.argmax(ious))
            if ious[best] >= iou_thr and not matched[gt_idx[best]]:
                matched[gt_idx[best]] = True
                cls_dets[cls].append((float(det[4]), True))
            else:
                cls_dets[cls].append((float(det[4]), False))

    aps = []
    for cls, n_gt in cls_ngt.items():
        if n_gt == 0:
            continue
        dets  = sorted(cls_dets.get(cls, []), key=lambda x: -x[0])
        tp    = np.array([d[1] for d in dets], dtype=float)
        fp    = 1 - tp
        tp_c  = np.cumsum(tp)
        fp_c  = np.cumsum(fp)
        rec   = tp_c / max(n_gt, 1)
        prec  = tp_c / np.maximum(tp_c + fp_c, 1e-9)
        aps.append(_ap_from_pr(rec, prec))

    return float(np.mean(aps)) if aps else 0.0


def _ap_from_pr(rec: np.ndarray, prec: np.ndarray) -> float:
    rec  = np.concatenate([[0.0], rec,  [1.0]])
    prec = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    idx = np.where(rec[1:] != rec[:-1])[0]
    return float(np.sum((rec[idx + 1] - rec[idx]) * prec[idx + 1]))


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = a + b - inter
    return np.where(union > 0, inter / union, 0.0)


# ── Core inference functions ──────────────────────────────────────────────────

def run_global(img: np.ndarray, model: YOLO, device: str) -> np.ndarray:
    preds = model.predict(img, device=device, verbose=False, conf=0.25)
    return result_to_boxes(preds[0])


def run_topk(img: np.ndarray, router: ImageRouter, specialists: dict[str, YOLO],
             k: int | str, device: str) -> np.ndarray:
    weights  = router.weights_from_img(img)
    selected = select_top_k(weights, top_k=k)
    rw = []
    for cond, w in selected:
        preds = specialists[cond].predict(img, device=device, verbose=False, conf=0.25)
        rw.append((result_to_boxes(preds[0]), w))
    return blend_detections(rw)


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(val_samples: list[dict],
              bench_imgs: list[str],
              global_ckpt: Path,
              router_ckpt: Path,
              device: str,
              large_ckpt: Path | None = None,
              n_warmup: int = 3) -> list[dict]:
    """
    For each strategy (global large + global distilled + k=1..5), compute mAP50 + ms/image + FPS.
    """
    rows: list[dict] = []

    # preload models once
    large_m = None
    if large_ckpt and large_ckpt.exists():
        print(f"  Loading Global Large model...  ({large_ckpt})")
        large_m = YOLO(str(large_ckpt))
    else:
        print("  No large checkpoint found — skipping Global Large")

    print("  Loading global model...")
    global_m = YOLO(str(global_ckpt))

    print("  Loading router + all specialists...")
    r_device = device if device else "cpu"
    router   = ImageRouter(str(router_ckpt), device=r_device)
    specialists: dict[str, YOLO] = {}
    for cond in CONDITIONS:
        ckpt = find_specialist_ckpt(cond) or global_ckpt
        specialists[cond] = YOLO(str(ckpt))
        print(f"    {cond}: {ckpt}")

    # Use canonical labels that match eval_full.py consolidated table
    # All strategy fns now accept a pre-loaded BGR numpy array (one disk read per image)
    strategies: list[tuple[str, callable]] = []
    if large_m is not None:
        strategies.append(("Global Large", lambda img: run_global(img, large_m, device)))
    strategies.append(("Global Distilled", lambda img: run_global(img, global_m, device)))
    for k in K_SETTINGS:
        label = f"Adaptive k={k}"
        strategies.append((label, lambda img, _k=k: run_topk(img, router, specialists, _k, device)))

    for label, fn in strategies:
        print(f"\n  [{label}]")

        # ── mAP on full val set ───────────────────────────────────────────
        print(f"    computing mAP50 on {len(val_samples)} images...")
        all_preds, all_gt_boxes, all_gt_labels = [], [], []
        for row in val_samples:
            img_path  = row["image_path"]
            lbl_path  = row.get("yolo_label", "")

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            pred_boxes = fn(img)          # numpy array — no second disk read
            gt_b, gt_l = load_gt_boxes(lbl_path, w, h)

            all_preds.append(pred_boxes)
            all_gt_boxes.append(gt_b)
            all_gt_labels.append(gt_l)

        map50 = compute_map50(all_preds, all_gt_boxes, all_gt_labels)
        print(f"    mAP50 = {map50:.4f}")

        # ── Speed + breakdown on bench subset ─────────────────────────────
        # Models already loaded above — warmup before timing
        print(f"    timing on {len(bench_imgs)} images (warmup={n_warmup})...")
        for img_path in bench_imgs[:n_warmup]:
            img_w = cv2.imread(img_path)
            if img_w is not None:
                fn(img_w)

        is_router = label not in ("Global Distilled", "Global Large")
        t_route_list, t_infer_list, t_nms_list, t_total_list = [], [], [], []

        for img_path in bench_imgs:
            img = cv2.imread(img_path)      # single disk read per image
            if img is None:
                continue
            if is_router:
                t_r0 = time.perf_counter()
                wts  = router.weights_from_img(img)
                sel  = select_top_k(wts, top_k=int(label.split("=")[1]))
                t_r1 = time.perf_counter()
                t_i0 = time.perf_counter()
                rw = []
                for cond, w in sel:
                    preds = specialists[cond].predict(
                        img, device=device, verbose=False, conf=0.25)
                    rw.append((result_to_boxes(preds[0]), w))
                t_i1 = time.perf_counter()
                t_n0 = time.perf_counter()
                blend_detections(rw)
                t_n1 = time.perf_counter()
                t_route_list.append((t_r1 - t_r0) * 1000)
                t_infer_list.append((t_i1 - t_i0) * 1000)
                t_nms_list.append(  (t_n1 - t_n0) * 1000)
                t_total_list.append((t_n1 - t_r0) * 1000)
            else:
                model = large_m if label == "Global Large" else global_m
                t0 = time.perf_counter()
                model.predict(img, device=device, verbose=False, conf=0.25)
                t1 = time.perf_counter()
                t_route_list.append(0.0)
                t_infer_list.append((t1 - t0) * 1000)
                t_nms_list.append(0.0)
                t_total_list.append((t1 - t0) * 1000)

        mean_ms = float(np.mean(t_total_list))
        std_ms  = float(np.std(t_total_list))
        fps     = round(1000.0 / mean_ms, 1) if mean_ms > 0 else 0.0

        print(f"    {mean_ms:.1f} ± {std_ms:.1f} ms/img  ({fps} FPS)  "
              f"[route={np.mean(t_route_list):.1f}  "
              f"infer={np.mean(t_infer_list):.1f}  "
              f"nms={np.mean(t_nms_list):.1f}]")
        rows.append({
            "strategy":      label,
            "map50":         round(map50, 4),
            "mean_ms":       round(mean_ms, 2),
            "std_ms":        round(std_ms, 2),
            "fps":           fps,
            "t_route_ms":    round(float(np.mean(t_route_list)), 2),
            "t_infer_ms":    round(float(np.mean(t_infer_list)), 2),
            "t_nms_ms":      round(float(np.mean(t_nms_list)),   2),
        })

    return rows


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_tradeoff(rows: list[dict]) -> None:
    """Accuracy–speed scatter: Global Distilled baseline + Adaptive k=1..5 curve."""
    _set_pub_rc()
    K_COLORS   = ["#6C757D", "#52B788", "#74C69D", "#F4A261", "#E76F51", "#C1121F"]
    K_MARKERS  = ["s",       "o",       "o",       "o",       "o",       "o"]

    all_fps = [float(r["fps"])   for r in rows]
    all_map = [float(r["map50"]) for r in rows]
    x_right = max(all_fps) * 1.30
    y_top   = min(1.0, max(all_map) + (max(all_map) - min(all_map)) * 0.55)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.grid(True, linestyle="--", zorder=0)
    ax.set_xlim(0, x_right)
    ax.set_ylim(0, y_top)

    global_row = next((r for r in rows if r["strategy"] == "Global Distilled"), None)
    if global_row:
        gx, gy = float(global_row["fps"]), float(global_row["map50"])
        ax.axvline(gx, color="#BBBBBB", linewidth=0.7, linestyle=":", zorder=2)
        ax.axhline(gy, color="#BBBBBB", linewidth=0.7, linestyle=":", zorder=2)

    # Connect router k=1..5 with dashed line
    router_rows = sorted(
        [r for r in rows if r["strategy"].startswith("Adaptive")],
        key=lambda r: float(r["fps"]), reverse=True
    )
    if len(router_rows) > 1:
        rx = [float(r["fps"])   for r in router_rows]
        ry = [float(r["map50"]) for r in router_rows]
        ax.plot(rx, ry, color="#999999", linewidth=1.3, linestyle="--",
                zorder=2, alpha=0.6)

    offsets = {
        "Global Distilled": (8, -14),
        "Adaptive k=1":     (8, 6),
        "Adaptive k=2":     (8, 6),
        "Adaptive k=3":     (8, -14),
        "Adaptive k=4":     (8, 6),
        "Adaptive k=5":     (-80, 6),
    }
    for i, r in enumerate(rows):
        x   = float(r["fps"])
        y   = float(r["map50"])
        col = K_COLORS[i % len(K_COLORS)]
        mrk = K_MARKERS[i % len(K_MARKERS)]
        sz  = 180 if r["strategy"] == "Global Distilled" else 150
        ax.scatter(x, y, s=sz, color=col, marker=mrk,
                   zorder=5, edgecolors="white", linewidths=1.5)
        dx, dy = offsets.get(r["strategy"], (8, 6))
        ax.annotate(
            f"{r['strategy']}\n{y:.3f} mAP · {x:.1f} FPS",
            xy=(x, y), xytext=(dx, dy), textcoords="offset points",
            fontsize=8.5, color=col, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=col, lw=0.7),
        )

    ax.set_xlabel("Throughput  (FPS — higher is better →)", labelpad=8)
    ax.set_ylabel("Detection Quality  (mAP50 — higher is better ↑)", labelpad=8)
    ax.set_title("DAFT Adaptive Routing: Accuracy–Speed Tradeoff  (k = 1 … 5)", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = OUT_DIR / "topk_tradeoff.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_map_bar(rows: list[dict]) -> None:
    _set_pub_rc()
    labels = [r["strategy"] for r in rows]
    vals   = [r["map50"]    for r in rows]
    colors = ["#6C757D"] + ["#52B788"] * len(K_SETTINGS)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, vals, color=colors[:len(labels)], alpha=0.88,
                  edgecolor="white", linewidth=1.2)
    bar_texts = ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9) or []
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_ylabel("mAP50")
    ax.set_title("mAP50 by Routing Strategy")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = OUT_DIR / "topk_map_bar.png"
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


def plot_speed_bar(rows: list[dict]) -> None:
    _set_pub_rc()
    labels = [r["strategy"] for r in rows]
    means  = [r["mean_ms"]  for r in rows]
    stds   = [r["std_ms"]   for r in rows]
    colors = ["#6C757D"] + ["#52B788"] * len(K_SETTINGS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bars = axes[0].bar(labels, means, yerr=stds, capsize=4,
                       color=colors[:len(labels)], alpha=0.88,
                       edgecolor="white", linewidth=1.2)
    axes[0].bar_label(bars, fmt="%.1f ms", padding=3, fontsize=8)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("Latency (ms/image)")
    axes[0].set_title("Inference Latency")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y")
    axes[0].spines[["top", "right"]].set_visible(False)

    fpss = [r["fps"] for r in rows]
    bars2 = axes[1].bar(labels, fpss, color=colors[:len(labels)], alpha=0.88,
                        edgecolor="white", linewidth=1.2)
    axes[1].bar_label(bars2, fmt="%.1f fps", padding=3, fontsize=8)
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel("Throughput (FPS)")
    axes[1].set_title("Inference Throughput")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("Speed by Routing Strategy", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / "topk_speed_bar.png"
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device",      default="cpu")
    p.add_argument("--n_bench",     type=int, default=30,
                   help="Images used for speed timing")
    p.add_argument("--n_map",       type=int, default=0,
                   help="Images used for mAP (0 = all val)")
    p.add_argument("--router_ckpt", default="checkpoints/router/best.pt")
    p.add_argument("--large_ckpt",  default="checkpoints/large/weights/best.pt")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args        = get_args()
    OUT_DIR.mkdir(exist_ok=True)
    global_ckpt = find_global_ckpt()
    router_ckpt = Path(args.router_ckpt)
    large_ckpt  = Path(args.large_ckpt) if args.large_ckpt else None

    if not router_ckpt.exists():
        print(f"ERROR: router checkpoint not found at {router_ckpt}")
        print("Train with: python train_router.py --device cuda")
        return

    print("\n===== Top-K Sweep Evaluation =====")
    print(f"  Global ckpt:  {global_ckpt}")
    print(f"  Router ckpt:  {router_ckpt}")
    if large_ckpt and large_ckpt.exists():
        print(f"  Large ckpt:   {large_ckpt}")
    else:
        print("  Large ckpt:   not found — will skip Global Large")

    val_samples = load_val_samples(max_samples=args.n_map, seed=args.seed)
    bench_imgs  = [r["image_path"] for r in val_samples]
    random.seed(args.seed)
    bench_imgs  = random.sample(bench_imgs, min(args.n_bench, len(bench_imgs)))

    rows = run_sweep(val_samples, bench_imgs, global_ckpt, router_ckpt, args.device,
                     large_ckpt=large_ckpt)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = OUT_DIR / "topk_sweep.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "strategy", "map50", "mean_ms", "std_ms", "fps",
            "t_route_ms", "t_infer_ms", "t_nms_ms",
        ])
        w.writeheader(); w.writerows(rows)
    print(f"\n  Saved: {out_csv}")

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'Strategy':<26} {'mAP50':>7} {'ms/img':>8} {'FPS':>7}")
    print("-" * 52)
    for r in rows:
        print(f"{r['strategy']:<26} {r['map50']:>7.4f} {r['mean_ms']:>8.1f} {r['fps']:>7.1f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n===== Generating plots =====")
    plot_tradeoff(rows)
    plot_map_bar(rows)
    plot_speed_bar(rows)
    print("\nDone.")


if __name__ == "__main__":
    main()
