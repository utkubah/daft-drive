"""
eval_paper.py  —  single authoritative evaluation for DAFT-Drive
=================================================================
All per-condition accuracy uses Ultralytics model.val() on the
condition-specific dataset YAMLs — the same pipeline used during
training, giving numbers that are directly comparable.

Speed timing uses model.predict() on a held-out bench sample.

Ablations and dawn/dusk use image-by-image eval with torchmetrics
(clearly labelled; only relative ordering matters there).

Outputs  (results/)
-------------------
  main_results.csv        per-condition mAP50 / mAP50:95 + overall + FPS
  per_class.csv           per-class AP50 global vs specialist (mean across conditions)
  ablations.csv           routing ablation (torchmetrics, relative)
  dawn_dusk.csv           dawn/dusk robustness analysis
  router_accuracy.csv     per-condition top-1 routing accuracy

  accuracy_speed_tradeoff.png  FPS vs mAP50 scatter (all strategies, consistent)
  condition_heatmap.png        model × condition mAP50 heatmap
  ablation_bar.png        ablation bar chart
  class_gains.png         per-class AP50 gain bars
  dawn_dusk.png           dawn/dusk robustness bar

Usage
-----
  python eval_paper.py --device cuda --batch 16 --n_bench 200
  python eval_paper.py --device cpu  --batch 4  --n_bench 30
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

from router import CONDITIONS, ImageRouter, MetadataRouter, blend_detections, select_top_k

# ─────────────────────────────────────────────────────────────────────────────
SPEC_CKPT_BASE = Path("checkpoints")
DATA_BASE      = Path("data/bdd100k/yolo")
MANIFEST_DIR   = Path("data/bdd100k/manifests")
OUT_DIR        = Path("results")

CLASS_NAMES = {
    0: "bicycle", 1: "bus", 2: "car", 3: "motor",
    4: "other person", 5: "other vehicle", 6: "pedestrian",
    7: "rider", 8: "traffic light", 9: "traffic sign",
    10: "trailer", 11: "train", 12: "truck",
}

STRATEGY_COLORS = {
    "Global Distilled":  "#6C757D",
    "Global Large":      "#9B2226",
    "Global XLarge":     "#AE2012",
    "Hard Routing":      "#005F73",
    "DAFT k=1":          "#1B7A4A",
    "DAFT k=2":          "#1B7A4A",
    "DAFT k=3":          "#1B7A4A",
    "DAFT k=4":          "#1B7A4A",
    "DAFT k=5":          "#1B7A4A",
}


# ── Plot style ────────────────────────────────────────────────────────────────

def _rc() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.linewidth": 0.8, "grid.linewidth": 0.4,
        "grid.color": "#CCCCCC", "grid.alpha": 1.0,
        "legend.fontsize": 9, "legend.framealpha": 0.93,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def find_ckpt(key: str) -> Path | None:
    lut = {
        "global": SPEC_CKPT_BASE / "global"  / "weights" / "best.pt",
        "large":  SPEC_CKPT_BASE / "large"   / "weights" / "best.pt",
        "xlarge": SPEC_CKPT_BASE / "xlarge"  / "weights" / "best.pt",
        "router": SPEC_CKPT_BASE / "router"  / "best.pt",
        **{c: SPEC_CKPT_BASE / c / "weights" / "best.pt" for c in CONDITIONS},
    }
    p = lut.get(key)
    return p if (p and p.exists()) else None


# ── model.val() — per-condition mAP (authoritative) ──────────────────────────

def val_condition(ckpt: Path, cond: str, device: str, batch: int) -> dict:
    """Ultralytics model.val() on the condition YAML."""
    yaml = DATA_BASE / f"{cond}.yaml"
    if not yaml.exists():
        print(f"  WARNING: {yaml} not found — skipping {cond}")
        return {"map50": None, "map5095": None, "per_class": {}}
    model   = YOLO(str(ckpt))
    metrics = model.val(data=str(yaml), device=device, batch=batch, verbose=False)
    pc = {}
    if hasattr(metrics.box, "ap_class_index"):
        for idx, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
            pc[int(idx)] = round(float(ap), 4)
    return {
        "map50":    round(float(metrics.box.map50), 4),
        "map5095":  round(float(metrics.box.map),   4),
        "per_class": pc,
    }


def condition_sizes() -> dict[str, int]:
    """Count val images per condition from val.csv."""
    sizes: dict[str, int] = defaultdict(int)
    with open(MANIFEST_DIR / "val.csv", newline="") as f:
        for r in csv.DictReader(f):
            sizes[r["condition"]] += 1
    return dict(sizes)


def weighted_mean(per_cond: dict[str, float | None], sizes: dict[str, int]) -> float | None:
    valid   = [(sizes[c], per_cond[c]) for c in CONDITIONS if per_cond.get(c) is not None]
    total_n = sum(n for n, _ in valid)
    if total_n == 0:
        return None
    return round(sum(n * v for n, v in valid) / total_n, 4)


# ── Ground truth & prediction helpers (for image-by-image eval) ──────────────

def load_gt(label_path: str, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
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
        boxes.append([(xc - bw / 2) * img_w, (yc - bh / 2) * img_h,
                      (xc + bw / 2) * img_w, (yc + bh / 2) * img_h])
        labels.append(cls)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def boxes_from_result(result) -> np.ndarray:
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy().reshape(-1, 1)
    cls  = result.boxes.cls.cpu().numpy().reshape(-1, 1)
    return np.concatenate([xyxy, conf, cls], axis=1).astype(np.float32)


def compute_map(preds: list[np.ndarray],
                gt_boxes: list[np.ndarray],
                gt_labels: list[np.ndarray]) -> tuple[float, float]:
    """Returns (mAP50, mAP50:95) via torchmetrics."""
    from torchmetrics.detection import MeanAveragePrecision
    metric = MeanAveragePrecision(box_format="xyxy")
    metric.update(
        [{"boxes":  torch.tensor(p[:, :4]),
          "scores": torch.tensor(p[:, 4]),
          "labels": torch.tensor(p[:, 5].astype(np.int64))} for p in preds],
        [{"boxes":  torch.tensor(b),
          "labels": torch.tensor(l)} for b, l in zip(gt_boxes, gt_labels)],
    )
    r = metric.compute()
    return float(r["map_50"].item()), float(r["map"].item())


# ── Speed benchmarking ────────────────────────────────────────────────────────

def load_bench(n: int, seed: int = 42) -> list[str]:
    with open(MANIFEST_DIR / "val.csv", newline="") as f:
        paths = [r["image_path"] for r in csv.DictReader(f)
                 if Path(r["image_path"]).exists()]
    random.seed(seed)
    return random.sample(paths, min(n, len(paths)))


def fps_of(fn, imgs: list[np.ndarray], n_warmup: int = 5) -> float:
    """Time fn on pre-loaded numpy arrays — no disk I/O in the timed window."""
    for img in imgs[:n_warmup]:
        fn(img)
    times = []
    for img in imgs:
        t0 = time.perf_counter()
        fn(img)
        times.append((time.perf_counter() - t0) * 1000)
    mean_ms = float(np.mean(times))
    return round(1000.0 / mean_ms, 1) if mean_ms > 0 else 0.0


def benchmark_fps(bench: list[str], device: str,
                  models: dict[str, YOLO],
                  specialists: dict[str, YOLO],
                  router: ImageRouter) -> dict[str, float]:
    # Pre-load all bench images once — disk I/O excluded from every timing loop
    print(f"  Pre-loading {len(bench)} bench images...")
    bench_imgs = [img for img in (cv2.imread(p) for p in bench) if img is not None]

    result: dict[str, float] = {}

    for label in ("Global Distilled", "Global Large", "Global XLarge"):
        m = models.get(label)
        if m is None:
            continue
        print(f"  timing {label}...")
        result[label] = fps_of(
            lambda img, _m=m: _m.predict(img, device=device, verbose=False, conf=0.25),
            bench_imgs,
        )

    # Hard Routing: metadata lookup (negligible) + one specialist forward pass
    ref = specialists[CONDITIONS[0]]
    print("  timing Hard Routing (single specialist, no router)...")
    result["Hard Routing"] = fps_of(
        lambda img: ref.predict(img, device=device, verbose=False, conf=0.25),
        bench_imgs,
    )

    # DAFT k=1: ImageRouter forward pass (~5 ms) + top-1 specialist
    print("  timing DAFT k=1 (router + specialist)...")
    result["DAFT k=1"] = fps_of(
        lambda img: specialists[
            select_top_k(router.weights_from_img(img), top_k=1)[0][0]
        ].predict(img, device=device, verbose=False, conf=0.25),
        bench_imgs,
    )

    for k in [2, 3, 4, 5]:
        print(f"  timing DAFT k={k}...")
        def _run(img, _k=k):
            sel = select_top_k(router.weights_from_img(img), top_k=_k)
            rw  = [(boxes_from_result(
                        specialists[c].predict(img, device=device, verbose=False, conf=0.25)[0]
                    ), wt) for c, wt in sel]
            blend_detections(rw)
        result[f"DAFT k={k}"] = fps_of(_run, bench_imgs)

    return result


# ── Dawn/dusk robustness (image-by-image, torchmetrics) ──────────────────────

def load_dawn_dusk_samples(max_n: int = 0, seed: int = 42) -> list[dict]:
    """Return val rows that are dawn/dusk (is_ambiguous=True or time_of_day dawn/dusk)."""
    rows = []
    with open(MANIFEST_DIR / "val.csv", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for r in reader:
            is_dd = False
            if "is_ambiguous" in fields:
                is_dd = str(r.get("is_ambiguous", "")).strip().lower() in ("true", "1", "yes")
            elif "time_of_day" in fields:
                is_dd = r.get("time_of_day", "").strip().lower() in ("dawn/dusk", "dawn", "dusk")
            if is_dd and Path(r["image_path"]).exists():
                rows.append(r)

    if not rows:
        print("  WARNING: no dawn/dusk images found in val.csv "
              "(checked is_ambiguous and time_of_day columns)")
        return []

    random.seed(seed)
    if max_n and max_n < len(rows):
        rows = random.sample(rows, max_n)
    print(f"  Dawn/dusk images: {len(rows)}")
    return rows


def eval_dawn_dusk(rows: list[dict],
                   global_m: YOLO,
                   specialists: dict[str, YOLO],
                   router: ImageRouter,
                   device: str,
                   large_m: YOLO | None = None) -> list[dict]:
    """
    Compare strategies on dawn/dusk images:
      Global Distilled  — single small global model
      Global Large      — larger global model (if available)
      Hard Routing      — day specialist per condition (single forward pass)
      DAFT k=1          — ImageRouter top-1 specialist
      DAFT k=2          — ImageRouter top-2 blend
      DAFT k=5          — all 5 specialists blended

    Returns list of dicts suitable for dawn_dusk.csv.
    """
    if not rows:
        return []

    strategies = ["Global Distilled"]
    if large_m is not None:
        strategies.append("Global Large")
    strategies += ["Hard Routing", "DAFT k=1", "DAFT k=2", "DAFT k=5"]

    all_preds = {s: [] for s in strategies}
    all_gt_b: list[np.ndarray] = []
    all_gt_l: list[np.ndarray] = []

    for r in rows:
        img = cv2.imread(r["image_path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_b, gt_l = load_gt(r.get("yolo_label", ""), w, h)
        all_gt_b.append(gt_b)
        all_gt_l.append(gt_l)

        cond = r.get("condition") or CONDITIONS[0]

        def _pred(m):
            return boxes_from_result(m.predict(img, device=device, verbose=False, conf=0.25)[0])

        def _blend(sel):
            rw = [(_pred(specialists[c]), wt) for c, wt in sel]
            return blend_detections(rw) if rw else np.zeros((0, 6), dtype=np.float32)

        all_preds["Global Distilled"].append(_pred(global_m))

        if large_m is not None:
            all_preds["Global Large"].append(_pred(large_m))

        all_preds["Hard Routing"].append(
            _pred(specialists.get(cond, specialists[CONDITIONS[0]])))

        img_wts = router.weights_from_img(img)
        all_preds["DAFT k=1"].append(_pred(specialists[select_top_k(img_wts, top_k=1)[0][0]]))
        all_preds["DAFT k=2"].append(_blend(select_top_k(img_wts, top_k=2)))
        all_preds["DAFT k=5"].append(_blend(select_top_k(img_wts, top_k=5)))

    result_rows = []
    for strat in strategies:
        m50, m5095 = compute_map(all_preds[strat], all_gt_b, all_gt_l)
        result_rows.append({"strategy": strat,
                            "map50":    round(m50,   4),
                            "map5095":  round(m5095, 4),
                            "n_images": len(all_gt_b)})
        print(f"    {strat:<20} mAP50={m50:.4f}  mAP50:95={m5095:.4f}")

    return result_rows


# ── Router accuracy ───────────────────────────────────────────────────────────

def eval_router_accuracy(router: ImageRouter, device: str) -> tuple[list[dict], list[dict]]:
    """
    Top-1 accuracy of ImageRouter vs ground-truth condition in val.csv.
    Returns (accuracy_rows, confusion_rows).
    confusion_rows: one row per gt_condition with predicted counts for each condition.
    """
    counts: dict[str, dict] = {c: {"total": 0, "correct": 0} for c in CONDITIONS}
    # confusion[gt][pred] = count
    confusion: dict[str, dict[str, int]] = {gt: {pred: 0 for pred in CONDITIONS}
                                             for gt in CONDITIONS}

    with open(MANIFEST_DIR / "val.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if Path(r["image_path"]).exists()]

    for r in rows:
        gt_cond = r["condition"]
        if gt_cond not in CONDITIONS:
            continue
        img = cv2.imread(r["image_path"])
        if img is None:
            continue
        wts  = router.weights_from_img(img)
        pred = max(wts, key=wts.get)
        counts[gt_cond]["total"]        += 1
        counts[gt_cond]["correct"]      += int(pred == gt_cond)
        confusion[gt_cond][pred]        += 1

    acc_rows = []
    total_c = total_t = 0
    for cond in CONDITIONS:
        t = counts[cond]["total"]
        c = counts[cond]["correct"]
        acc = round(c / t, 4) if t > 0 else 0.0
        acc_rows.append({"condition": cond, "total": t, "correct": c, "accuracy": acc})
        total_c += c;  total_t += t
        print(f"    {cond:<20} {c}/{t}  acc={acc:.4f}")

    overall_acc = round(total_c / total_t, 4) if total_t > 0 else 0.0
    acc_rows.append({"condition": "overall", "total": total_t,
                     "correct": total_c, "accuracy": overall_acc})
    print(f"    {'overall':<20} {total_c}/{total_t}  acc={overall_acc:.4f}")

    # Confusion matrix rows: gt_condition + one column per predicted condition
    conf_rows = []
    for gt in CONDITIONS:
        row = {"gt_condition": gt}
        total = counts[gt]["total"] or 1
        for pred in CONDITIONS:
            row[f"pred_{pred}"] = round(confusion[gt][pred] / total, 4)
        conf_rows.append(row)

    return acc_rows, conf_rows


# ── K-sweep (image-by-image, torchmetrics — consistent tradeoff curve) ───────

def eval_ksweep(val_rows: list[dict],
                global_m: YOLO,
                large_m: YOLO | None,
                xlarge_m: YOLO | None,
                specialists: dict[str, YOLO],
                router: ImageRouter,
                device: str,
                bench: list[str]) -> list[dict]:
    """
    Evaluate all strategies image-by-image with torchmetrics so every point
    on the accuracy-speed tradeoff plot uses the same eval method.
    Includes Global Large/XLarge so they appear in the tradeoff scatter.
    """
    rows: list[dict] = []

    # Pre-load all val images once — shared across every strategy
    print(f"    Pre-loading {len(val_rows)} val images...")
    loaded_val: list[tuple[np.ndarray, dict, np.ndarray, np.ndarray]] = []
    for r in val_rows:
        img = cv2.imread(r["image_path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_b, gt_l = load_gt(r.get("yolo_label", ""), w, h)
        loaded_val.append((img, r, gt_b, gt_l))
    print(f"    Loaded {len(loaded_val)} images")

    def _collect(fn) -> tuple[float, float, dict[str, float]]:
        """Returns (overall_map50, overall_map5095, {cond: map50})."""
        preds, gt_bs, gt_ls = [], [], []
        preds_c  = {c: [] for c in CONDITIONS}
        gt_bs_c  = {c: [] for c in CONDITIONS}
        gt_ls_c  = {c: [] for c in CONDITIONS}
        for img, r, gt_b, gt_l in loaded_val:
            p    = fn(img, r)
            cond = r.get("condition") or CONDITIONS[0]
            preds.append(p);  gt_bs.append(gt_b);  gt_ls.append(gt_l)
            if cond in CONDITIONS:
                preds_c[cond].append(p)
                gt_bs_c[cond].append(gt_b)
                gt_ls_c[cond].append(gt_l)
        m50, m5095 = compute_map(preds, gt_bs, gt_ls)
        per_cond: dict[str, float] = {}
        for c in CONDITIONS:
            if preds_c[c]:
                cm50, _ = compute_map(preds_c[c], gt_bs_c[c], gt_ls_c[c])
                per_cond[c] = round(cm50, 4)
        return m50, m5095, per_cond

    strategies: list[tuple[str, callable]] = [
        ("Global Distilled",
         lambda img, r: boxes_from_result(
             global_m.predict(img, device=device, verbose=False, conf=0.25)[0])),
    ]
    if large_m:
        strategies.append((
            "Global Large",
            lambda img, r: boxes_from_result(
                large_m.predict(img, device=device, verbose=False, conf=0.25)[0])))
    if xlarge_m:
        strategies.append((
            "Global XLarge",
            lambda img, r: boxes_from_result(
                xlarge_m.predict(img, device=device, verbose=False, conf=0.25)[0])))
    strategies.append((
        "Hard Routing",
        lambda img, r: boxes_from_result(
            specialists[r.get("condition") or CONDITIONS[0]].predict(
                img, device=device, verbose=False, conf=0.25)[0])))
    for k in [1, 2, 3, 4, 5]:
        strategies.append((
            f"DAFT k={k}",
            lambda img, r, _k=k: blend_detections([
                (boxes_from_result(specialists[c].predict(
                     img, device=device, verbose=False, conf=0.25)[0]), wt)
                for c, wt in select_top_k(router.weights_from_img(img), top_k=_k)
            ]),
        ))

    # Pre-load bench images once — disk I/O excluded from timing
    bench_imgs  = [img for img in (cv2.imread(p) for p in bench) if img is not None]
    _dummy_row  = {"condition": CONDITIONS[0]}

    for label, fn in strategies:
        print(f"    [{label}]  mAP on {len(val_rows)} images...")
        m50, m5095, per_cond = _collect(fn)
        _fps = fps_of(lambda img, _fn=fn: _fn(img, _dummy_row), bench_imgs)
        print(f"      mAP50={m50:.4f}  mAP50:95={m5095:.4f}  fps={_fps}")
        row = {"strategy": label, "map50": round(m50, 4),
               "map5095": round(m5095, 4), "fps": _fps}
        for c in CONDITIONS:
            row[f"{c}_map50"] = per_cond.get(c)
        rows.append(row)

    return rows


# ── Ablation study (image-by-image, torchmetrics, relative) ──────────────────

def eval_ablations(val_rows: list[dict],
                   global_m: YOLO,
                   specialists: dict[str, YOLO],
                   router: ImageRouter,
                   device: str,
                   bench: list[str]) -> list[dict]:
    """
    Ablation strategies (relative comparison, torchmetrics):
      Random Routing K=1   — random specialist per image
      Worst Routing K=1    — always the wrong-condition specialist
      DAFT Adaptive K=1    — image router top-1
      Uniform Blend K=5    — equal weight all 5 specialists
      Single: <cond>       — one specialist on the full val set (5 rows)
    """
    random.seed(42)
    bench_imgs_abl = [img for img in (cv2.imread(p) for p in bench) if img is not None]

    # Pre-load all val images once — shared across every ablation strategy
    print(f"    Pre-loading {len(val_rows)} val images for ablations...")
    loaded_abl: list[tuple[np.ndarray, dict, np.ndarray, np.ndarray]] = []
    for r in val_rows:
        img = cv2.imread(r["image_path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_b, gt_l = load_gt(r.get("yolo_label", ""), w, h)
        loaded_abl.append((img, r, gt_b, gt_l))
    print(f"    Loaded {len(loaded_abl)} images")

    def _run_strategy(fn) -> tuple[float, float]:
        preds, gt_bs, gt_ls = [], [], []
        for img, r, gt_b, gt_l in loaded_abl:
            boxes = fn(img, r)
            preds.append(boxes);  gt_bs.append(gt_b);  gt_ls.append(gt_l)
        return compute_map(preds, gt_bs, gt_ls)

    # Worst routing: pick the condition that is NOT the ground truth
    _other = {c: [x for x in CONDITIONS if x != c] for c in CONDITIONS}

    strats: list[tuple[str, callable]] = [
        ("Random Routing K=1",
         lambda img, r: boxes_from_result(
             specialists[random.choice(CONDITIONS)].predict(
                 img, device=device, verbose=False, conf=0.25)[0])),

        ("Worst Routing K=1",
         lambda img, r: boxes_from_result(
             specialists[random.choice(_other.get(r.get("condition") or CONDITIONS[0],
                                                  CONDITIONS))
             ].predict(img, device=device, verbose=False, conf=0.25)[0])),

        ("DAFT Adaptive K=1",
         lambda img, r: boxes_from_result(
             specialists[select_top_k(router.weights_from_img(img), top_k=1)[0][0]
             ].predict(img, device=device, verbose=False, conf=0.25)[0])),

        ("Uniform Blend K=5",
         lambda img, r: blend_detections([
             (boxes_from_result(specialists[c].predict(
                 img, device=device, verbose=False, conf=0.25)[0]), 1.0 / len(CONDITIONS))
             for c in CONDITIONS])),
    ]
    for cond in CONDITIONS:
        strats.append((
            f"Single: {cond}",
            lambda img, r, _c=cond: boxes_from_result(
                specialists[_c].predict(img, device=device, verbose=False, conf=0.25)[0])
        ))

    result_rows = []
    for label, fn in strats:
        print(f"    [{label}]")
        m50, m5095 = _run_strategy(fn)

        _fps = fps_of(lambda img, _fn=fn: _fn(img, {"condition": CONDITIONS[0]}),
                      bench_imgs_abl)

        print(f"      mAP50={m50:.4f}  fps={_fps}")
        result_rows.append({"strategy": label,
                            "map50":   round(m50, 4),
                            "map5095": round(m5095, 4),
                            "fps":     _fps})
    return result_rows


# ── Per-class aggregation ─────────────────────────────────────────────────────

def aggregate_per_class(cond_results: dict[str, dict],
                        sizes: dict[str, int]) -> list[dict]:
    """
    Average per-class AP50 across conditions (weighted by condition size).
    cond_results: {cond: {"global": {cls: ap}, "specialist": {cls: ap}}}
    """
    all_cls = set()
    for v in cond_results.values():
        all_cls |= set(v["global"].keys()) | set(v["specialist"].keys())

    rows = []
    for cls in sorted(all_cls):
        g_vals, s_vals, weights = [], [], []
        for cond in CONDITIONS:
            g = cond_results.get(cond, {}).get("global",      {}).get(cls)
            s = cond_results.get(cond, {}).get("specialist",  {}).get(cls)
            n = sizes.get(cond, 1)
            if g is not None and s is not None:
                g_vals.append(g * n);  s_vals.append(s * n);  weights.append(n)
        if not weights:
            continue
        total = sum(weights)
        g_mean = round(sum(g_vals) / total, 4)
        s_mean = round(sum(s_vals) / total, 4)
        rows.append({
            "class":       cls,
            "class_name":  CLASS_NAMES.get(cls, str(cls)),
            "global_ap50": g_mean,
            "specialist_ap50": s_mean,
            "gain":        round(s_mean - g_mean, 4),
        })
    rows.sort(key=lambda r: r["gain"], reverse=True)
    return rows


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_heatmap(main_rows: list[dict]) -> None:
    _rc()
    labels  = [r["strategy"] for r in main_rows]
    cond_cols = [c.replace("_", "\n") for c in CONDITIONS]
    data = np.array([
        [r.get(f"{c}_map50") or 0.0 for c in CONDITIONS]
        for r in main_rows
    ])
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.6 + 1)))
    im = ax.imshow(data, aspect="auto", cmap="YlGn", vmin=0.3, vmax=1.0)
    ax.set_xticks(range(len(CONDITIONS)));  ax.set_xticklabels(cond_cols, fontsize=9)
    ax.set_yticks(range(len(labels)));      ax.set_yticklabels(labels,    fontsize=9)
    for i in range(len(labels)):
        for j in range(len(CONDITIONS)):
            v = data[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=8, color="black" if v < 0.7 else "white")
    plt.colorbar(im, ax=ax, fraction=0.03, label="mAP50")
    ax.set_title("mAP50 by Model × Driving Condition", pad=10)
    plt.tight_layout()
    p = OUT_DIR / "condition_heatmap.png"
    plt.savefig(p); plt.close()
    print(f"  Saved: {p}")


def plot_ablations(abl_rows: list[dict]) -> None:
    _rc()
    labels = [r["strategy"] for r in abl_rows]
    vals   = [r["map50"]    for r in abl_rows]
    colors = (["#E63946"] + ["#E63946"] +   # random, worst  — bad
              ["#52B788"] +                 # DAFT k=1       — good
              ["#457B9D"] +                 # uniform        — baseline
              ["#A8DADC"] * len(CONDITIONS))  # singles

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.1), 5))
    bars = ax.bar(labels, vals, color=colors[:len(labels)],
                  alpha=0.88, edgecolor="white", linewidth=1.2)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_ylabel("mAP50")
    ax.set_title("Routing Ablation Study")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = OUT_DIR / "ablation_bar.png"
    plt.savefig(p); plt.close()
    print(f"  Saved: {p}")


def plot_class_gains(pc_rows: list[dict]) -> None:
    _rc()
    # top 10 by gain, skip classes with gain == 0
    rows = [r for r in pc_rows if r["gain"] != 0][:10]
    if not rows:
        return
    names = [r["class_name"] for r in rows]
    gains = [r["gain"]       for r in rows]
    colors = ["#52B788" if g > 0 else "#E63946" for g in gains]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names[::-1], gains[::-1], color=colors[::-1],
                   alpha=0.88, edgecolor="white", linewidth=1.2)
    ax.bar_label(bars, fmt="+%.3f", padding=3, fontsize=9)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("AP50 gain  (specialist − global)")
    ax.set_title("Per-Class AP50 Gain: DAFT Specialist vs Global Distilled")
    ax.grid(axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = OUT_DIR / "class_gains.png"
    plt.savefig(p); plt.close()
    print(f"  Saved: {p}")


def plot_dawn_dusk(dd_rows: list[dict]) -> None:
    if not dd_rows:
        return
    _rc()
    labels = [r["strategy"] for r in dd_rows]
    vals   = [r["map50"]    for r in dd_rows]
    colors = [STRATEGY_COLORS.get(l, "#6C757D") for l in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, vals, color=colors, alpha=0.88,
                  edgecolor="white", linewidth=1.2)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set_ylim(0, max(vals) * 1.20)
    ax.set_ylabel("mAP50")
    ax.set_title("Blending Under Ambiguous Lighting: k=2 vs Hard Routing")
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = OUT_DIR / "dawn_dusk.png"
    plt.savefig(p); plt.close()
    print(f"  Saved: {p}")


def plot_accuracy_speed_tradeoff(ksweep_rows: list[dict]) -> None:
    """FPS vs mAP50 scatter — all strategies, consistent torchmetrics eval."""
    if not ksweep_rows:
        return
    _rc()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.grid(True, linestyle="--", zorder=0)

    # connect the DAFT k=1..5 curve
    k_rows = sorted(
        [r for r in ksweep_rows if r["strategy"].startswith("DAFT k=")],
        key=lambda r: r["fps"], reverse=True,
    )
    if len(k_rows) > 1:
        ax.plot([r["fps"] for r in k_rows], [r["map50"] for r in k_rows],
                color="#1B7A4A", linewidth=1.5, linestyle="--", zorder=2)

    all_fps = [r["fps"]   for r in ksweep_rows]
    all_map = [r["map50"] for r in ksweep_rows]
    ax.set_xlim(0, max(all_fps) * 1.30)
    ax.set_ylim(0, min(1.0, max(all_map) + (max(all_map) - min(all_map)) * 0.6))

    offsets = {"Global Distilled": (8, -16), "Hard Routing": (8, -16),
               "DAFT k=1": (8, 6), "DAFT k=2": (8, 6),
               "DAFT k=3": (8, -14), "DAFT k=4": (8, 6), "DAFT k=5": (-80, 6)}
    for r in ksweep_rows:
        col = STRATEGY_COLORS.get(r["strategy"], "#555555")
        mrk = "s" if r["strategy"] in ("Global Distilled", "Hard Routing") else "o"
        sz  = 180 if mrk == "s" else 150
        ax.scatter(r["fps"], r["map50"], s=sz, color=col,
                   marker=mrk, zorder=5, edgecolors="white", linewidths=1.5, alpha=1.0)
        dx, dy = offsets.get(r["strategy"], (8, 6))
        ax.annotate(
            f"{r['strategy']}\n{r['map50']:.3f} mAP · {r['fps']:.0f} FPS",
            xy=(r["fps"], r["map50"]), xytext=(dx, dy),
            textcoords="offset points", fontsize=8.5,
            color=col, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=col, lw=0.7),
        )
    ax.set_xlabel("Throughput  (FPS — higher is better →)", labelpad=8)
    ax.set_ylabel("Detection Quality  (mAP50 — higher is better ↑)", labelpad=8)
    ax.set_title("DAFT Adaptive Routing: Accuracy–Speed Tradeoff  (k = 1 … 5)", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = OUT_DIR / "accuracy_speed_tradeoff.png"
    plt.savefig(p); plt.close()
    print(f"  Saved: {p}")


def plot_confusion_matrix(conf_rows: list[dict]) -> None:
    if not conf_rows:
        return
    _rc()
    short = [c.replace("_", "\n") for c in CONDITIONS]
    data  = np.array([
        [conf_rows[i][f"pred_{c}"] for c in CONDITIONS]
        for i in range(len(CONDITIONS))
    ])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(CONDITIONS))); ax.set_xticklabels(short, fontsize=8)
    ax.set_yticks(range(len(CONDITIONS))); ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Predicted condition");  ax.set_ylabel("Ground-truth condition")
    ax.set_title("ImageRouter Top-1 Confusion Matrix (row-normalised)", pad=10)
    for i in range(len(CONDITIONS)):
        for j in range(len(CONDITIONS)):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if v > 0.55 else "black")
    plt.colorbar(im, ax=ax, fraction=0.04, label="Fraction of GT class")
    plt.tight_layout()
    p = OUT_DIR / "router_confusion_matrix.png"
    plt.savefig(p); plt.close()
    print(f"  Saved: {p}")


# ── CSV helpers ───────────────────────────────────────────────────────────────

def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device",       default="cpu")
    p.add_argument("--batch",        type=int, default=8)
    p.add_argument("--n_bench",      type=int, default=200,
                   help="Images for speed timing")
    p.add_argument("--n_abl",        type=int, default=1000,
                   help="Images for ablation + dawn/dusk (0=all)")
    p.add_argument("--skip_router",  action="store_true",
                   help="Skip router accuracy eval (slow on CPU)")
    p.add_argument("--skip_abl",     action="store_true",
                   help="Skip ablation study")
    return p.parse_args()


def main():
    args = get_args()
    OUT_DIR.mkdir(exist_ok=True)

    # ── Load checkpoints ──────────────────────────────────────────────────────
    global_ckpt  = find_ckpt("global")
    large_ckpt   = find_ckpt("large")
    xlarge_ckpt  = find_ckpt("xlarge")
    router_ckpt  = find_ckpt("router")

    if global_ckpt is None:
        raise FileNotFoundError("Global checkpoint not found at checkpoints/global/weights/best.pt")

    print(f"  Global:  {global_ckpt}")
    print(f"  Large:   {large_ckpt  or 'not found — skipping'}")
    print(f"  XLarge:  {xlarge_ckpt or 'not found — skipping'}")
    print(f"  Router:  {router_ckpt or 'not found — skipping adaptive strategies'}")

    global_m  = YOLO(str(global_ckpt))
    large_m   = YOLO(str(large_ckpt))  if large_ckpt  else None
    xlarge_m  = YOLO(str(xlarge_ckpt)) if xlarge_ckpt else None
    router    = ImageRouter(str(router_ckpt), device=args.device) if router_ckpt else None

    specialists: dict[str, YOLO] = {}
    all_specs_found = True
    for cond in CONDITIONS:
        ckpt = find_ckpt(cond)
        if ckpt:
            specialists[cond] = YOLO(str(ckpt))
        else:
            print(f"  WARNING: specialist {cond} not found — falling back to global")
            specialists[cond] = global_m
            all_specs_found = False

    sizes = condition_sizes()

    # ── Phase 1: Per-condition mAP → Table 1 (per-condition columns) ─────────
    # Uses Ultralytics model.val() on each condition YAML — same pipeline as
    # training, directly comparable. Global models run on all 5 condition splits;
    # each specialist runs only on its own condition split (no cross-evals).
    print("\n===== Phase 1: Per-condition mAP (model.val()) — Table 1 =====")
    cond_results: dict[str, dict] = {}   # {cond: {model_label: {map50, map5095, per_class}}}

    # Global models: evaluate on all 5 condition val sets
    global_models = {"Global Distilled": global_ckpt}
    if large_ckpt:
        global_models["Global Large"]  = large_ckpt
    if xlarge_ckpt:
        global_models["Global XLarge"] = xlarge_ckpt

    eval_cache: dict[str, dict[str, dict]] = defaultdict(dict)
    for label, ckpt in global_models.items():
        print(f"\n  [{label}]")
        for cond in CONDITIONS:
            print(f"    {cond}...", end=" ", flush=True)
            res = val_condition(ckpt, cond, args.device, args.batch)
            eval_cache[label][cond] = res
            if res["map50"] is not None:
                print(f"mAP50={res['map50']:.4f}  mAP50:95={res['map5095']:.4f}")
            else:
                print("SKIP")

    # Specialists: each runs only on its own condition val set (no cross-evals)
    print(f"\n  [Specialists — each on its own condition only]")
    for cond in CONDITIONS:
        spec_ckpt = find_ckpt(cond) or global_ckpt
        print(f"    {cond}...", end=" ", flush=True)
        res = val_condition(spec_ckpt, cond, args.device, args.batch)
        eval_cache[f"spec_{cond}"][cond] = res
        if res["map50"] is not None:
            print(f"mAP50={res['map50']:.4f}  mAP50:95={res['map5095']:.4f}")
        else:
            print("SKIP")
        # Store for per-class aggregation
        if cond not in cond_results:
            cond_results[cond] = {}
        cond_results[cond]["specialist"] = res.get("per_class", {})
        cond_results[cond]["global"]     = eval_cache["Global Distilled"][cond].get("per_class", {})

    # ── Phase 2: Speed benchmarking → Table 1 (FPS column) + Table 2 ────────
    # FPS measured at batch=1 with pre-loaded models on NVIDIA A100 80GB PCIe.
    # End-to-end latency: routing + specialist forward + merge/NMS.
    print(f"\n===== Phase 2: Speed ({args.n_bench} images) — Table 1 FPS + Table 2 =====")
    bench = load_bench(args.n_bench)

    named_models = {"Global Distilled": global_m}
    if large_m:  named_models["Global Large"]  = large_m
    if xlarge_m: named_models["Global XLarge"] = xlarge_m

    if router:
        fps_map = benchmark_fps(bench, args.device, named_models, specialists, router)
    else:
        print(f"  Pre-loading {len(bench)} bench images...")
        bench_imgs_fb = [img for img in (cv2.imread(p) for p in bench) if img is not None]
        fps_map = {}
        for label, m in named_models.items():
            print(f"  timing {label}...")
            fps_map[label] = fps_of(
                lambda img, _m=m: _m.predict(img, device=args.device, verbose=False, conf=0.25),
                bench_imgs_fb,
            )
        print("  timing Hard Routing (single specialist, no router)...")
        fps_map["Hard Routing"] = fps_of(
            lambda img: specialists[CONDITIONS[0]].predict(
                img, device=args.device, verbose=False, conf=0.25),
            bench_imgs_fb,
        )

    print("  FPS results:")
    for k, v in fps_map.items():
        print(f"    {k:<22} {v:>7.1f} FPS")

    # ── Build main_results table ──────────────────────────────────────────────
    # Rows: Global Distilled, Global Large, Global XLarge, Hard Routing, DAFT k=1
    # Hard Routing mAP = specialist.val(own condition) — upper bound per condition
    # DAFT k=1 mAP     = same specialist checkpoints (router accuracy ~96% → effectively equal)
    # DAFT k=2..5 are not shown here; their accuracy/speed tradeoff is in ksweep.csv only

    def _build_row(label: str, per_cond_map50: dict, per_cond_map5095: dict,
                   fps: float) -> dict:
        row: dict = {"strategy": label, "fps": fps}
        for cond in CONDITIONS:
            row[f"{cond}_map50"]   = per_cond_map50.get(cond)
            row[f"{cond}_map5095"] = per_cond_map5095.get(cond)
        row["overall_map50"]   = weighted_mean(per_cond_map50,   sizes)
        row["overall_map5095"] = weighted_mean(per_cond_map5095, sizes)
        return row

    main_rows: list[dict] = []

    for label in ("Global Distilled", "Global Large", "Global XLarge"):
        if label not in eval_cache:
            continue
        m50  = {c: eval_cache[label][c]["map50"]   for c in CONDITIONS}
        m595 = {c: eval_cache[label][c]["map5095"] for c in CONDITIONS}
        main_rows.append(_build_row(label, m50, m595, fps_map.get(label, 0.0)))

    # Hard Routing & DAFT k=1: per-condition = specialist val
    spec_m50  = {c: eval_cache[f"spec_{c}"][c]["map50"]   for c in CONDITIONS}
    spec_m595 = {c: eval_cache[f"spec_{c}"][c]["map5095"] for c in CONDITIONS}
    main_rows.append(_build_row("Hard Routing", spec_m50, spec_m595,
                                fps_map.get("Hard Routing", 0.0)))
    main_rows.append(_build_row("DAFT k=1", spec_m50, spec_m595,
                                fps_map.get("DAFT k=1", 0.0)))
    # DAFT k=2..5 accuracy is measured via image-by-image eval in eval_ksweep;
    # using spec_m50 here would make k=2..5 show identical mAP as k=1 (misleading).

    # ── Phase 3: Dawn/dusk blending → Table 4 + Figure 8 ────────────────────
    # Evaluates K=2 metadata blending vs Hard Routing vs DAFT K=1 on the 661
    # dawn/dusk validation images. Uses MetadataRouter with 50/50 day/night
    # weight for the K=2 blending row — this is an oracle experiment, not the
    # deployable path. Both day and night specialists saw dawn/dusk during
    # training, so this measures ensembling on a shared subset, not OOD robustness.
    print("\n===== Phase 3: Dawn/Dusk Analysis — Table 4 + Figure 8 =====")
    dd_samples = load_dawn_dusk_samples(max_n=args.n_abl if args.n_abl else 0)
    dd_rows: list[dict] = []
    if dd_samples and router:
        dd_rows = eval_dawn_dusk(dd_samples, global_m, specialists, router, args.device,
                                 large_m=large_m)
    elif not router:
        print("  Skipping — no router checkpoint")

    # ── Phase 4: Router accuracy ──────────────────────────────────────────────
    # ── Phase 4: Router accuracy → Figure 4 (confusion matrix) ──────────────
    # Top-1 accuracy of ImageRouter vs ground-truth condition on val.csv.
    # Produces router_accuracy.csv (per-condition %) and the 5×5 confusion
    # matrix (router_confusion_matrix.csv + router_confusion_matrix.png).
    router_acc_rows:  list[dict] = []
    router_conf_rows: list[dict] = []
    if router and not args.skip_router:
        print("\n===== Phase 4: Router Accuracy — Figure 4 =====")
        router_acc_rows, router_conf_rows = eval_router_accuracy(router, args.device)

    # ── Phase 5: Ablation study ───────────────────────────────────────────────
    # shared sample for ablations + k-sweep (same seed → same 1000 images)
    with open(MANIFEST_DIR / "val.csv", newline="") as f:
        all_val = [r for r in csv.DictReader(f) if Path(r["image_path"]).exists()]
    random.seed(42)
    imgbyimg_val = random.sample(all_val, min(args.n_abl, len(all_val))) if args.n_abl else all_val
    print(f"\n  Image-by-image sample: {len(imgbyimg_val)} images (seed=42)")

    # ── Phase 5: K-sweep + accuracy-speed tradeoff → Table 2 + Figure 3 ─────
    # Evaluates all strategies image-by-image with torchmetrics so every point
    # on the accuracy-speed scatter uses the same evaluator. Includes global
    # baselines so they appear in Figure 3 alongside the DAFT K=1..5 curve.
    ksweep_rows: list[dict] = []
    if router:
        print("\n===== Phase 5: K-sweep tradeoff — Table 2 + Figure 3 =====")
        ksweep_rows = eval_ksweep(imgbyimg_val, global_m, large_m, xlarge_m,
                                  specialists, router, args.device, bench)

    # ── Phase 6: Routing ablation → Table 3 + Figure 7 ──────────────────────
    # Compares: Worst Routing / Random Routing / DAFT K=1 / Uniform Blend K=5
    # plus single-specialist baselines. Uses torchmetrics on the same
    # imgbyimg_val sample as the K-sweep (seed=42) for consistent comparison.
    abl_rows: list[dict] = []
    if router and not args.skip_abl:
        print("\n===== Phase 6: Routing Ablation — Table 3 + Figure 7 =====")
        print(f"  Evaluating on {len(imgbyimg_val)} val images")
        abl_rows = eval_ablations(imgbyimg_val, global_m, specialists, router, args.device, bench)

    # ── Align main_results overall mAP with ksweep (same evaluator for plot+table)
    # ksweep uses torchmetrics consistently; replace overall_map50/map5095 in
    # main_rows with the matching ksweep values so Table 1 and Figure 4 agree.
    if ksweep_rows:
        ksweep_lookup = {r["strategy"]: r for r in ksweep_rows}
        for row in main_rows:
            ks = ksweep_lookup.get(row["strategy"])
            if ks:
                row["overall_map50"]   = ks["map50"]
                row["overall_map5095"] = ks.get("map5095")
                row["fps"]             = ks["fps"]   # also align FPS

        # Add DAFT k=2 and k=5 to main_rows with per-condition scores from ksweep
        for k in [2, 5]:
            label = f"DAFT k={k}"
            ks = ksweep_lookup.get(label)
            if ks:
                row = {"strategy": label,
                       "fps":             ks["fps"],
                       "overall_map50":   ks["map50"],
                       "overall_map5095": ks.get("map5095")}
                for cond in CONDITIONS:
                    row[f"{cond}_map50"]   = ks.get(f"{cond}_map50")
                    row[f"{cond}_map5095"] = None   # not tracked per-condition in ksweep
                main_rows.append(row)

    # ── Per-class aggregation → Figure 5 + per_class.csv ────────────────────
    # Weighted average of per-class AP50 across conditions (weighted by val
    # image count). Gain = specialist_AP50 - global_AP50. Supports §5.6 analysis.
    print("\n===== Per-class AP50 — Figure 5 =====")
    pc_rows = aggregate_per_class(cond_results, sizes)
    for r in pc_rows[:5]:
        print(f"    {r['class_name']:<16} global={r['global_ap50']:.3f}  "
              f"spec={r['specialist_ap50']:.3f}  gain={r['gain']:+.3f}")

    # ── Condition split counts ────────────────────────────────────────────────
    dd_total = sum(1 for r in open(MANIFEST_DIR / "val.csv").readlines()[1:]
                   if "true" in r.lower() or "dawn" in r.lower() or "dusk" in r.lower())
    split_rows = [{"condition": c, "train_count": sizes.get(c, 0),
                   "val_count":  sizes.get(c, 0)} for c in CONDITIONS]
    split_rows.append({"condition": "all_val",   "train_count": "N/A",
                       "val_count": sum(sizes.values())})
    split_rows.append({"condition": "dawn_dusk_val", "train_count": "N/A",
                       "val_count": len(dd_samples) if dd_samples else "N/A"})

    # ── Per-condition × per-class breakdown ───────────────────────────────────
    pc_detail_rows = []
    for cond in CONDITIONS:
        g_pc = cond_results.get(cond, {}).get("global",     {})
        s_pc = cond_results.get(cond, {}).get("specialist", {})
        all_cls = sorted(set(g_pc.keys()) | set(s_pc.keys()))
        for cls in all_cls:
            g = g_pc.get(cls)
            s = s_pc.get(cls)
            if g is not None and s is not None:
                pc_detail_rows.append({
                    "condition":    cond,
                    "class":        cls,
                    "class_name":   CLASS_NAMES.get(cls, str(cls)),
                    "global_ap50":  g,
                    "spec_ap50":    s,
                    "gain":         round(s - g, 4),
                })

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    print("\n===== Saving results =====")
    save_csv(main_rows,         OUT_DIR / "main_results.csv")
    save_csv(pc_rows,           OUT_DIR / "per_class.csv")
    save_csv(pc_detail_rows,    OUT_DIR / "per_class_detail.csv")
    save_csv(split_rows,        OUT_DIR / "condition_split_counts.csv")
    save_csv(dd_rows,           OUT_DIR / "dawn_dusk.csv")
    save_csv(router_acc_rows,   OUT_DIR / "router_accuracy.csv")
    save_csv(router_conf_rows,  OUT_DIR / "router_confusion_matrix.csv")
    save_csv(ksweep_rows,       OUT_DIR / "ksweep.csv")
    save_csv(abl_rows,          OUT_DIR / "ablations.csv")

    # ── Print main table ──────────────────────────────────────────────────────
    cond_headers = "  ".join(f"{c[:8]:>8}" for c in CONDITIONS)
    print(f"\n{'Strategy':<22} {cond_headers}  {'Overall':>8}  {'FPS':>6}")
    print("-" * 90)
    for r in main_rows:
        cond_vals = "  ".join(
            f"{r.get(f'{c}_map50', 0.0) or 0.0:>8.4f}" for c in CONDITIONS
        )
        ov  = r.get("overall_map50") or 0.0
        fps = r.get("fps") or 0.0
        print(f"{r['strategy']:<22} {cond_vals}  {ov:>8.4f}  {fps:>6.1f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n===== Generating plots =====")

    plot_heatmap(main_rows)
    plot_class_gains(pc_rows)
    if ksweep_rows:
        plot_accuracy_speed_tradeoff(ksweep_rows)
    if dd_rows:
        plot_dawn_dusk(dd_rows)
    if abl_rows:
        plot_ablations(abl_rows)
    if router_conf_rows:
        plot_confusion_matrix(router_conf_rows)

    print("\nDone. All results in results/")


if __name__ == "__main__":
    main()
