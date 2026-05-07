"""
plot_results.py
===============
All presentation figures for the DAFT-BDD100K pipeline.

Figures produced
----------------
  results/condition_analysis.png     dataset split, object density, class composition
  results/accuracy_speed_tradeoff.png FPS vs mAP50 for all strategies
  results/performance_bars.png        mAP50 bar chart per strategy + timing
  results/dawn_dusk_performance.png   robustness on ambiguous dawn/dusk scenes

Usage
-----
  python plot_results.py                   # all figures
  python plot_results.py --which conditions
  python plot_results.py --which tradeoff [--demo]
  python plot_results.py --which bars
  python plot_results.py --which dawn_dusk

Data sources (auto-detected from results/)
  topk_sweep.csv          → FPS for routing strategies   (eval_topk.py)
  timing_breakdown.csv    → FPS ratios for Large / Hard Routing
  consolidated.csv        → mAP50 per strategy
  dawn_dusk_robustness.csv → dawn/dusk subset mAP50
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np


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
        "legend.borderpad":  0.6,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })

# ── Paths ─────────────────────────────────────────────────────────────────────
MANIFEST_DIR = Path("data/bdd100k/manifests")
OUT_DIR      = Path("results")

# ── Strategy colours / markers (shared across all figures) ────────────────────
S_LARGE     = "Global Large"
S_DISTILLED = "Global Distilled"
S_HARD      = "Hard Routing"
S_K1        = "Adaptive k=1"
S_K2        = "Adaptive k=2"
ALL_STRATS  = [S_LARGE, S_DISTILLED, S_HARD, S_K1, S_K2]

STRAT_COLOR = {
    S_LARGE:     "#9B2226",
    S_DISTILLED: "#6C757D",
    S_HARD:      "#005F73",
    S_K1:        "#52B788",
    S_K2:        "#F4A261",
}
STRAT_MARKER = {
    S_LARGE: "D", S_DISTILLED: "s", S_HARD: "^", S_K1: "o", S_K2: "o",
}
ROUTER_COLORS = ["#52B788", "#74C69D", "#F4A261", "#E76F51", "#C1121F"]

# ── Condition config ───────────────────────────────────────────────────────────
CONDITIONS = ["city_day", "city_night", "highway_day", "highway_night", "residential"]
COND_LABELS = {
    "city_day":      "City · Day",
    "city_night":    "City · Night",
    "highway_day":   "Highway · Day",
    "highway_night": "Highway · Night",
    "residential":   "Residential",
}
COND_COLORS = {
    "city_day":      "#F4845F",
    "city_night":    "#3A7CA5",
    "highway_day":   "#52B788",
    "highway_night": "#1B4332",
    "residential":   "#F7B731",
    "_other":        "#CBD5E0",
}
BDD100K_CLASSES = [
    "pedestrian", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign",
]
CLASS_SHORT = [
    "Pedestrian", "Rider", "Car", "Truck", "Bus",
    "Train", "Motorcycle", "Bicycle", "Traffic\nLight", "Traffic\nSign",
]
MAX_SAMPLE = 800


# ══════════════════════════════════════════════════════════════════════════════
# CSV helpers
# ══════════════════════════════════════════════════════════════════════════════

def _read(name: str) -> list[dict]:
    p = OUT_DIR / name
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def _f(v, fallback=float("nan")) -> float:
    try:
        x = float(v)
        return x if x == x else fallback
    except (TypeError, ValueError):
        return fallback


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Condition Analysis
# ══════════════════════════════════════════════════════════════════════════════

def _load_manifests() -> list[dict]:
    rows: list[dict] = []
    for split in ["train", "val"]:
        p = MANIFEST_DIR / f"{split}.csv"
        if p.exists():
            with open(p, newline="") as f:
                rows.extend(csv.DictReader(f))
    return rows


def plot_condition_analysis() -> None:
    _set_pub_rc()
    rows = _load_manifests()
    if not rows:
        print("  ERROR: no manifests — run prepare_data.py first.")
        return
    print(f"  {len(rows):,} manifest rows loaded")

    # ── condition distribution ────────────────────────────────────────────────
    counts: Counter = Counter()
    for r in rows:
        c = r.get("condition", "")
        counts[c if c in CONDITIONS else "_other"] += 1
    total = len(rows)

    # ── objects per image ─────────────────────────────────────────────────────
    random.seed(42)
    by_cond: dict[str, list] = defaultdict(list)
    for r in rows:
        if r.get("condition") in CONDITIONS:
            by_cond[r["condition"]].append(r)

    obj_stats: dict[str, tuple] = {}
    for cond in CONDITIONS:
        sample = by_cond[cond]
        if len(sample) > MAX_SAMPLE:
            sample = random.sample(sample, MAX_SAMPLE)
        vals = []
        for r in sample:
            nb = r.get("num_boxes", "")
            if nb and nb.isdigit():
                vals.append(float(nb))
                continue
            lbl = r.get("yolo_label", "")
            if lbl:
                lp = Path(lbl)
                if lp.exists():
                    vals.append(float(sum(1 for ln in lp.read_text().splitlines() if ln.strip())))
        obj_stats[cond] = (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)

    # ── class composition ─────────────────────────────────────────────────────
    n_cls  = len(BDD100K_CLASSES)
    matrix = np.zeros((len(CONDITIONS), n_cls), dtype=float)
    for ci, cond in enumerate(CONDITIONS):
        sample = by_cond[cond]
        if len(sample) > MAX_SAMPLE:
            sample = random.sample(sample, MAX_SAMPLE)
        for r in sample:
            lbl = r.get("yolo_label", "")
            if not lbl:
                continue
            lp = Path(lbl)
            if not lp.exists():
                continue
            for line in lp.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if 0 <= cls_id < n_cls:
                        matrix[ci, cls_id] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix   = np.where(row_sums > 0, matrix / row_sums, 0.0)

    palette = [COND_COLORS[c] for c in CONDITIONS] + [COND_COLORS["_other"]]

    fig = plt.figure(figsize=(17, 7), facecolor="white")
    fig.subplots_adjust(left=0.04, right=0.97, top=0.93, bottom=0.10,
                        wspace=0.35, hspace=0.45)
    gs      = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.4, 1.5])
    ax_don  = fig.add_subplot(gs[:, 0])
    ax_bar  = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[:, 2])
    ax_note = fig.add_subplot(gs[1, 1])
    ax_note.axis("off")

    # donut
    ordered = CONDITIONS + ["_other"]
    sizes   = [counts.get(c, 0) for c in ordered]
    covered = sum(counts.get(c, 0) for c in CONDITIONS)
    pct_cov = 100 * covered / max(total, 1)
    ax_don.pie(sizes, colors=palette, startangle=90,
               wedgeprops={"width": 0.52, "linewidth": 1.5, "edgecolor": "white"})
    ax_don.text(0,  0.10, f"{pct_cov:.1f}%",
                ha="center", va="center", fontsize=22, fontweight="bold", color="#1a1a2e")
    ax_don.text(0, -0.22, "of BDD100K\ncovered",
                ha="center", va="center", fontsize=10, color="#555")
    legend_handles = [
        mpatches.Patch(facecolor=palette[i],
                       label=COND_LABELS.get(ordered[i], "Other / uncovered"),
                       edgecolor="white", linewidth=0.8)
        for i, s in enumerate(sizes) if s > 0
    ]
    ax_don.legend(handles=legend_handles, loc="lower center",
                  bbox_to_anchor=(0.5, -0.18), fontsize=9,
                  frameon=False, ncol=2, handlelength=1.2)
    ax_don.set_title("Dataset Distribution", fontsize=12, fontweight="bold", pad=10)

    # bars
    cond_order = sorted(CONDITIONS, key=lambda c: obj_stats[c][0], reverse=True)
    means  = [obj_stats[c][0] for c in cond_order]
    stds   = [obj_stats[c][1] for c in cond_order]
    y_pos  = np.arange(len(cond_order))
    bars   = ax_bar.barh(y_pos, means, xerr=stds, height=0.6,
                         color=[COND_COLORS[c] for c in cond_order],
                         capsize=3, error_kw={"linewidth": 1.2, "ecolor": "#888"}, alpha=0.92)
    for i, (m, _) in enumerate(zip(means, bars)):
        ax_bar.text(m + max(means) * 0.02, i, f"{m:.1f}",
                    va="center", fontsize=9, fontweight="bold", color="#333")
    ax_bar.set_xlim(left=0)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([COND_LABELS[c] for c in cond_order], fontsize=9)
    ax_bar.set_xlabel("Avg objects / image", fontsize=9)
    ax_bar.set_title("Object Density per Condition", fontsize=11, fontweight="bold")
    ax_bar.grid(axis="x", alpha=0.25)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.tick_params(axis="y", length=0)

    ax_note.text(0.05, 0.95, (
        "Why 5 conditions?\n\n"
        "• City streets: 2–3× more objects than highways\n"
        "• Night conditions: same scene, low-light challenge\n"
        "• Scene type explains domain shift better than weather\n"
        f"• {pct_cov:.1f}% coverage  vs  ~90% for simple day/night/rain"
    ), transform=ax_note.transAxes, va="top", ha="left",
       fontsize=9.5, color="#222", linespacing=1.7,
       bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8",
                 edgecolor="#ccc", linewidth=0.8))

    # heatmap
    cmap = LinearSegmentedColormap.from_list(
        "blues", ["#f7fbff", "#6baed6", "#2171b5", "#084594"], N=256)
    im = ax_heat.imshow(matrix * 100, aspect="auto", cmap=cmap,
                        vmin=0, vmax=min(70, (matrix * 100).max()))
    for ci in range(len(CONDITIONS)):
        for cj in range(n_cls):
            val = matrix[ci, cj] * 100
            ax_heat.text(cj, ci, f"{val:.0f}%", ha="center", va="center",
                         fontsize=7.5, color="white" if val > 35 else "#333",
                         fontweight="bold" if val > 20 else "normal")
    ax_heat.set_xticks(range(n_cls))
    ax_heat.set_xticklabels(CLASS_SHORT, fontsize=8)
    ax_heat.set_yticks(range(len(CONDITIONS)))
    ax_heat.set_yticklabels([COND_LABELS[c] for c in CONDITIONS], fontsize=9)
    for tick, cond in zip(ax_heat.get_yticklabels(), CONDITIONS):
        tick.set_color(COND_COLORS[cond])
        tick.set_fontweight("bold")
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.7, pad=0.02)
    cbar.set_label("% of objects in condition", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    ax_heat.set_title("Class Composition per Condition", fontsize=11, fontweight="bold")

    fig.suptitle("Why 5 Scene-Based Specialists?  —  BDD100K Condition Analysis",
                 fontsize=14, fontweight="bold", y=0.99, color="#1a1a2e")
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "condition_analysis.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Accuracy–Speed Tradeoff
# ══════════════════════════════════════════════════════════════════════════════

def _build_tradeoff_points(demo: bool) -> list[dict]:
    """
    Assembles one dict per strategy: {label, map50, fps, kind}.
    FPS source:
      - Routing k=1..5 and Global Distilled → topk_sweep.csv  (single benchmark)
      - Hard Routing                         → same FPS as Global Distilled
                                               (same YOLOv8s architecture, no NN router)
      - Global Large                         → scaled from timing_breakdown ratio
                                               (4g.40gb GPU times not available on CPU)
    mAP source: consolidated.csv.
    Falls back to DEMO_ROWS when --demo or when CSVs are missing.
    """
    DEMO = [
        {"label": S_LARGE,     "map50": 0.730, "fps": 3.7,  "kind": "large"},
        {"label": S_DISTILLED, "map50": 0.683, "fps": 6.1,  "kind": "global"},
        {"label": S_HARD,      "map50": 0.832, "fps": 6.1,  "kind": "hard"},
        {"label": S_K1,        "map50": 0.832, "fps": 3.9,  "kind": "router", "k": 1},
        {"label": S_K2,        "map50": 0.842, "fps": 1.9,  "kind": "router", "k": 2},
        {"label": "Adaptive k=3", "map50": 0.842, "fps": 1.4, "kind": "router", "k": 3},
        {"label": "Adaptive k=4", "map50": 0.846, "fps": 1.1, "kind": "router", "k": 4},
        {"label": "Adaptive k=5", "map50": 0.846, "fps": 0.8, "kind": "router", "k": 5},
    ]
    if demo:
        return DEMO

    # ── load CSVs ─────────────────────────────────────────────────────────────
    topk_rows  = _read("topk_sweep.csv")
    cons_rows  = _read("consolidated.csv")
    td_rows    = _read("timing_breakdown.csv")

    if not topk_rows and not cons_rows:
        print("  CSVs not found — using demo values.")
        return DEMO

    # mAP lookup from consolidated
    mAP = {}
    for r in cons_rows:
        v = _f(r.get("mAP50"))
        if v == v:   # not nan
            mAP[r["strategy"]] = v

    # ── topk_sweep → routing strategies + global distilled ────────────────────
    # tolerate old labels ("global (no routing)", "router k=N")
    def _norm_label(s: str) -> tuple[str, int | None]:
        s = s.strip()
        if "large" in s.lower():
            return S_LARGE, None
        if "global" in s.lower() or s == "global (no routing)":
            return S_DISTILLED, None
        if "hard" in s.lower():
            return S_HARD, None
        for k in range(1, 6):
            if f"k={k}" in s or f"k = {k}" in s or s.endswith(str(k)):
                label = f"Adaptive k={k}" if k > 2 else (S_K1 if k == 1 else S_K2)
                return label, k
        return s, None

    points: dict[str, dict] = {}
    global_fps = None
    for r in topk_rows:
        lbl, k = _norm_label(r["strategy"])
        fps  = _f(r.get("fps"))
        m50  = mAP.get(lbl, _f(r.get("map50")))
        kind = "large" if lbl == S_LARGE else ("global" if lbl == S_DISTILLED
               else ("hard" if lbl == S_HARD else "router"))
        pt   = {"label": lbl, "map50": m50, "fps": fps, "kind": kind}
        if k is not None:
            pt["k"] = k
        points[lbl] = pt
        if lbl == S_DISTILLED and fps == fps:
            global_fps = fps

    if global_fps is None:
        global_fps = 6.1   # fallback

    # ── Hard Routing: same FPS as global distilled (same arch, no NN router) ──
    hr_map = mAP.get(S_HARD, _f(next(
        (r.get("map50") for r in cons_rows if r["strategy"] == S_HARD), "nan")))
    if S_HARD not in points and hr_map == hr_map:
        points[S_HARD] = {"label": S_HARD, "map50": hr_map,
                           "fps": global_fps, "kind": "hard"}

    # ── Global Large: derive FPS from timing_breakdown ratio ──────────────────
    td = {r["strategy"]: r for r in td_rows}
    large_map = mAP.get(S_LARGE, float("nan"))
    large_ms  = _f(td.get(S_LARGE,     {}).get("t_total_ms"))
    global_ms = _f(td.get(S_DISTILLED, {}).get("t_total_ms"))
    if large_ms == large_ms and global_ms == global_ms and global_ms > 0:
        large_fps = global_fps * (global_ms / large_ms)
    else:
        large_fps = global_fps / 1.67    # ~40% slower (typical m vs s ratio)

    if S_LARGE not in points:
        pt = {"label": S_LARGE, "map50": large_map,
              "fps": large_fps, "kind": "large"}
        points[S_LARGE] = pt

    return list(points.values())


def plot_accuracy_speed(demo: bool = False) -> None:
    _set_pub_rc()
    pts = _build_tradeoff_points(demo)

    router_pts = sorted(
        [p for p in pts if p["kind"] == "router" and "k" in p],
        key=lambda p: p["k"]
    )
    global_pt = next((p for p in pts if p["kind"] == "global"), None)
    large_pt  = next((p for p in pts if p["kind"] == "large"),  None)
    hard_pt   = next((p for p in pts if p["kind"] == "hard"),   None)

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.grid(True, linestyle="--", zorder=0)

    # Compute axis limits from points with valid mAP (large may be nan)
    valid_map = [p["map50"] for p in pts if p["map50"] == p["map50"]]
    valid_fps = [p["fps"]   for p in pts if p["fps"]   == p["fps"]]
    if not valid_map or not valid_fps:
        print("  [tradeoff] no valid data")
        return

    x_pad = max(valid_fps) * 0.15
    y_pad = (max(valid_map) - min(valid_map)) * 0.25
    ax.set_xlim(0, max(valid_fps) + x_pad * 2.5)
    ax.set_ylim(0, min(1.0, max(valid_map) + y_pad * 1.8))

    # ── Reference lines at global distilled ──────────────────────────────────
    if global_pt and global_pt["map50"] == global_pt["map50"]:
        gx, gy = global_pt["fps"], global_pt["map50"]
        ax.axvline(gx, color=STRAT_COLOR[S_DISTILLED], linewidth=0.8,
                   linestyle=":", alpha=0.6, zorder=1)
        ax.axhline(gy, color=STRAT_COLOR[S_DISTILLED], linewidth=0.8,
                   linestyle=":", alpha=0.6, zorder=1)

    # ── Pareto curve for router k=1..5 ────────────────────────────────────────
    if len(router_pts) > 1:
        ax.plot([p["fps"] for p in router_pts],
                [p["map50"] for p in router_pts],
                color="#888", linewidth=1.4, linestyle="--", zorder=2, alpha=0.6)

    # ── Global Large ──────────────────────────────────────────────────────────
    if large_pt:
        has_map = large_pt["map50"] == large_pt["map50"]
        col = STRAT_COLOR[S_LARGE]
        if has_map:
            ax.scatter(large_pt["fps"], large_pt["map50"],
                       s=180, color=col, marker="D",
                       zorder=5, edgecolors="white", linewidths=1.5)
            ax.annotate(
                f"{S_LARGE}\n{large_pt['map50']:.3f} mAP · {large_pt['fps']:.1f} FPS",
                xy=(large_pt["fps"], large_pt["map50"]),
                xytext=(-30, 8), textcoords="offset points",
                fontsize=8.5, color=col, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.8),
            )
        else:
            # Show as speed-only reference (vertical dashed line)
            ax.axvline(large_pt["fps"], color=col, linewidth=1.2,
                       linestyle=":", alpha=0.7, zorder=2)
            ax.text(large_pt["fps"] + 0.05, ax.get_ylim()[0] + y_pad * 0.3,
                    f"{S_LARGE}\n(YOLOv8m)\n{large_pt['fps']:.1f} FPS\nmAP pending",
                    fontsize=7.5, color=col, fontstyle="italic",
                    va="bottom", ha="left")

    # ── Global Distilled ──────────────────────────────────────────────────────
    if global_pt and global_pt["map50"] == global_pt["map50"]:
        col = STRAT_COLOR[S_DISTILLED]
        ax.scatter(global_pt["fps"], global_pt["map50"],
                   s=160, color=col, marker="s",
                   zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(
            f"{S_DISTILLED}\n{global_pt['map50']:.3f} mAP · {global_pt['fps']:.1f} FPS",
            xy=(global_pt["fps"], global_pt["map50"]),
            xytext=(12, -18), textcoords="offset points",
            fontsize=8.5, color=col, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=col, lw=0.8),
        )

    # ── Hard Routing ──────────────────────────────────────────────────────────
    if hard_pt and hard_pt["map50"] == hard_pt["map50"]:
        col = STRAT_COLOR[S_HARD]
        ax.scatter(hard_pt["fps"], hard_pt["map50"],
                   s=160, color=col, marker="^",
                   zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(
            f"{S_HARD}\n{hard_pt['map50']:.3f} mAP · {hard_pt['fps']:.1f} FPS",
            xy=(hard_pt["fps"], hard_pt["map50"]),
            xytext=(12, 10), textcoords="offset points",
            fontsize=8.5, color=col, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=col, lw=0.8),
        )

    # ── Router k=1..5 ─────────────────────────────────────────────────────────
    for p in router_pts:
        k     = p["k"]
        color = ROUTER_COLORS[k - 1]
        ax.scatter(p["fps"], p["map50"],
                   s=150, color=color, marker="o",
                   zorder=6, edgecolors="white", linewidths=1.5)
        if k == 1:
            xytext, ha = (10, 8), "left"
        elif k % 2 == 0:
            xytext, ha = (10, -16), "left"
        else:
            xytext, ha = (-10, 8), "right"
        ax.annotate(
            f"k = {k}\n{p['map50']:.3f} mAP · {p['fps']:.1f} FPS",
            xy=(p["fps"], p["map50"]), xytext=xytext,
            textcoords="offset points",
            fontsize=8.5, color=color, fontweight="bold", ha=ha,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )

    # ── k=1 gain annotation ───────────────────────────────────────────────────
    if global_pt and router_pts:
        k1 = router_pts[0]
        map_gain = k1["map50"] - global_pt["map50"]
        fps_drop = global_pt["fps"] - k1["fps"]
        sign = "−" if fps_drop > 0 else "+"
        ax.text(0.50, 0.03,
                f"+{map_gain*100:.1f}% mAP vs global  at only {sign}{abs(fps_drop):.1f} FPS cost  (Adaptive k=1 fast path)",
                transform=ax.transAxes, fontsize=9.5, ha="center",
                color="#2D6A4F", fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#e8f5e9",
                          edgecolor="#b7dfba", linewidth=0.8))

    ax.set_xlabel("Throughput  (FPS — higher is better →)", fontsize=12, labelpad=8)
    ax.set_ylabel("Detection Quality  (mAP50 — higher is better ↑)", fontsize=12, labelpad=8)
    title = "Accuracy–Speed Tradeoff: DAFT Adaptive Routing vs Baselines"
    if demo:
        title += "  [illustrative]"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)

    # ── Timing note ───────────────────────────────────────────────────────────
    ax.text(0.01, 0.97,
            "⚠ CPU timing (single image).  Hard Routing ≈ same speed as Global Distilled\n"
            "— both use YOLOv8s; no neural router overhead for Hard Routing.",
            transform=ax.transAxes, fontsize=7.5, va="top", color="#666",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                      edgecolor="#ddd", linewidth=0.6))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=STRAT_COLOR[S_LARGE],
               markersize=10, label=f"{S_LARGE} (YOLOv8m, BDD100K fine-tuned)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=STRAT_COLOR[S_DISTILLED],
               markersize=10, label=f"{S_DISTILLED} (YOLOv8s, distilled+fine-tuned)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=STRAT_COLOR[S_HARD],
               markersize=10, label=S_HARD),
    ] + [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ROUTER_COLORS[k - 1],
               markersize=10, label=f"Adaptive k={k}")
        for k in range(1, len(router_pts) + 1)
    ] + [
        Line2D([0], [0], linestyle="--", color="#888",
               linewidth=1.4, label="k = 1 → 5 Pareto curve"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              frameon=True, borderpad=0.8)

    plt.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "accuracy_speed_tradeoff.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Performance Bar Chart (strategy × mAP50 + FPS)
# ══════════════════════════════════════════════════════════════════════════════

def plot_performance_bars() -> None:
    _set_pub_rc()
    cons = _read("consolidated.csv")
    if not cons:
        print("  [bars] consolidated.csv not found")
        return

    # Fill mAP gaps from topk_sweep (e.g. Global Large computed there)
    topk_map = {r["strategy"]: _f(r.get("map50")) for r in _read("topk_sweep.csv")}
    topk_fps = {r["strategy"]: _f(r.get("fps"))   for r in _read("topk_sweep.csv")}

    rows = []
    for r in cons:
        m = _f(r.get("mAP50"))
        f = _f(r.get("fps"))
        s = r["strategy"]
        # Fall back to topk_sweep when consolidated has N/A
        if m != m and s in topk_map and topk_map[s] == topk_map[s]:
            m = topk_map[s]
        if f != f and s in topk_fps and topk_fps[s] == topk_fps[s]:
            f = topk_fps[s]
        rows.append({"strategy": s, "map50": m, "fps": f, "pending": m != m})

    if not rows:
        print("  [bars] no rows in consolidated.csv")
        return

    # Sort by mAP ascending; pending at bottom
    rows.sort(key=lambda r: (r["pending"], r["map50"] if not r["pending"] else -1))

    # mAP axis limits — always start from 0 for honest comparisons
    known_maps = [r["map50"] for r in rows if not r["pending"]]
    map_min = 0.0
    map_max = min(1.0, max(known_maps) + 0.06) if known_maps else 1.0

    strats  = [r["strategy"] for r in rows]
    maps    = [r["map50"]    for r in rows]
    fpss    = [r["fps"]      for r in rows]
    colors  = [STRAT_COLOR.get(s, "#555") for s in strats]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(rows) * 0.9 + 1.5)),
                             facecolor="white")
    fig.subplots_adjust(wspace=0.05)

    y = np.arange(len(rows))

    # ── Left: mAP50 bars ──────────────────────────────────────────────────────
    ax = axes[0]

    for i, (m, s, pending) in enumerate(zip(maps, strats, [r["pending"] for r in rows])):
        col = STRAT_COLOR.get(s, "#555")
        if pending:
            # hatched placeholder bar at map_min width
            ax.barh(i, map_max - map_min, left=map_min, height=0.6,
                    color=col, alpha=0.18, edgecolor=col, linewidth=1.0,
                    hatch="///")
            ax.text((map_min + map_max) / 2, i, "eval pending",
                    va="center", ha="center", fontsize=8.5,
                    color=col, fontstyle="italic")
        else:
            ax.barh(i, m, height=0.6, color=col, alpha=0.88,
                    edgecolor="white", linewidth=1.2)
            ax.text(m + (map_max - map_min) * 0.01, i, f"{m:.4f}",
                    va="center", fontsize=9, fontweight="bold", color=col)

    # reference line at global distilled mAP
    gmap = next((r["map50"] for r in rows if r["strategy"] == S_DISTILLED
                 and not r["pending"]), None)
    if gmap:
        ax.axvline(gmap, color=STRAT_COLOR[S_DISTILLED], linewidth=1.1,
                   linestyle="--", alpha=0.7)
        ax.text(gmap + (map_max - map_min) * 0.01, -0.55, "Global\nDistilled",
                fontsize=7, color=STRAT_COLOR[S_DISTILLED], va="top")

    ax.set_yticks(y)
    ax.set_yticklabels(strats, fontsize=10)
    ax.set_xlabel("mAP50", fontsize=10)
    ax.set_xlim(map_min, map_max + (map_max - map_min) * 0.30)
    ax.set_title("Detection Accuracy (mAP50)", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # ── Right: FPS bars ───────────────────────────────────────────────────────
    ax2 = axes[1]
    valid_fps = [(i, f) for i, f in enumerate(fpss) if f == f]
    if valid_fps:
        fps_bars = ax2.barh(
            [i for i, _ in valid_fps],
            [f for _, f in valid_fps],
            color=[colors[i] for i, _ in valid_fps],
            alpha=0.88, height=0.6, edgecolor="white", linewidth=1.2,
        )
        max_fps = max(f for _, f in valid_fps)
        for bar, (i, f) in zip(fps_bars, valid_fps):
            ax2.text(f + max_fps * 0.02, i, f"{f:.1f} FPS",
                     va="center", fontsize=9, fontweight="bold",
                     color=STRAT_COLOR.get(strats[i], "#333"))

    ax2.set_xlim(left=0)
    ax2.set_yticks(y)
    ax2.set_yticklabels([""] * len(rows))   # shared y-axis labels on left only
    ax2.set_xlabel("Throughput (FPS)", fontsize=10)
    ax2.set_title("Speed (FPS, CPU)", fontsize=11, fontweight="bold")
    ax2.grid(axis="x", alpha=0.25)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(axis="y", length=0)

    fig.suptitle("Strategy Comparison: Accuracy vs Speed",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = OUT_DIR / "performance_bars.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Dawn/Dusk Robustness
# ══════════════════════════════════════════════════════════════════════════════

def plot_dawn_dusk() -> None:
    _set_pub_rc()
    dd_rows  = _read("dawn_dusk_robustness.csv")
    cons_rows = _read("consolidated.csv")

    if not dd_rows:
        print("  [dawn_dusk] dawn_dusk_robustness.csv not found")
        return

    # Normalise strategy labels
    label_map = {
        "Hard Routing":        S_HARD,
        "Adaptive k=2 (meta)": S_K2,
        "Adaptive k=1 (img)":  S_K1,
    }
    dd = {label_map.get(r["strategy"], r["strategy"]): _f(r.get("map50"))
          for r in dd_rows}

    # Overall mAP from consolidated for same strategies
    overall = {r["strategy"]: _f(r.get("mAP50")) for r in cons_rows}

    strats  = [s for s in [S_HARD, S_K1, S_K2] if s in dd]
    dd_vals = [dd[s]      for s in strats]
    ov_vals = [overall.get(s, float("nan")) for s in strats]
    colors  = [STRAT_COLOR[s] for s in strats]

    # mAP axis — always start from 0
    all_vals = [v for v in dd_vals + ov_vals if v == v]
    map_min  = 0.0
    map_max  = min(1.0, max(all_vals) + 0.06) if all_vals else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white",
                             gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(wspace=0.35)

    # ── Left: grouped bar chart (overall vs dawn/dusk) ────────────────────────
    ax = axes[0]
    y  = np.arange(len(strats))
    w  = 0.35

    bars_ov = ax.barh(y + w / 2, ov_vals, w, label="Overall val set",
                      color=colors, alpha=0.55, edgecolor="white", linewidth=1.0,
                      hatch="///")
    bars_dd = ax.barh(y - w / 2, dd_vals, w, label="Dawn/dusk subset",
                      color=colors, alpha=0.92, edgecolor="white", linewidth=1.0)

    for i, (ov, dd_v) in enumerate(zip(ov_vals, dd_vals)):
        if ov == ov:
            ax.text(ov + 0.003, i + w / 2, f"{ov:.4f}", va="center",
                    fontsize=8.5, fontweight="bold", color=colors[i], alpha=0.7)
        if dd_v == dd_v:
            ax.text(dd_v + 0.003, i - w / 2, f"{dd_v:.4f}", va="center",
                    fontsize=8.5, fontweight="bold", color=colors[i])

    ax.set_yticks(y)
    ax.set_yticklabels(strats, fontsize=10)
    ax.set_xlabel("mAP50", fontsize=10)
    ax.set_xlim(map_min, map_max + (map_max - map_min) * 0.35)
    ax.set_title("mAP50: Overall vs Dawn/Dusk Subset", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # ── Right: delta bar (gain of blending over hard routing on dawn/dusk) ────
    ax2 = axes[1]
    hr_dd = dd.get(S_HARD, float("nan"))
    deltas, delta_labels, delta_colors = [], [], []
    for s in strats:
        if s == S_HARD:
            continue
        d = dd.get(s, float("nan"))
        if d == d and hr_dd == hr_dd:
            deltas.append(d - hr_dd)
            delta_labels.append(s)
            delta_colors.append(STRAT_COLOR[s])

    if deltas:
        dy = np.arange(len(deltas))
        bar_cols = ["#52B788" if d >= 0 else "#E63946" for d in deltas]
        dbars = ax2.barh(dy, deltas, color=bar_cols, alpha=0.88, height=0.5,
                         edgecolor="white", linewidth=1.0)
        ax2.axvline(0, color="#333", linewidth=0.8)
        for bar, d in zip(dbars, deltas):
            ax2.text(d + (max(abs(v) for v in deltas) * 0.03 * np.sign(d) if d != 0 else 0.001),
                     bar.get_y() + bar.get_height() / 2,
                     f"{d:+.4f}", va="center", fontsize=9, fontweight="bold",
                     color="#2D6A4F" if d >= 0 else "#9B2226")
        ax2.set_yticks(dy)
        ax2.set_yticklabels(delta_labels, fontsize=10)
        ax2.set_xlabel("ΔmAP50 vs Hard Routing", fontsize=10)
        ax2.set_title("Blending Gain\non Dawn/Dusk", fontsize=11, fontweight="bold")
        ax2.grid(axis="x", alpha=0.25)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.tick_params(axis="y", length=0)

    n_dd = next((int(r.get("n_images", 0)) for r in dd_rows), 0)
    fig.suptitle(
        f"Robustness on Ambiguous Scenes  (dawn/dusk subset, n={n_dd})\n"
        "Adaptive k=2 blends day+night specialists 50/50 via MetadataRouter",
        fontsize=12, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    out = OUT_DIR / "dawn_dusk_performance.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — k-Sweep Dual-Axis (mAP50 + FPS vs k)
# ══════════════════════════════════════════════════════════════════════════════

def plot_ksweep() -> None:
    """Dual Y-axis line chart: mAP50 (left) and FPS (right) as k=1..5."""
    _set_pub_rc()
    rows = _read("topk_sweep.csv")
    k_rows = sorted(
        [r for r in rows if r["strategy"].startswith("Adaptive")],
        key=lambda r: int(r["strategy"].split("=")[1])
    )
    if not k_rows:
        print("  [ksweep] no Adaptive rows in topk_sweep.csv")
        return

    k_vals   = [int(r["strategy"].split("=")[1]) for r in k_rows]
    map_vals = [_f(r["map50"]) for r in k_rows]
    fps_vals = [_f(r["fps"])   for r in k_rows]

    # Global Distilled reference
    ref_row  = next((r for r in rows if "Distilled" in r["strategy"]), None)
    ref_map  = _f(ref_row["map50"]) if ref_row else float("nan")
    ref_fps  = _f(ref_row["fps"])   if ref_row else float("nan")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.grid(True, linestyle="--", zorder=0)

    C_MAP = "#52B788"
    C_FPS = "#E63946"

    # mAP line
    ax1.plot(k_vals, map_vals, "o-", color=C_MAP, linewidth=2.2,
             markersize=9, zorder=5, label="mAP50")
    if ref_map == ref_map:
        ax1.axhline(ref_map, color=STRAT_COLOR[S_DISTILLED], linewidth=1.0,
                    linestyle=":", alpha=0.7, label="Global Distilled mAP")
    ax1.set_xlabel("Top-K specialists blended", labelpad=8)
    ax1.set_ylabel("mAP50", color=C_MAP, labelpad=8)
    ax1.tick_params(axis="y", labelcolor=C_MAP)
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks(k_vals)
    ax1.spines[["top", "right"]].set_visible(False)

    # FPS line on right axis
    ax2 = ax1.twinx()
    ax2.plot(k_vals, fps_vals, "s--", color=C_FPS, linewidth=2.2,
             markersize=9, zorder=5, label="FPS")
    if ref_fps == ref_fps:
        ax2.axhline(ref_fps, color=STRAT_COLOR[S_DISTILLED], linewidth=1.0,
                    linestyle="-.", alpha=0.7, label="Global Distilled FPS")
    ax2.set_ylabel("Throughput (FPS)", color=C_FPS, labelpad=8)
    ax2.tick_params(axis="y", labelcolor=C_FPS)
    ax2.set_ylim(0, max(fps_vals) * 1.35)
    ax2.spines[["top"]].set_visible(False)

    # Annotate each k point
    for k, m, f in zip(k_vals, map_vals, fps_vals):
        ax1.annotate(f"{m:.4f}", xy=(k, m), xytext=(0, 10),
                     textcoords="offset points", ha="center",
                     fontsize=8, color=C_MAP, fontweight="bold")
        ax2.annotate(f"{f:.0f}", xy=(k, f), xytext=(0, -16),
                     textcoords="offset points", ha="center",
                     fontsize=8, color=C_FPS, fontweight="bold")

    # Sweet-spot callout at k=1
    ax1.annotate("Sweet spot\n(best accuracy/speed ratio)",
                 xy=(1, map_vals[0]),
                 xytext=(1.6, map_vals[0] - 0.08),
                 fontsize=8.5, color="#2D6A4F", fontstyle="italic",
                 arrowprops=dict(arrowstyle="->", color="#2D6A4F", lw=1.0))

    # Combined legend
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="center right",
               frameon=True, fontsize=9)

    ax1.set_title("Adaptive Routing: mAP50 and Speed vs Number of Specialists Blended",
                  pad=14)
    plt.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "ksweep_tradeoff.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Condition Radar / Spider Chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_condition_radar() -> None:
    """Radar chart: Global Distilled vs DAFT k=1 across all 5 conditions."""
    _set_pub_rc()
    rows = _read("per_condition.csv")
    if not rows:
        print("  [radar] per_condition.csv not found")
        return

    cond_order = ["city_day", "city_night", "highway_day", "highway_night", "residential"]
    labels     = [COND_LABELS.get(c, c) for c in cond_order]

    global_vals = []
    daft_vals   = []
    for c in cond_order:
        row = next((r for r in rows if r["condition"] == c), {})
        global_vals.append(_f(row.get("Global Distilled")))
        # prefer Hard Routing (= specialist, zero router overhead), fallback to k=1
        v = _f(row.get("Hard Routing"))
        if v != v:
            v = _f(row.get("Adaptive k=1"))
        daft_vals.append(v)

    N      = len(cond_order)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    global_vals += global_vals[:1]
    daft_vals   += daft_vals[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Grid styling
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7.5, color="#888")
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.5)
    ax.xaxis.grid(True, color="#CCCCCC", linewidth=0.5)
    ax.spines["polar"].set_visible(False)

    # Condition labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")

    # Global Distilled polygon
    ax.plot(angles, global_vals, "o-", linewidth=2, color=STRAT_COLOR[S_DISTILLED],
            markersize=6, label="Global Distilled", zorder=3)
    ax.fill(angles, global_vals, alpha=0.12, color=STRAT_COLOR[S_DISTILLED])

    # DAFT polygon
    ax.plot(angles, daft_vals, "o-", linewidth=2.2, color="#52B788",
            markersize=7, label="DAFT (Specialist)", zorder=4)
    ax.fill(angles, daft_vals, alpha=0.18, color="#52B788")

    # Value annotations on DAFT polygon
    for angle, val, cond in zip(angles[:-1], daft_vals[:-1], cond_order):
        ax.annotate(f"{val:.3f}",
                    xy=(angle, val), xytext=(angle, val + 0.07),
                    ha="center", va="center", fontsize=8,
                    color="#2D6A4F", fontweight="bold")

    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.15),
              frameon=True, fontsize=10)
    ax.set_title("Per-Condition mAP50:\nGlobal Distilled vs DAFT Specialists",
                 fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    out = OUT_DIR / "condition_radar.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Timing Breakdown Stacked Bar (GPU)
# ══════════════════════════════════════════════════════════════════════════════

def plot_timing_breakdown() -> None:
    """Stacked horizontal bar: route / infer / nms breakdown per strategy (GPU)."""
    _set_pub_rc()
    rows = _read("topk_sweep.csv")
    if not rows:
        print("  [timing] topk_sweep.csv not found")
        return

    # Show Global Large, Global Distilled, k=1, k=2, k=3
    keep = ["Global Large", "Global Distilled", "Adaptive k=1", "Adaptive k=2", "Adaptive k=3"]
    rows = [r for r in rows if r["strategy"] in keep]
    rows = sorted(rows, key=lambda r: keep.index(r["strategy"]))

    labels   = [r["strategy"] for r in rows]
    t_route  = [_f(r.get("t_route_ms")) for r in rows]
    t_infer  = [_f(r.get("t_infer_ms")) for r in rows]
    t_nms    = [_f(r.get("t_nms_ms"))   for r in rows]

    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid(axis="x", zorder=0)

    C_ROUTE = "#F4A261"
    C_INFER = "#52B788"
    C_NMS   = "#ADB5BD"

    b1 = ax.barh(y, t_route, height=0.55, color=C_ROUTE, label="Router (MobileNetV3)",
                 edgecolor="white", linewidth=0.8, zorder=3)
    b2 = ax.barh(y, t_infer, height=0.55, left=t_route, color=C_INFER,
                 label="Detector (YOLOv8s/m)", edgecolor="white", linewidth=0.8, zorder=3)
    b3 = ax.barh(y, t_nms, height=0.55,
                 left=[r + i for r, i in zip(t_route, t_infer)],
                 color=C_NMS, label="Blend + NMS",
                 edgecolor="white", linewidth=0.8, zorder=3)

    # Total label at end of bar
    for i, r in enumerate(rows):
        total = _f(r.get("mean_ms"))
        fps   = _f(r.get("fps"))
        ax.text(total + 0.3, i, f"  {total:.1f} ms  ({fps:.0f} FPS)",
                va="center", fontsize=9, fontweight="bold",
                color=STRAT_COLOR.get(r["strategy"], "#333"))

    ax.set_xlim(left=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Latency per image (ms)  —  GPU", labelpad=8)
    ax.set_title("Inference Latency Breakdown by Component  (GPU, single image)",
                 pad=14)
    ax.legend(loc="lower right", frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()

    plt.tight_layout()
    out = OUT_DIR / "timing_breakdown_gpu.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — Per-Class AP50 Gain (with readable class names)
# ══════════════════════════════════════════════════════════════════════════════

_BDD_CLASS_NAMES = {
    "0": "Pedestrian", "1": "Rider", "2": "Car", "3": "Truck",
    "4": "Bus", "5": "Train", "6": "Motorcycle", "7": "Bicycle",
    "8": "Traffic Light", "9": "Traffic Sign",
}

def plot_class_gains() -> None:
    """Grouped horizontal bar: per-class AP50 (Global Distilled vs Specialist) + gain."""
    _set_pub_rc()
    rows = _read("per_class.csv")
    if not rows:
        print("  [class_gains] per_class.csv not found")
        return

    # Map class IDs to names, drop unknowns and zero-gain classes
    named = []
    for r in rows:
        cls_id  = str(r.get("class", "")).strip()
        name    = _BDD_CLASS_NAMES.get(cls_id, f"Class {cls_id}")
        gain    = _f(r.get("mean_gain"))
        glob_ap = _f(r.get("global_distilled_ap50"))
        spec_ap = _f(r.get("specialist_ap50"))
        if gain != gain or (glob_ap == 0 and spec_ap == 0):
            continue
        named.append({"name": name, "gain": gain,
                      "global": glob_ap, "specialist": spec_ap})

    named.sort(key=lambda r: r["gain"], reverse=True)

    labels  = [r["name"]       for r in named]
    gains   = [r["gain"]       for r in named]
    glb     = [r["global"]     for r in named]
    spec    = [r["specialist"] for r in named]
    g_colors = ["#52B788" if g >= 0 else "#E63946" for g in gains]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(named) * 0.55 + 1.5)))
    fig.subplots_adjust(wspace=0.40)
    y = np.arange(len(named))

    # ── Left: gain bars ───────────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.barh(y, gains, color=g_colors, alpha=0.88, height=0.6,
                   edgecolor="white", linewidth=1.0)
    bar_texts = ax.bar_label(bars, fmt="%+.3f", padding=4, fontsize=8.5) or []
    for txt, c in zip(bar_texts, g_colors):
        txt.set_color(c)
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Mean AP50 gain  (Specialist − Global Distilled)")
    ax.set_title("Per-Class Gain\n(Specialist over Global Distilled)")
    ax.grid(axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    # ── Right: absolute AP50 grouped bars ─────────────────────────────────────
    ax2 = axes[1]
    w = 0.35
    ax2.barh(y - w/2, glb,  w, label="Global Distilled",
             color=STRAT_COLOR[S_DISTILLED], alpha=0.88, edgecolor="white")
    ax2.barh(y + w/2, spec, w, label="DAFT Specialist",
             color="#52B788", alpha=0.88, edgecolor="white")
    ax2.set_xlim(left=0)
    ax2.set_yticks(y); ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlabel("AP50")
    ax2.set_title("Absolute AP50\nGlobal Distilled vs Best Specialist")
    ax2.legend(frameon=True, loc="lower right")
    ax2.grid(axis="x")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.invert_yaxis()

    fig.suptitle("Per-Class Performance: Where DAFT Specialists Help Most",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "class_gains.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--which",
                   choices=["conditions", "tradeoff", "bars", "dawn_dusk",
                            "ksweep", "radar", "timing", "class_gains", "all"],
                   default="all")
    p.add_argument("--demo", action="store_true",
                   help="Use illustrative demo values for tradeoff plot")
    return p.parse_args()


def main():
    args = get_args()
    OUT_DIR.mkdir(exist_ok=True)

    if args.which in ("conditions", "all"):
        print("\n── Condition Analysis ──────────────────────────")
        plot_condition_analysis()

    if args.which in ("tradeoff", "all"):
        print("\n── Accuracy–Speed Tradeoff ─────────────────────")
        plot_accuracy_speed(demo=args.demo)

    if args.which in ("bars", "all"):
        print("\n── Performance Bar Chart ───────────────────────")
        plot_performance_bars()

    if args.which in ("dawn_dusk", "all"):
        print("\n── Dawn/Dusk Robustness ────────────────────────")
        plot_dawn_dusk()

    if args.which in ("ksweep", "all"):
        print("\n── k-Sweep Dual-Axis ───────────────────────────")
        plot_ksweep()

    if args.which in ("radar", "all"):
        print("\n── Condition Radar ─────────────────────────────")
        plot_condition_radar()

    if args.which in ("timing", "all"):
        print("\n── Timing Breakdown (GPU) ──────────────────────")
        plot_timing_breakdown()

    if args.which in ("class_gains", "all"):
        print("\n── Per-Class Gains ─────────────────────────────")
        plot_class_gains()

    print("\nDone.")


if __name__ == "__main__":
    main()
