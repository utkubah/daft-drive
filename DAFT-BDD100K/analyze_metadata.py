"""
analyze_metadata.py
===================
Analyze BDD100K metadata from existing manifests to decide
the best routing strategy for DAFT specialists.

Reads from data/bdd100k/manifests/*.csv (produced by prepare_data.py).
No model loading, no HuggingFace — fast local analysis only.

Outputs
-------
  results/metadata/summary.txt       printed stats saved to file
  results/metadata/scene_dist.png    scene distribution
  results/metadata/condition_dist.png  timeofday x weather heatmap
  results/metadata/class_per_scene.png  object class counts per scene

Usage
-----
  python analyze_metadata.py
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MANIFEST_DIR = Path("data/bdd100k/manifests")
OUT_DIR      = Path("results/metadata")


def load_all_manifests() -> list[dict]:
    """Load train + val full manifests (not condition-specific ones)."""
    rows = []
    for split in ["train", "val"]:
        p = MANIFEST_DIR / f"{split}.csv"
        if not p.exists():
            print(f"  WARNING: {p} not found, skipping")
            continue
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def count(rows: list[dict], field: str) -> Counter:
    return Counter(r.get(field) or "undefined" for r in rows)


def cross_tab(rows: list[dict], field_a: str, field_b: str) -> dict[str, Counter]:
    """Returns {value_a: Counter(value_b)}."""
    tab: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        a = r.get(field_a) or "undefined"
        b = r.get(field_b) or "undefined"
        tab[a][b] += 1
    return tab


def print_counter(name: str, c: Counter, lines: list[str]):
    total = sum(c.values())
    lines.append(f"\n{name} (total={total:,})")
    lines.append("-" * 45)
    for val, n in c.most_common():
        pct = 100 * n / total if total else 0
        lines.append(f"  {val:<25}  {n:>6,}  ({pct:5.1f}%)")


def plot_bar(counter: Counter, title: str, path: Path, color: str = "#5577aa"):
    labels = [k for k, _ in counter.most_common()]
    values = [counter[k] for k in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=color)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set_title(title)
    ax.set_ylabel("Image count")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_heatmap(tab: dict[str, Counter], title: str, path: Path):
    rows_k = sorted(tab.keys())
    all_cols = sorted({v for c in tab.values() for v in c})
    matrix = np.array([[tab[r][c] for c in all_cols] for r in rows_k])

    fig, ax = plt.subplots(figsize=(max(8, len(all_cols) * 1.5), max(4, len(rows_k) * 0.8)))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(all_cols))); ax.set_xticklabels(all_cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows_k)));  ax.set_yticklabels(rows_k)
    for i in range(len(rows_k)):
        for j in range(len(all_cols)):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] > matrix.max() * 0.6 else "black")
    plt.colorbar(im, ax=ax, label="Image count")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_all_manifests()
    if not rows:
        print("No manifests found. Run prepare_data.py first.")
        return

    print(f"\nLoaded {len(rows):,} samples from manifests.\n")
    lines = [f"BDD100K Metadata Analysis — {len(rows):,} total samples"]

    # ── 1. Per-field distributions ──────────────────────────────────────────
    scene_c   = count(rows, "scene")
    tod_c     = count(rows, "timeofday")
    weather_c = count(rows, "weather")
    cond_c    = count(rows, "condition")

    print_counter("Scene",     scene_c,   lines)
    print_counter("Timeofday", tod_c,     lines)
    print_counter("Weather",   weather_c, lines)
    print_counter("Current condition (day/night/rain)", cond_c, lines)

    # ── 2. Cross-tabs ───────────────────────────────────────────────────────
    lines.append("\n\nScene × Timeofday")
    lines.append("=" * 60)
    tab_st = cross_tab(rows, "scene", "timeofday")
    all_tod = sorted({t for c in tab_st.values() for t in c})
    header = f"  {'scene':<25}" + "".join(f"{t:>12}" for t in all_tod)
    lines.append(header)
    for sc in sorted(tab_st):
        row_s = f"  {sc:<25}" + "".join(f"{tab_st[sc].get(t, 0):>12,}" for t in all_tod)
        lines.append(row_s)

    lines.append("\n\nScene × Weather")
    lines.append("=" * 60)
    tab_sw = cross_tab(rows, "scene", "weather")
    all_wx = sorted({w for c in tab_sw.values() for w in c})
    header = f"  {'scene':<25}" + "".join(f"{w:>12}" for w in all_wx)
    lines.append(header)
    for sc in sorted(tab_sw):
        row_s = f"  {sc:<25}" + "".join(f"{tab_sw[sc].get(w, 0):>12,}" for w in all_wx)
        lines.append(row_s)

    # ── 3. Object count per scene ───────────────────────────────────────────
    boxes_per_scene: Counter = Counter()
    imgs_per_scene:  Counter = Counter()
    for r in rows:
        sc = r.get("scene") or "undefined"
        n  = int(r.get("num_boxes") or 0)
        boxes_per_scene[sc] += n
        imgs_per_scene[sc]  += 1

    lines.append("\n\nAvg objects per image per scene")
    lines.append("-" * 45)
    for sc, total_boxes in boxes_per_scene.most_common():
        n_imgs = imgs_per_scene[sc]
        avg    = total_boxes / n_imgs if n_imgs else 0
        lines.append(f"  {sc:<25}  {avg:5.1f} avg  ({n_imgs:,} images)")

    # ── 4. Routing recommendation ───────────────────────────────────────────
    lines.append("\n\n" + "=" * 60)
    lines.append("ROUTING RECOMMENDATION")
    lines.append("=" * 60)

    top_scenes = [s for s, n in scene_c.most_common() if n >= 200]
    lines.append(f"\nScenes with >= 200 images: {top_scenes}")
    lines.append("\nCurrent conditions coverage:")
    total = len(rows)
    for cond, n in cond_c.most_common():
        if cond:
            lines.append(f"  {cond}: {n:,} ({100*n/total:.1f}%)")
    unmapped = cond_c.get("None", 0) + cond_c.get("", 0)
    lines.append(f"  unmapped (dawn/dusk, overcast, etc): {unmapped:,} ({100*unmapped/total:.1f}%)")

    # Print and save summary
    summary = "\n".join(lines)
    print(summary)
    summary_path = OUT_DIR / "summary.txt"
    summary_path.write_text(summary)
    print(f"\n  Saved: {summary_path}")

    # ── 5. Plots ────────────────────────────────────────────────────────────
    plot_bar(scene_c,   "Scene distribution",     OUT_DIR / "scene_dist.png",   "#5577aa")
    plot_bar(tod_c,     "Time of day distribution", OUT_DIR / "timeofday_dist.png", "#44aa66")
    plot_bar(weather_c, "Weather distribution",    OUT_DIR / "weather_dist.png", "#aa7744")
    plot_heatmap(tab_st, "Scene × Timeofday (image count)", OUT_DIR / "scene_x_timeofday.png")
    plot_heatmap(tab_sw, "Scene × Weather (image count)",   OUT_DIR / "scene_x_weather.png")

    print("\nDone. Check results/metadata/ for plots and summary.txt.")


if __name__ == "__main__":
    main()
