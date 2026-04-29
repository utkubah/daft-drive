from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import csv

ROOT = Path.home() / "datasets" / "hf_bdd100k_full_repo"
OUT_DIR = Path("analysis/hf_bdd100k_repo_inspect")


def find_split_dirs(root: Path):
    out = []
    for p in root.rglob("samples.json"):
        split_dir = p.parent
        out.append(split_dir)
    return sorted(set(out))


def load_samples(samples_path: Path):
    with open(samples_path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "samples" in obj:
        return obj["samples"]
    if isinstance(obj, list):
        return obj
    raise RuntimeError(f"Unexpected samples.json format: {samples_path}")


def get_label(sample, field):
    x = sample.get(field)
    if not x:
        return None
    if isinstance(x, dict):
        return x.get("label")
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("ROOT:", ROOT)
    print("exists:", ROOT.exists())
    if not ROOT.exists():
        raise SystemExit("Dataset root not found")

    print("\n=== TOP LEVEL ===")
    for p in sorted(ROOT.iterdir()):
        print(("DIR " if p.is_dir() else "FILE"), p.name)

    print("\n=== FILE EXTENSION COUNTS ===")
    ext_counter = Counter(p.suffix.lower() for p in ROOT.rglob("*") if p.is_file())
    for ext, count in ext_counter.most_common():
        print(repr(ext), count)

    split_dirs = find_split_dirs(ROOT)
    print("\n=== SPLIT DIRS FOUND ===")
    for d in split_dirs:
        print(d)

    summary_rows = []
    manifest_rows = []

    for split_dir in split_dirs:
        split_name = split_dir.name
        samples_path = split_dir / "samples.json"
        samples = load_samples(samples_path)

        data_dir = split_dir / "data"
        drivable_dir = split_dir / "fields" / "drivable"

        n_jpg = len(list(data_dir.glob("*.jpg"))) if data_dir.exists() else 0
        n_png = len(list(drivable_dir.glob("*.png"))) if drivable_dir.exists() else 0

        weather_counts = Counter()
        time_counts = Counter()
        scene_counts = Counter()
        det_count_dist = Counter()

        for s in samples:
            filepath = s.get("filepath")
            image_name = Path(filepath).name if filepath else None

            weather = get_label(s, "weather")
            timeofday = get_label(s, "timeofday")
            scene = get_label(s, "scene")

            detections = s.get("detections", {})
            dets = detections.get("detections", []) if isinstance(detections, dict) else []
            num_det = len(dets)

            has_drivable = bool(s.get("drivable"))
            drivable_mask_path = None
            if isinstance(s.get("drivable"), dict):
                drivable_mask_path = s["drivable"].get("mask_path")

            weather_counts[weather or "MISSING"] += 1
            time_counts[timeofday or "MISSING"] += 1
            scene_counts[scene or "MISSING"] += 1
            det_count_dist[num_det] += 1

            manifest_rows.append({
                "split": split_name,
                "image_name": image_name,
                "filepath": filepath,
                "weather": weather,
                "timeofday": timeofday,
                "scene": scene,
                "num_detections": num_det,
                "has_drivable": has_drivable,
                "drivable_mask_path": drivable_mask_path,
            })

        summary_rows.append({
            "split": split_name,
            "samples_json_rows": len(samples),
            "jpg_files": n_jpg,
            "drivable_png_files": n_png,
            "num_unique_weather": len(weather_counts),
            "num_unique_timeofday": len(time_counts),
            "num_unique_scene": len(scene_counts),
        })

        print(f"\n=== SPLIT: {split_name} ===")
        print("samples_json_rows:", len(samples))
        print("jpg_files:", n_jpg)
        print("drivable_png_files:", n_png)
        print("\nweather counts:")
        print(dict(weather_counts.most_common()))
        print("\ntimeofday counts:")
        print(dict(time_counts.most_common()))
        print("\nscene counts:")
        print(dict(scene_counts.most_common()))
        print("\ndetection-count distribution (top 20):")
        print(dict(det_count_dist.most_common(20)))

    summary_csv = OUT_DIR / "repo_summary.csv"
    manifest_csv = OUT_DIR / "repo_manifest.csv"

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\nSaved:")
    print("-", summary_csv)
    print("-", manifest_csv)


if __name__ == "__main__":
    main()