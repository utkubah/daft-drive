from __future__ import annotations

import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path.home() / "datasets" / "hf_bdd100k_full_repo"
SAMPLES_JSON = ROOT / "samples.json"
OUT_DIR = Path("data/hf_bdd100k_od")
SEED = 42

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10

MAIN_SCENES = ["city street", "highway", "residential"]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_label(sample: dict, field: str):
    x = sample.get(field)
    if not x:
        return None
    if isinstance(x, dict):
        return x.get("label")
    return None


def normalize_scene(scene: str | None) -> str:
    if scene in MAIN_SCENES:
        return scene
    return "other"


def yolo_line(det: dict, class_to_idx: dict[str, int]) -> str | None:
    label = det.get("label")
    box = det.get("bounding_box")

    if label is None or box is None or len(box) != 4:
        return None

    x, y, w, h = box
    xc = x + w / 2.0
    yc = y + h / 2.0

    return f"{class_to_idx[label]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def link_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def stratified_split(rows: list[dict], key: str):
    random.seed(SEED)
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)

    train, val, test = [], [], []

    for _, items in groups.items():
        random.shuffle(items)
        n = len(items)

        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))

        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            if n_train + n_val >= n:
                n_val = max(1, n - n_train - 1)
        else:
            n_train = max(1, n - 1)
            n_val = 0

        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    random.seed(SEED)

    if not SAMPLES_JSON.exists():
        raise FileNotFoundError(f"Missing: {SAMPLES_JSON}")

    ensure_dir(OUT_DIR)
    manifests_dir = OUT_DIR / "manifests"
    yolo_dir = OUT_DIR / "yolo"
    ensure_dir(manifests_dir)
    ensure_dir(yolo_dir)

    with open(SAMPLES_JSON, "r") as f:
        obj = json.load(f)

    samples = obj["samples"]
    print("samples loaded:", len(samples))

    class_counter = Counter()
    scene_counter = Counter()
    weather_counter = Counter()
    time_counter = Counter()

    rows = []

    for s in samples:
        rel_fp = s.get("filepath")
        if not rel_fp:
            continue

        img_path = ROOT / rel_fp
        if not img_path.exists():
            continue

        image_name = Path(rel_fp).name
        scene = safe_label(s, "scene")
        timeofday = safe_label(s, "timeofday")
        weather = safe_label(s, "weather")
        scene_group = normalize_scene(scene)

        detections = s.get("detections", {})
        dets = detections.get("detections", []) if isinstance(detections, dict) else []
        labels = [d.get("label") for d in dets if d.get("label") is not None]

        if len(labels) == 0:
            # object detection dataset için boş kutulu sample'ları şu an dışarıda bırakıyoruz
            continue

        for lab in labels:
            class_counter[lab] += 1

        scene_counter[scene_group] += 1
        weather_counter[weather or "MISSING"] += 1
        time_counter[timeofday or "MISSING"] += 1

        rows.append({
            "image_name": image_name,
            "image_path": str(img_path),
            "scene": scene,
            "scene_group": scene_group,
            "timeofday": timeofday,
            "weather": weather,
            "num_detections": len(dets),
            "has_drivable": bool(s.get("drivable")),
            "detections_json": json.dumps(dets),
        })

    print("usable detection samples:", len(rows))
    print("\nscene_group counts:", dict(scene_counter))
    print("\ntimeofday counts:", dict(time_counter))
    print("\nweather counts:", dict(weather_counter))

    classes = sorted(class_counter.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    with open(OUT_DIR / "classes.txt", "w") as f:
        for c in classes:
            f.write(c + "\n")

    train_rows, val_rows, test_rows = stratified_split(rows, key="scene_group")

    for split_name, split_rows in [
        ("train", train_rows),
        ("val", val_rows),
        ("test", test_rows),
    ]:
        print(f"\n{split_name}: {len(split_rows)}")

    save_csv(manifests_dir / "all.csv", rows)
    save_csv(manifests_dir / "train.csv", train_rows)
    save_csv(manifests_dir / "val.csv", val_rows)
    save_csv(manifests_dir / "test.csv", test_rows)

    # scene-specific manifests for future experts
    for scene_name in MAIN_SCENES:
        for split_name, split_rows in [
            ("train", train_rows),
            ("val", val_rows),
            ("test", test_rows),
        ]:
            sub = [r for r in split_rows if r["scene_group"] == scene_name]
            safe_name = scene_name.replace(" ", "_")
            save_csv(manifests_dir / f"{safe_name}.{split_name}.csv", sub)

    # YOLO export
    for split_name, split_rows in [
        ("train", train_rows),
        ("val", val_rows),
        ("test", test_rows),
    ]:
        img_out = yolo_dir / "images" / split_name
        lbl_out = yolo_dir / "labels" / split_name
        ensure_dir(img_out)
        ensure_dir(lbl_out)

        for row in split_rows:
            src_img = Path(row["image_path"])
            dst_img = img_out / row["image_name"]
            link_or_copy(src_img, dst_img)

            dets = json.loads(row["detections_json"])
            lines = []
            for det in dets:
                line = yolo_line(det, class_to_idx)
                if line is not None:
                    lines.append(line)

            label_path = lbl_out / f"{Path(row['image_name']).stem}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(lines))

    dataset_yaml = yolo_dir / "dataset.yaml"
    with open(dataset_yaml, "w") as f:
        f.write(f"path: {yolo_dir.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write(f"nc: {len(classes)}\n")
        f.write("names:\n")
        for i, c in enumerate(classes):
            f.write(f"  {i}: {json.dumps(c)}\n")

    print("\nSaved:")
    print("-", manifests_dir / "all.csv")
    print("-", manifests_dir / "train.csv")
    print("-", manifests_dir / "val.csv")
    print("-", manifests_dir / "test.csv")
    print("-", dataset_yaml)
    print("-", OUT_DIR / "classes.txt")
    print("\nDone.")


if __name__ == "__main__":
    main()