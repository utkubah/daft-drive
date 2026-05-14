"""
collect_missing_data.py
=======================
Generates all data still missing for the paper in one run.

Outputs (all written to results/):
  results/global_models_comparison.csv    -- mAP50 + FPS for global baselines AND DAFT k=2/k=5
  results/condition_split_counts.csv      -- train/val image counts per condition
  results/class_names.csv                 -- class index → name mapping
  results/router_accuracy.csv            -- top-1 accuracy + per-class breakdown
  results/router_confusion_matrix.csv    -- 5x5 confusion matrix (true vs predicted)

The comparison CSV rows are:
  Global Distilled (YOLOv8s), Global YOLOv8m, Global YOLOv8x (optional),
  DAFT Adaptive k=2, DAFT Adaptive k=5

Usage (on HPC, inside conda activate daft):
  python collect_missing_data.py
  python collect_missing_data.py --skip_global   # skip global model evals
  python collect_missing_data.py --skip_daft     # skip DAFT k=2/k=5 eval
  python collect_missing_data.py --skip_router   # skip router eval
"""

from __future__ import annotations
import argparse
import csv
import json
import random
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

CONDITIONS = ["city_day", "city_night", "highway_day", "highway_night", "residential"]
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Arg parsing ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--distilled_ckpt", default="checkpoints/global/weights/best.pt",
                   help="Global Distilled (YOLOv8s via KD) checkpoint")
    p.add_argument("--large_ckpt",    default="checkpoints/large/weights/best.pt",
                   help="Global YOLOv8m checkpoint")
    p.add_argument("--xlarge_ckpt",   default="checkpoints/xlarge/weights/best.pt",
                   help="Global YOLOv8x checkpoint (optional, skipped if not found)")
    p.add_argument("--router_ckpt",    default="checkpoints/router/best.pt",
                   help="Path to ImageRouter checkpoint")
    p.add_argument("--data_dir",       default="data/bdd100k",
                   help="Root of prepared BDD100K data")
    p.add_argument("--device",         default="cuda")
    p.add_argument("--imgsz",          type=int, default=640)
    p.add_argument("--n_samples",      type=int, default=100,
                   help="Images per condition for quick eval (default 100 → 500 total)")
    p.add_argument("--ckpt_dir",        default="checkpoints",
                   help="Root dir for specialist checkpoints ({cond}/weights/best.pt)")
    p.add_argument("--skip_global",    action="store_true")
    p.add_argument("--skip_daft",      action="store_true")
    p.add_argument("--skip_router",    action="store_true")
    return p.parse_args()


# ── 1. Condition split counts + class names ───────────────────────────────────

def collect_split_counts(data_dir: Path):
    """Read manifest CSVs, count images per condition per split."""
    man_dir = data_dir / "manifests"
    if not man_dir.exists():
        print(f"[WARN] manifests dir not found: {man_dir}. Skipping split counts.")
        return

    rows = []
    for cond in CONDITIONS:
        counts = {}
        for split in ["train", "val"]:
            csv_path = man_dir / f"{cond}.{split}.csv"
            if not csv_path.exists():
                counts[split] = "N/A"
                continue
            with open(csv_path) as f:
                counts[split] = sum(1 for _ in csv.DictReader(f))
        rows.append({"condition": cond, "train_count": counts["train"], "val_count": counts["val"]})

    # also get total val count and dawn/dusk count
    for tag, fname in [("all_val", "val.csv"), ("dawn_dusk_val", "dawn_dusk.val.csv")]:
        csv_path = man_dir / fname
        if csv_path.exists():
            with open(csv_path) as f:
                n = sum(1 for _ in csv.DictReader(f))
            rows.append({"condition": tag, "train_count": "N/A", "val_count": n})

    out = RESULTS_DIR / "condition_split_counts.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "train_count", "val_count"])
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] {out}")


def collect_class_names(data_dir: Path):
    """Read classes.txt and write index→name CSV."""
    cls_file = data_dir / "classes.txt"
    if not cls_file.exists():
        print(f"[WARN] {cls_file} not found. Skipping class names.")
        return

    names = cls_file.read_text().strip().splitlines()
    out = RESULTS_DIR / "class_names.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "name"])
        for i, name in enumerate(names):
            w.writerow([i, name])
    print(f"[OK] {out} ({len(names)} classes)")
    return names


# ── 2. Global model comparison (distilled / medium / xlarge) ─────────────────

def _sample_val_images(data_dir: Path, n_per_condition: int, seed: int = 42) -> dict:
    """Sample n images per condition from val manifest. Returns {cond: [path, ...]}."""
    man_path = data_dir / "manifests" / "val.csv"
    if not man_path.exists():
        print(f"[WARN] {man_path} not found — cannot sample val images.")
        return {}
    by_cond = {c: [] for c in CONDITIONS}
    with open(man_path) as f:
        for row in csv.DictReader(f):
            cond = row.get("condition")
            if cond in by_cond:
                by_cond[cond].append(row["image_path"])
    rng = random.Random(seed)
    return {c: rng.sample(paths, min(n_per_condition, len(paths)))
            for c, paths in by_cond.items() if paths}


def _write_temp_yaml(img_paths: list, classes: list) -> str:
    """Write a throwaway YOLO yaml pointing to a custom image list.

    YOLO's model.val() needs a yaml file — there's no way to pass a raw
    image list directly. We write one to a temp file and clean it up after.
    The txt file uses absolute paths, so the 'path:' root field is ignored.
    """
    txt = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    txt.write("\n".join(img_paths))
    txt.close()

    yml = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yml.write(f"path: /\n")          # absolute paths in txt make this irrelevant
    yml.write(f"train: {txt.name}\n")
    yml.write(f"val:   {txt.name}\n")
    yml.write(f"nc: {len(classes)}\n")
    yml.write("names:\n")
    for i, c in enumerate(classes):
        yml.write(f"  {i}: {json.dumps(c)}\n")
    yml.close()
    return yml.name


def _eval_one_model(model, label: str, sampled: dict, classes: list, args) -> dict:
    """Evaluate one model: per-condition mAP50, overall mAP50, FPS."""
    row = {"model": label}
    all_imgs = []

    for cond in CONDITIONS:
        imgs = sampled.get(cond, [])
        if not imgs:
            row[cond] = "N/A"
            continue
        all_imgs.extend(imgs)
        yaml_path = _write_temp_yaml(imgs, classes)
        try:
            m = model.val(data=yaml_path, imgsz=args.imgsz, device=args.device,
                          verbose=False, split="val")
            row[cond] = round(float(m.box.map50), 4)
        finally:
            Path(yaml_path).unlink(missing_ok=True)
        print(f"    {cond}: mAP50={row[cond]}")

    # Overall mAP50 on pooled images
    if all_imgs:
        yaml_path = _write_temp_yaml(all_imgs, classes)
        try:
            m = model.val(data=yaml_path, imgsz=args.imgsz, device=args.device,
                          verbose=False, split="val")
            row["overall_mAP50"] = round(float(m.box.map50), 4)
            # FPS from YOLO's own speed dict (ms/image → FPS)
            speed = m.speed   # {'preprocess': X, 'inference': Y, 'postprocess': Z}
            ms_per_img = sum(speed.values())
            row["fps"] = round(1000.0 / ms_per_img, 1) if ms_per_img > 0 else "N/A"
        finally:
            Path(yaml_path).unlink(missing_ok=True)
        print(f"    overall: mAP50={row['overall_mAP50']}  FPS={row['fps']}")

    return row


def eval_global_models(args):
    """Evaluate all available global baselines and write comparison CSV."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.")
        return

    # Load class names from classes.txt
    cls_file = Path(args.data_dir) / "classes.txt"
    if not cls_file.exists():
        print(f"[WARN] {cls_file} not found — using empty class list.")
        classes = []
    else:
        classes = cls_file.read_text().strip().splitlines()

    sampled = _sample_val_images(Path(args.data_dir), args.n_samples)
    if not sampled:
        print("[WARN] No val images sampled. Skipping global model eval.")
        return

    total = sum(len(v) for v in sampled.values())
    print(f"  Sampled {total} val images across {len(sampled)} conditions "
          f"({args.n_samples} per condition)")

    # Required models — warn loudly if missing
    required = [
        ("Global Distilled (YOLOv8s)", args.distilled_ckpt),
        ("Global YOLOv8m",             args.large_ckpt),
    ]
    # Optional — silently skip if not found
    optional = [
        ("Global YOLOv8x",             args.xlarge_ckpt),
    ]
    candidates = [(label, ckpt, True)  for label, ckpt in required] + \
                 [(label, ckpt, False) for label, ckpt in optional]

    rows = []
    for label, ckpt_str, required_flag in candidates:
        ckpt = Path(ckpt_str)
        if not ckpt.exists():
            tag = "[WARN]" if required_flag else "[SKIP]"
            print(f"{tag} {label}: checkpoint not found at {ckpt_str}")
            continue
        print(f"\n  Evaluating {label} ...")
        model = YOLO(str(ckpt))
        row = _eval_one_model(model, label, sampled, classes, args)
        rows.append(row)

    if not rows:
        print("[WARN] No models evaluated.")
    return rows


# ── 3. DAFT Adaptive k=2 and k=5 per-condition mAP50 ─────────────────────────

def _load_gt_boxes(label_path: str, img_w: int, img_h: int):
    """Read a YOLO label file and return (boxes_xyxy, labels) in pixel coords."""
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


def _result_to_boxes(result) -> np.ndarray:
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy().reshape(-1, 1)
    cls  = result.boxes.cls.cpu().numpy().reshape(-1, 1)
    return np.concatenate([xyxy, conf, cls], axis=1).astype(np.float32)


def _compute_map50(all_preds, all_gt_boxes, all_gt_labels) -> float:
    """Compute mAP50 over a list of images using torchmetrics."""
    try:
        from torchmetrics.detection import MeanAveragePrecision
        metric = MeanAveragePrecision(iou_thresholds=[0.5], box_format="xyxy")
        metric.update(
            [{"boxes":  torch.tensor(p[:, :4]),
              "scores": torch.tensor(p[:, 4]),
              "labels": torch.tensor(p[:, 5].astype(np.int64))} for p in all_preds],
            [{"boxes":  torch.tensor(b),
              "labels": torch.tensor(l)} for b, l in zip(all_gt_boxes, all_gt_labels)],
        )
        return float(metric.compute()["map_50"].item())
    except ImportError:
        print("[WARN] torchmetrics not available — mAP50 computation skipped.")
        return float("nan")


def eval_daft_routing(args) -> list[dict]:
    """
    Compute per-condition mAP50 for DAFT Adaptive k=2 and k=5.

    Loads the five specialist models and the ImageRouter once, then for each
    condition's sampled images runs the full routing + blend pipeline and
    computes mAP50 with torchmetrics.  Returns rows ready to append to the
    global_models_comparison.csv table.
    """
    try:
        from ultralytics import YOLO
        from router import ImageRouter, blend_detections, select_top_k
    except ImportError as e:
        print(f"[ERROR] Missing import for DAFT eval: {e}")
        return []

    router_ckpt = Path(args.router_ckpt)
    if not router_ckpt.exists():
        print(f"[WARN] Router checkpoint not found at {args.router_ckpt}. Skipping DAFT eval.")
        return []

    ckpt_dir = Path(args.ckpt_dir)
    man_path = Path(args.data_dir) / "manifests" / "val.csv"
    if not man_path.exists():
        print(f"[WARN] {man_path} not found. Skipping DAFT eval.")
        return []

    # Build {cond: [{"image_path": ..., "yolo_label": ...}, ...]} from val manifest
    # We need yolo_label paths to compute ground truth — read from the manifest
    cond_samples: dict[str, list[dict]] = {c: [] for c in CONDITIONS}
    with open(man_path) as f:
        for row in csv.DictReader(f):
            cond = row.get("condition")
            if cond in cond_samples:
                cond_samples[cond].append(row)

    # Subsample to n_samples per condition (same seed as global eval for consistency)
    rng = random.Random(42)
    for cond in CONDITIONS:
        rows_c = cond_samples[cond]
        cond_samples[cond] = rng.sample(rows_c, min(args.n_samples, len(rows_c)))

    total = sum(len(v) for v in cond_samples.values())
    print(f"  Sampled {total} val images for DAFT eval "
          f"({args.n_samples} per condition)")

    # Load router + specialists once
    print("  Loading ImageRouter ...")
    router = ImageRouter(str(router_ckpt), device=args.device)

    print("  Loading specialists ...")
    specialists: dict[str, YOLO] = {}
    for cond in CONDITIONS:
        sp_ckpt = ckpt_dir / cond / "weights" / "best.pt"
        if not sp_ckpt.exists():
            # Fall back to the global distilled model so eval still runs
            sp_ckpt = Path(args.distilled_ckpt)
            print(f"    [WARN] {cond} specialist not found — falling back to global")
        specialists[cond] = YOLO(str(sp_ckpt))
        print(f"    {cond}: {sp_ckpt}")

    result_rows = []
    for k in (2, 5):
        label = f"DAFT Adaptive k={k}"
        print(f"\n  [{label}]")
        row: dict = {"model": label, "fps": "N/A"}
        all_preds_overall, all_gt_b_overall, all_gt_l_overall = [], [], []

        for cond in CONDITIONS:
            samples = cond_samples.get(cond, [])
            if not samples:
                row[cond] = "N/A"
                continue

            all_preds_c, all_gt_b_c, all_gt_l_c = [], [], []
            for s in samples:
                img_path = s.get("image_path", "")
                lbl_path = s.get("yolo_label", "")

                img = cv2.imread(img_path)
                if img is None:
                    continue
                h, w = img.shape[:2]

                # Route + blend
                weights  = router.weights_from_img(img)
                selected = select_top_k(weights, top_k=k)
                rw = []
                for sel_cond, sel_w in selected:
                    preds = specialists[sel_cond].predict(
                        img, device=args.device, verbose=False, conf=0.25)
                    rw.append((_result_to_boxes(preds[0]), sel_w))
                pred_boxes = blend_detections(rw)

                gt_b, gt_l = _load_gt_boxes(lbl_path, w, h)
                all_preds_c.append(pred_boxes)
                all_gt_b_c.append(gt_b)
                all_gt_l_c.append(gt_l)

                all_preds_overall.append(pred_boxes)
                all_gt_b_overall.append(gt_b)
                all_gt_l_overall.append(gt_l)

            map50_c = _compute_map50(all_preds_c, all_gt_b_c, all_gt_l_c)
            row[cond] = round(map50_c, 4)
            print(f"    {cond}: mAP50={row[cond]}")

        map50_all = _compute_map50(all_preds_overall, all_gt_b_overall, all_gt_l_overall)
        row["overall_mAP50"] = round(map50_all, 4)
        print(f"    overall: mAP50={row['overall_mAP50']}")
        result_rows.append(row)

    return result_rows


# ── 4. Router accuracy + confusion matrix ─────────────────────────────────────

def eval_router(args):
    """Compute ImageRouter top-1 accuracy and 5x5 confusion matrix on val."""
    ckpt = Path(args.router_ckpt)
    if not ckpt.exists():
        print(f"[WARN] Router checkpoint not found at {args.router_ckpt}. Skipping.")
        return

    man_path = Path(args.data_dir) / "manifests" / "val.csv"
    if not man_path.exists():
        print(f"[WARN] {man_path} not found. Skipping router eval.")
        return

    # Read val manifest — keep only rows with a known condition
    samples = []
    with open(man_path) as f:
        for row in csv.DictReader(f):
            if row.get("condition") in CONDITIONS:
                samples.append(row)

    if not samples:
        print("[WARN] No usable samples in val manifest. Skipping router eval.")
        return

    print(f"  Evaluating router on {len(samples)} val images ...")

    try:
        from router import ImageRouter
    except ImportError:
        print("[ERROR] Could not import router.py. Make sure you run from the project root.")
        return

    router = ImageRouter(str(ckpt), device=args.device)

    # confusion matrix: rows = true condition, cols = predicted condition
    cond_idx = {c: i for i, c in enumerate(CONDITIONS)}
    conf_matrix = [[0] * len(CONDITIONS) for _ in CONDITIONS]
    correct = 0

    for i, row in enumerate(samples):
        if i % 500 == 0:
            print(f"    {i}/{len(samples)} ...")
        img_path = row["image_path"]
        true_cond = row["condition"]
        weights = router.weights(img_path)
        pred_cond = max(weights, key=weights.get)

        ti = cond_idx[true_cond]
        pi = cond_idx[pred_cond]
        conf_matrix[ti][pi] += 1
        if pred_cond == true_cond:
            correct += 1

    top1_acc = correct / len(samples)
    print(f"  Router top-1 accuracy: {top1_acc:.4f} ({correct}/{len(samples)})")

    # Per-condition accuracy
    per_cond_rows = []
    for i, cond in enumerate(CONDITIONS):
        total = sum(conf_matrix[i])
        acc = conf_matrix[i][i] / total if total else 0.0
        per_cond_rows.append({"condition": cond, "total": total,
                               "correct": conf_matrix[i][i],
                               "accuracy": round(acc, 4)})

    # Write accuracy summary
    acc_out = RESULTS_DIR / "router_accuracy.csv"
    with open(acc_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "total", "correct", "accuracy"])
        w.writeheader()
        w.writerows(per_cond_rows)
        # append overall row
        f.write(f"overall,{len(samples)},{correct},{round(top1_acc, 4)}\n")
    print(f"[OK] {acc_out}")

    # Write confusion matrix
    cm_out = RESULTS_DIR / "router_confusion_matrix.csv"
    with open(cm_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true \\ pred"] + CONDITIONS)
        for i, cond in enumerate(CONDITIONS):
            w.writerow([cond] + conf_matrix[i])
    print(f"[OK] {cm_out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    print("=== Step 1: Condition split counts + class names ===")
    collect_split_counts(data_dir)
    collect_class_names(data_dir)

    # Collect all comparison rows into one list so the final CSV has everything
    comparison_rows: list[dict] = []

    if not args.skip_global:
        print("\n=== Step 2: Global model comparison (distilled / medium / xlarge) ===")
        comparison_rows += eval_global_models(args) or []
    else:
        print("\n[SKIP] Global model eval (--skip_global)")

    if not args.skip_daft:
        print("\n=== Step 3: DAFT Adaptive k=2 and k=5 per-condition mAP50 ===")
        comparison_rows += eval_daft_routing(args) or []
    else:
        print("\n[SKIP] DAFT routing eval (--skip_daft)")

    # Write the unified comparison CSV once all rows are ready
    if comparison_rows:
        fieldnames = ["model", "overall_mAP50"] + CONDITIONS + ["fps"]
        out = RESULTS_DIR / "global_models_comparison.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(comparison_rows)
        print(f"\n[OK] {out}")

    if not args.skip_router:
        print("\n=== Step 4: Router accuracy + confusion matrix ===")
        eval_router(args)
    else:
        print("\n[SKIP] Router eval (--skip_router)")

    print("\nDone. Check results/ for new CSVs.")


if __name__ == "__main__":
    main()
