"""
visualize.py
============
For each condition (day / night / rain), pick N sample images and
produce a 3-panel comparison:

    [ Ground Truth ]  |  [ Global model ]  |  [ Specialist ]

Outputs saved to results/viz/{condition}/{image_name}.jpg

Usage
-----
  python visualize.py
  python visualize.py --n_samples 5 --device cpu
  python visualize.py --condition night --n_samples 3
"""

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CONDITIONS       = ["city_day", "city_night", "highway_day", "highway_night", "residential"]
GLOBAL_CKPT_BASE = Path("runs/detect/checkpoints")
SPEC_CKPT_BASE   = Path("checkpoints")
DATA_BASE        = Path("data/bdd100k/yolo")
MANIFEST_BASE    = Path("data/bdd100k/manifests")
OUT_DIR          = Path("results/viz")

COLORS = {
    "gt":         (0,   200,   0),   # green
    "global":     (200,  80,   0),   # blue-ish (BGR)
    "specialist": (0,    80, 200),   # orange-ish (BGR)
}


def load_model(ckpt: Path) -> YOLO:
    return YOLO(str(ckpt))


def find_ckpt(bases: list[Path], cond: str) -> Path | None:
    for base in bases:
        p = base / cond / "weights" / "best.pt"
        if p.exists():
            return p
    return None


def load_classes(data_base: Path) -> list[str]:
    classes_file = data_base.parent / "classes.txt"
    if classes_file.exists():
        return [l.strip() for l in classes_file.read_text().splitlines() if l.strip()]
    return []


def draw_boxes(img: np.ndarray, boxes, color: tuple, label_prefix: str, classes: list[str]) -> np.ndarray:
    """Draw bounding boxes on image. boxes can be GT (normalized) or ultralytics Results."""
    h, w = img.shape[:2]
    out = img.copy()

    if boxes is None:
        return out

    for box in boxes:
        if len(box) == 5:
            # YOLO label format: cls xc yc bw bh (normalized)
            cls_id = int(box[0])
            xc, yc, bw, bh = box[1:]
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            label = classes[cls_id] if cls_id < len(classes) else str(cls_id)
            conf_str = ""
        else:
            # ultralytics box: xyxy + conf + cls
            x1, y1, x2, y2 = map(int, box[:4])
            conf  = float(box[4])
            cls_id = int(box[5])
            label = classes[cls_id] if cls_id < len(classes) else str(cls_id)
            conf_str = f" {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{label}{conf_str}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, text, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


def make_panel(img: np.ndarray, boxes, color: tuple, title: str, classes: list[str]) -> np.ndarray:
    panel = draw_boxes(img, boxes, color, title, classes)
    cv2.putText(panel, title, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return panel


def process_image(img_path: Path, lbl_path: Path,
                  global_model: YOLO, spec_model: YOLO | None,
                  classes: list[str], device: str) -> np.ndarray:
    img = cv2.imread(str(img_path))

    # ground truth
    gt_boxes = []
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            parts = list(map(float, line.split()))
            if parts:
                gt_boxes.append(parts)

    # global prediction
    g_results = global_model.predict(str(img_path), device=device, verbose=False, conf=0.25)
    g_boxes = g_results[0].boxes.data.cpu().numpy() if g_results[0].boxes else []

    # specialist prediction
    if spec_model is not None:
        s_results = spec_model.predict(str(img_path), device=device, verbose=False, conf=0.25)
        s_boxes = s_results[0].boxes.data.cpu().numpy() if s_results[0].boxes else []
        s_title = "Specialist"
    else:
        s_boxes = []
        s_title = "Specialist (N/A)"

    p1 = make_panel(img, gt_boxes,   COLORS["gt"],         "Ground Truth", classes)
    p2 = make_panel(img, g_boxes,    COLORS["global"],     "Global",       classes)
    p3 = make_panel(img, s_boxes,    COLORS["specialist"], s_title,        classes)

    return np.concatenate([p1, p2, p3], axis=1)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--condition",  default=None,
                   help="Single condition to visualize (default: all)")
    p.add_argument("--n_samples",  type=int, default=5)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)

    classes = load_classes(DATA_BASE)

    global_ckpt = next(
        (p for p in [
            GLOBAL_CKPT_BASE / "global" / "weights" / "best.pt",
            SPEC_CKPT_BASE   / "global" / "weights" / "best.pt",
        ] if p.exists()),
        None,
    )
    if global_ckpt is None:
        raise FileNotFoundError("Global checkpoint not found.")

    print(f"Loading global model from {global_ckpt}")
    global_model = load_model(global_ckpt)

    conds = [args.condition] if args.condition else CONDITIONS

    for cond in conds:
        manifest = MANIFEST_BASE / f"{cond}.val.csv"
        if not manifest.exists():
            print(f"  Skipping {cond} — manifest not found")
            continue

        with open(manifest, newline="") as f:
            rows = list(csv.DictReader(f))

        sample = random.sample(rows, min(args.n_samples, len(rows)))

        spec_ckpt = find_ckpt([SPEC_CKPT_BASE, GLOBAL_CKPT_BASE], cond)
        spec_model = load_model(spec_ckpt) if spec_ckpt else None
        if spec_model:
            print(f"  [{cond}] specialist: {spec_ckpt}")
        else:
            print(f"  [{cond}] no specialist checkpoint found")

        out_dir = OUT_DIR / cond
        out_dir.mkdir(parents=True, exist_ok=True)

        for row in sample:
            img_path = Path(row["image_path"])
            lbl_path = Path(row["yolo_label"])

            if not img_path.exists():
                continue

            print(f"  {cond}/{img_path.name}")
            panel = process_image(img_path, lbl_path, global_model, spec_model, classes, args.device)

            out_path = out_dir / img_path.name
            cv2.imwrite(str(out_path), panel)

        print(f"  Saved {len(sample)} images -> {out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
