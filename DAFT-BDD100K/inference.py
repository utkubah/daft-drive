"""
inference.py
============
Adaptive DAFT routing for BDD100K object detection.

Supports:
  - metadata-driven soft weights
  - top-K selection: auto / 1 / 2 / 5
  - weighted blending via blend_detections
  - specialist fallback to global if missing
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from router import MetadataRouter, CONDITIONS, blend_detections, select_top_k

CKPT_DIR = Path("checkpoints")
ALT_CKPT_DIR = Path("runs/detect/checkpoints")

_MODEL_CACHE: dict[str, YOLO] = {}
_router = MetadataRouter()


def resolve_checkpoint(condition: str) -> tuple[Path, str]:
    candidates = [
        (CKPT_DIR / condition / "weights" / "best.pt", condition),
        (ALT_CKPT_DIR / condition / "weights" / "best.pt", condition),
        (CKPT_DIR / "global" / "weights" / "best.pt", "global"),
        (ALT_CKPT_DIR / "global" / "weights" / "best.pt", "global"),
    ]
    for path, used in candidates:
        if path.exists():
            return path, used

    raise FileNotFoundError(
        f"No checkpoint found for '{condition}'. Searched:\n"
        + "\n".join(f"  {p}" for p, _ in candidates)
    )


def load_model_from_path(path: Path) -> YOLO:
    key = str(path.resolve())
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    model = YOLO(str(path))
    _MODEL_CACHE[key] = model
    return model


def build_manifest_index(manifest_csv: str) -> dict[str, dict]:
    index: dict[str, dict] = {}
    with open(manifest_csv, newline="") as f:
        for row in csv.DictReader(f):
            index[row["image_name"]] = {
                "scene": row.get("scene", ""),
                "timeofday": row.get("timeofday", ""),
                "condition": row.get("condition", ""),
            }
    return index


def one_hot_condition(condition: str) -> dict[str, float]:
    w = {c: 0.0 for c in CONDITIONS}
    if condition in w:
        w[condition] = 1.0
    else:
        for c in CONDITIONS:
            w[c] = 1.0 / len(CONDITIONS)
    return w


def choose_routes(img_name: str, args, manifest: dict[str, dict]) -> tuple[dict[str, float], list[tuple[str, float]], dict]:
    if args.condition:
        if args.condition == "global":
            weights = {"global": 1.0}
            selected = [("global", 1.0)]
        else:
            weights = {args.condition: 1.0}
            selected = [(args.condition, 1.0)]
        info = {"mode": "forced", "scene": None, "timeofday": None}
        return weights, selected, info

    if img_name in manifest:
        meta = manifest[img_name]
        scene = meta.get("scene", "")
        timeofday = meta.get("timeofday", "")
        condition = meta.get("condition", "")

        if scene or timeofday:
            weights = _router.weights(scene, timeofday)
        elif condition:
            weights = one_hot_condition(condition)
        else:
            weights = {c: 1.0 / len(CONDITIONS) for c in CONDITIONS}

        selected = select_top_k(weights, top_k=args.top_k)
        info = {
            "mode": "metadata",
            "scene": scene,
            "timeofday": timeofday,
            "condition_hint": condition,
        }
        return weights, selected, info

    weights = {"global": 1.0}
    selected = [("global", 1.0)]
    info = {"mode": "fallback-global", "scene": None, "timeofday": None}
    return weights, selected, info


def collapse_selected_to_runs(selected: list[tuple[str, float]]) -> list[dict]:
    plans: dict[str, dict] = {}

    for requested_condition, weight in selected:
        path, used = resolve_checkpoint(requested_condition)
        key = str(path.resolve())

        if key not in plans:
            plans[key] = {
                "path": path,
                "used": used,
                "requested": [],
                "weight": 0.0,
            }

        plans[key]["requested"].append(requested_condition)
        plans[key]["weight"] += float(weight)

    out = list(plans.values())
    out.sort(key=lambda x: x["weight"], reverse=True)
    return out


def result_to_boxes(result) -> np.ndarray:
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy().reshape(-1, 1)
    cls = result.boxes.cls.cpu().numpy().reshape(-1, 1)
    return np.concatenate([xyxy, conf, cls], axis=1).astype(np.float32)


def draw_boxes(image: np.ndarray, boxes: np.ndarray, route_text: str) -> np.ndarray:
    canvas = image.copy()

    for row in boxes:
        x1, y1, x2, y2, conf, cls = row
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(canvas, p1, p2, (0, 255, 0), 2)
        label = f"c{int(cls)} {conf:.2f}"
        cv2.putText(
            canvas,
            label,
            (p1[0], max(18, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        route_text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def save_outputs(img_path: Path, pred_dir: Path, boxes: np.ndarray, route_payload: dict) -> None:
    image = cv2.imread(str(img_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    route_text = " | ".join(
        f"{x['used']}:{x['weight']:.2f}" for x in route_payload["effective_runs"]
    )
    drawn = draw_boxes(image, boxes, route_text)

    out_img = pred_dir / img_path.name
    cv2.imwrite(str(out_img), drawn)

    out_json = pred_dir / f"{img_path.stem}.routing.json"
    with open(out_json, "w") as f:
        json.dump(route_payload, f, indent=2)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Image file or directory of .jpg files")
    p.add_argument("--condition", default=None, help="Force a single route: global or one specialist")
    p.add_argument("--manifest", default=None, help="Manifest CSV for metadata routing")
    p.add_argument("--pred_dir", default="data/predictions")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou_nms", type=float, default=0.50, help="IoU threshold for weighted NMS blending")
    p.add_argument("--top_k", default="auto", choices=["auto", "1", "2", "5"])
    p.add_argument("--device", default="")
    return p.parse_args()


def main():
    args = get_args()
    source = Path(args.source)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(source.rglob("*.jpg")) if source.is_dir() else [source]
    manifest = build_manifest_index(args.manifest) if args.manifest else {}

    print(f"Found {len(imgs)} images.")
    for img_path in imgs:
        weights, selected, info = choose_routes(img_path.name, args, manifest)
        effective_runs = collapse_selected_to_runs(selected)

        results_and_weights = []
        for run in effective_runs:
            model = load_model_from_path(run["path"])
            preds = model.predict(
                source=str(img_path),
                conf=args.conf,
                device=args.device,
                save=False,
                verbose=False,
            )
            boxes = result_to_boxes(preds[0])
            results_and_weights.append((boxes, run["weight"]))

        merged = blend_detections(results_and_weights, iou_threshold=args.iou_nms)

        payload = {
            "image_name": img_path.name,
            "raw_weights": weights,
            "selected": [{"condition": c, "weight": float(w)} for c, w in selected],
            "effective_runs": [
                {
                    "used": run["used"],
                    "requested": run["requested"],
                    "weight": float(run["weight"]),
                    "path": str(run["path"]),
                }
                for run in effective_runs
            ],
            "meta": info,
            "n_detections": int(len(merged)),
        }

        save_outputs(img_path, pred_dir, merged, payload)

        selected_str = ", ".join(f"{c}:{w:.2f}" for c, w in selected)
        print(f"  {img_path.name}  [{selected_str}]  -> {len(merged)} detections")

    print("\nDone.")


if __name__ == "__main__":
    main()
