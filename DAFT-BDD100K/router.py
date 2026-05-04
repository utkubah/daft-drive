"""
router.py
=========
Adaptive soft router for DAFT-BDD100K specialists.

Determines blend weights over the 5 specialists given metadata or image features.
At inference, supports configurable top-K:
  - auto -> if max_weight >= threshold: top-1, else top-2
  - 1    -> always top-1
  - 2    -> always top-2
  - 5    -> blend all 5 specialists
"""

from __future__ import annotations

import numpy as np

CONDITIONS = ["city_day", "city_night", "highway_day", "highway_night", "residential"]
CONFIDENT_THRESHOLD = 0.70


def _normalize_selection(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    total = sum(w for _, w in items)
    if total <= 0:
        return [("city_day", 1.0)]
    return [(c, w / total) for c, w in items]


def select_top_k(
    weights: dict[str, float],
    top_k: str | int = "auto",
    threshold: float = CONFIDENT_THRESHOLD,
) -> list[tuple[str, float]]:
    ranked = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    ranked = [(c, float(w)) for c, w in ranked if w > 0]

    if not ranked:
        return [("city_day", 1.0)]

    if str(top_k) == "auto":
        if ranked[0][1] >= threshold:
            return [(ranked[0][0], 1.0)]
        k = min(2, len(ranked))
        return _normalize_selection(ranked[:k])

    k = int(top_k)
    k = max(1, min(k, len(ranked)))
    return _normalize_selection(ranked[:k])


class MetadataRouter:
    def weights(self, scene: str | None, timeofday: str | None) -> dict[str, float]:
        scene = (scene or "").lower().strip()
        timeofday = (timeofday or "").lower().strip()

        w = {c: 0.0 for c in CONDITIONS}

        if "city" in scene:
            if timeofday == "night":
                w["city_night"] = 1.0
            elif timeofday == "daytime":
                w["city_day"] = 1.0
            elif timeofday == "dawn/dusk":
                w["city_day"] = 0.5
                w["city_night"] = 0.5
            else:
                w["city_day"] = 0.6
                w["city_night"] = 0.4

        elif "highway" in scene:
            if timeofday == "night":
                w["highway_night"] = 1.0
            elif timeofday == "daytime":
                w["highway_day"] = 1.0
            elif timeofday == "dawn/dusk":
                w["highway_day"] = 0.5
                w["highway_night"] = 0.5
            else:
                w["highway_day"] = 0.6
                w["highway_night"] = 0.4

        elif "residential" in scene:
            w["residential"] = 1.0

        else:
            for c in CONDITIONS:
                w[c] = 1.0 / len(CONDITIONS)

        return w

    def select(
        self,
        weights: dict[str, float],
        threshold: float = CONFIDENT_THRESHOLD,
        top_k: str | int = "auto",
    ) -> list[tuple[str, float]]:
        return select_top_k(weights, top_k=top_k, threshold=threshold)


class ImageRouter:
    def __init__(self, ckpt_path: str | None = None):
        self.model = None
        if ckpt_path:
            import torch
            self.model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.model.eval()

    def weights_from_features(self, sppf_feat) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("ImageRouter: no checkpoint loaded.")
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            pooled = sppf_feat.mean(dim=[2, 3])
            logits = self.model(pooled)
            probs = F.softmax(logits, dim=-1).squeeze()
        return {c: float(probs[i]) for i, c in enumerate(CONDITIONS)}

    def select(
        self,
        weights: dict[str, float],
        threshold: float = CONFIDENT_THRESHOLD,
        top_k: str | int = "auto",
    ) -> list[tuple[str, float]]:
        return select_top_k(weights, top_k=top_k, threshold=threshold)


def blend_detections(
    results_and_weights: list[tuple[np.ndarray, float]],
    iou_threshold: float = 0.5,
) -> np.ndarray:
    all_boxes: list[np.ndarray] = []
    for boxes, w in results_and_weights:
        if boxes is None or len(boxes) == 0:
            continue
        scaled = boxes.copy().astype(np.float32)
        scaled[:, 4] *= float(w)
        all_boxes.append(scaled)

    if not all_boxes:
        return np.zeros((0, 6), dtype=np.float32)

    merged = np.concatenate(all_boxes, axis=0)
    keep = _greedy_nms(merged, iou_threshold)
    return merged[keep]


def _greedy_nms(boxes: np.ndarray, iou_thr: float) -> list[int]:
    if len(boxes) == 0:
        return []

    order = np.argsort(-boxes[:, 4])
    keep = []

    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break

        rest = order[1:]
        ious = _iou(boxes[i], boxes[rest])
        same_cls = boxes[rest, 5] == boxes[i, 5]
        suppress = same_cls & (ious > iou_thr)
        order = rest[~suppress]

    return keep


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)
