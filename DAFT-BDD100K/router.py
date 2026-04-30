"""
router.py
=========
Adaptive soft router for DAFT-BDD100K specialists.

Determines blend weights over the 5 specialists given metadata or image features.
At inference, uses adaptive top-K:
  - max_weight > CONFIDENT_THRESHOLD  ->  top-1 only   (single forward pass)
  - otherwise                         ->  top-2 blended (two forward passes)

Specialists
-----------
  city_day     city street, daytime / dawn-dusk
  city_night   city street, night
  highway_day  highway, daytime / dawn-dusk
  highway_night highway, night
  residential  residential (all times)

Two routing modes
-----------------
  MetadataRouter  -- rule-based soft weights from scene + timeofday labels.
                     Used when metadata is available at inference time.

  ImageRouter     -- placeholder for a learned MLP trained on backbone features.
                     Swap in by implementing `weights_from_features(feat)`.

blend_detections  -- weighted NMS to merge detections from selected specialists.
"""

from __future__ import annotations

import numpy as np

CONDITIONS = ["city_day", "city_night", "highway_day", "highway_night", "residential"]

# If the top specialist weight exceeds this, run only that one specialist.
CONFIDENT_THRESHOLD = 0.70


# ── Metadata router ──────────────────────────────────────────────────────────

class MetadataRouter:
    """
    Rule-based soft weights from BDD100K scene + timeofday metadata.

    Edge cases handled as soft blends:
      dawn/dusk  -> 50/50 between the day and night variant of that scene
      residential night -> residential (not enough data to split)
      undefined scene   -> uniform weights over all 5 specialists
    """

    def weights(self, scene: str | None, timeofday: str | None) -> dict[str, float]:
        scene     = (scene     or "").lower().strip()
        timeofday = (timeofday or "").lower().strip()

        w = {c: 0.0 for c in CONDITIONS}

        if "city" in scene:
            if timeofday == "night":
                w["city_night"] = 1.0
            elif timeofday in ("daytime",):
                w["city_day"] = 1.0
            elif timeofday == "dawn/dusk":
                w["city_day"]   = 0.5
                w["city_night"] = 0.5
            else:                          # undefined timeofday
                w["city_day"]   = 0.6
                w["city_night"] = 0.4

        elif "highway" in scene:
            if timeofday == "night":
                w["highway_night"] = 1.0
            elif timeofday in ("daytime",):
                w["highway_day"] = 1.0
            elif timeofday == "dawn/dusk":
                w["highway_day"]   = 0.5
                w["highway_night"] = 0.5
            else:
                w["highway_day"]   = 0.6
                w["highway_night"] = 0.4

        elif "residential" in scene:
            w["residential"] = 1.0

        else:
            # Unknown / parking lot / tunnel / gas station — uniform fallback
            for c in CONDITIONS:
                w[c] = 1.0 / len(CONDITIONS)

        return w

    def select(
        self,
        weights: dict[str, float],
        threshold: float = CONFIDENT_THRESHOLD,
    ) -> list[tuple[str, float]]:
        """
        Adaptive top-K selection.
        Returns [(condition, normalized_weight), ...] — 1 or 2 entries.
        """
        ranked = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        ranked = [(c, w) for c, w in ranked if w > 0]

        if not ranked:
            return [("city_day", 1.0)]  # last-resort fallback

        # Top-1 fast path
        if ranked[0][1] >= threshold:
            return [(ranked[0][0], 1.0)]

        # Top-2 blend
        top2 = ranked[:2]
        total = sum(w for _, w in top2)
        return [(c, w / total) for c, w in top2]


# ── Image router (learned — future) ─────────────────────────────────────────

class ImageRouter:
    """
    Placeholder for a learned router trained on backbone (SPPF) features.
    Implement `weights_from_features` to activate.

    Architecture sketch:
      SPPF output (BxCxHxW) -> GlobalAvgPool -> Linear(C, 64) -> ReLU
                             -> Linear(64, 5) -> Softmax
    Trained on scene × timeofday labels with cross-entropy loss.
    """

    def __init__(self, ckpt_path: str | None = None):
        self.model = None
        if ckpt_path:
            import torch
            self.model = torch.load(ckpt_path, map_location="cpu")
            self.model.eval()

    def weights_from_features(self, sppf_feat) -> dict[str, float]:
        """
        sppf_feat: torch.Tensor of shape (1, C, H, W)
        Returns soft weight dict over CONDITIONS.
        """
        if self.model is None:
            raise RuntimeError("ImageRouter: no checkpoint loaded.")
        import torch, torch.nn.functional as F
        with torch.no_grad():
            pooled = sppf_feat.mean(dim=[2, 3])          # (1, C)
            logits = self.model(pooled)                   # (1, 5)
            probs  = F.softmax(logits, dim=-1).squeeze()  # (5,)
        return {c: float(probs[i]) for i, c in enumerate(CONDITIONS)}

    def select(
        self,
        weights: dict[str, float],
        threshold: float = CONFIDENT_THRESHOLD,
    ) -> list[tuple[str, float]]:
        """Same adaptive top-K logic as MetadataRouter."""
        ranked = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        if ranked[0][1] >= threshold:
            return [(ranked[0][0], 1.0)]
        top2  = ranked[:2]
        total = sum(w for _, w in top2)
        return [(c, w / total) for c, w in top2]


# ── Detection blending ───────────────────────────────────────────────────────

def blend_detections(
    results_and_weights: list[tuple[np.ndarray, float]],
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """
    Merge detections from multiple specialists via weighted NMS.

    Parameters
    ----------
    results_and_weights : [(boxes_Nx6, weight), ...]
        boxes_Nx6: each row is [x1, y1, x2, y2, conf, cls]
        weight   : router weight for this specialist (already normalized)

    Returns
    -------
    np.ndarray of shape (M, 6) after weighted NMS
    """
    all_boxes: list[np.ndarray] = []
    for boxes, w in results_and_weights:
        if boxes is None or len(boxes) == 0:
            continue
        scaled = boxes.copy().astype(np.float32)
        scaled[:, 4] *= w            # scale confidence by router weight
        all_boxes.append(scaled)

    if not all_boxes:
        return np.zeros((0, 6), dtype=np.float32)

    merged = np.concatenate(all_boxes, axis=0)

    # Greedy NMS per class
    keep = _greedy_nms(merged, iou_threshold)
    return merged[keep]


def _greedy_nms(boxes: np.ndarray, iou_thr: float) -> list[int]:
    """Simple greedy NMS. boxes: Nx6 [x1,y1,x2,y2,conf,cls]."""
    if len(boxes) == 0:
        return []

    # Sort by confidence descending
    order = np.argsort(-boxes[:, 4])
    keep  = []

    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break

        # IoU of box i against remaining
        rest   = order[1:]
        ious   = _iou(boxes[i], boxes[rest])
        # Only suppress same class
        same_cls = boxes[rest, 5] == boxes[i, 5]
        suppress = same_cls & (ious > iou_thr)
        order    = rest[~suppress]

    return keep


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union  = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)
