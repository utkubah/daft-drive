"""
router.py
=========
Routing logic for DAFT-BDD100K adaptive specialists.

Two router types
----------------
MetadataRouter  Rule-based soft weights from BDD100K scene + timeofday labels.
                Used when metadata is available (e.g. from manifest CSV).

ImageRouter     MobileNetV3-small classifier trained on condition labels.
                Used when no metadata is available — image routes itself.
                Load with ImageRouter(ckpt_path="checkpoints/router/best.pt").

Top-K selection
---------------
select_top_k(weights, top_k=1)
  1     always top-1  ← default; used in all paper experiments
  2     always top-2
  5     blend all 5 specialists
  "auto"  dynamic: top-1 if max_weight >= 0.70, else top-2.
          NOT used in any reported experiment — K is a fixed
          hyperparameter throughout the paper.

Detection blending
------------------
blend_detections([(boxes_Nx6, weight), ...])
  Scales each specialist's box confidences by its router weight,
  pools all boxes, then applies greedy NMS per class.
  boxes_Nx6 format: [x1, y1, x2, y2, conf, cls]
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

CONDITIONS          = ["city_day", "city_night", "highway_day", "highway_night", "residential"]
CONFIDENT_THRESHOLD = 0.70


# ── Top-K selection ───────────────────────────────────────────────────────────

def select_top_k(
    weights: dict[str, float],
    top_k: str | int = 1,
    threshold: float = CONFIDENT_THRESHOLD,
) -> list[tuple[str, float]]:
    """
    Return [(condition, normalised_weight), ...] for the top-K specialists.

    top_k=1 is the default and is used in all paper experiments (Tables 1–4).
    top_k="auto" is a convenience mode (top-1 if confident, else top-2) that
    was NOT evaluated in the paper; K is always fixed in reported experiments.
    """
    ranked = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    ranked = [(c, float(w)) for c, w in ranked if w > 0]

    if not ranked:
        return [("city_day", 1.0)]

    if str(top_k) == "auto":
        if ranked[0][1] >= threshold:
            return [(ranked[0][0], 1.0)]
        k = min(2, len(ranked))
    else:
        k = max(1, min(int(top_k), len(ranked)))

    top = ranked[:k]
    total = sum(w for _, w in top)
    return [(c, w / total) for c, w in top]


# ── Metadata router ───────────────────────────────────────────────────────────

class MetadataRouter:
    """
    Rule-based soft weights from BDD100K scene + timeofday labels.

    This is an upper-bound oracle used for controlled experiments only.
    Metadata (scene, timeofday) is unavailable from a live camera stream,
    so MetadataRouter is NOT the deployable path — ImageRouter is.
    In the paper it appears as "Hard Routing (oracle, K=1)" in Table 1.

    dawn/dusk receives 50/50 day/night weight (used in Table 4 only).
    Unknown or unsupported scenes fall back to uniform weights.
    """

    def weights(self, scene: str | None, timeofday: str | None) -> dict[str, float]:
        scene     = (scene     or "").lower().strip()
        timeofday = (timeofday or "").lower().strip()
        w = {c: 0.0 for c in CONDITIONS}

        if "city" in scene:
            if timeofday == "night":
                w["city_night"] = 1.0
            elif timeofday == "daytime":
                w["city_day"]   = 1.0
            elif timeofday == "dawn/dusk":
                w["city_day"]   = 0.5
                w["city_night"] = 0.5
            else:
                w["city_day"]   = 0.6
                w["city_night"] = 0.4

        elif "highway" in scene:
            if timeofday == "night":
                w["highway_night"] = 1.0
            elif timeofday == "daytime":
                w["highway_day"]   = 1.0
            elif timeofday == "dawn/dusk":
                w["highway_day"]   = 0.5
                w["highway_night"] = 0.5
            else:
                w["highway_day"]   = 0.6
                w["highway_night"] = 0.4

        elif "residential" in scene:
            w["residential"] = 1.0

        else:                              # tunnel, parking lot, undefined
            for c in CONDITIONS:
                w[c] = 1.0 / len(CONDITIONS)

        return w

    def select(self, weights: dict[str, float], top_k: str | int = "auto") -> list[tuple[str, float]]:
        return select_top_k(weights, top_k=top_k)


# ── Image router ──────────────────────────────────────────────────────────────

class ImageRouter:
    """
    MobileNetV3-small classifier that routes images without metadata.

    Train with train_router.py, then load here via ckpt_path.
    Input: any image file path.  Output: soft weight dict over CONDITIONS.

    Architecture (trained in train_router.py):
      MobileNetV3-small (pretrained ImageNet)
          └── classifier[-1] replaced with Linear(576, 5)
    """

    IMG_SIZE = 224

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        import torchvision.models as models

        self.device = torch.device(device)

        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features   # 1024 for MobileNetV3-small
        model.classifier[-1] = torch.nn.Linear(in_features, len(CONDITIONS))
        model.load_state_dict(
            torch.load(ckpt_path, map_location=self.device, weights_only=True)
        )
        model.eval()
        self.model = model.to(self.device)

    def weights_from_img(self, img: np.ndarray) -> dict[str, float]:
        """Accept a pre-loaded BGR numpy array — avoids double disk read."""
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # ImageNet mean/std normalisation — matches the transforms used during training
        tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.model(tensor), dim=-1).squeeze()
        return {c: float(probs[i]) for i, c in enumerate(CONDITIONS)}

    def weights(self, img_path: str) -> dict[str, float]:
        img = cv2.imread(img_path)
        if img is None:
            return {c: 1.0 / len(CONDITIONS) for c in CONDITIONS}
        return self.weights_from_img(img)

    def select(self, img_path: str, top_k: str | int = "auto") -> list[tuple[str, float]]:
        return select_top_k(self.weights(img_path), top_k=top_k)


# ── Detection blending ────────────────────────────────────────────────────────

def blend_detections(
    results_and_weights: list[tuple[np.ndarray, float]],
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """
    Merge detections from multiple specialists via weighted NMS.

    Each specialist's box confidences are scaled by its router weight,
    then all boxes are pooled and greedy NMS is applied per class.

    Parameters
    ----------
    results_and_weights : [(boxes_Nx6, weight), ...]
        boxes_Nx6 rows: [x1, y1, x2, y2, conf, cls]
    """
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
    return merged[_greedy_nms(merged, iou_threshold)]


def _greedy_nms(boxes: np.ndarray, iou_thr: float) -> list[int]:
    # Sort by confidence descending, then iteratively suppress lower-confidence
    # boxes that overlap the current best box AND share the same class label.
    # Cross-class suppression is intentionally skipped (a car and a bus can overlap).
    order = np.argsort(-boxes[:, 4])
    keep  = []
    while len(order):
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest     = order[1:]
        ious     = _iou(boxes[i], boxes[rest])
        same_cls = boxes[rest, 5] == boxes[i, 5]
        order    = rest[~(same_cls & (ious > iou_thr))]
    return keep


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    # Standard intersection-over-union between one box and an array of boxes.
    # Returns 0 for degenerate (zero-area) pairs to avoid divide-by-zero.
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter  = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union  = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)
