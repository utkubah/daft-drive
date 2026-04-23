"""
dataset.py
==========
PyTorch Dataset for medical image segmentation (.npz format).

Each .npz file has two arrays:
  imgs  : image data — (H, W) grayscale or (H, W, 3) RGB for 2D,
                        (D, H, W) for 3D volumetric scans
  gts   : ground-truth label map — same shape, integer class IDs
          (0 = background, 1, 2, 3, ... = different anatomical structures)

Per-sample pipeline:
  1. Load file; if 3D pick one random slice
  2. Ensure 3-channel image (repeat grayscale 3×)
  3. Resize so longest side = 256, keeping aspect ratio
  4. Zero-pad shorter side to reach 256×256
  5. Normalize pixel values to [0, 1]
  6. Pick ONE random foreground label → binary mask
  7. Random horizontal / vertical flips (augmentation only)
  8. Compute bounding box from mask + small random jitter

Why bounding boxes?
-------------------
This is a *prompt-based* model (SAM-style). At training time we derive
the box from the ground truth mask. At inference time the box comes from
the .npz file directly (provided by the dataset).
"""

import random
import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class MedSegDataset(Dataset):
    """
    Args:
        csv_path   : CSV with a 'file' column listing .npz file paths
        img_size   : resize + pad to (img_size × img_size)    default 256
        bbox_shift : max pixel jitter on each bbox edge        default 5
        augment    : random flips during training              default True
    """

    def __init__(self, csv_path, img_size=256, bbox_shift=5, augment=True):
        self.files      = sorted(pd.read_csv(csv_path)["file"].tolist())
        self.img_size   = img_size
        self.bbox_shift = bbox_shift
        self.augment    = augment
        print(f"  Dataset: {len(self.files)} files | augment={augment}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        imgs = data["imgs"]
        gts  = data["gts"]

        # 1. For 3D volumes pick a random 2D slice
        if gts.ndim == 3:
            z    = random.randint(0, gts.shape[0] - 1)
            img  = imgs[z]    # (H, W)
            mask = gts[z]     # (H, W)
        else:
            img  = imgs       # (H, W) or (H, W, 3)
            mask = gts        # (H, W)

        # 2. Convert grayscale to 3-channel
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)   # (H, W, 3)

        # 3 & 4. Resize longest-side → 256, then zero-pad to square
        img  = self._resize(img)
        mask = cv2.resize(
            mask.astype(np.float32),
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(gts.dtype)
        img  = self._pad(img)
        mask = self._pad(mask)

        # 5. Normalize to [0, 1]
        img  = img.astype(np.float32)
        vmin, vmax = img.min(), img.max()
        img  = (img - vmin) / max(vmax - vmin, 1e-8)

        # 6. Pick one random foreground label → binary mask
        label_ids = np.unique(mask)[1:]    # drop 0 (background)
        if len(label_ids) == 0:
            return self.__getitem__(random.randint(0, len(self) - 1))
        chosen      = random.choice(label_ids.tolist())
        binary_mask = (mask == chosen).astype(np.uint8)

        # 7. Data augmentation
        if self.augment:
            if random.random() > 0.5:
                img         = np.ascontiguousarray(np.fliplr(img))
                binary_mask = np.ascontiguousarray(np.fliplr(binary_mask))
            if random.random() > 0.5:
                img         = np.ascontiguousarray(np.flipud(img))
                binary_mask = np.ascontiguousarray(np.flipud(binary_mask))

        # 8. Bounding box with random jitter
        bbox = self._bbox(binary_mask)    # shape (4,)  [x0, y0, x1, y1]

        # Build tensors
        # img:  (3, 256, 256)
        # mask: (1, 256, 256)
        # bbox: (1, 1, 4)   — two extra dims match EfficientViT-SAM prompt encoder input
        img_t  = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(binary_mask[None]).long()
        bbox_t = torch.from_numpy(bbox[None, None]).float()

        return {"image": img_t, "mask": mask_t, "bbox": bbox_t}

    # ── helpers ────────────────────────────────────────────────────────────────

    def _resize(self, image):
        h, w  = image.shape[:2]
        scale = self.img_size / max(h, w)
        nh    = int(h * scale + 0.5)
        nw    = int(w * scale + 0.5)
        return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)

    def _pad(self, image):
        h, w = image.shape[:2]
        ph, pw = self.img_size - h, self.img_size - w
        if image.ndim == 3:
            return np.pad(image, ((0, ph), (0, pw), (0, 0)))
        return np.pad(image, ((0, ph), (0, pw)))

    def _bbox(self, binary_mask):
        H, W   = binary_mask.shape
        ys, xs = np.where(binary_mask > 0)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        x0 = max(0,   x0 - random.randint(0, self.bbox_shift))
        x1 = min(W-1, x1 + random.randint(0, self.bbox_shift))
        y0 = max(0,   y0 - random.randint(0, self.bbox_shift))
        y1 = min(H-1, y1 + random.randint(0, self.bbox_shift))
        return np.array([x0, y0, x1, y1], dtype=np.float32)
