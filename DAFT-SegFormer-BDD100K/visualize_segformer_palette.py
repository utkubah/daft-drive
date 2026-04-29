from __future__ import annotations

import os
from collections import Counter

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from dataset_bdd100k import BDD100KSegDataset


SEG_ROOT = "/Users/berkay/datasets/bdd100k_seg/bdd100k/seg"
CKPT = "checkpoints_segformer/global_b2/best_final_snapshot.pt"
OUT_DIR = "viz_segformer/global_b2_final_palette"
NUM_LABELS = 19
N = 6

PALETTE = {
    0: (128, 64, 128),    # road
    1: (244, 35, 232),    # sidewalk
    2: (70, 70, 70),      # building
    3: (102, 102, 156),   # wall
    4: (190, 153, 153),   # fence
    5: (153, 153, 153),   # pole
    6: (250, 170, 30),    # traffic light
    7: (220, 220, 0),     # traffic sign
    8: (107, 142, 35),    # vegetation
    9: (152, 251, 152),   # terrain
    10: (70, 130, 180),   # sky
    11: (220, 20, 60),    # person
    12: (255, 0, 0),      # rider
    13: (0, 0, 142),      # car
    14: (0, 0, 70),       # truck
    15: (0, 60, 100),     # bus
    16: (0, 80, 100),     # train
    17: (0, 0, 230),      # motorcycle
    18: (119, 11, 32),    # bicycle
    255: (0, 0, 0),       # ignore
}

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_palette(seg_root: str, split: str = "val", max_files: int = 50):
    """
    Auto-derive a label->RGB palette by pairing:
      labels/<split>/*_train_id.png
      color_labels/<split>/*_train_color.png
    """
    label_dir = os.path.join(seg_root, "labels", split)
    color_dir = os.path.join(seg_root, "color_labels", split)

    palette_votes = {}

    label_files = sorted(
        [f for f in os.listdir(label_dir) if f.endswith("_train_id.png")]
    )[:max_files]

    for lf in label_files:
        stem = lf.replace("_train_id.png", "")
        cf = f"{stem}_train_color.png"
        color_path = os.path.join(color_dir, cf)
        label_path = os.path.join(label_dir, lf)

        if not os.path.exists(color_path):
            continue

        label = np.array(Image.open(label_path))
        color = np.array(Image.open(color_path).convert("RGB"))

        for lab in np.unique(label):
            if lab == 255:
                continue
            mask = label == lab
            if mask.sum() == 0:
                continue

            pixels = color[mask]
            if len(pixels) == 0:
                continue

            # Most common RGB for this label in this image
            rgb = Counter(map(tuple, pixels.tolist())).most_common(1)[0][0]
            palette_votes.setdefault(int(lab), []).append(rgb)

    palette = {}
    for lab, rgbs in palette_votes.items():
        palette[lab] = Counter(rgbs).most_common(1)[0][0]

    # fallback colors for unseen labels
    fallback = {
        0: (0, 0, 0),
        1: (128, 64, 128),
        2: (244, 35, 232),
        3: (70, 70, 70),
        4: (102, 102, 156),
        5: (190, 153, 153),
        6: (153, 153, 153),
        7: (250, 170, 30),
        8: (220, 220, 0),
        9: (107, 142, 35),
        10: (152, 251, 152),
        11: (0, 130, 180),
        12: (220, 20, 60),
        13: (255, 0, 0),
        14: (0, 0, 142),
        15: (0, 0, 70),
        16: (0, 60, 100),
        17: (0, 80, 100),
        18: (119, 11, 32),
    }
    for k, v in fallback.items():
        palette.setdefault(k, v)

    return palette


def colorize_mask(mask: np.ndarray, palette: dict[int, tuple[int, int, int]]) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    for lab, rgb in palette.items():
        out[mask == lab] = rgb

    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    device = get_device()
    print("device:", device)

    processor = SegformerImageProcessor(
        do_resize=True,
        size={"height": 512, "width": 512},
        do_reduce_labels=False,
    )

    palette = PALETTE
    print("using fixed palette")

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )
    state = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()

    ds = BDD100KSegDataset(
        root=SEG_ROOT,
        split="val",
        processor=processor,
        max_items=N,
    )

    for i in range(min(N, len(ds))):
        sample = ds[i]
        pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
        labels = sample["labels"].cpu().numpy()

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            pred = torch.nn.functional.interpolate(
                logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pred = pred.argmax(dim=1)[0].cpu().numpy()

        img = sample["pixel_values"].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / max(img.max() - img.min(), 1e-8)

        gt_color = colorize_mask(labels, palette)
        pred_color = colorize_mask(pred, palette)

        valid = labels != 255
        diff = np.zeros((*labels.shape, 3), dtype=np.float32)
        diff[(pred == labels) & valid] = [0, 1, 0]
        diff[(pred != labels) & valid] = [1, 0, 0]

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))

        axes[0].imshow(img)
        axes[0].set_title("Image")

        axes[1].imshow(img)
        axes[1].imshow(gt_color, alpha=0.55)
        axes[1].set_title("Ground Truth (palette)")

        axes[2].imshow(img)
        axes[2].imshow(pred_color, alpha=0.55)
        axes[2].set_title("Prediction (palette)")

        axes[3].imshow(img)
        axes[3].imshow(diff, alpha=0.5)
        axes[3].set_title("Green ok / Red wrong")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"sample_{i:02d}.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print("saved:", out_path)

    print("done")


if __name__ == "__main__":
    main()