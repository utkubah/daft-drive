from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from dataset_bdd100k import BDD100KSegDataset


SEG_ROOT = "/Users/berkay/datasets/bdd100k_seg/bdd100k/seg"
CKPT = "checkpoints_segformer/global_b2/best_final_snapshot.pt"
OUT_DIR = "viz_segformer/global_b2_final"
NUM_LABELS = 19
N = 6


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    device = get_device()
    print("device:", device)

    processor = SegformerImageProcessor(
        do_resize=True,
        size={"height": 512, "width": 512},
        do_reduce_labels=False,
    )

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

        valid_gt = labels != 255
        valid_pred = pred != 255

        # Simple correctness map on valid pixels
        diff = np.zeros((*labels.shape, 3), dtype=np.float32)
        correct = (pred == labels) & valid_gt
        wrong = (pred != labels) & valid_gt
        pred_only = valid_pred & (~valid_gt)

        diff[correct] = [0, 1, 0]   # green
        diff[wrong] = [1, 0, 0]     # red
        diff[pred_only] = [1, 1, 0] # yellow

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(img)
        axes[0].set_title("Image")

        axes[1].imshow(labels, interpolation="nearest")
        axes[1].set_title("Ground Truth")

        axes[2].imshow(pred, interpolation="nearest")
        axes[2].set_title("Prediction")

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