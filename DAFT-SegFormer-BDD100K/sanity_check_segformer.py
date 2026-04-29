from __future__ import annotations

import torch

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from dataset_bdd100k import BDD100KSegDataset


SEG_ROOT = "/Users/berkay/Downloads/archive (2) 2/bdd100k_seg/bdd100k/seg"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    device = get_device()
    print("device:", device)

    processor = SegformerImageProcessor(
        do_resize=True,
        size={"height": 512, "width": 512},
        do_reduce_labels=False,
    )

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=19,
        ignore_mismatched_sizes=True,
    ).to(device)
    model.eval()

    ds = BDD100KSegDataset(
        root=SEG_ROOT,
        split="train",
        processor=processor,
        max_items=4,
    )

    sample = ds[0]
    pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
    labels = sample["labels"].unsqueeze(0).to(device)

    print("image_name:", sample["image_name"])
    print("pixel_values shape:", tuple(pixel_values.shape), pixel_values.dtype)
    print("labels shape:", tuple(labels.shape), labels.dtype)
    print("labels unique:", torch.unique(labels).detach().cpu().tolist()[:50])

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, labels=labels)

    print("logits shape:", tuple(outputs.logits.shape))
    print("loss:", float(outputs.loss))
    print("forward OK")


if __name__ == "__main__":
    main()