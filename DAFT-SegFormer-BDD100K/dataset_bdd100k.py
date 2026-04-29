from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset


class BDD100KSegDataset(Dataset):
    def __init__(self, root: str, split: str, processor, max_items: int = 0):
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.processor = processor

        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split

        self.images = sorted(self.image_dir.glob("*.jpg"))
        if max_items > 0:
            self.images = self.images[:max_items]

        print(f"[{split}] images: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        label_path = self.label_dir / f"{image_path.stem}_train_id.png"

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        encoded = self.processor(
            images=image,
            segmentation_maps=label,
            return_tensors="pt",
        )

        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"].squeeze(0),
            "image_name": image_path.name,
        }