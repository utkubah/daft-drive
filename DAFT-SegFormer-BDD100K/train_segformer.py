from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from dataset_bdd100k import BDD100KSegDataset


SEG_ROOT = "/Users/berkay/Downloads/archive (2) 2/bdd100k_seg/bdd100k/seg"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Config:
    train_items: int = 50
    val_items: int = 10
    batch_size: int = 1
    epochs: int = 1
    lr: float = 6e-5
    num_labels: int = 19
    out_dir: str = "checkpoints_segformer/bdd100k_tiny"


def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch], dim=0)
    labels = torch.stack([x["labels"] for x in batch], dim=0)
    names = [x["image_name"] for x in batch]
    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "image_name": names,
    }


def evaluate(model, loader, device):
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            losses.append(float(outputs.loss.detach().cpu()))

    return sum(losses) / max(len(losses), 1)


def main():
    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = get_device()
    print("device:", device)

    processor = SegformerImageProcessor(
        do_resize=True,
        size={"height": 512, "width": 512},
        do_reduce_labels=False,
    )

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=cfg.num_labels,
        ignore_mismatched_sizes=True,
    ).to(device)

    train_ds = BDD100KSegDataset(
        root=SEG_ROOT,
        split="train",
        processor=processor,
        max_items=cfg.train_items,
    )
    val_ds = BDD100KSegDataset(
        root=SEG_ROOT,
        split="val",
        processor=processor,
        max_items=cfg.val_items,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_val = float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        train_losses = []

        for step, batch in enumerate(train_loader, start=1):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.detach().cpu()))

            if step % 10 == 0 or step == len(train_loader):
                print(f"epoch {epoch+1} step {step}/{len(train_loader)} loss {train_losses[-1]:.4f}")

        train_loss = sum(train_losses) / max(len(train_losses), 1)
        val_loss = evaluate(model, val_loader, device)

        print(f"epoch {epoch+1}: train {train_loss:.4f} | val {val_loss:.4f}")

        latest_path = os.path.join(cfg.out_dir, "latest.pt")
        torch.save(model.state_dict(), latest_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(cfg.out_dir, "best.pt")
            torch.save(model.state_dict(), best_path)
            print("saved best:", best_path)

    print("done")


if __name__ == "__main__":
    main()