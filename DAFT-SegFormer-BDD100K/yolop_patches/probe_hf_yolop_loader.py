from __future__ import annotations

from pathlib import Path
from collections import Counter

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from lib.config import cfg
from lib.dataset import HfBddTxtDataset


DATA_ROOT = Path.home() / "projects" / "daft-drive" / "DAFT-SegFormer-BDD100K" / "data" / "hf_bdd100k_yolop_global"


def main():
    cfg.defrost()

    cfg.DATASET.DATAROOT = str(DATA_ROOT / "images")
    cfg.DATASET.LABELROOT = str(DATA_ROOT / "labels")
    cfg.DATASET.MASKROOT = str(DATA_ROOT / "masks")
    cfg.DATASET.LANEROOT = str(DATA_ROOT / "lanes")
    cfg.DATASET.TRAIN_SET = "train"
    cfg.DATASET.TEST_SET = "val"
    cfg.DATASET.DATASET = "HfBddTxtDataset"
    cfg.DATASET.DATA_FORMAT = "jpg"
    cfg.DATASET.ORG_IMG_SIZE = [720, 1280]

    cfg.num_seg_class = 2
    cfg.TRAIN.DET_ONLY = True
    cfg.TRAIN.DRIVABLE_ONLY = False
    cfg.TRAIN.LANE_ONLY = False

    cfg.freeze()

    tfm = transforms.Compose([transforms.ToTensor()])

    ds = HfBddTxtDataset(cfg=cfg, is_train=True, inputsize=640, transform=tfm)

    print("dataset len:", len(ds))
    print("img root   :", ds.img_root)
    print("label root :", ds.label_root)
    print("mask root  :", ds.mask_root)
    print("lane root  :", ds.lane_root)

    print("\n=== FIRST 3 DB ENTRIES ===")
    for i in range(3):
        item = ds.db[i]
        print(f"\nDB[{i}] keys:", list(item.keys()))
        print("image:", item["image"])
        print("mask :", item["mask"])
        print("lane :", item["lane"])
        lab = item["label"]
        print("label shape:", getattr(lab, "shape", None))
        if getattr(lab, "shape", None) is not None and lab.size > 0:
            print("first labels:\n", lab[:5])
            print("unique class ids:", sorted(set(lab[:, 0].astype(int).tolist())))
        else:
            print("no labels")

    print("\n=== CLASS ID DISTRIBUTION OVER FIRST 200 SAMPLES ===")
    ctr = Counter()
    for item in ds.db[:200]:
        lab = item["label"]
        if getattr(lab, "shape", None) is not None and lab.size > 0:
            ctr.update(lab[:, 0].astype(int).tolist())
    print(dict(sorted(ctr.items())))

    print("\n=== DATALOADER SMOKE TEST ===")
    dl = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=ds.collate_fn,
    )

    imgs, targets, paths, shapes = next(iter(dl))
    det_t, seg_t, lane_t = targets

    print("imgs shape      :", tuple(imgs.shape))
    print("det target shape:", tuple(det_t.shape))
    print("seg target shape:", tuple(seg_t.shape))
    print("lane target shape:", tuple(lane_t.shape))
    print("first batch paths:", paths[:2])

    print("\nprobe OK")


if __name__ == "__main__":
    main()
