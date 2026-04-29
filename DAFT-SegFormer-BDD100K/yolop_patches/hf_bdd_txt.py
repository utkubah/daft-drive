from __future__ import annotations

from pathlib import Path
import numpy as np
from tqdm import tqdm

from .AutoDriveDataset import AutoDriveDataset


class HfBddTxtDataset(AutoDriveDataset):
    def __init__(self, cfg, is_train, inputsize=640, transform=None):
        super().__init__(cfg, is_train, inputsize=inputsize, transform=transform)
        self.db = self._get_db()

    def _get_db(self):
        print("building HF txt database...")
        db = []

        img_paths = sorted(self.img_root.glob(f"*.{self.data_format}"))
        missing = {"label": 0, "mask": 0, "lane": 0}

        for img_path in tqdm(img_paths):
            stem = img_path.stem

            label_path = self.label_root / f"{stem}.txt"
            mask_path = self.mask_root / f"{stem}.png"
            lane_path = self.lane_root / f"{stem}.png"

            if not label_path.exists():
                missing["label"] += 1
                continue
            if not mask_path.exists():
                missing["mask"] += 1
                continue
            if not lane_path.exists():
                missing["lane"] += 1
                continue

            labels = []
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cls_id, xc, yc, w, h = map(float, parts)
                    labels.append([cls_id, xc, yc, w, h])

            if labels:
                label_arr = np.array(labels, dtype=np.float32)
            else:
                label_arr = np.zeros((0, 5), dtype=np.float32)

            rec = {
                "image": str(img_path),
                "label": label_arr,
                "mask": str(mask_path),
                "lane": str(lane_path),
            }
            db.append(rec)

        print("database build done:", len(db))
        print("missing counts:", missing)
        return db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError("Custom eval not implemented for HfBddTxtDataset")
