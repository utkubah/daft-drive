"""
make_demo_splits.py
===================
Build the demo training dataset from the 10 test demo files we already have.

The training pipeline (dataset.py) expects each .npz file to contain BOTH:
  imgs  -- the image data
  gts   -- the ground-truth label map

But the demo files are split across two folders:
  data/test_imgs/X.npz  ->  keys: imgs, boxes
  data/test_gts/X.npz   ->  keys: gts

This script merges them into combined files in:
  data/demo_train/      (8 files, 80%)
  data/demo_val/        (2 files, 20%)

And writes the CSV lists that train.py / distill.py consume:
  data/datasplit/demo_train.csv
  data/datasplit/demo_val.csv

Usage (run from daft-clean/):
  python data/datasplit/make_demo_splits.py
"""

import os
import glob
import numpy as np

# Paths (relative to daft-clean/)
IMGS_DIR  = "data/test_imgs"
GTS_DIR   = "data/test_gts"
TRAIN_OUT = "data/demo_train"
VAL_OUT   = "data/demo_val"
CSV_DIR   = "data/datasplit"

os.makedirs(TRAIN_OUT, exist_ok=True)
os.makedirs(VAL_OUT,   exist_ok=True)
os.makedirs(CSV_DIR,   exist_ok=True)

# All 10 demo files, sorted for reproducibility
img_files = sorted(glob.glob(os.path.join(IMGS_DIR, "*.npz")))

# Fixed 80/20 split: first 8 train, last 2 val
train_files_src = img_files[:8]
val_files_src   = img_files[8:]   # last 2

def merge_and_save(src_img_path, out_dir):
    name    = os.path.basename(src_img_path)
    gt_path = os.path.join(GTS_DIR, name)

    img_data = np.load(src_img_path, allow_pickle=True)
    gt_data  = np.load(gt_path,      allow_pickle=True)

    imgs = img_data["imgs"]   # (H,W,3) or (D,H,W)
    gts  = gt_data["gts"]    # (H,W)   or (D,H,W)

    out_path = os.path.join(out_dir, name)

    # Save combined file -- dataset.py only needs 'imgs' and 'gts'
    if "spacing" in gt_data.files:
        np.savez_compressed(out_path, imgs=imgs, gts=gts,
                            spacing=gt_data["spacing"])
    else:
        np.savez_compressed(out_path, imgs=imgs, gts=gts)

    return out_path

train_paths = [merge_and_save(f, TRAIN_OUT) for f in train_files_src]
val_paths   = [merge_and_save(f, VAL_OUT)   for f in val_files_src]

# Write CSVs
import pandas as pd
pd.DataFrame({"file": train_paths}).to_csv(
    os.path.join(CSV_DIR, "demo_train.csv"), index=False)
pd.DataFrame({"file": val_paths}).to_csv(
    os.path.join(CSV_DIR, "demo_val.csv"), index=False)

print("Demo dataset ready:")
print(f"  Train ({len(train_paths)} files): {TRAIN_OUT}/")
for p in train_paths:
    print(f"    {os.path.basename(p)}")
print(f"  Val   ({len(val_paths)} files): {VAL_OUT}/")
for p in val_paths:
    print(f"    {os.path.basename(p)}")
print()
print("CSVs written:")
print(f"  data/datasplit/demo_train.csv")
print(f"  data/datasplit/demo_val.csv")
print()
print("To simulate training (2 quick epochs, batch=1):")
print("  python train.py \\")
print("    --train_csv data/datasplit/demo_train.csv \\")
print("    --val_csv   data/datasplit/demo_val.csv   \\")
print("    --weights   checkpoints/global.pth        \\")
print("    --name      demo                          \\")
print("    --epochs    2 --batch_size 1 --num_workers 0")
