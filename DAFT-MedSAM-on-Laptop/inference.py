"""
inference.py
============
PyTorch inference for EfficientViT-SAM with DAFT routing.

DAFT routing
------------
The filename prefix determines which specialist checkpoint to load.
For example "2DBox_US_demo.npz" maps to the "US" specialist.
If no specialist exists for a modality, the global model is used.

Handles two cases:
  2D files (prefix "2DBox_*") -- one image, one or more bounding boxes
  3D files (prefix "3DBox_*") -- volumetric stack; slices are processed
                                 from the midpoint outward, with the
                                 previous slice's mask providing the
                                 bounding box for the next (propagation).

Input npz format (test_imgs/*.npz)
-----------------------------------
  2D:  imgs  (H, W, 3)  uint8 RGB
        boxes (N, 4)    int64  [x0, y0, x1, y1] in original image coords

  3D:  imgs  (D, H, W)  uint8 grayscale
        boxes (N, 6)    int64  [x0, y0, z0, x1, y1, z1] in original coords
        spacing (3,)    float  voxel spacing (mm) -- not used here

Output npz format (test_preds/*.npz)
--------------------------------------
  segs  same shape as input  uint16  (0 = background, 1/2/... = structure idx)

Usage
-----
  python inference.py \\
    --img_dir    data/test_imgs \\
    --pred_dir   data/test_preds \\
    --ckpt_dir   checkpoints \\
    --device     cpu
"""

import os
import glob
import argparse
import numpy as np
import cv2
import torch

from model import MedSAM


# --- DAFT routing ---
# Directly translated from PerfectMetaOpenVINO.py in the original repo.
# Maps filename prefix -> specialist model name.

def filename_to_modelname(filename):
    """Return the specialist model name for a given .npz filename."""
    if filename.startswith("3DBox_PET"): return "3D"
    if filename.startswith("3DBox_MR"):  return "3D"
    if filename.startswith("3DBox_CT"):  return "3D"

    if filename.startswith("2DBox_X-Ray"):    return "XRay"
    if filename.startswith("2DBox_XRay"):     return "XRay"
    if filename.startswith("2DBox_CXR"):      return "XRay"
    if filename.startswith("2DBox_XR"):       return "XRay"
    if filename.startswith("2DBox_US"):       return "US"
    if filename.startswith("2DBox_Ultra"):    return "US"
    if filename.startswith("2DBox_Fundus"):   return "Fundus"
    if filename.startswith("2DBox_Endoscopy"):  return "Endoscopy"
    if filename.startswith("2DBox_Endoscope"):  return "Endoscopy"
    if filename.startswith("2DBox_Dermoscope"): return "Dermoscopy"
    if filename.startswith("2DBox_Dermoscopy"): return "Dermoscopy"
    if filename.startswith("2DBox_Microscope"): return "Microscopy"
    if filename.startswith("2DBox_Microscopy"): return "Microscopy"
    if filename.startswith("2DBox_CT"):  return "3D"
    if filename.startswith("2DBox_MR"):  return "3D"
    if filename.startswith("2DBox_PET"): return "3D"
    if filename.startswith("2DBox_Mamm"):return "Mammography"
    if filename.startswith("2DBox_OCT"): return "OCT"

    # fallback substring checks
    if "Microscope"  in filename or "Microscopy"  in filename: return "Microscopy"
    if "Dermoscopy"  in filename:                              return "Dermoscopy"
    if "Endoscopy"   in filename:                              return "Endoscopy"
    if "Fundus"      in filename:                              return "Fundus"
    if "X-Ray"       in filename or "XRay" in filename:        return "XRay"
    if "PET"         in filename:                              return "3D"
    if "OCT"         in filename:                              return "OCT"
    if "MR"          in filename:                              return "3D"
    if "Mamm"        in filename:                              return "Mammography"
    if "US"          in filename:                              return "US"
    if "CT"          in filename:                              return "3D"

    print(f"  WARNING: no routing match for '{filename}', using global model")
    return "global"


# --- model cache -- load each specialist once ---
_MODEL_CACHE = {}

def load_model(model_name, ckpt_dir, device):
    """
    Return a cached MedSAM model.
    Tries checkpoints/<model_name>/best.pth first,
    then checkpoints/<model_name>.pth,
    then falls back to checkpoints/global.pth.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    candidates = [
        os.path.join(ckpt_dir, model_name, "best.pth"),
        os.path.join(ckpt_dir, f"{model_name}.pth"),
        os.path.join(ckpt_dir, "global.pth"),
    ]
    ckpt_path = next((p for p in candidates if os.path.isfile(p)), None)

    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint found for '{model_name}'. "
            f"Searched:\n" + "\n".join(f"  {p}" for p in candidates)
        )

    print(f"  Loading model '{model_name}' from {ckpt_path}")
    model = MedSAM(ckpt_path).to(device).eval()
    _MODEL_CACHE[model_name] = model
    return model


# --- image preprocessing ---
def preprocess(img_hwc):
    """
    Resize longest side to 256, normalize to [0,1], zero-pad to 256x256.

    Args:
        img_hwc  (H, W, 3)  uint8 or float

    Returns:
        tensor   (1, 3, 256, 256)  float32 ready for model
        new_h    int  height after resize (before padding)
        new_w    int  width  after resize (before padding)
    """
    H, W = img_hwc.shape[:2]
    scale = 256 / max(H, W)
    new_h = int(H * scale + 0.5)
    new_w = int(W * scale + 0.5)

    resized = cv2.resize(img_hwc, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized = resized.astype(np.float32)

    vmin, vmax = resized.min(), resized.max()
    normed = (resized - vmin) / max(vmax - vmin, 1e-8)

    padded = np.zeros((256, 256, 3), dtype=np.float32)
    padded[:new_h, :new_w] = normed

    tensor = torch.from_numpy(padded.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, new_h, new_w


def scale_box(box_orig, orig_H, orig_W):
    """
    Scale a bounding box from original image coordinates to 256-padded space.

    box_orig : array-like (4,)  [x0, y0, x1, y1]
    Returns  : (4,) float32
    """
    ratio = 256 / max(orig_H, orig_W)
    return (np.array(box_orig, dtype=np.float32) * ratio)


def bbox_from_mask(mask_256):
    """
    Get the tight bounding box of a binary mask in 256-space.
    Returns (4,) float32 or None if mask is empty.
    """
    ys, xs = np.where(mask_256 > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


# --- single-slice inference ---
def infer_slice(model, img_tensor, box_256, new_h, new_w, orig_H, orig_W, device):
    """
    Run one forward pass and return the binary mask in original image space.

    img_tensor : (1, 3, 256, 256)  CPU tensor
    box_256    : (4,)  float32  [x0, y0, x1, y1] in 256-padded space
    Returns    : (orig_H, orig_W)  uint8
    """
    box_t = torch.from_numpy(box_256).float().reshape(1, 1, 1, 4).to(device)

    with torch.no_grad():
        logits, _ = model(img_tensor.to(device), box_t)   # (1, 1, 256, 256)

    # Remove padding -> resize back to original resolution
    logits_np = logits[0, 0].cpu().numpy()          # (256, 256)
    logits_np = logits_np[:new_h, :new_w]            # trim pad
    logits_up = cv2.resize(logits_np, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
    return (logits_up > 0).astype(np.uint8)


# --- 2D file inference ---
def process_2d(model, npz_path, pred_dir, device):
    """
    Segment a 2D image.
    Each row in npz['boxes'] is one structure; the output labels them 1, 2, ?
    """
    npz_name = os.path.basename(npz_path)
    data  = np.load(npz_path, allow_pickle=True)
    imgs  = data["imgs"]    # (H, W, 3)  uint8
    boxes = data["boxes"]   # (N, 4)     [x0, y0, x1, y1] original coords

    orig_H, orig_W = imgs.shape[:2]
    segs = np.zeros((orig_H, orig_W), dtype=np.uint16)

    img_tensor, new_h, new_w = preprocess(imgs)

    for idx, box in enumerate(boxes, start=1):
        box256 = scale_box(box, orig_H, orig_W)
        mask   = infer_slice(model, img_tensor, box256, new_h, new_w, orig_H, orig_W, device)
        segs[mask > 0] = idx

    np.savez_compressed(os.path.join(pred_dir, npz_name), segs=segs)
    print(f"    {npz_name}  ->  {int(segs.max())} structure(s)")


# --- 3D file inference (slice propagation) ---
def process_3d(model, npz_path, pred_dir, device):
    """
    Segment a 3D volume using midpoint-outward slice propagation.

    For each 3-D bounding box:
      1. Start at the midpoint slice (z_mid) using the scaled 2-D box.
      2. Propagate forward  (z_mid -> z_max): use previous mask's bbox.
      3. Propagate backward (z_mid -> z_min): use next mask's bbox.
    """
    npz_name  = os.path.basename(npz_path)
    data      = np.load(npz_path, allow_pickle=True)
    imgs_3d   = data["imgs"]    # (D, H, W)  uint8
    boxes_3d  = data["boxes"]   # (N, 6)     [x0,y0,z0, x1,y1,z1]

    D, orig_H, orig_W = imgs_3d.shape
    segs = np.zeros((D, orig_H, orig_W), dtype=np.uint16)

    # --- which slices do we need? ---
    needed = set()
    for box in boxes_3d:
        z0, z1 = int(box[2]), int(box[5])
        for z in range(max(0, z0), min(D, z1)):
            needed.add(z)

    print(f"    {npz_name}: pre-computing {len(needed)} slice embeddings ?")

    # Pre-compute (img_tensor, new_h, new_w) for every needed slice
    slice_data = {}
    for z in sorted(needed):
        sl    = imgs_3d[z]                                         # (H, W)
        sl_3c = np.stack([sl, sl, sl], axis=-1)                    # (H, W, 3)
        t, nh, nw = preprocess(sl_3c)
        slice_data[z] = (t, nh, nw)

    # --- process each 3-D bounding box ---
    for idx, box in enumerate(boxes_3d, start=1):
        x0, y0, z0, x1, y1, z1 = box
        z0i   = max(0, int(z0))
        z1i   = min(D, int(z1))
        z_mid = (z0i + z1i) // 2
        mid_box_2d = np.array([x0, y0, x1, y1], dtype=np.float32)

        segs_tmp = np.zeros((D, orig_H, orig_W), dtype=np.uint16)

        # --- forward: z_mid -> z1 ---
        for z in range(z_mid, z1i):
            t, nh, nw = slice_data[z]

            if z == z_mid:
                box256 = scale_box(mid_box_2d, orig_H, orig_W)
            else:
                prev_mask = segs_tmp[z - 1]
                if prev_mask.max() > 0:
                    # Resize prev mask to 256-space to get bbox
                    prev_256 = cv2.resize(
                        prev_mask.astype(np.float32), (nw, nh),
                        interpolation=cv2.INTER_NEAREST
                    )
                    padded_256 = np.zeros((256, 256), dtype=np.float32)
                    padded_256[:nh, :nw] = prev_256
                    b = bbox_from_mask(padded_256)
                    box256 = b if b is not None else scale_box(mid_box_2d, orig_H, orig_W)
                else:
                    box256 = scale_box(mid_box_2d, orig_H, orig_W)

            mask = infer_slice(model, t, box256, nh, nw, orig_H, orig_W, device)
            segs_tmp[z, mask > 0] = idx

        # --- backward: z_mid-1 -> z0 ---
        for z in range(z_mid - 1, z0i - 1, -1):
            t, nh, nw = slice_data[z]

            next_mask = segs_tmp[z + 1]
            if next_mask.max() > 0:
                next_256 = cv2.resize(
                    next_mask.astype(np.float32), (nw, nh),
                    interpolation=cv2.INTER_NEAREST
                )
                padded_256 = np.zeros((256, 256), dtype=np.float32)
                padded_256[:nh, :nw] = next_256
                b = bbox_from_mask(padded_256)
                box256 = b if b is not None else scale_box(mid_box_2d, orig_H, orig_W)
            else:
                box256 = scale_box(mid_box_2d, orig_H, orig_W)

            mask = infer_slice(model, t, box256, nh, nw, orig_H, orig_W, device)
            segs_tmp[z, mask > 0] = idx

        segs[segs_tmp > 0] = idx

    np.savez_compressed(os.path.join(pred_dir, npz_name), segs=segs)
    print(f"    {npz_name}  ->  done ({int(segs.max())} structure(s))")


# --- main ---
def get_args():
    p = argparse.ArgumentParser(description="DAFT inference on .npz test images.")
    p.add_argument("--img_dir",  default="data/test_imgs",
                   help="Directory containing input .npz files (imgs + boxes)")
    p.add_argument("--pred_dir", default="data/test_preds",
                   help="Directory where prediction .npz files will be saved")
    p.add_argument("--ckpt_dir", default="checkpoints",
                   help="Root checkpoint directory (searched for specialist + global)")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.pred_dir, exist_ok=True)

    npz_files = sorted(glob.glob(os.path.join(args.img_dir, "*.npz")))
    if not npz_files:
        print(f"No .npz files found in {args.img_dir}")
        return

    print(f"Found {len(npz_files)} files. Device: {args.device}")

    for npz_path in npz_files:
        name       = os.path.basename(npz_path)
        model_name = filename_to_modelname(name)
        model      = load_model(model_name, args.ckpt_dir, args.device)

        if name.startswith("3D"):
            process_3d(model, npz_path, args.pred_dir, args.device)
        else:
            process_2d(model, npz_path, args.pred_dir, args.device)

    print("\nInference complete.")


if __name__ == "__main__":
    main()
