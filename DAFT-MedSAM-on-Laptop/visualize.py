"""
visualize.py
============
Visualize the demo dataset: images, ground-truth masks, bounding boxes,
and the reference predictions from the paper.

For each .npz file this produces a PNG with four panels:
  [Image + boxes]  [Ground-truth mask]  [Reference prediction]  [Overlay]

3D files (CT/MR) show the middle slice of the bounding box range.

Output saved to:  data/viz/

Usage
-----
  python visualize.py                      # visualize all 10 demo files
  python visualize.py --file 2DBox_US      # one file by name prefix
"""

import os
import glob
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def normalize_for_display(img):
    """Normalize any image to [0, 1] float for matplotlib."""
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-6:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def mask_to_color(mask, alpha=0.45):
    """
    Convert a uint label map to an RGBA overlay.
    Background stays transparent; each label gets a distinct colour.
    """
    H, W = mask.shape
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    colours = [
        [1.00, 0.85, 0.00],   # yellow
        [0.00, 0.70, 1.00],   # blue
        [1.00, 0.30, 0.30],   # red
        [0.30, 1.00, 0.30],   # green
        [0.90, 0.40, 0.90],   # purple
    ]
    for i, label in enumerate(np.unique(mask)):
        if label == 0:
            continue
        c = colours[i % len(colours)]
        where = mask == label
        rgba[where, :3] = c
        rgba[where, 3]  = alpha
    return rgba


def get_middle_slice(imgs_3d, boxes_3d):
    """Pick the slice index at the midpoint of the first 3-D bounding box."""
    z0 = int(boxes_3d[0, 2])
    z1 = int(boxes_3d[0, 5])
    return (z0 + z1) // 2


# ---------------------------------------------------------------------------
# 2D visualisation
# ---------------------------------------------------------------------------

def viz_2d(name, imgs_dir, gts_dir, segs_dir, out_dir):
    img_data = np.load(os.path.join(imgs_dir, name), allow_pickle=True)
    gt_data  = np.load(os.path.join(gts_dir,  name), allow_pickle=True)

    img   = img_data["imgs"]              # (H, W, 3)  uint8
    boxes = img_data["boxes"]             # (N, 4)
    gts   = gt_data["gts"]               # (H, W)

    ref_segs = None
    seg_path = os.path.join(segs_dir, name)
    if os.path.isfile(seg_path):
        ref_segs = np.load(seg_path, allow_pickle=True)["segs"]

    img_disp = normalize_for_display(img)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(name.replace(".npz", ""), fontsize=13, fontweight="bold")

    # Panel 1: image + bounding boxes
    axes[0].imshow(img_disp)
    axes[0].set_title("Image  +  GT boxes")
    for box in boxes:
        x0, y0, x1, y1 = box
        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2, edgecolor="cyan", facecolor="none"
        )
        axes[0].add_patch(rect)

    # Panel 2: ground-truth mask
    axes[1].imshow(img_disp)
    axes[1].imshow(mask_to_color(gts))
    axes[1].set_title("Ground-truth mask")

    # Panel 3: reference prediction (from paper)
    if ref_segs is not None:
        axes[2].imshow(img_disp)
        axes[2].imshow(mask_to_color(ref_segs))
        axes[2].set_title("Paper reference prediction")
    else:
        axes[2].imshow(img_disp)
        axes[2].set_title("Reference prediction\n(not available)")

    # Panel 4: side-by-side GT vs ref
    if ref_segs is not None:
        diff = np.zeros((*gts.shape, 3), dtype=np.float32)
        diff[(gts > 0) & (ref_segs > 0)]  = [0, 1, 0]   # green  = correct
        diff[(gts > 0) & (ref_segs == 0)] = [1, 0, 0]   # red    = missed
        diff[(gts == 0) & (ref_segs > 0)] = [1, 1, 0]   # yellow = false positive
        axes[3].imshow(img_disp)
        axes[3].imshow(diff, alpha=0.5)
        axes[3].set_title("Green=correct  Red=missed  Yellow=FP")
    else:
        axes[3].imshow(img_disp)
        axes[3].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(out_dir, name.replace(".npz", ".png"))
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# 3D visualisation  (shows middle slice of the first box)
# ---------------------------------------------------------------------------

def viz_3d(name, imgs_dir, gts_dir, segs_dir, out_dir):
    img_data = np.load(os.path.join(imgs_dir, name), allow_pickle=True)
    gt_data  = np.load(os.path.join(gts_dir,  name), allow_pickle=True)

    imgs_3d  = img_data["imgs"]           # (D, H, W)
    boxes_3d = img_data["boxes"]          # (N, 6)
    gts_3d   = gt_data["gts"]            # (D, H, W)

    ref_segs_3d = None
    seg_path = os.path.join(segs_dir, name)
    if os.path.isfile(seg_path):
        ref_segs_3d = np.load(seg_path, allow_pickle=True)["segs"]

    z = get_middle_slice(imgs_3d, boxes_3d)
    img   = imgs_3d[z]                    # (H, W)
    gts   = gts_3d[z]                    # (H, W)
    ref_segs = ref_segs_3d[z] if ref_segs_3d is not None else None

    D = imgs_3d.shape[0]
    img_disp = normalize_for_display(np.stack([img, img, img], axis=-1))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    title = f"{name.replace('.npz','')}  [slice {z}/{D-1}]"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Panel 1: image + 2-D projection of 3-D boxes
    axes[0].imshow(img_disp, cmap="gray")
    axes[0].set_title(f"Slice {z}  +  box projection")
    for box3d in boxes_3d:
        x0, y0, z0, x1, y1, z1 = box3d
        if int(z0) <= z <= int(z1):
            rect = patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=2, edgecolor="cyan", facecolor="none"
            )
            axes[0].add_patch(rect)

    # Panel 2: ground truth
    axes[1].imshow(img_disp, cmap="gray")
    axes[1].imshow(mask_to_color(gts))
    axes[1].set_title("Ground-truth mask")

    # Panel 3: reference prediction
    if ref_segs is not None:
        axes[2].imshow(img_disp, cmap="gray")
        axes[2].imshow(mask_to_color(ref_segs))
        axes[2].set_title("Paper reference prediction")
    else:
        axes[2].imshow(img_disp, cmap="gray")
        axes[2].set_title("Reference prediction\n(not available)")

    # Panel 4: error map
    if ref_segs is not None:
        diff = np.zeros((*gts.shape, 3), dtype=np.float32)
        diff[(gts > 0) & (ref_segs > 0)]  = [0, 1, 0]
        diff[(gts > 0) & (ref_segs == 0)] = [1, 0, 0]
        diff[(gts == 0) & (ref_segs > 0)] = [1, 1, 0]
        axes[3].imshow(img_disp, cmap="gray")
        axes[3].imshow(diff, alpha=0.5)
        axes[3].set_title("Green=correct  Red=missed  Yellow=FP")
    else:
        axes[3].imshow(img_disp, cmap="gray")
        axes[3].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(out_dir, name.replace(".npz", ".png"))
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# summary strip  (one row per file, for a quick overview)
# ---------------------------------------------------------------------------

def viz_summary(imgs_dir, gts_dir, segs_dir, out_dir):
    """Single figure showing one representative image per demo file."""
    names = sorted(os.path.basename(f)
                   for f in glob.glob(os.path.join(imgs_dir, "*.npz")))

    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        img_data = np.load(os.path.join(imgs_dir, name), allow_pickle=True)
        gt_data  = np.load(os.path.join(gts_dir,  name), allow_pickle=True)

        imgs = img_data["imgs"]
        gts  = gt_data["gts"]

        # Pick representative slice
        if imgs.ndim == 3 and gts.ndim == 3 and imgs.shape[0] == gts.shape[0]:
            # 3D: pick middle slice
            z    = imgs.shape[0] // 2
            img  = imgs[z]
            mask = gts[z]
        else:
            img  = imgs
            mask = gts

        # Make RGB
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        img_disp = normalize_for_display(img)
        ax.imshow(img_disp)
        ax.imshow(mask_to_color(mask))
        ax.set_title(name.replace("Box_", "\n").replace("_demo.npz", ""),
                     fontsize=8)
        ax.axis("off")

    plt.suptitle("Demo dataset overview  (image + GT mask)", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "00_overview.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Visualize demo dataset.")
    p.add_argument("--imgs_dir", default="data/test_imgs")
    p.add_argument("--gts_dir",  default="data/test_gts")
    p.add_argument("--segs_dir", default="data/test_demo_segs",
                   help="Optional: reference predictions from the paper")
    p.add_argument("--out_dir",  default="data/viz")
    p.add_argument("--file",     default=None,
                   help="Visualize only one file (match by name prefix, e.g. '2DBox_US')")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_names = sorted(
        os.path.basename(f)
        for f in glob.glob(os.path.join(args.imgs_dir, "*.npz"))
    )

    if args.file:
        all_names = [n for n in all_names if args.file in n]
        if not all_names:
            print(f"No file matching '{args.file}' in {args.imgs_dir}")
            return

    print(f"\nVisualizing {len(all_names)} file(s) -> {args.out_dir}/\n")

    # Overview strip first
    viz_summary(args.imgs_dir, args.gts_dir, args.segs_dir, args.out_dir)

    # Individual panels
    for name in all_names:
        if name.startswith("3D"):
            viz_3d(name, args.imgs_dir, args.gts_dir, args.segs_dir, args.out_dir)
        else:
            viz_2d(name, args.imgs_dir, args.gts_dir, args.segs_dir, args.out_dir)

    print(f"\nDone. Open {args.out_dir}/ to see the visualisations.")


if __name__ == "__main__":
    main()
