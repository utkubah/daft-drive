"""
demo.py
=======
Full end-to-end DAFT demonstration using only the 10 demo files.

DAFT = Data-Aware Fine-Tuning
------------------------------
The key idea: instead of one model for everything, train a SPECIALIST model
for each imaging modality. Each specialist is fine-tuned on its modality's
data only, so it learns modality-specific features.

Full DAFT training pipeline (for reference):
  1. distill.py        -- knowledge distillation: EfficientViT copies
                          TinyViT features  ->  distilled.pth
  2. merge_weights.py  -- glue distilled encoder + LiteMedSAM decoder
                          ->  checkpoints/merged.pth
  3. train.py          -- GLOBAL fine-tune on ALL modalities combined
                          ->  checkpoints/global/best.pth
                          (this is what we downloaded as global.pth)
  4. train.py x9       -- PER-MODALITY DAFT fine-tune, each starting from
                          global.pth, seeing only its own modality data
                          ->  checkpoints/US/best.pth
                          ->  checkpoints/XRay/best.pth  etc.
  5. inference.py      -- filename routing: "2DBox_US_foo.npz" -> US specialist
                          -> checkpoints/US/best.pth (falls back to global.pth)

Why do DAFT specialists start from global.pth, NOT distilled.pth?
  distilled.pth is only the encoder -- it has not seen any medical image
  segmentation data yet. global.pth has already been trained on thousands of
  multi-modal medical images. Specialists fine-tune FROM this strong starting
  point, adapting the already-capable model to their specific modality.
  Starting from distilled.pth would throw away all that learning.

How does modality routing work?
  Routing is 100% filename-based. The function filename_to_modelname()
  inspects the filename prefix (e.g. "2DBox_US", "3DBox_CT") and returns
  the modality name ("US", "3D", "XRay", ...).  The same function is used
  in BOTH directions:
    - Training (Step 3): group files by modality -> train one specialist each
    - Inference (Step 5): route each test file -> load correct specialist

What this script does, step by step
------------------------------------
  Step 1  Prepare data     Merge imgs+gts, group by modality using
                           filename_to_modelname() -- same routing as inference
  Step 2  Visualize        Save overview plots to data/viz/
  Step 3  DAFT training    Train one specialist per modality (1 epoch each)
                           Starts from global.pth (correct -- see above).
                           Saved to checkpoints/demo_{modality}/ to avoid
                           overwriting real inference checkpoints.
  Step 4  Inference        Run with the DOWNLOADED global model (global.pth).
                           Demo specialists from Step 3 are intentionally NOT
                           used -- 1 training file per modality is far too few
                           to improve on a model trained on thousands of samples.
  Step 5  Evaluate         Compute DSC vs ground truth, compare with paper.

Requirements
------------
  conda activate daft       (see README.md Section 8 for setup)
  Run from inside daft-clean/:
    python demo.py
    python demo.py --skip_viz          # faster, skip plots
    python demo.py --skip_training     # skip Step 3, inference + eval only
"""

import os
import sys
import glob
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import monai
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import our own modules
from model     import MedSAM
from dataset   import MedSegDataset
from train     import cal_iou, run_epoch, set_seeds
from inference import (filename_to_modelname, preprocess,
                       scale_box, bbox_from_mask, infer_slice,
                       process_2d, process_3d)
from evaluate  import file_dsc

def get_best_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
# ===========================================================================
# CONFIG
# ===========================================================================

DEVICE = get_best_device()
GLOBAL_PTH = "checkpoints/global.pth"       # the downloaded model
IMGS_DIR   = "data/test_imgs"
GTS_DIR    = "data/test_gts"
SEGS_DIR   = "data/test_demo_segs"          # paper reference predictions
VIZ_DIR    = "data/viz"
TRAIN_DIR  = "data/demo_combined"           # merged imgs+gts for training
PRED_DIR   = "data/demo_preds"              # our inference output
CSV_DIR    = "data/datasplit"

SEED       = 2024
DAFT_EPOCHS     = 1     # 1 epoch per modality (pipeline demo only)
DAFT_BATCH_SIZE = 1     # 1 sample at a time (demo data is tiny)
DAFT_LR         = 5e-5


# ===========================================================================
# STEP 0 -- banner
# ===========================================================================

def banner(text):
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


# ===========================================================================
# STEP 1 -- prepare demo data
# ===========================================================================

def step1_prepare_data():
    banner("STEP 1: Prepare demo data")

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(CSV_DIR,   exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(IMGS_DIR, "*.npz")))
    print(f"  Found {len(img_files)} demo files in {IMGS_DIR}/")

    combined_paths = []
    for img_path in img_files:
        name    = os.path.basename(img_path)
        gt_path = os.path.join(GTS_DIR, name)

        img_data = np.load(img_path, allow_pickle=True)
        gt_data  = np.load(gt_path,  allow_pickle=True)

        out_path = os.path.join(TRAIN_DIR, name)
        save_kwargs = {"imgs": img_data["imgs"], "gts": gt_data["gts"]}
        if "spacing" in gt_data.files:
            save_kwargs["spacing"] = gt_data["spacing"]
        np.savez_compressed(out_path, **save_kwargs)
        combined_paths.append(os.path.abspath(out_path))

    # --- DAFT ROUTING ---
    # filename_to_modelname() maps each filename to a modality string.
    # This is the exact same function used by inference.py to route test files
    # to the correct specialist checkpoint. Here we use it to group training
    # files so each specialist is trained on its own modality only.
    #
    # Example routing for our 10 demo files:
    #   2DBox_CXR_demo.npz        -> "XRay"
    #   2DBox_Dermoscopy_demo.npz -> "Dermoscopy"
    #   2DBox_Endoscopy_demo.npz  -> "Endoscopy"
    #   2DBox_Fundus_demo.npz     -> "Fundus"
    #   2DBox_Microscopy_demo.npz -> "Microscopy"
    #   2DBox_OCT_demo.npz        -> "OCT"
    #   2DBox_US_demo.npz         -> "US"
    #   3DBox_CT_demo.npz         -> "3D"
    #   3DBox_MR_demo.npz         -> "3D"
    #   3DBox_PET_demo.npz        -> "3D"  (same specialist handles all 3D)
    modality_files = {}
    for path in combined_paths:
        name     = os.path.basename(path)
        modality = filename_to_modelname(name)
        modality_files.setdefault(modality, []).append(path)

    print("\n  DAFT routing: filename -> modality -> specialist")
    print(f"  {'Modality':<15} {'Files'}")
    print(f"  {'-'*15} {'-'*40}")
    for mod, files in sorted(modality_files.items()):
        fnames = [os.path.basename(f) for f in files]
        ckpt   = f"checkpoints/{mod}/best.pth"
        print(f"  {mod:<15} {fnames}")
        print(f"  {'':15} -> specialist: {ckpt}")

    # Write one CSV per modality (train = val = same file for demo purposes)
    mod_csv_dir = os.path.join(CSV_DIR, "demo_modalities")
    os.makedirs(mod_csv_dir, exist_ok=True)

    mod_csvs = {}
    for mod, files in modality_files.items():
        csv_path = os.path.join(mod_csv_dir, f"{mod}.csv")
        pd.DataFrame({"file": files}).to_csv(csv_path, index=False)
        mod_csvs[mod] = csv_path

    print(f"\n  Per-modality CSVs written to {mod_csv_dir}/")
    return mod_csvs, modality_files


# ===========================================================================
# STEP 2 -- visualize
# ===========================================================================

def step2_visualize():
    banner("STEP 2: Visualize demo data")

    try:
        from visualize import main as viz_main
        # Patch sys.argv so visualize.main() picks up our paths
        old_argv = sys.argv
        sys.argv  = ["visualize.py",
                     "--imgs_dir", IMGS_DIR,
                     "--gts_dir",  GTS_DIR,
                     "--segs_dir", SEGS_DIR,
                     "--out_dir",  VIZ_DIR]
        viz_main()
        sys.argv = old_argv
    except Exception as e:
        print(f"  Visualisation skipped: {e}")


# ===========================================================================
# STEP 3 -- DAFT training (one specialist per modality)
# ===========================================================================

def step3_daft_training(mod_csvs):
    """
    Data-Aware Fine-Tuning: train one specialist model per modality.

    Each specialist starts from global.pth (the GLOBALLY fine-tuned model)
    and is then fine-tuned on ONLY its own modality's data. This is the
    core idea of DAFT: route each data type to a specialist fine-tuned on
    that type.

    Why global.pth and NOT distilled.pth?
    --------------------------------------
    distilled.pth only has a distilled encoder -- it has never seen medical
    segmentation data. global.pth has been globally fine-tuned on thousands
    of multi-modal medical images and is already a strong model. DAFT
    specialists adapt this strong foundation to specific modalities.
    Starting from distilled.pth would throw away all that learning.

    DAFT order:  distilled.pth  ->  global.pth  ->  per-modality specialists
                 (encoder only)    (all modalities   (each modality only)
                                    combined)

    Checkpoint save location:
    -------------------------
    Demo specialists are saved to checkpoints/demo_{modality}/ (not
    checkpoints/{modality}/) so they do NOT interfere with real inference.
    In a full DAFT run you would save to checkpoints/{modality}/best.pth
    so inference.py automatically picks them up via filename routing.

    With only 1 file per modality this is a PIPELINE DEMO.
    Real DAFT training uses hundreds of files per modality.
    """
    banner("STEP 3: DAFT training -- one specialist per modality")
    print("  Starting model: checkpoints/global.pth  (downloaded globally fine-tuned)")
    print("  Each specialist: global.pth + fine-tune on ONE modality's data only")
    print(f"  Epochs per modality: {DAFT_EPOCHS}  (demo -- just verifying pipeline)")
    print(f"  Device: {DEVICE}")

    set_seeds(SEED)

    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss  = nn.BCEWithLogitsLoss(reduction="mean")
    iou_loss = nn.MSELoss(reduction="mean")

    trained_ckpts = {}   # modality -> checkpoint path

    for modality, csv_path in sorted(mod_csvs.items()):
        print(f"\n  --- Modality: {modality} ---")
        print(f"      CSV: {csv_path}")

        # Each specialist starts fresh from the global model
        model = MedSAM(GLOBAL_PTH).to(DEVICE)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=DAFT_LR, betas=(0.9, 0.999), weight_decay=0.01,
        )

        # Use the same file for train and val (1 file per modality in demo)
        ds    = MedSegDataset(csv_path, bbox_shift=5, augment=True)
        loader = DataLoader(ds, batch_size=DAFT_BATCH_SIZE,
                            shuffle=True, num_workers=0)

        for epoch in range(1, DAFT_EPOCHS + 1):
            tr_loss = run_epoch(
                model, loader, optimizer,
                seg_loss, ce_loss, iou_loss,
                1.0, 1.0, 1.0,
                DEVICE, train=True,
            )
            print(f"      Epoch {epoch}/{DAFT_EPOCHS}  loss={tr_loss:.4f}")

        # Save specialist checkpoint.
        # Saved to checkpoints/demo_{modality}/ (with "demo_" prefix) so that
        # real inference.py routing (which looks for checkpoints/{modality}/)
        # is not affected by this demo training run.
        # In a production DAFT run, save to checkpoints/{modality}/best.pth
        # so that inference.py automatically routes to the specialist.
        ckpt_dir  = os.path.join("checkpoints", f"demo_{modality}")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "best.pth")
        torch.save({"model": model.state_dict(), "epoch": DAFT_EPOCHS,
                    "best_val_loss": tr_loss}, ckpt_path)
        trained_ckpts[modality] = ckpt_path
        print(f"      Saved specialist -> {ckpt_path}")

    print("\n  DAFT training complete.")
    print("  NOTE: These specialists are not used for final inference.")
    print("        The downloaded global model is used instead (see Step 4).")
    return trained_ckpts


# ===========================================================================
# STEP 4 -- inference with the downloaded global model
# ===========================================================================

def step4_inference():
    banner("STEP 4: Inference using the DOWNLOADED model (checkpoints/global.pth)")

    # We intentionally use global.pth here, NOT the demo specialists from Step 3.
    # Reason: with only 1 training file per modality, the demo specialists have
    # not learned anything meaningful. global.pth was trained on thousands of
    # samples and is the best model we have.
    #
    # In a full DAFT run you would call inference.py directly, which auto-routes
    # each file to its specialist via filename_to_modelname():
    #   "2DBox_US_demo.npz"  ->  "US"  ->  checkpoints/US/best.pth
    #   "3DBox_CT_demo.npz"  ->  "3D"  ->  checkpoints/3D/best.pth
    #   (falls back to global.pth if specialist does not exist)

    os.makedirs(PRED_DIR, exist_ok=True)
    npz_files = sorted(glob.glob(os.path.join(IMGS_DIR, "*.npz")))
    print(f"  Running on {len(npz_files)} files  ->  {PRED_DIR}/")
    print(f"  Model: {GLOBAL_PTH}  (device: {DEVICE})")
    print(f"  (Demo specialists from Step 3 intentionally skipped -- see docstring)")

    # Load the global model once
    model = MedSAM(GLOBAL_PTH).to(DEVICE).eval()

    for npz_path in npz_files:
        name = os.path.basename(npz_path)
        if name.startswith("3D"):
            process_3d(model, npz_path, PRED_DIR, DEVICE)
        else:
            process_2d(model, npz_path, PRED_DIR, DEVICE)

    print("  Inference done.")


# ===========================================================================
# STEP 5 -- evaluate
# ===========================================================================

def step5_evaluate():
    banner("STEP 5: Evaluation  (DSC vs ground truth)")

    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.npz")))
    gt_dir     = GTS_DIR
    ref_dir    = SEGS_DIR

    rows_ours = []
    rows_ref  = []

    for pred_path in pred_files:
        name    = os.path.basename(pred_path)
        gt_path = os.path.join(gt_dir, name)
        if not os.path.isfile(gt_path):
            continue

        dsc_ours = file_dsc(pred_path, gt_path)
        rows_ours.append({"file": name, "dsc_global_model": round(dsc_ours, 4)})

        ref_path = os.path.join(ref_dir, name)
        if os.path.isfile(ref_path):
            dsc_ref = file_dsc(ref_path, gt_path)
            rows_ref.append({"file": name, "dsc_paper": round(dsc_ref, 4)})

    df_ours = pd.DataFrame(rows_ours).sort_values("file").reset_index(drop=True)
    df_ref  = pd.DataFrame(rows_ref ).sort_values("file").reset_index(drop=True)

    # Merge for display
    df = df_ours.copy()
    if not df_ref.empty:
        df = df.merge(df_ref, on="file", how="left")

    print("\n  Results:")
    print(df.to_string(index=False))

    mean_ours = df["dsc_global_model"].mean()
    print(f"\n  Mean DSC  (our global model) : {mean_ours:.4f}")
    if "dsc_paper" in df.columns:
        mean_ref = df["dsc_paper"].mean()
        print(f"  Mean DSC  (paper reference)  : {mean_ref:.4f}")

    # Save
    df.to_csv("results_demo.csv", index=False)
    print("\n  Saved -> results_demo.csv")

    # Bar chart
    _plot_results(df)


def _plot_results(df):
    names = [n.replace("Box_", "\n").replace("_demo.npz", "")
             for n in df["file"]]
    x = range(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    bars1 = ax.bar([i - width/2 for i in x], df["dsc_global_model"],
                   width, label="Global model (downloaded)", color="#2196F3")
    if "dsc_paper" in df.columns:
        bars2 = ax.bar([i + width/2 for i in x], df["dsc_paper"],
                       width, label="Paper reference", color="#4CAF50")

    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Dice Score")
    ax.set_title("DAFT Demo -- Segmentation Results\n"
                 "(Inference with downloaded global model vs paper reference)")
    ax.legend()
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="0.8 threshold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("results_demo.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Plot saved -> results_demo.png")


# ===========================================================================
# MAIN
# ===========================================================================

def get_args():
    p = argparse.ArgumentParser(description="DAFT end-to-end demo.")
    p.add_argument("--skip_viz",      action="store_true",
                   help="Skip visualisation step (faster)")
    p.add_argument("--skip_training", action="store_true",
                   help="Skip DAFT training step (use inference + eval only)")
    return p.parse_args()


def main():
    args = get_args()

    print("\n" + "#" * 60)
    print("  DAFT -- Data-Aware Fine-Tuning  (Demo)")
    print("  EfficientViT-SAM  /  Ma et al., CVPR 2024")
    print("#" * 60)
    print(f"  Device : {DEVICE}")
    print(f"  Model  : {GLOBAL_PTH}")

    if not os.path.isfile(GLOBAL_PTH):
        print(f"\n  ERROR: {GLOBAL_PTH} not found.")
        print("  Make sure checkpoints/global.pth exists.")
        sys.exit(1)

    # Step 1 -- always needed
    mod_csvs, _ = step1_prepare_data()

    # Step 2 -- visualise
    if not args.skip_viz:
        step2_visualize()
    else:
        print("\n  [Step 2 skipped -- --skip_viz]")

    # Step 3 -- DAFT training (demonstrates the concept, model not used after)
    if not args.skip_training:
        step3_daft_training(mod_csvs)
    else:
        print("\n  [Step 3 skipped -- --skip_training]")

    # Step 4 -- inference with DOWNLOADED model
    step4_inference()

    # Step 5 -- evaluate
    step5_evaluate()

    print("\n" + "#" * 60)
    print("  Demo complete.")
    print("  Key outputs:")
    print(f"    data/viz/            -- visualisations")
    print(f"    data/demo_preds/     -- segmentation predictions")
    print(f"    results_demo.csv     -- DSC scores")
    print(f"    results_demo.png     -- bar chart")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
