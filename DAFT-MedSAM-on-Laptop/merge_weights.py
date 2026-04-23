"""
merge_weights.py
================
Combine the distilled EfficientViT-l0 image encoder with the prompt encoder
and mask decoder from the original LiteMedSAM (TinyViT-based) checkpoint.

Background
----------
After knowledge distillation (distill.py), we have a trained EfficientViT
image encoder, but we still need the SAM prompt encoder and mask decoder.
These are taken from the TinyViT-based LiteMedSAM because:
  * They are standard SAM components (same architecture in both models).
  * The TinyViT model was trained end-to-end and its decoder is well-tuned.

The merged model can be loaded with:  MedSAM("checkpoints/merged.pth")

lite_medsam.pth layout
-----------------------
  image_encoder.*   <- TinyViT weights  (we SKIP these)
  prompt_encoder.*  <- SAM weights      (we KEEP these)
  mask_decoder.*    <- SAM weights      (we KEEP these)

Usage
-----
  # After running distill.py:
  python merge_weights.py \\
    --lite_medsam   ../lite_medsam.pth                   \\
    --encoder       checkpoints/distilled_encoder.pth    \\
    --output        checkpoints/merged.pth

  # Then fine-tune the merged model:
  python train.py --weights checkpoints/merged.pth --name global ...
"""

import os
import argparse
import torch
from efficientvit.sam_model_zoo import create_efficientvit_sam_model


def get_args():
    p = argparse.ArgumentParser(
        description="Merge distilled EfficientViT encoder + LiteMedSAM decoder."
    )
    p.add_argument("--lite_medsam", default="lite_medsam.pth",
                   help="Original LiteMedSAM checkpoint (TinyViT-based)")
    p.add_argument("--encoder",     default="checkpoints/distilled_encoder.pth",
                   help="Distilled EfficientViT image encoder (from distill.py)")
    p.add_argument("--output",      default="checkpoints/merged.pth",
                   help="Where to save the merged EfficientViT-SAM model")
    return p.parse_args()


def main():
    args = get_args()

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    # --- Step 1: Extract prompt encoder + mask decoder from LiteMedSAM ---
    print(f"  Loading LiteMedSAM from: {args.lite_medsam}")
    lite_state = torch.load(args.lite_medsam, map_location="cpu", weights_only=False)

    # Filter by prefix and strip the prefix to get bare sub-module state dicts
    pe_state = {k[len("prompt_encoder."):]: v
                for k, v in lite_state.items()
                if k.startswith("prompt_encoder.")}
    md_state = {k[len("mask_decoder."):]: v
                for k, v in lite_state.items()
                if k.startswith("mask_decoder.")}

    if not pe_state:
        raise ValueError(
            "No 'prompt_encoder.*' keys found in the LiteMedSAM checkpoint.\n"
            f"Available top-level prefixes: "
            f"{set(k.split('.')[0] for k in lite_state.keys())}"
        )

    print(f"    prompt_encoder keys : {len(pe_state)}")
    print(f"    mask_decoder   keys : {len(md_state)}")

    # --- Step 2: Load distilled EfficientViT image encoder ---
    print(f"  Loading distilled encoder from: {args.encoder}")
    ie_state = torch.load(args.encoder, map_location="cpu", weights_only=False)
    print(f"    image_encoder  keys : {len(ie_state)}")

    # --- Step 3: Assemble full EfficientViT-SAM l0 ---
    print("  Building EfficientViT-SAM l0 ...")
    model = create_efficientvit_sam_model("efficientvit-sam-l0", pretrained=False)
    model.prompt_encoder.input_image_size = (256, 256)

    model.image_encoder.load_state_dict(ie_state, strict=True)
    model.prompt_encoder.load_state_dict(pe_state, strict=True)
    model.mask_decoder.load_state_dict(md_state,   strict=True)

    # --- Step 4: Save ---
    torch.save(model.state_dict(), args.output)
    print(f"\n  Merged model saved -> {args.output}")
    print(f"  Load with:  MedSAM('{args.output}')")
    print()
    print("  Next step -- global fine-tuning:")
    print(f"    python train.py --weights {args.output} "
          f"--train_csv data/datasplit/train.csv "
          f"--val_csv data/datasplit/val.csv --name global")


if __name__ == "__main__":
    main()
