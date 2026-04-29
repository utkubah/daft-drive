"""
model.py
========
The segmentation model: EfficientViT-SAM (l0 variant).

Why EfficientViT-SAM instead of original SAM?
----------------------------------------------
Original SAM uses a large ViT-H image encoder (632M params) — way too slow on
a laptop CPU. EfficientViT-l0 achieves similar segmentation quality in ~1/10
of the time, while producing feature maps in the same (B, 256, 64, 64) format
that the SAM mask decoder expects.

Three components
----------------
  image_encoder   EfficientViT-l0 backbone
                  input:  (B, 3, 256, 256) normalized image
                  output: (B, 256, 64, 64) image features

  prompt_encoder  Converts a bounding box into prompt embeddings.
                  FROZEN during all training — weights come from LiteMedSAM and
                  don't need to change. Freezing also speeds up training.

  mask_decoder    Cross-attends image features with prompt embeddings → mask.
                  output: (B, 1, 256, 256) logit map  +  (B, 1) IoU score

Usage
-----
  from model import MedSAM

  model = MedSAM()                               # no weights
  model = MedSAM("checkpoints/global.pth")       # load weights

  logits, iou = model(images, bboxes)
  masks = torch.sigmoid(logits) > 0.5

Required packages:  efficientvit (pip install efficientvit)
"""

import torch
import torch.nn as nn
from efficientvit.sam_model_zoo import create_efficientvit_sam_model


class MedSAM(nn.Module):
    """
    EfficientViT-SAM l0 wrapper for medical image segmentation.

    Checkpoint formats accepted:
      - plain state dict (e.g. general_finetuned.pth from the paper's authors)
      - dict with a "model" key (checkpoints saved by train.py in this project)
    """

    def __init__(self, checkpoint_path=None):
        super().__init__()

        # Build EfficientViT-SAM l0 (architecture only, no pretrained weights).
        # pretrained=False so it does not try to download weights from the internet.
        # Note: newer efficientvit renamed create_sam_model -> create_efficientvit_sam_model
        #       and the model key changed from "l0" to "efficientvit-sam-l0".
        base = create_efficientvit_sam_model("efficientvit-sam-l0", pretrained=False)

        # The original model expects 1024×1024 input; we work at 256×256
        base.prompt_encoder.input_image_size = (256, 256)

        self.image_encoder  = base.image_encoder
        self.prompt_encoder = base.prompt_encoder
        self.mask_decoder   = base.mask_decoder

        # Load weights if supplied
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            # Handle both plain state dicts and {"model": state_dict, ...} format
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            base.load_state_dict(state, strict=True)
            print(f"  Weights loaded: {checkpoint_path}")

        # Freeze the prompt encoder — no gradient updates
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, images, bboxes):
        """
        Args:
            images  (B, 3, 256, 256)  normalized to [0, 1]
            bboxes  (B, 1, 4)         [x_min, y_min, x_max, y_max] in 256-space

        Returns:
            logits  (B, 1, 256, 256)  raw segmentation logits
            iou     (B, 1)            predicted IoU quality score

        Note: the mask_decoder's repeat_interleave is designed for (1 image,
        N prompts) at a time, so we loop per image.  The image encoder still
        runs on the full batch for efficiency.
        """
        # Step 1: encode all images at once (efficient)
        features = self.image_encoder(images)    # (B, 256, 64, 64)

        # Steps 2+3: prompt-encode + decode one image at a time
        all_logits, all_iou = [], []
        for feat, bbox in zip(features, bboxes):
            # feat: (256, 64, 64)   bbox: (1, 4)
            with torch.no_grad():
                sparse_emb, dense_emb = self.prompt_encoder(
                    points=None,
                    boxes=bbox.unsqueeze(0),   # (1, 1, 4)
                    masks=None,
                )
            logit, pred_iou = self.mask_decoder(
                image_embeddings=feat.unsqueeze(0),   # (1, 256, 64, 64)
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            all_logits.append(logit)
            all_iou.append(pred_iou)

        return torch.cat(all_logits, dim=0), torch.cat(all_iou, dim=0)
