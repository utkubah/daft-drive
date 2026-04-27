#!/bin/bash
# Run ONCE on the login node -- no GPU needed, just pip installs.
# Creates conda env "daft" and downloads the LiteMedSAM teacher checkpoint
# (used by distill.py).
#
# Usage:  bash hpc/setup_env.sh

set -e

source /software/miniconda3/etc/profile.d/conda.sh

if conda env list | awk '{print $1}' | grep -qx "daft"; then
    echo "  conda env 'daft' already exists"
else
    echo "  Creating conda env 'daft' ..."
    conda create -n daft python=3.10 -y
fi
conda activate daft

module load cuda/12.4 || true

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install monai opencv-python pandas numpy tqdm matplotlib gdown
pip install git+https://github.com/mit-han-lab/efficientvit.git
pip install git+https://github.com/facebookresearch/segment-anything.git

# Verify env
python - <<'PY'
import torch
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
print("  PyTorch:", torch.__version__)
print("  CUDA available:", torch.cuda.is_available())
m = create_efficientvit_sam_model("efficientvit-sam-l0", pretrained=False)
print("  Model OK -", sum(p.numel() for p in m.parameters())//1_000_000, "M params")
PY

# Download LiteMedSAM teacher (single small file -- gdown is reliable for this)
if [ ! -f lite_medsam.pth ]; then
    echo "  Downloading LiteMedSAM teacher checkpoint ..."
    gdown 18Zed-TUTsmr2zc5CHUWd5Tu13nb6vq6z -O lite_medsam.pth
else
    echo "  lite_medsam.pth already present."
fi

echo ""
echo "Setup complete. Next steps (after rsync'ing ~/medsam-data/ from local):"
echo "  sbatch hpc/pretrain.sh        # distill + merge + general fine-tune"
echo "  sbatch hpc/train_daft.sh      # 9 DAFT specialists in parallel"
