#!/bin/bash
# Run ONCE on the login node -- no GPU needed, just pip installs.
# Creates conda env "daft" and checks that global.pth is in place.
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

# Verify
python - <<'PY'
import torch
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
print("  PyTorch:", torch.__version__)
print("  CUDA available:", torch.cuda.is_available())
m = create_efficientvit_sam_model("efficientvit-sam-l0", pretrained=False)
print("  Model OK -", sum(p.numel() for p in m.parameters())//1_000_000, "M params")
PY

# Check for global.pth
if [ ! -f checkpoints/global.pth ]; then
    echo ""
    echo "  WARNING: checkpoints/global.pth not found."
    echo "  Copy it from your local machine:"
    echo "    scp ./checkpoints/global.pth 3223837@slogin.hpc.unibocconi.it:~/daft-drive/DAFT-MedSAM-on-Laptop/checkpoints/"
else
    echo "  checkpoints/global.pth found."
fi

echo ""
echo "Setup complete. Next steps:"
echo "  sbatch hpc/download_data.sh"
