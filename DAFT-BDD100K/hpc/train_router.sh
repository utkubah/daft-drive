#!/bin/bash
# Trains the MobileNetV3-small image router that classifies each frame
# into one of the five driving conditions at inference time.
# Needs the global checkpoint and the val manifests from prepare_data.sh.
#
# Usage: sbatch hpc/train_router.sh

#SBATCH --job-name=bdd100k-router
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/router_%j.out
#SBATCH --error=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/router_%j.err

set -e
mkdir -p /mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs

source /software/miniconda3/etc/profile.d/conda.sh
conda activate daft
module load cuda/12.4 || true

cd /mnt/beegfsstudents/home/3223837/DAFT-BDD100K

test -f checkpoints/global/weights/best.pt || {
    echo "ERROR: checkpoints/global/weights/best.pt not found."
    echo "Run sbatch hpc/pretrain_bdd100k.sh first."
    exit 1
}

test -f data/bdd100k/manifests/train.csv || {
    echo "ERROR: data/bdd100k/manifests/train.csv not found."
    echo "Run sbatch hpc/prepare_data.sh first."
    exit 1
}

echo "===== Training RouterHead classifier ====="
python train_router.py \
    --out_dir  checkpoints/router \
    --epochs   20 \
    --batch    64 \
    --workers  2 \
    --device   cuda

echo ""
echo "RouterHead saved to: checkpoints/router/router_head.pt"
echo "Run inference with:"
echo "  python inference.py --source <imgs> \\"
echo "    --router_ckpt checkpoints/router/router_head.pt \\"
echo "    --top_k auto"
