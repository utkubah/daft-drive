#!/bin/bash
# BDD100K Stage 2: train all 3 DAFT specialists sequentially in one job.
# Avoids QOSMaxSubmitJobPerUserLimit (no job array).
#
# Usage: sbatch hpc/train_daft_bdd100k.sh

#SBATCH --job-name=bdd100k-daft
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --output=/home/3223837/daft-drive/DAFT-BDD100K/logs/daft_%j.out
#SBATCH --error=/home/3223837/daft-drive/DAFT-BDD100K/logs/daft_%j.err

set -e
mkdir -p /home/3223837/daft-drive/DAFT-BDD100K/logs

source /software/miniconda3/etc/profile.d/conda.sh
conda activate daft
module load cuda/12.4 || true

cd $HOME/daft-drive/DAFT-BDD100K

test -f checkpoints/global/weights/best.pt || {
    echo "ERROR: checkpoints/global/weights/best.pt not found."
    echo "Run sbatch hpc/pretrain_bdd100k.sh first."
    exit 1
}

CONDITIONS=(day night rain)

for COND in "${CONDITIONS[@]}"; do
    echo "===== Training specialist: $COND ====="

    test -f "data/bdd100k/yolo/${COND}.yaml" || {
        echo "ERROR: data/bdd100k/yolo/${COND}.yaml not found."
        echo "Run prepare_data.py first."
        exit 1
    }

    python train.py \
        --data     "data/bdd100k/yolo/${COND}.yaml" \
        --weights  checkpoints/global/weights/best.pt \
        --name     "${COND}" \
        --epochs   30 \
        --batch    8 \
        --imgsz    640 \
        --lr       5e-5 \
        --patience 10 \
        --device   0

    echo "===== Done: $COND ====="
done

echo ""
echo "All DAFT specialists trained."
echo "Checkpoints:"
echo "  checkpoints/day/weights/best.pt"
echo "  checkpoints/night/weights/best.pt"
echo "  checkpoints/rain/weights/best.pt"
