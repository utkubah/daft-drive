#!/bin/bash
# BDD100K xlarge baseline: fine-tune YOLOv8x on all conditions.
# YOLOv8x is the largest/slowest YOLOv8 — optional reference point.
# Runs independently of the DAFT pipeline — no distillation step needed.
#
# Usage: sbatch hpc/pretrain_bdd100k_xlarge.sh
# Checkpoint: checkpoints/xlarge/weights/best.pt

#SBATCH --job-name=bdd100k-xlarge
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --output=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/xlarge_%j.out
#SBATCH --error=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/xlarge_%j.err

set -e
mkdir -p /mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs

source /software/miniconda3/etc/profile.d/conda.sh
conda activate daft
module load cuda/12.4 || true

cd /mnt/beegfsstudents/home/3223837/DAFT-BDD100K

test -f data/bdd100k/yolo/dataset.yaml || {
    echo "ERROR: data/bdd100k/yolo/dataset.yaml not found."
    echo "Run sbatch hpc/prepare_data.sh first."
    exit 1
}

echo "===== Fine-tuning YOLOv8x (xlarge baseline) ====="
python train.py \
    --data     data/bdd100k/yolo/dataset.yaml \
    --weights  yolov8x.pt \
    --name     xlarge \
    --epochs   50 \
    --batch    4 \
    --imgsz    640 \
    --lr       5e-5 \
    --patience 20 \
    --cos_lr \
    --workers  2 \
    --device   cuda

echo ""
echo "XLarge global model saved to: checkpoints/xlarge/weights/best.pt"
