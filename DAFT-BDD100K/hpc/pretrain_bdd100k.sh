#!/bin/bash
# Trains the global YOLOv8s model used as the starting point for all specialists.
# Step 1: distill YOLOv8m → YOLOv8s so the small model inherits good feature representations.
# Step 2: fine-tune the distilled student on all of BDD100K.
#
# Run prepare_data.sh first. Output: checkpoints/global/weights/best.pt
#
# Usage: sbatch hpc/pretrain_bdd100k.sh

#SBATCH --job-name=bdd100k-pretrain
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --output=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/pretrain_%j.out
#SBATCH --error=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/pretrain_%j.err

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

echo "===== Step 1/2: Distillation (YOLOv8m -> YOLOv8s) ====="
python distill.py \
    --img_dir data/bdd100k/yolo/images/train \
    --out_dir checkpoints/distilled \
    --teacher yolov8m.pt \
    --student yolov8s.pt \
    --epochs  20 \
    --batch   16 \
    --workers 2 \
    --device  cuda

echo "===== Step 2/2: Global fine-tune ====="
python train.py \
    --data     data/bdd100k/yolo/dataset.yaml \
    --weights  checkpoints/distilled/distilled.pt \
    --name     global \
    --epochs   50 \
    --batch    16 \
    --imgsz    640 \
    --lr       5e-5 \
    --patience 20 \
    --cos_lr \
    --workers  2 \
    --device   cuda

echo ""
echo "Global model saved to: checkpoints/global/weights/best.pt"
echo "Next: sbatch hpc/train_daft_bdd100k.sh"
