#!/bin/bash
# BDD100K large baseline: fine-tune YOLOv8m on all conditions.
# Produces the "Global YOLOv8m" reference point for the comparison table.
# Runs independently of the DAFT pipeline — no distillation step needed.
#
# Usage: sbatch hpc/pretrain_bdd100k_large.sh
# Checkpoint: checkpoints/large/weights/best.pt
# Pass to eval:  python eval_full.py --large_ckpt checkpoints/large/weights/best.pt

#SBATCH --job-name=bdd100k-large
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --output=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/large_%j.out
#SBATCH --error=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/large_%j.err

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

echo "===== Fine-tuning YOLOv8m (large baseline) ====="
python train.py \
    --data     data/bdd100k/yolo/dataset.yaml \
    --weights  yolov8m.pt \
    --name     large \
    --epochs   50 \
    --batch    8 \
    --imgsz    640 \
    --lr       5e-5 \
    --patience 20 \
    --cos_lr \
    --workers  2 \
    --device   cuda

echo ""
echo "Large global model saved to: checkpoints/large/weights/best.pt"
echo "Pass to evaluation with:"
echo "  python eval.py --large_ckpt checkpoints/large/weights/best.pt"