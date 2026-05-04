#!/bin/bash
# BDD100K Stage 1: distillation + global fine-tune.
# Requires data to be prepared first via sbatch hpc/prepare_data.sh
#
# Usage: sbatch hpc/pretrain_bdd100k.sh

#SBATCH --job-name=bdd100k-pretrain
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --output=/home/3223837/DAFT-BDD100K/logs/pretrain_%j.out
#SBATCH --error=/home/3223837/DAFT-BDD100K/logs/pretrain_%j.err

set -e
mkdir -p /home/3223837/DAFT-BDD100K/logs

source /software/miniconda3/etc/profile.d/conda.sh
conda activate daft
module load cuda/12.4 || true

cd $HOME/DAFT-BDD100K

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
    --epochs  15 \
    --batch   16 \
    --workers 2 \
    --device  cuda

echo "===== Step 2/2: Global fine-tune ====="
python train.py \
    --data     data/bdd100k/yolo/dataset.yaml \
    --weights  checkpoints/distilled/distilled.pt \
    --name     global \
    --epochs   30 \
    --batch    8 \
    --imgsz    640 \
    --lr       5e-5 \
    --patience 15 \
    --device   cuda

echo ""
echo "Global model saved to: checkpoints/global/weights/best.pt"
echo "Next: sbatch hpc/train_daft_bdd100k.sh"
