#!/bin/bash
# BDD100K Stage 1: data prep + distillation + global fine-tune
# Run this first, then sbatch hpc/train_daft_bdd100k.sh
#
# Usage: sbatch hpc/pretrain_bdd100k.sh

#SBATCH --job-name=bdd100k-pretrain
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:00:00
#SBATCH --output=/home/3223837/daft-drive/DAFT-BDD100K/logs/pretrain_%j.out
#SBATCH --error=/home/3223837/daft-drive/DAFT-BDD100K/logs/pretrain_%j.err

set -e
mkdir -p /home/3223837/daft-drive/DAFT-BDD100K/logs

source /software/miniconda3/etc/profile.d/conda.sh
conda activate daft
module load cuda/12.4 || true

cd $HOME/daft-drive/DAFT-BDD100K

echo "===== Step 1/3: Prepare data ====="
python prepare_data.py --hf_token "$HF_TOKEN"

echo "===== Step 2/3: Distillation (YOLOv8m -> YOLOv8n) ====="
python distill.py \
    --img_dir  data/bdd100k/yolo/images/train \
    --out_dir  checkpoints/distilled \
    --teacher  yolov8m.pt \
    --student  yolov8n.pt \
    --epochs   20 \
    --batch    16 \
    --workers  4 \
    --device   0

echo "===== Step 3/3: Global fine-tune ====="
python train.py \
    --data     data/bdd100k/yolo/dataset.yaml \
    --weights  checkpoints/distilled/distilled.pt \
    --name     global \
    --epochs   30 \
    --batch    8 \
    --imgsz    640 \
    --lr       5e-5 \
    --patience 10 \
    --device   0

echo ""
echo "Global model saved to: checkpoints/global/weights/best.pt"
echo "Next: sbatch hpc/train_daft_bdd100k.sh"
