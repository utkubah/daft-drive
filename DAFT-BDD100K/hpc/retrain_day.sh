#!/bin/bash
# Retrain the three day/residential specialists that were accidentally trained
# with the YOLOv8n global checkpoint (6 MB) instead of the YOLOv8s global (22 MB).
#
# city_night and highway_night are already correct (22 MB, retrained in job 489711).
# This job retrains only: city_day, highway_day, residential.
#
# Usage:
#   sbatch hpc/retrain_day.sh
#
# Or chain after a dependency:
#   sbatch --dependency=afterok:<JOB_ID> hpc/retrain_day.sh

#SBATCH --job-name=bdd100k-retrain-day
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/3223837/DAFT-BDD100K/logs/retrain_day_%j.out
#SBATCH --error=/home/3223837/DAFT-BDD100K/logs/retrain_day_%j.err

set -e
mkdir -p /home/3223837/DAFT-BDD100K/logs

source /software/miniconda3/etc/profile.d/conda.sh
conda activate daft
module load cuda/12.4 || true

cd $HOME/DAFT-BDD100K

# Verify global checkpoint is YOLOv8s (should be ~22 MB)
GLOBAL=checkpoints/global/weights/best.pt
test -f "$GLOBAL" || {
    echo "ERROR: $GLOBAL not found."
    exit 1
}
SIZE_MB=$(du -m "$GLOBAL" | cut -f1)
echo "Global checkpoint: $GLOBAL  (${SIZE_MB} MB)"
if [ "$SIZE_MB" -lt 15 ]; then
    echo "ERROR: global checkpoint is ${SIZE_MB} MB — looks like YOLOv8n, not YOLOv8s."
    echo "Re-run pretrain_bdd100k.sh first."
    exit 1
fi
echo "Size check OK (YOLOv8s confirmed)."

echo ""
echo "===== Retraining city_day (from YOLOv8s global) ====="
python train.py \
    --data     data/bdd100k/yolo/city_day.yaml \
    --weights  checkpoints/global/weights/best.pt \
    --name     city_day \
    --epochs   40 --batch 16 --lr 5e-5 \
    --patience 15 --mosaic 0 --cos_lr --workers 2 \
    --device   cuda

echo ""
echo "===== Retraining highway_day (from YOLOv8s global) ====="
python train.py \
    --data     data/bdd100k/yolo/highway_day.yaml \
    --weights  checkpoints/global/weights/best.pt \
    --name     highway_day \
    --epochs   40 --batch 16 --lr 5e-5 \
    --patience 15 --mosaic 0 --cos_lr --workers 2 \
    --device   cuda

echo ""
echo "===== Retraining residential (from YOLOv8s global) ====="
python train.py \
    --data     data/bdd100k/yolo/residential.yaml \
    --weights  checkpoints/global/weights/best.pt \
    --name     residential \
    --epochs   40 --batch 16 --lr 5e-5 \
    --patience 15 --mosaic 0 --cos_lr --workers 2 \
    --device   cuda

echo ""
echo "Done. All three day specialists retrained from YOLOv8s global."
echo "Verify with: ls -lh checkpoints/{city_day,highway_day,residential}/weights/best.pt"
echo "Expected: ~22 MB each"
