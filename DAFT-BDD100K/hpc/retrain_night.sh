#!/bin/bash
#SBATCH --job-name=bdd100k-night
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/retrain_night_%j.out
#SBATCH --error=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/retrain_night_%j.err

set -e
mkdir -p /mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs

source /software/miniconda3/etc/profile.d/conda.sh
conda activate daft
module load cuda/12.4 || true

cd /mnt/beegfsstudents/home/3223837/DAFT-BDD100K

test -f checkpoints/global/weights/best.pt || {
    echo "ERROR: checkpoints/global/weights/best.pt not found."
    exit 1
}

echo "===== Retraining city_night (with dawn/dusk data) ====="
python train.py \
    --data     data/bdd100k/yolo/city_night.yaml \
    --weights  checkpoints/global/weights/best.pt \
    --name     city_night \
    --epochs   40 --batch 16 --lr 5e-5 \
    --patience 15 --mosaic 0 --cos_lr --workers 2 \
    --device   cuda

echo "===== Retraining highway_night (with dawn/dusk data) ====="
python train.py \
    --data     data/bdd100k/yolo/highway_night.yaml \
    --weights  checkpoints/global/weights/best.pt \
    --name     highway_night \
    --epochs   40 --batch 16 --lr 5e-5 \
    --patience 15 --mosaic 0 --cos_lr --workers 2 \
    --device   cuda

echo "Done. Both night specialists retrained with dawn/dusk augmentation."
