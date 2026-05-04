#!/bin/bash
#SBATCH --job-name=bdd100k-large
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --output=/home/3223837/DAFT-BDD100K/logs/large_%j.out
#SBATCH --error=/home/3223837/DAFT-BDD100K/logs/large_%j.err

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

python train.py \
    --data     data/bdd100k/yolo/dataset.yaml \
    --weights  yolov8m.pt \
    --name     global_yolov8m \
    --epochs   50 \
    --batch    8 \
    --imgsz    640 \
    --lr       5e-5 \
    --patience 10 \
    --device   cuda

echo ""
echo "Large global model saved to: checkpoints/global_yolov8m/weights/best.pt"