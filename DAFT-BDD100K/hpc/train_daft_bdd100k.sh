#!/bin/bash
# Fine-tunes one specialist per driving condition (city_day, city_night,
# highway_day, highway_night, residential), all in a single job to stay
# within the per-user job submission limit.
# Needs the global checkpoint from pretrain_bdd100k.sh first.
#
# Usage: sbatch hpc/train_daft_bdd100k.sh

#SBATCH --job-name=bdd100k-daft
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --output=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/daft_%j.out
#SBATCH --error=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/daft_%j.err

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

CONDITIONS=(city_day city_night highway_day highway_night residential)

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
        --epochs   40 \
        --batch    16 \
        --imgsz    640 \
        --lr       5e-5 \
        --patience 15 \
        --mosaic   0 \
        --cos_lr \
        --workers  2 \
        --device   cuda

    echo "===== Done: $COND ====="
done

echo ""
echo "All DAFT specialists trained."
