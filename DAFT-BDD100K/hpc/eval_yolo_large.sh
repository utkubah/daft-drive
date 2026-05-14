#!/bin/bash
# Evaluate the fine-tuned YOLOv8m large baseline alongside the full DAFT suite.
# Run AFTER pretrain_bdd100k_large.sh AND the main eval_full.sh have completed,
# or chain both evals together (see below).
#
# Usage:
#   sbatch hpc/eval_yolo_large.sh
#
# Or chain after large fine-tune:
#   JOB_L=$(sbatch --parsable hpc/pretrain_bdd100k_large.sh)
#   sbatch --dependency=afterok:$JOB_L hpc/eval_yolo_large.sh

#SBATCH --job-name=bdd100k-y8m-eval
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/eval_y8m_%j.out
#SBATCH --error=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/eval_y8m_%j.err

set -e
mkdir -p /mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs

source /software/miniconda3/etc/profile.d/conda.sh
conda activate daft
module load cuda/12.4 || true

cd /mnt/beegfsstudents/home/3223837/DAFT-BDD100K

test -f checkpoints/large/weights/best.pt || {
    echo "ERROR: checkpoints/large/weights/best.pt not found."
    echo "Run sbatch hpc/pretrain_bdd100k_large.sh first."
    exit 1
}

echo "===== Evaluating large baseline + full DAFT suite ====="
python eval_full.py \
    --device      cpu \
    --batch       4 \
    --n_bench     100 \
    --router_ckpt checkpoints/router/best.pt \
    --large_ckpt  checkpoints/large/weights/best.pt \
    --force_timing

echo ""
echo "Results updated in results/ (large baseline now included in all tables)"