#!/bin/bash
# BDD100K evaluation: compare global vs specialists + visualization.
# No GPU needed — runs on CPU.
# Run after train_daft_bdd100k.sh completes.
#
# Usage:
#   sbatch hpc/eval_bdd100k.sh
#
# Or chain after training:
#   JOB1=$(sbatch --parsable hpc/train_daft_bdd100k.sh)
#   sbatch --dependency=afterok:$JOB1 hpc/eval_bdd100k.sh

#SBATCH --job-name=bdd100k-eval
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/3223837/DAFT-BDD100K/logs/eval_%j.out
#SBATCH --error=/home/3223837/DAFT-BDD100K/logs/eval_%j.err

set -e
mkdir -p /home/3223837/DAFT-BDD100K/logs

source /software/miniconda3/etc/profile.d/conda.sh
conda activate daft

cd $HOME/DAFT-BDD100K

ls checkpoints/city_day/weights/best.pt 2>/dev/null || \
ls checkpoints/highway_day/weights/best.pt 2>/dev/null || {
    echo "ERROR: No specialist checkpoints found."
    echo "Run sbatch hpc/train_daft_bdd100k.sh first."
    exit 1
}

echo "===== Step 1/2: Compare global vs specialists ====="
python compare.py \
    --device cpu \
    --batch 4 \
    --n_bench 20 \
    --router_ckpt checkpoints/router/best.pt

echo ""
echo "===== Step 2/2: Visualize GT | Global | Specialist ====="
python visualize.py --device cpu --n_samples 5

echo ""
echo "Results saved to results/"
echo "  results/compare.csv   results/compare.png"
echo "  results/speed.csv     results/speed.png"
echo "  results/viz/{city_day,city_night,highway_day,highway_night,residential}/"
