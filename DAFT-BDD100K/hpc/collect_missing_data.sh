#!/bin/bash
# Collect all missing paper data in one job.
# Runs: Global Large per-condition eval + router accuracy + split counts.
#
# Usage: sbatch hpc/collect_missing_data.sh

#SBATCH --job-name=bdd100k-collect
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/collect_%j.out
#SBATCH --error=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/collect_%j.err

set -e
mkdir -p /mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs

source /software/miniconda3/etc/profile.d/conda.sh
conda activate daft
module load cuda/12.4 || true

cd /mnt/beegfsstudents/home/3223837/DAFT-BDD100K

# Print GPU info into the log so we have it for the paper
echo "=== GPU INFO ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "Ultralytics: $(python -c 'import ultralytics; print(ultralytics.__version__)')"
echo "================"

python collect_missing_data.py \
    --distilled_ckpt checkpoints/global/weights/best.pt \
    --large_ckpt     checkpoints/large/weights/best.pt \
    --xlarge_ckpt    checkpoints/xlarge/weights/best.pt \
    --router_ckpt    checkpoints/router/best.pt \
    --ckpt_dir       checkpoints \
    --data_dir       data/bdd100k \
    --device         cuda \
    --imgsz          640 \
    --n_samples      100

echo ""
echo "Results written to results/:"
ls -lh results/global_models_comparison.csv \
        results/condition_split_counts.csv \
        results/class_names.csv \
        results/router_accuracy.csv \
        results/router_confusion_matrix.csv 2>/dev/null || true
