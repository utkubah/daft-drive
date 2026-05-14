#!/bin/bash
# BDD100K data preparation — no GPU needed.
# Downloads BDD100K via FiftyOne, exports YOLO format,
# builds 5-condition splits (city_day/city_night/highway_day/highway_night/residential).
#
# Usage: sbatch hpc/prepare_data.sh
# Set HF_TOKEN env var before submitting:
#   export HF_TOKEN=hf_xxx && sbatch hpc/prepare_data.sh

#SBATCH --job-name=bdd100k-data
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/prepare_%j.out
#SBATCH --error=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/prepare_%j.err

set -e
mkdir -p /mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs

source /software/miniconda3/etc/profile.d/conda.sh
conda activate daft
cd /mnt/beegfsstudents/home/3223837/DAFT-BDD100K

export HF_TOKEN= hf_xxx 

echo "===== Preparing BDD100K (full dataset) ====="
python prepare_data.py --max_samples 0

echo ""
echo "Data ready. Next: sbatch hpc/pretrain_bdd100k.sh"
