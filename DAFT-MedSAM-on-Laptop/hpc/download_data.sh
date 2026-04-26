#!/bin/bash
# Download the full CVPR24 MedSAM-on-Laptop training data from Google Drive
# directly to scratch on beegfs. ~70 GB total.
#
# Usage:  sbatch hpc/download_data.sh

#SBATCH --job-name=medsam-download
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=23:00:00
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err

set -e
mkdir -p logs

DATA_DIR=/mnt/beegfsstudents/scratch_3223837/medsam-data

source /software/miniconda3/etc/profile.d/conda.sh
conda activate daft

mkdir -p $DATA_DIR
cd $DATA_DIR

echo "  Downloading training data to $DATA_DIR ..."
# Source: https://drive.google.com/drive/folders/1LCux2WYYQC9Kh3JpX_kONs4pOyd43PjR
gdown --folder 1LCux2WYYQC9Kh3JpX_kONs4pOyd43PjR --remaining-ok

echo ""
echo "  Done. Per-folder disk usage:"
du -sh $DATA_DIR/*
echo ""
echo "  Total: $(du -sh $DATA_DIR | cut -f1)"
