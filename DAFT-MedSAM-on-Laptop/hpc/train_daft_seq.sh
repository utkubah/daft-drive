#!/bin/bash
# Train all 8 DAFT specialists sequentially in a single SLURM job.
# Avoids QOSMaxSubmitJobPerUserLimit hit by job arrays.
#
# Usage:  sbatch hpc/train_daft_seq.sh

#SBATCH --job-name=medsam-daft
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --output=/home/3223837/daft-drive/DAFT-MedSAM-on-Laptop/logs/daft_seq_%j.out
#SBATCH --error=/home/3223837/daft-drive/DAFT-MedSAM-on-Laptop/logs/daft_seq_%j.err

set -e
mkdir -p /home/3223837/daft-drive/DAFT-MedSAM-on-Laptop/logs

source /software/miniconda3/etc/profile.d/conda.sh
conda activate daft
module load cuda/12.4 || true

cd $HOME/daft-drive/DAFT-MedSAM-on-Laptop

MODALITIES=(Endoscopy Fundus Mammography Microscopy OCT US XRay MR PET)

for MOD in "${MODALITIES[@]}"; do
    echo "===== Training specialist: $MOD ====="
    python train.py \
        --train_csv data/datasplit/modalities/${MOD}.train.csv \
        --val_csv   data/datasplit/modalities/${MOD}.val.csv   \
        --weights   checkpoints/global/best.pth                \
        --name      ${MOD}                                     \
        --epochs    30                                         \
        --batch_size 8                                         \
        --num_workers 4
    echo "===== Done: $MOD ====="
done

echo ""
echo "All DAFT specialists trained."
