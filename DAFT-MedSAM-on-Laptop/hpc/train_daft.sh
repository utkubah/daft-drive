#!/bin/bash
# Train all 9 DAFT specialists in parallel as a SLURM job array.
# Each array task fine-tunes from global.pth on one modality subset.
#
# Prerequisites:
#   - conda env "daft" is set up (run hpc/setup_env.sh)
#   - checkpoints/global/best.pth exists (run hpc/pretrain.sh first)
#   - data/datasplit/modalities/ CSVs exist (built by pretrain.sh)
#
# Usage:  sbatch hpc/train_daft.sh
#
# Array index -> modality:
#   0 Dermoscopy  1 Endoscopy  2 Fundus  3 Mammography  4 Microscopy
#   5 OCT         6 US         7 XRay    8 3D (CT+MR+PET)

#SBATCH --job-name=medsam-daft
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

set -e
mkdir -p logs

MODALITIES=(Dermoscopy Endoscopy Fundus Mammography Microscopy OCT US XRay 3D)
MOD=${MODALITIES[$SLURM_ARRAY_TASK_ID]}

source /software/miniconda3/etc/profile.d/conda.sh
conda activate daft
module load cuda/12.4 || true

cd $HOME/daft-drive/DAFT-MedSAM-on-Laptop

echo "  Training specialist: $MOD  (array task $SLURM_ARRAY_TASK_ID)"
python train.py \
    --train_csv data/datasplit/modalities/${MOD}.train.csv \
    --val_csv   data/datasplit/modalities/${MOD}.val.csv   \
    --weights   checkpoints/global/best.pth                \
    --name      ${MOD}                                     \
    --epochs    20                                         \
    --batch_size 8                                         \
    --num_workers 4
