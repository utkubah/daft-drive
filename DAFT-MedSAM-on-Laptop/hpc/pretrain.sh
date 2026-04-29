#!/bin/bash
# Full pre-DAFT pipeline in one SLURM job:
#   1. Distillation     (TinyViT teacher -> EfficientViT student)
#   2. Merge weights    (distilled encoder + LiteMedSAM decoder)
#   3. General fine-tune (on all modalities combined)
#
# Inputs required in repo root:
#   lite_medsam.pth                     (downloaded by setup_env.sh)
#   ~/medsam-data/train_npz/...         (uploaded from local via rsync)
#   ~/medsam-data/validation-box/...    (optional)
#
# Outputs:
#   checkpoints/distilled_encoder.pth
#   checkpoints/merged.pth
#   checkpoints/global/best.pth         (the input to DAFT)
#
# Usage:  sbatch hpc/pretrain.sh

#SBATCH --job-name=medsam-pretrain
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --output=/home/3223837/daft-drive/DAFT-MedSAM-on-Laptop/logs/pretrain_%j.out
#SBATCH --error=/home/3223837/daft-drive/DAFT-MedSAM-on-Laptop/logs/pretrain_%j.err

set -e
mkdir -p logs

DATA_DIR=$HOME/medsam-data

source /software/miniconda3/etc/profile.d/conda.sh
conda activate daft
module load cuda/12.4 || true

cd $HOME/daft-drive/DAFT-MedSAM-on-Laptop

# Build CSV splits from the (subsampled) data
echo "===== Building CSV splits ====="
python prepare_data.py --data_dir $DATA_DIR

# Step 1: Knowledge distillation
echo "===== Step 1/3: Distillation ====="
python distill.py \
    --train_csv data/datasplit/train.csv \
    --val_csv   data/datasplit/val.csv   \
    --lite_medsam lite_medsam.pth        \
    --epochs    24  --batch_size 8  --num_workers 4

# Step 2: Merge distilled encoder + LiteMedSAM decoder
echo "===== Step 2/3: Merge weights ====="
python merge_weights.py \
    --lite_medsam lite_medsam.pth                  \
    --encoder     checkpoints/distilled_encoder.pth \
    --output      checkpoints/merged.pth

# Step 3: General fine-tune on all data
echo "===== Step 3/3: General fine-tune ====="
python train.py \
    --train_csv data/datasplit/train.csv \
    --val_csv   data/datasplit/val.csv   \
    --weights   checkpoints/merged.pth   \
    --name      global                   \
    --epochs    24  --batch_size 8  --num_workers 4

echo ""
echo "Pretrain pipeline complete.  Next:"
echo "  sbatch hpc/train_daft.sh"
