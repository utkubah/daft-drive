#!/bin/bash
#SBATCH --job-name=bdd100k-y8m-eval
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/3223837/DAFT-BDD100K/logs/eval_y8m_%j.out
#SBATCH --error=/home/3223837/DAFT-BDD100K/logs/eval_y8m_%j.err

set -e
mkdir -p /home/3223837/DAFT-BDD100K/logs

source /software/miniconda3/etc/profile.d/conda.sh
conda activate daft

cd $HOME/DAFT-BDD100K

python evaluate.py \
  --weights yolov8m.pt \
  --data data/bdd100k/yolo/dataset.yaml \
  --device cpu