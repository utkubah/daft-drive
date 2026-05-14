#!/bin/bash
# Full evaluation pipeline for DAFT-BDD100K.
# Run this after all training jobs are done (specialists + router).
#
# Phases:
#   0 — compare.py            per-condition mAP, global vs each specialist
#   1 — eval_topk.py          k=1..5 router sweep, mAP + GPU timing
#   2 — eval_full.py          consolidated table, plots, timing breakdown
#   3 — collect_missing_data.py  global baselines comparison (distilled/m/x)
#   4 — visualize.py          side-by-side GT / Global / Specialist panels
#
# eval_topk must finish before eval_full (reads topk_sweep.csv).
# compare.py must finish before eval_full (reads map_comparison.csv).
#
# Usage:
#   sbatch hpc/eval_full.sh

#SBATCH --job-name=bdd100k-eval-full
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/eval_full_%j.out
#SBATCH --error=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/eval_full_%j.err

set -e
mkdir -p /mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs

# Cap DataLoader workers to match --cpus-per-task=4 (2 per dataloader × train+val = 4 total)
export OMP_NUM_THREADS=2

source /software/miniconda3/etc/profile.d/conda.sh
conda activate daft
module load cuda/12.4 || true

cd /mnt/beegfsstudents/home/3223837/DAFT-BDD100K

# Check the global checkpoint exists before doing anything else
test -f checkpoints/global/weights/best.pt || {
    echo "ERROR: checkpoints/global/weights/best.pt not found."
    echo "Run sbatch hpc/pretrain_bdd100k.sh first."
    exit 1
}

# Make sure specialist checkpoints exist and are the right size (~22 MB for YOLOv8s).
# Catches the common mistake of accidentally leaving YOLOv8n weights in place.
echo "===== Checkpoint size verification ====="
ALL_OK=1
for COND in city_day city_night highway_day highway_night residential; do
    CKPT="checkpoints/${COND}/weights/best.pt"
    if [ ! -f "$CKPT" ]; then
        echo "  MISSING  $CKPT"
        ALL_OK=0
    else
        SIZE_MB=$(du -m "$CKPT" | cut -f1)
        if [ "$SIZE_MB" -lt 15 ]; then
            echo "  ⚠ WRONG  $CKPT  (${SIZE_MB} MB — looks like YOLOv8n, expected ~22 MB)"
            ALL_OK=0
        else
            echo "  OK       $CKPT  (${SIZE_MB} MB)"
        fi
    fi
done
if [ "$ALL_OK" -eq 0 ]; then
    echo ""
    echo "ERROR: one or more specialists are the wrong architecture."
    echo "Run sbatch hpc/retrain_day.sh first, then resubmit this job."
    exit 1
fi
echo ""

echo "===== Phase 0: Per-condition mAP (compare.py) ====="
python -u compare.py \
    --device      cuda \
    --batch       16 \
    --router_ckpt checkpoints/router/best.pt

echo ""
echo "===== Phase 1: Top-K router sweep (eval_topk.py) ====="
python -u eval_topk.py \
    --device      cuda \
    --n_bench     200 \
    --n_map       1000 \
    --router_ckpt checkpoints/router/best.pt \
    --large_ckpt  checkpoints/large/weights/best.pt

echo ""
echo "===== Phase 2: Consolidated table + plots (eval_full.py) ====="
# Pass the YOLOv8m checkpoint if available; eval_full.py handles it being missing gracefully
LARGE_CKPT=""
if [ -f checkpoints/large/weights/best.pt ]; then
    LARGE_CKPT="--large_ckpt checkpoints/large/weights/best.pt"
fi

python -u eval_full.py \
    --device      cuda \
    --batch       16 \
    --n_bench     200 \
    --router_ckpt checkpoints/router/best.pt \
    --force_timing \
    $LARGE_CKPT

echo ""
echo "===== Phase 3: Global baselines comparison (collect_missing_data.py) ====="
# Evaluates distilled / YOLOv8m / YOLOv8x on a 100-image-per-condition sample.
# YOLOv8x is optional — skipped if the checkpoint isn't there yet.
python -u collect_missing_data.py \
    --distilled_ckpt checkpoints/global/weights/best.pt \
    --large_ckpt     checkpoints/large/weights/best.pt \
    --xlarge_ckpt    checkpoints/xlarge/weights/best.pt \
    --router_ckpt    checkpoints/router/best.pt \
    --data_dir       data/bdd100k \
    --device         cuda \
    --imgsz          640 \
    --n_samples      100

echo ""
echo "===== Phase 4: Qualitative visualizations (visualize.py) ====="
python -u visualize.py --device cuda --n_samples 5

echo ""
echo "Done. Results are in results/:"
echo "  consolidated.csv            main accuracy + speed table"
echo "  per_condition.csv           per-condition mAP50 breakdown"
echo "  per_class.csv               per-class AP50 gain"
echo "  timing_breakdown.csv        router / detector / NMS timing"
echo "  global_models_comparison.csv  distilled vs YOLOv8m vs YOLOv8x"
echo "  eval_report.txt             human-readable summary"
echo "  accuracy_speed.png          FPS vs mAP50 scatter"
echo "  condition_heatmap.png       condition × strategy heatmap"
echo "  class_gain.png              per-class gain bars"
echo "  topk_sweep.csv              k=1..5 tradeoff numbers"
