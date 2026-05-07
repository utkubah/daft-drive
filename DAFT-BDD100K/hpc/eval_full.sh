#!/bin/bash
# Full DAFT evaluation suite.
#
# Runs in four phases (ORDER MATTERS — eval_topk must precede eval_full.py):
#   Phase 0 — compare.py     : per-condition mAP (global vs specialist) +
#                               per-class AP50. Generates map_comparison.csv
#                               and perclass_map.csv needed by eval_full.py.
#   Phase 1 — eval_topk.py   : sweeps adaptive router k=1..5, writes
#                               topk_sweep.csv (mAP + timing per k).
#                               Must run BEFORE eval_full.py which reads it.
#   Phase 2 — eval_full.py   : loads accuracy CSVs (map_comparison.csv +
#                               topk_sweep.csv), re-runs timing benchmark,
#                               produces consolidated table + plots.
#   Phase 3 — visualize.py   : GT | Global | Specialist panels.
#
# Run after train_daft_bdd100k.sh AND train_router.sh have completed.
#
# Usage:
#   sbatch hpc/eval_full.sh
#
# Or chain after specialist retraining:
#   JOB_D=$(sbatch --parsable hpc/retrain_day.sh)
#   sbatch --dependency=afterok:$JOB_D hpc/eval_full.sh

#SBATCH --job-name=bdd100k-eval-full
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/home/3223837/DAFT-BDD100K/logs/eval_full_%j.out
#SBATCH --error=/home/3223837/DAFT-BDD100K/logs/eval_full_%j.err

set -e
mkdir -p /home/3223837/DAFT-BDD100K/logs

# Cap DataLoader workers to match --cpus-per-task=4 (2 per dataloader × train+val = 4 total)
export OMP_NUM_THREADS=2

source /software/miniconda3/etc/profile.d/conda.sh
conda activate daft
module load cuda/12.4 || true

cd $HOME/DAFT-BDD100K

# Sanity checks
test -f checkpoints/global/weights/best.pt || {
    echo "ERROR: checkpoints/global/weights/best.pt not found."
    echo "Run sbatch hpc/pretrain_bdd100k.sh first."
    exit 1
}

# Verify all specialists are YOLOv8s (>15 MB); abort early if any are wrong.
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
echo "  Runs model.val() for each condition × global + specialist."
echo "  Generates map_comparison.csv and perclass_map.csv."
python -u compare.py \
    --device      cuda \
    --batch       16 \
    --router_ckpt checkpoints/router/best.pt

echo ""
echo "===== Phase 1: Top-K sweep (eval_topk.py) ====="
echo "  Sweeps Global Large + adaptive router k=1..5, measures mAP50 + timing on GPU."
echo "  Uses 1000 val images for mAP (fast). Writes topk_sweep.csv."
python -u eval_topk.py \
    --device      cuda \
    --n_bench     200 \
    --n_map       1000 \
    --router_ckpt checkpoints/router/best.pt \
    --large_ckpt  checkpoints/large/weights/best.pt

echo ""
echo "===== Phase 2: Consolidated evaluation (eval_full.py) ====="
echo "  Reads map_comparison.csv + topk_sweep.csv (both now up-to-date)."
echo "  Re-runs timing benchmark on GPU."
echo "  Hard routing uses MetadataRouter (per-image manifest lookup)."
# Add --large_ckpt if the fine-tuned large model exists
LARGE_CKPT=""
if [ -f checkpoints/large/weights/best.pt ]; then
    LARGE_CKPT="--large_ckpt checkpoints/large/weights/best.pt"
    echo "  Large baseline found: checkpoints/large/weights/best.pt"
else
    echo "  No large baseline found (run hpc/pretrain_bdd100k_large.sh to add it)"
fi

python -u eval_full.py \
    --device      cuda \
    --batch       16 \
    --n_bench     200 \
    --router_ckpt checkpoints/router/best.pt \
    --force_timing \
    $LARGE_CKPT

echo ""
echo "===== Phase 3: Qualitative visualizations ====="
python -u visualize.py --device cuda --n_samples 5

echo ""
echo "All results saved to results/"
echo "  results/consolidated.csv          <- main accuracy+speed table"
echo "  results/per_condition.csv         <- condition breakdown"
echo "  results/per_class.csv             <- class breakdown"
echo "  results/timing_breakdown.csv      <- router/detector/nms timing"
echo "  results/eval_report.txt           <- full human-readable report"
echo "  results/accuracy_speed.png        <- tradeoff scatter plot"
echo "  results/condition_heatmap.png     <- condition × strategy heatmap"
echo "  results/class_gain.png            <- per-class AP50 gain chart"
echo "  results/topk_sweep.csv            <- k=1..5 mAP + timing"
echo "  results/topk_tradeoff.png         <- k=1..5 tradeoff curve"
echo "  results/viz/{condition}/          <- GT|Global|Specialist panels"
