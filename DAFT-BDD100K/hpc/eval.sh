#!/bin/bash
# eval_paper.sh  —  single evaluation job for DAFT-Drive paper
#
# Runs eval_paper.py which produces every table and figure the paper needs:
#   results/main_results.csv        per-condition + overall mAP50/mAP50:95 + FPS
#   results/per_class.csv           per-class AP50 gains
#   results/ablations.csv           routing ablation study
#   results/dawn_dusk.csv           dawn/dusk robustness
#   results/router_accuracy.csv     top-1 routing accuracy
#
#   results/accuracy_speed.png      main paper scatter figure
#   results/condition_heatmap.png   model × condition heatmap
#   results/ablation_bar.png        ablation bar chart
#   results/class_gains.png         per-class gain bars
#   results/dawn_dusk.png           dawn/dusk bar chart
#
# Usage:
#   sbatch hpc/eval_paper.sh

#SBATCH --job-name=daft-eval
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/eval_paper_%j.out
#SBATCH --error=/mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs/eval_paper_%j.err

set -e
mkdir -p /mnt/beegfsstudents/home/3223837/DAFT-BDD100K/logs

export OMP_NUM_THREADS=2

source /software/miniconda3/etc/profile.d/conda.sh
conda activate daft
module load cuda/12.4 || true

cd /mnt/beegfsstudents/home/3223837/DAFT-BDD100K

# Verify required checkpoints
test -f checkpoints/global/weights/best.pt || {
    echo "ERROR: checkpoints/global/weights/best.pt not found"; exit 1
}
for COND in city_day city_night highway_day highway_night residential; do
    test -f checkpoints/${COND}/weights/best.pt || {
        echo "ERROR: checkpoints/${COND}/weights/best.pt not found"; exit 1
    }
done
test -f checkpoints/router/best.pt || {
    echo "ERROR: checkpoints/router/best.pt not found"; exit 1
}

echo "===== Running eval_paper.py ====="
python -u eval_paper.py \
    --device  cuda \
    --batch   16   \
    --n_bench 200  \
    --n_abl   1000

echo ""
echo "===== Running visualize.py ====="
python -u visualize.py --device cuda --n_samples 5

echo ""
echo "Done. Results in results/:"
echo "  main_results.csv        accuracy + speed table"
echo "  per_class.csv           per-class AP50 gains"
echo "  ablations.csv           routing ablation"
echo "  dawn_dusk.csv           dawn/dusk analysis"
echo "  router_accuracy.csv          routing accuracy per condition"
echo "  router_confusion_matrix.csv  5x5 confusion matrix"
echo "  accuracy_speed.png           FPS vs mAP scatter"
echo "  condition_heatmap.png        model x condition heatmap"
echo "  ablation_bar.png             ablation chart"
echo "  class_gains.png              per-class gain bars"
echo "  dawn_dusk.png                dawn/dusk robustness"
echo "  router_confusion_matrix.png  router confusion matrix heatmap"
