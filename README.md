# DAFT-Drive: Condition-Aware Specialist Routing for Autonomous Driving Object Detection

DAFT-Drive trains five YOLOv8s condition specialists (city-day, city-night, highway-day, highway-night, residential) on BDD100K and routes each image to the right specialist at inference time. A MobileNetV3-small image router selects one or more specialists without requiring metadata.

## Results

| System | Overall mAP50 | mAP50-95 | FPS |
|---|---|---|---|
| Global Distilled (YOLOv8s baseline) | 0.4809 | 0.3029 | 159.7 |
| Global Large (YOLOv8m) | 0.5283 | 0.3392 | 134.7 |
| Global XLarge (YOLOv8x) | 0.6262 | 0.3954 | 77.0 |
| Hard Routing (oracle, uses metadata) | 0.6859 | 0.4771 | 159.6 |
| **DAFT K=1 (ours)** | **0.6819** | **0.4686** | **84.6** |
| DAFT K=2 | 0.6917 | — | 53.0 |
| DAFT K=3 | 0.6949 | — | 40.4 |
| DAFT K=4 | 0.6975 | — | 31.7 |
| DAFT K=5 | 0.6991 | — | 26.5 |

DAFT K=1 improves over Global Distilled by **41.8%** relative mAP50 and recovers **99.4%** of the oracle metadata router's accuracy using only the raw image. The learned ImageRouter achieves **96.4%** top-1 condition accuracy on the held-out validation set.

Per-condition mAP50 (specialist vs global distilled):

| Condition | Global | Specialist | Gain |
|---|---|---|---|
| city_day | 0.517 | 0.629 | +11.2 pp |
| city_night | 0.565 | 0.728 | +16.3 pp |
| highway_day | 0.485 | 0.715 | +23.0 pp |
| highway_night | 0.606 | 0.756 | +15.0 pp |
| residential | 0.655 | 0.781 | +12.6 pp |

Evaluated on BDD100K validation split (9,864 images). GPU: NVIDIA A100 80GB PCIe, CUDA 12.4, batch size 1, models pre-loaded.

## Architecture

```
YOLOv8m (teacher, frozen)
    └── feature-mimicking distillation
YOLOv8s (global student)
    ├── city_day specialist
    ├── city_night specialist
    ├── highway_day specialist
    ├── highway_night specialist
    └── residential specialist

At inference:
  image → ImageRouter (MobileNetV3-small, 4.97 ms)
        → top-K specialist(s)
        → weighted NMS → detections
```

Three training stages:
1. **Distillation** — YOLOv8m teacher → YOLOv8s student via feature mimicking (20 epochs)
2. **Global fine-tuning** — student trained on all BDD100K conditions (50 epochs)
3. **Specialist fine-tuning** — five copies, each on one condition subset (40 epochs, mosaic disabled)

## Quickstart

### 1. Environment

```bash
conda env create -f environment.yml
conda activate daft
```

### 2. Data

Download BDD100K from [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu/). Place the images and JSON annotations under `data/bdd100k/`. Then run:

```bash
python prepare_data.py
# or on HPC:
sbatch hpc/prepare_data.sh
```

This generates per-condition YAML files and the manifest CSVs used by all training and evaluation scripts.

### 3. Checkpoints

Checkpoints are available on request. Place them as:

```
checkpoints/
  global/weights/best.pt        YOLOv8s global distilled
  large/weights/best.pt         YOLOv8m global fine-tuned
  xlarge/weights/best.pt        YOLOv8x global fine-tuned
  city_day/weights/best.pt
  city_night/weights/best.pt
  highway_day/weights/best.pt
  highway_night/weights/best.pt
  residential/weights/best.pt
  router/best.pt                MobileNetV3-small image router
```

Alternatively, reproduce all checkpoints from scratch using the steps below.

### 4. Reproduce training from scratch

```bash
# Stage 1+2: distill and globally fine-tune
sbatch hpc/pretrain_bdd100k.sh

# Stage 3: train five specialists
sbatch hpc/train_daft_bdd100k.sh

# Train image router
sbatch hpc/train_router.sh
```

### 5. Evaluation

```bash
# Full paper evaluation (all tables + figures)
sbatch hpc/eval_paper.sh

# Results appear in results/
#   main_results.csv        per-condition mAP50 + FPS
#   ksweep.csv              full k=1..5 tradeoff
#   dawn_dusk.csv           dawn/dusk robustness
#   router_accuracy.csv     routing accuracy per condition
#   router_confusion_matrix.csv  5x5 confusion matrix
#   ablations.csv           routing ablation study
#   per_class.csv           per-class AP50 gains
#   per_class_detail.csv    per-condition x per-class breakdown
#   condition_split_counts.csv   val images per condition
#   accuracy_speed_tradeoff.png
#   condition_heatmap.png
#   class_gains.png
#   dawn_dusk.png
#   ablation_bar.png
#   router_confusion_matrix.png
```

### 5b. Running without SLURM (single GPU)

If you are not on an HPC cluster, run each stage directly:

```bash
# Stage 1: distill YOLOv8m → YOLOv8s
python distill.py --img_dir data/bdd100k/yolo/images/train --device cuda

# Stage 2: global fine-tuning
python train.py --data data/bdd100k/yolo/dataset.yaml \
                --weights checkpoints/distilled/distilled.pt \
                --name global --device cuda

# Stage 3: specialist fine-tuning (run once per condition)
for COND in city_day city_night highway_day highway_night residential; do
  python train.py --data data/bdd100k/yolo/${COND}.yaml \
                  --weights checkpoints/global/weights/best.pt \
                  --name ${COND} --mosaic 0 --device cuda
done

# Train image router
python train_router.py --device cuda

# Full evaluation (reproduces all paper tables and figures)
python eval_paper.py --device cuda --batch 8
```

### 5c. Script → paper output mapping

| Paper element | Script | Key output file |
|---|---|---|
| Table 1 (main results) | `eval_paper.py` Phase 1+2 | `results/main_results.csv` |
| Table 2 (K-sweep) | `eval_paper.py` Phase 5 | `results/ksweep.csv` |
| Table 3 (ablation) | `eval_paper.py` Phase 6 | `results/ablations.csv` |
| Table 4 (dawn/dusk) | `eval_paper.py` Phase 3 | `results/dawn_dusk.csv` |
| Figure 3 (accuracy-speed) | `eval_paper.py` Phase 5 | `results/accuracy_speed_tradeoff.png` |
| Figure 4 (router confusion) | `eval_paper.py` Phase 4 | `results/router_confusion_matrix.png` |
| Figure 5 (per-class gains) | `eval_paper.py` per-class | `results/class_gains.png` |
| Figure 6 (condition heatmap) | `eval_paper.py` plots | `results/condition_heatmap.png` |
| Figure 7 (ablation bar) | `eval_paper.py` Phase 6 | `results/ablation_bar.png` |
| Figure 8 (dawn/dusk bar) | `eval_paper.py` Phase 3 | `results/dawn_dusk.png` |
| Figure 2 (qualitative) | `visualize.py` | `results/viz/{condition}/` |

### 6. Inference on a single image

```bash
# Image-based routing (deployable — no metadata needed):
python inference.py --source path/to/image.jpg --router_ckpt checkpoints/router/best.pt --device cuda

# Force a specific specialist:
python inference.py --source path/to/image.jpg --condition city_night --device cuda
```

## Repository structure

```
train.py                  specialist + global fine-tuning
train_router.py           MobileNetV3-small router training
distill.py                feature-mimicking distillation (Stage 1)
prepare_data.py           BDD100K → per-condition YOLO format
router.py                 MetadataRouter (oracle), ImageRouter, blend_detections
eval_paper.py             single authoritative evaluation script (all tables + figures)
visualize.py              qualitative GT / Global / Specialist panels
inference.py              single-image inference demo
hpc/                      SLURM job scripts
environment.yml           conda environment
classes.txt               BDD100K object class names (13 classes)
results/                  pre-computed CSVs, plots, and qualitative visualizations
```

## Condition partitioning

Five mutually exclusive conditions derived from BDD100K `scene` and `timeofday` metadata fields:

| Condition | Scene | Time of day | Val images |
|---|---|---|---|
| city_day | city | daytime | 3,530 |
| city_night | city | night | 2,582 |
| highway_day | highway | daytime | 1,492 |
| highway_night | highway | night | 1,007 |
| residential | residential | any | 1,253 |
| **Total** | | | **9,864** |

These five conditions cover 98.6% of the sampled images. Dawn/dusk images (661 in val) appear in both day and night specialist training splits.

## Router accuracy

| Condition | Accuracy |
|---|---|
| city_day | 93.2% |
| city_night | 96.6% |
| highway_day | 98.8% |
| highway_night | 99.6% |
| residential | 99.5% |
| **Overall** | **96.4%** |

Main confusion: city_day ↔ residential (similar daytime lighting and object density).

## Citation

```bibtex
@article{bahcivanoglu2025daftdrive,
  title   = {DAFT-Drive: Condition-Aware Specialist Routing for Autonomous Driving Object Detection},
  author  = {Bah\c{c}{\i}vano\u{g}lu, Utku and Demirkaz{\i}k, Berkay and Karsavurano\u{g}lu, Eren},
  year    = {2025}
}
```
