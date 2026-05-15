# DAFT-Drive: Condition-Aware Specialist Routing for Autonomous Driving Object Detection

DAFT-Drive trains five YOLOv8s condition specialists (city-day, city-night, highway-day, highway-night, residential) on BDD100K and routes each image to the right specialist at inference time. A MobileNetV3-small image router selects one or more specialists without requiring metadata.

## Results

| System | Overall mAP50 | FPS |
|---|---|---|
| Global Distilled (YOLOv8s baseline) | 0.481 | 164.4 |
| Global Large (YOLOv8m) | — | — |
| Hard Routing | 0.682 | 165.9 |
| **DAFT K=1** | **0.682** | **85.1** |
| DAFT K=2 | 0.692 | 55.4 |
| DAFT K=5 | 0.699 | 27.1 |

Per-condition mAP50 (specialist vs global distilled):

| Condition | Global | Specialist | Gain |
|---|---|---|---|
| city_day | 0.517 | 0.629 | +11.2 pp |
| city_night | 0.565 | 0.728 | +16.3 pp |
| highway_day | 0.485 | 0.715 | +23.0 pp |
| highway_night | 0.606 | 0.756 | +14.9 pp |
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
  image → ImageRouter (MobileNetV3-small, 4.86 ms)
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
#   ablations.csv           routing ablation study
#   accuracy_speed_tradeoff.png
#   condition_heatmap.png
#   class_gains.png
#   dawn_dusk.png
#   ablation_bar.png
#   router_confusion_matrix.png
```

### 6. Inference on a single image

```bash
python inference.py --image path/to/image.jpg --device cuda
```

## Repository structure

```
train.py                  specialist + global fine-tuning
train_router.py           MobileNetV3-small router training
distill.py                feature-mimicking distillation (Stage 1)
prepare_data.py           BDD100K → per-condition YOLO format
router.py                 MetadataRouter, ImageRouter, blend_detections
eval_paper.py             single authoritative evaluation script
compare.py                standalone per-condition model.val() utility
visualize.py              qualitative GT / Global / Specialist panels
inference.py              single-image inference demo
hpc/                      SLURM job scripts
environment.yml           conda environment
classes.txt               BDD100K object class names (13 classes)
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

These five conditions cover 98.6% of the sampled images. Dawn/dusk images (661 in val) are included in both the day and night specialist training splits.

## Citation

```bibtex
@article{bahcivanoglu2025daftdrive,
  title   = {DAFT-Drive: Condition-Aware Specialist Routing for Autonomous Driving Object Detection},
  author  = {Bah\c{c}{\i}vano\u{g}lu, Utku and Demirkaz{\i}k, Berkay and Karsavurano\u{g}lu, Eren},
  year    = {2025}
}
```
