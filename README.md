# DAFT-BDD100K: Condition-Aware Object Detection for Autonomous Driving

**DAFT** (Data-Aware Fine-Tuning) trains five YOLOv8s specialists — one per driving condition — and routes each frame to the right expert at inference time. On BDD100K, this improves mAP50 by **+44% over the global baseline** while staying real-time at **83 FPS**.

> Project for the Computer Vision course, Bocconi University (20878).

---

## Results

| Model | mAP50 | FPS |
|---|---|---|
| Global Distilled (YOLOv8s) | 0.470 | 158 |
| Global YOLOv8m | 0.549 | 135 |
| Hard Routing (metadata) | 0.677 | 158 |
| **DAFT Adaptive k=1** | **0.677** | **83** |
| DAFT Adaptive k=2 | 0.684 | 53 |

Per-condition gains over the global baseline (mAP50):

| Condition | Global | Specialist | Gain |
|---|---|---|---|
| City · Day | 0.564 | 0.651 | +0.087 |
| City · Night | 0.547 | 0.689 | +0.143 |
| Highway · Day | 0.476 | 0.732 | +0.256 |
| Highway · Night | 0.515 | 0.762 | +0.247 |
| Residential | 0.667 | 0.784 | +0.117 |

---

## Method

The pipeline has three stages:

1. **Knowledge distillation** — YOLOv8m teacher → YOLOv8s student, so the compact model starts with strong feature representations.
2. **Global fine-tuning** — the YOLOv8s student is fine-tuned on all BDD100K images.
3. **Specialist fine-tuning** — five separate models are trained, each on one driving condition (city/highway/residential × day/night).

At inference a lightweight **MobileNetV3-small router** (4.97 ms GPU) classifies the incoming frame and dispatches it to the top-K specialists. With K=1 this matches hard metadata routing; K=2 helps on ambiguous dawn/dusk scenes by blending two specialists.

```
BDD100K frames
     │
     ├─ ImageRouter (MobileNetV3-small, 2.5M params)
     │        └─ top-K condition weights
     │
     ├─ Specialist k₁ ──┐
     ├─ Specialist k₂ ──┤ blend_detections() + NMS
     └─ ...             └─► final detections
```

---

## Setup

**Requirements:** Python 3.10+, CUDA 12.4, ~50 GB disk for BDD100K.

```bash
# Clone and create the environment
git clone https://github.com/<your-org>/DAFT-BDD100K.git
cd DAFT-BDD100K
conda env create -f environment.yml
conda activate daft
```

Key packages: `ultralytics`, `torch`, `fiftyone`, `torchmetrics`, `opencv-python`, `matplotlib`.

---

## Reproducing the Experiments

All steps can be run locally or submitted to a SLURM cluster via the scripts in `hpc/`.

### 1. Prepare the data

Downloads BDD100K via FiftyOne (HuggingFace), exports YOLO-format labels, and builds per-condition splits.

```bash
# Set a HuggingFace token to avoid rate limits (~40 GB download)
export HF_TOKEN=hf_your_token_here
python prepare_data.py --max_samples 0   # 0 = full dataset
```

### 2. Train

```bash
# Step 1: distill YOLOv8m → YOLOv8s, then fine-tune on all BDD100K
python distill.py --img_dir data/bdd100k/yolo/images/train --teacher yolov8m.pt --student yolov8s.pt
python train.py --data data/bdd100k/yolo/dataset.yaml --weights checkpoints/distilled/distilled.pt --name global

# Step 2: fine-tune one specialist per condition
python train.py --data data/bdd100k/yolo/city_day.yaml     --weights checkpoints/global/weights/best.pt --name city_day
python train.py --data data/bdd100k/yolo/city_night.yaml   --weights checkpoints/global/weights/best.pt --name city_night
python train.py --data data/bdd100k/yolo/highway_day.yaml  --weights checkpoints/global/weights/best.pt --name highway_day
python train.py --data data/bdd100k/yolo/highway_night.yaml --weights checkpoints/global/weights/best.pt --name highway_night
python train.py --data data/bdd100k/yolo/residential.yaml  --weights checkpoints/global/weights/best.pt --name residential

# Step 3: train the image router
python train_router.py --out_dir checkpoints/router
```

On a SLURM cluster, submit the full chain in one go:

```bash
PREP=$(sbatch --parsable hpc/prepare_data.sh)
PRE=$(sbatch --parsable --dependency=afterok:$PREP hpc/pretrain_bdd100k.sh)
LRG=$(sbatch --parsable --dependency=afterok:$PREP hpc/pretrain_bdd100k_large.sh)
DAFT=$(sbatch --parsable --dependency=afterok:$PRE hpc/train_daft_bdd100k.sh)
RTR=$(sbatch --parsable --dependency=afterok:$PRE hpc/train_router.sh)
sbatch --dependency=afterok:$DAFT:$RTR:$LRG hpc/eval_full.sh
```

### 3. Evaluate

```bash
# Runs all evaluation phases: per-condition mAP, k=1..5 sweep,
# consolidated table + plots, global baselines comparison
sbatch hpc/eval_full.sh

# Or locally (no GPU required for the plots and report):
python compare.py --device cpu
python eval_topk.py --device cpu --n_bench 50 --n_map 500
python eval_full.py --device cpu --n_bench 50
python collect_missing_data.py --device cpu --n_samples 50
```

### 4. Run inference on new images

```bash
python inference.py \
    --source path/to/images \
    --router_ckpt checkpoints/router/best.pt \
    --top_k auto
```

---

## Repository Structure

```
DAFT-BDD100K/
├── prepare_data.py        # BDD100K download, YOLO export, condition splits
├── distill.py             # knowledge distillation (YOLOv8m → YOLOv8s)
├── train.py               # YOLOv8 fine-tuning wrapper
├── train_router.py        # MobileNetV3-small router training
├── router.py              # MetadataRouter, ImageRouter, blend_detections
├── inference.py           # single-image / folder inference with routing
├── compare.py             # per-condition mAP: global vs specialist
├── eval_topk.py           # k=1..5 sweep: mAP50 + GPU timing
├── eval_full.py           # consolidated table, plots, eval report
├── collect_missing_data.py # global baselines comparison (distilled/m/x)
├── visualize.py           # GT / Global / Specialist side-by-side panels
├── plot_results.py        # regenerate all figures from saved CSVs
├── environment.yml        # conda environment
├── hpc/                   # SLURM job scripts
│   ├── prepare_data.sh
│   ├── pretrain_bdd100k.sh
│   ├── pretrain_bdd100k_large.sh
│   ├── pretrain_bdd100k_xlarge.sh
│   ├── train_daft_bdd100k.sh
│   ├── train_router.sh
│   └── eval_full.sh
├── checkpoints/           # model weights (see Checkpoints section)
│   ├── global/            # YOLOv8s global model
│   ├── large/             # YOLOv8m global baseline
│   ├── xlarge/            # YOLOv8x global baseline (optional)
│   ├── router/            # MobileNetV3 image router
│   └── {condition}/       # five condition specialists
├── data/bdd100k/          # prepared dataset (generated by prepare_data.py)
│   ├── yolo/              # YOLO-format images + labels + yamls
│   └── manifests/         # per-condition CSVs for routing and eval
└── results/               # all outputs (CSVs, PNGs, eval_report.txt)
```

---

## Checkpoints

Pre-trained checkpoints are available at: **[link placeholder]**

| Checkpoint | Description | Size |
|---|---|---|
| `global/weights/best.pt` | YOLOv8s distilled + BDD100K global fine-tune | ~22 MB |
| `large/weights/best.pt` | YOLOv8m BDD100K fine-tune | ~52 MB |
| `router/best.pt` | MobileNetV3-small image router | ~10 MB |
| `city_day/weights/best.pt` | City daytime specialist | ~22 MB |
| `city_night/weights/best.pt` | City night specialist | ~22 MB |
| `highway_day/weights/best.pt` | Highway daytime specialist | ~22 MB |
| `highway_night/weights/best.pt` | Highway night specialist | ~22 MB |
| `residential/weights/best.pt` | Residential specialist | ~22 MB |

---

## Data

BDD100K is sourced from [dgural/bdd100k](https://huggingface.co/datasets/dgural/bdd100k) on HuggingFace. `prepare_data.py` handles the full pipeline: download → YOLO export → condition splits using BDD100K's scene and timeofday metadata fields.

Condition assignment:

| Scene | Time of day | Condition |
|---|---|---|
| city | daytime / dawn/dusk | city_day |
| city | night | city_night |
| highway | daytime / dawn/dusk | highway_day |
| highway | night | highway_night |
| residential | any | residential |

Dawn/dusk images are included in training (assigned to the day condition) but flagged separately in the manifests for robustness evaluation.
