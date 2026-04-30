# DAFT-Drive

Implementations of **Data-Aware Fine-Tuning (DAFT)** with adaptive specialist routing,
applied to autonomous driving object detection on BDD100K.

Based on: Pfefferle et al., *DAFT: Data-Aware Fine-Tuning of Foundation Models for
Efficient and Effective Medical Image Segmentation*, CVPR 2024 Workshop — 1st place.

---

## Repository Structure

```
daft-drive/
├── DAFT-BDD100K/               ← Autonomous driving object detection (active)
│   ├── prepare_data.py         load BDD100K via FiftyOne, export to YOLO format
│   ├── distill.py              YOLOv8m → YOLOv8n backbone distillation
│   ├── train.py                global fine-tune + per-condition specialist training
│   ├── evaluate.py             mAP50 / mAP50-95 evaluation
│   ├── inference.py            DAFT routing → specialist prediction
│   ├── compare.py              global vs specialist mAP comparison + bar chart
│   ├── visualize.py            GT | Global | Specialist 3-panel comparison images
│   ├── router.py               MetadataRouter + adaptive top-K blending
│   ├── analyze_metadata.py     BDD100K metadata distribution analysis
│   └── hpc/
│       ├── pretrain_bdd100k.sh prepare data + distill + global fine-tune (SLURM)
│       └── train_daft_bdd100k.sh  train all 5 specialists sequentially (SLURM)
│
└── DAFT-MedSAM-on-Laptop/     ← Medical image segmentation (reference implementation)
```

---

## What is DAFT?

Instead of one model for everything, DAFT trains a **specialist model per data subset**
and routes each input to the right specialist at inference time.

```
Input image
    │
    ▼
[Router]  reads scene + lighting metadata
    │           │            │           │           │
    ▼           ▼            ▼           ▼           ▼
[city_day] [city_night] [highway_day] [highway_night] [residential]
    │
    ▼
Detections
```

---

## The Five Specialists

BDD100K metadata analysis showed that **scene type** causes the largest domain shift
for object detection — city streets have ~47% more objects per image than highways,
and object class distributions differ significantly across scenes.

| Specialist | Scene | Lighting | Key challenge |
|------------|-------|----------|---------------|
| `city_day` | city street | daytime | dense pedestrians, cyclists, buses |
| `city_night` | city street | night | low visibility, headlights, same density |
| `highway_day` | highway | daytime | cars at distance, high speed, scale variation |
| `highway_night` | highway | night | low visibility + sparse objects |
| `residential` | residential | any | parked cars, low density, mixed |

These 5 conditions cover **98.6%** of BDD100K (vs 90.5% for day/night/rain).

---

## The Three-Step DAFT Pipeline

```
Step 1  Backbone Distillation
        Teacher: YOLOv8m  [frozen]
        Student: YOLOv8n  [backbone trained to match teacher SPPF features]
        Loss:    MSE between projected feature maps
        Output:  checkpoints/distilled/distilled.pt

Step 2  Global Fine-Tune
        Start:   distilled YOLOv8n backbone
        Train:   on all BDD100K conditions combined
        Output:  checkpoints/global/weights/best.pt

Step 3  DAFT Specialists
        Start:   global best.pt
        Train:   one specialist per scene condition (5 total)
        Route:   scene + timeofday metadata → correct specialist
        Output:  checkpoints/city_day/weights/best.pt
                 checkpoints/city_night/weights/best.pt  etc.
```

---

## Adaptive Routing

`router.py` implements **metadata-driven soft routing** with adaptive top-K selection:

- Router computes soft weights over all 5 specialists from `scene` + `timeofday` labels
- If `max_weight > 0.7` → **top-1 only** (single forward pass — fast path)
- Otherwise → **top-2 blended** with weighted NMS (handles ambiguous scenes like dawn/dusk)

```python
from router import MetadataRouter

router = MetadataRouter()
weights  = router.weights(scene="city street", timeofday="dawn/dusk")
# → {"city_day": 0.5, "city_night": 0.5, ...}

selected = router.select(weights)
# → [("city_day", 0.5), ("city_night", 0.5)]  ← blend top-2
```

A placeholder `ImageRouter` is also provided for a future learned MLP on backbone features
(no metadata required at inference).

---

## Environment Setup

```bash
conda create -n daft python=3.10 -y
conda activate daft

# PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Dependencies
pip install ultralytics fiftyone opencv-python matplotlib numpy
```

---

## Running the Pipeline

### Prepare data
```bash
python prepare_data.py --hf_token YOUR_HF_TOKEN
# Downloads BDD100K via FiftyOne, exports YOLO format, writes 5-condition splits
```

### Distill + global fine-tune (HPC)
```bash
sbatch hpc/pretrain_bdd100k.sh
```

### Train specialists (HPC)
```bash
sbatch hpc/train_daft_bdd100k.sh
```

### Evaluate
```bash
# Single model on a dataset:
python evaluate.py --weights checkpoints/city_night/weights/best.pt \
                   --data data/bdd100k/yolo/city_night.yaml --device cpu

# All specialists vs global:
python compare.py --device cpu
```

### Inference
```bash
# Route per-image via manifest metadata:
python inference.py --source data/bdd100k/yolo/images/val/ \
                    --manifest data/bdd100k/manifests/val.csv --device cpu
```

### Visualize (GT | Global | Specialist)
```bash
python visualize.py --n_samples 5 --device cpu
# Saves to results/viz/{condition}/
```

---

## Full Workflow Reference

```
prepare_data.py     BDD100K → YOLO format, 5-condition splits  → data/bdd100k/
distill.py          YOLOv8m → YOLOv8n backbone distillation    → checkpoints/distilled/
train.py            Global fine-tune (all conditions)           → checkpoints/global/
train.py ×5         DAFT specialists (one per condition)        → checkpoints/{cond}/
inference.py        Metadata routing → specialist prediction    → data/predictions/
evaluate.py         mAP50 / mAP50-95                           → stdout
compare.py          Global vs specialist comparison             → results/compare.csv+png
visualize.py        3-panel GT|Global|Specialist images         → results/viz/
analyze_metadata.py BDD100K condition distribution analysis     → results/metadata/
```

---

## Model Weights

Weights are not stored in this repo. They are produced by running the pipeline above,
or available from the repo owner.

| File | Purpose |
|------|---------|
| `checkpoints/distilled/distilled.pt` | Post-distillation YOLOv8n backbone |
| `checkpoints/global/weights/best.pt` | Global fine-tuned model |
| `checkpoints/{cond}/weights/best.pt` | Per-condition specialist (×5) |
