# DAFT-Drive

Implementations of **Data-Aware Fine-Tuning (DAFT)** with adaptive specialist routing,
applied to image segmentation across multiple domains.

Based on: Pfefferle et al., *DAFT: Data-Aware Fine-Tuning of Foundation Models for
Efficient and Effective Medical Image Segmentation*, CVPR 2024 Workshop — 1st place.

---

## Repository Structure

```
daft-drive/
├── DAFT-MedSAM-on-Laptop/    ← Medical image segmentation (this paper, complete)
│   ├── model.py
│   ├── train.py
│   ├── demo.py
│   ├── data/
│   │   ├── test_imgs/        ← 10 demo input files (included)
│   │   ├── test_gts/         ← 10 ground-truth masks (included)
│   │   └── test_demo_segs/   ← paper reference predictions (included)
│   └── checkpoints/          ← model weights (not in repo, see below)
│
└── DAFT-BDD100K/             ← Autonomous driving on BDD100K (in progress)
```

---

## What is DAFT?

Instead of one model for everything, DAFT trains a **specialist model per data subset**
and routes each input to the right specialist at inference time — automatically.

```
Input image
    |
    v
[Routing / Meta-model]       reads filename or image metadata
    |         |         |
    v         v         v
[Specialist A] [Specialist B] [Specialist C]
    |
    v
Segmentation mask
```

---

## How SAM Works

**SAM (Segment Anything Model)** by Meta is a prompt-based segmentation model.
You give it a bounding box and it figures out what to segment inside it.

```
Image ──► [1. Image Encoder] ──► image features ──►
                                                    [3. Mask Decoder] ──► mask + IoU
Bounding Box ──► [2. Prompt Encoder] ──► prompt embeddings ──►
```

**Image Encoder** — compresses the image into a feature map. Runs once per image.
**Prompt Encoder** — converts the bounding box into embeddings. Always frozen.
**Mask Decoder** — combines features + prompt → segmentation mask + confidence score.

The original SAM encoder (ViT-H, 632M params) is too slow for a laptop.
DAFT replaces it with **EfficientViT-L0** (~6x faster) via knowledge distillation.

---

## The Three-Step DAFT Pipeline

```
Step 1  Knowledge Distillation
        Teacher: LiteMedSAM (TinyViT)  [frozen]
        Student: EfficientViT-L0        [trained to copy teacher feature maps]
        Loss:    MSE between feature maps
        Output:  distilled encoder weights

Step 2  General Fine-Tuning
        Start:   distilled encoder + LiteMedSAM decoder  (merge_weights.py)
        Train:   on ALL modalities combined
        Output:  checkpoints/global.pth   ← already done, we downloaded this

Step 3  Data-Aware Fine-Tuning  (the DAFT step)
        Start:   global.pth  (not distilled.pth — global already knows medicine)
        Train:   one specialist per modality, each sees only its own data
        Route:   filename prefix → correct specialist at inference time
        Output:  checkpoints/US/best.pth
                 checkpoints/XRay/best.pth  etc.
```

**Loss function** (Steps 2 and 3):
```
Total Loss = Dice Loss + BCE Loss + IoU Loss   (equal weights, 1:1:1)
```

---

## How NPZ Files Work

`.npz` = a ZIP of NumPy arrays, like a dictionary saved to disk.

```python
import numpy as np

data  = np.load("file.npz", allow_pickle=True)
image = data["imgs"]    # array stored under key "imgs"
mask  = data["gts"]
print(data.files)       # list all keys
```

**Training files** — image + mask in one file:
```
imgs  →  (H, W, 3)   uint8   2D RGB image
      →  (D, H, W)   uint8   3D volume (D slices)
gts   →  (H, W)      uint8   label map  (0=background, 1/2/3...=structures)
      →  (D, H, W)   uint8   same for 3D
```

**Test files** — split across two folders:
```
test_imgs/X.npz:  imgs + boxes  (bounding box prompts for SAM)
test_gts/X.npz:   gts           (ground truth, used only for evaluation)
```

---

## Results on the 10 Demo Files

| File | Our model (global.pth) | Paper specialist | Notes |
|------|------------------------|------------------|-------|
| CXR (chest X-ray) | 0.914 | 0.925 | Close |
| Dermoscopy | **0.978** | 0.971 | We beat specialist |
| Endoscopy | 0.962 | 0.987 | Small gap |
| Fundus | 0.970 | 0.982 | Small gap |
| Mammography | **0.778** | 0.748 | We beat specialist |
| Microscopy | 0.912 | 0.957 | Moderate gap |
| OCT | **0.888** | 0.875 | We beat specialist |
| Ultrasound | **0.900** | 0.875 | We beat specialist |
| CT (3D) | **0.836** | 0.720 | We beat specialist |
| MR (3D) | 0.389 | **0.899** | Gap — MR needs specialist |

MR is the clearest case for DAFT: the global model struggles,
the MR specialist (trained only on MRI data) scores 0.899.

---

## Model Weights

Weights are **not stored in this repo** (too large for GitHub).

| File | Size | Purpose | Where to get it |
|------|------|---------|-----------------|
| `checkpoints/global.pth` | 133 MB | Main inference model | See download note below |
| `lite_medsam.pth` | ~30 MB | Teacher for distillation | See download note below |
| `distilled.pth` | ~50 MB | Post-distillation encoder | Produced by `distill.py` |

**Download note:** The weights come from the original paper's release.
Place them in `DAFT-MedSAM-on-Laptop/` (for `lite_medsam.pth`) and
`DAFT-MedSAM-on-Laptop/checkpoints/` (for `global.pth`).
Ask the repo owner for the shared download link.

---

## Environment Setup

```bash
# Create and activate conda environment
conda create -n daft python=3.10 -y
conda activate daft

# PyTorch with CUDA
# Check your CUDA version first: nvidia-smi (top-right corner shows CUDA Version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.x
# CPU only (slow but works):
pip install torch torchvision

# Core dependencies
pip install monai opencv-python pandas numpy tqdm matplotlib

# Model architecture packages
pip install git+https://github.com/mit-han-lab/efficientvit.git
pip install git+https://github.com/facebookresearch/segment-anything.git
```

**Windows only — triton patch** (triton is Linux-only, this replaces it):

```bash
# Find where efficientvit is installed
python -c "import efficientvit, os; print(os.path.dirname(efficientvit.__file__))"
# It will print something like:
# C:\Users\...\miniconda3\envs\daft\Lib\site-packages\efficientvit
# Use that path below:

cat > "<ABOVE_PATH>/models/nn/triton_rms_norm.py" << 'EOF'
"""Pure-PyTorch fallback -- triton is Linux-only, not needed for our use case."""
import torch
from torch.autograd import Function

class TritonRMSNorm2dFunc(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        rms = x.pow(2).mean(dim=1, keepdim=True).add(eps).sqrt()
        y = x / rms
        if weight is not None:
            y = y * weight.view(1, -1, 1, 1)
        if bias is not None:
            y = y + bias.view(1, -1, 1, 1)
        ctx.save_for_backward(x, weight, bias, rms)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, rms = ctx.saved_tensors
        grad_x = grad_output / rms
        grad_weight = (grad_output * x / rms).sum(dim=[0, 2, 3]) if weight is not None else None
        grad_bias = grad_output.sum(dim=[0, 2, 3]) if bias is not None else None
        return grad_x, grad_weight, grad_bias, None
EOF
```

**Verify setup:**
```bash
python -c "
import torch
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
m = create_efficientvit_sam_model('efficientvit-sam-l0', pretrained=False)
print('Model OK -', sum(p.numel() for p in m.parameters())//1_000_000, 'M params')
"
```

---

## Model Caching

By default, model weights are loaded directly from local `.pth` files — no internet
caching is involved. When you call `MedSAM("checkpoints/global.pth")` it reads
the file from disk every time.

**To avoid reloading the model on every run** (useful when running inference in a
loop or a notebook), keep the model object in memory:

```python
from model import MedSAM
import torch

# Load once
model = MedSAM("checkpoints/global.pth").to("cuda").eval()

# Reuse for many images — no reload
for npz_path in my_files:
    process_2d(model, npz_path, pred_dir, "cuda")
```

The `inference.py` script already does this via `_MODEL_CACHE` — each specialist
is loaded once and reused for all files of that modality.

**Cache location for pip packages** (efficientvit downloads nothing automatically):
```bash
# PyTorch model cache (if torch.hub is ever used):
# Windows: C:\Users\<you>\.cache\torch\hub\
# Linux:   ~/.cache/torch/hub/

# To change the cache directory:
set TORCH_HOME=D:\my_cache    # Windows
export TORCH_HOME=/data/cache  # Linux
```

---

## Running the Demo

```bash
conda activate daft
cd DAFT-MedSAM-on-Laptop/

# Fast: inference + evaluation only (~1-2 min on GPU)
python demo.py --skip_training

# Full pipeline: visualise + DAFT training demo + inference + evaluate
python demo.py
```

**Output:**
```
data/viz/            — visualisation plots (4 panels per file)
data/demo_preds/     — predicted segmentation masks
results_demo.csv     — DSC scores per file
results_demo.png     — bar chart vs paper reference
```

---

## Running Individual Scripts

```bash
cd DAFT-MedSAM-on-Laptop/

# Visualise demo data
python visualize.py
python visualize.py --file 2DBox_US    # one file only

# Inference with the global model
python inference.py --img_dir data/test_imgs --pred_dir data/test_preds

# Evaluate predictions
python evaluate.py --pred_dir data/test_preds --gt_dir data/test_gts --out_csv results.csv

# Distillation (needs full training data for real results; demo data for pipeline check)
python distill.py \
  --train_csv data/datasplit/demo_train.csv \
  --val_csv   data/datasplit/demo_val.csv   \
  --lite_medsam lite_medsam.pth \
  --epochs 2 --batch_size 1 --num_workers 0

# Merge distilled encoder + LiteMedSAM decoder
python merge_weights.py \
  --lite_medsam lite_medsam.pth \
  --encoder     checkpoints/distilled_encoder.pth \
  --output      checkpoints/merged.pth

# Train a DAFT specialist (example: MR)
python train.py \
  --train_csv data/datasplit/modalities/3D.train.csv \
  --val_csv   data/datasplit/modalities/3D.val.csv   \
  --weights   checkpoints/global.pth \
  --name      3D \
  --epochs    20 --batch_size 4
```

---

## Full Workflow Reference

```
distill.py        EfficientViT learns TinyViT features    → checkpoints/distilled_encoder.pth
merge_weights.py  Encoder + LiteMedSAM decoder            → checkpoints/merged.pth
prepare_data.py   Build train/val CSV splits              → data/datasplit/
train.py          Global fine-tune (all modalities)       → checkpoints/global/best.pth
train.py ×11      DAFT specialists (one per modality)     → checkpoints/US/best.pth ...
inference.py      Filename routing → correct specialist   → data/test_preds/
evaluate.py       Compute DSC                             → results.csv
```

Steps 1–4 are already complete: `global.pth` is the output.

---

## Next Steps

### Get training data without downloading everything

- **MR subset only** — closes the biggest gap (0.389 → ~0.899 expected)
  Download just the MR folder from the competition dataset (Zenodo)
- **Public alternatives:**
  - CT/MR: [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
  - X-ray: [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)

### Train on the cloud (no local disk space needed)

| Platform | GPU | Disk | Cost |
|----------|-----|------|------|
| Kaggle | P100 16GB | 30 GB | Free |
| Google Colab | T4 15GB | 15 GB | Free (session limits) |
| RunPod / Vast.ai | A100 / RTX 4090 | Flexible | ~$0.50–1.50/hr |

```bash
# On any cloud machine:
git clone https://github.com/YOUR_USERNAME/daft-drive.git
cd daft-drive/DAFT-MedSAM-on-Laptop
pip install torch torchvision monai opencv-python pandas numpy tqdm matplotlib
pip install git+https://github.com/mit-han-lab/efficientvit.git
pip install git+https://github.com/facebookresearch/segment-anything.git
# place global.pth in checkpoints/ then:
python train.py --train_csv ... --val_csv ... --weights checkpoints/global.pth --name MR
```

### OpenVINO for faster CPU inference (6.5× speedup)

```bash
pip install openvino

# Export to ONNX
python -c "
import torch
from model import MedSAM
model = MedSAM('checkpoints/global.pth').eval()
torch.onnx.export(model,
    (torch.zeros(1,3,256,256), torch.zeros(1,1,1,4)),
    'global.onnx', opset_version=17,
    input_names=['image','box'], output_names=['logits','iou'])
"

# Convert to OpenVINO IR
mo --input_model global.onnx --output_dir checkpoints/openvino/
# Produces: checkpoints/openvino/global.xml + global.bin
```

---

## Adapting to Autonomous Driving (BDD100K)

| What | Medical (this project) | Driving (BDD100K) |
|------|----------------------|-------------------|
| Data format | NPZ (NumPy archives) | JPEG + JSON annotations |
| Modality splits | CT / MRI / X-ray / ... | Day / Night / Rain / Fog / ... |
| Routing | Filename prefix | JSON metadata (time_of_day, weather) |
| Input size | 256×256 | 1024×1024 (or 512×512) |
| Bounding boxes | Provided in NPZ | Provided in JSON annotations |
| What to specialise | Imaging modality | Driving conditions |

Best candidates for DAFT specialisation on BDD100K:
night-time, heavy rain, and dense urban scenes — these differ most
from the average training distribution and benefit most from a specialist.
