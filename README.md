# End-to-End CV Pipeline with PyTorch

A complete computer vision pipeline for training, evaluating, and deploying deep learning models. Features swappable models and datasets via YAML configuration, comprehensive metrics computation, and ONNX quantization for deployment.

## Overview

This project demonstrates an end-to-end workflow for image classification:

- **Part 1**: Train and evaluate models with detailed metrics (per-class accuracy, precision, recall, F1, confusion matrices)
- **Part 2**: Export to ONNX and compare model variants (FP32, INT8 dynamic quantization, INT8 static quantization)

## Features

✅ **Modular Architecture**
- Swap models and datasets via YAML configuration
- No code changes required for different experiments

✅ **Comprehensive Metrics**
- Per-class: accuracy, precision, recall, F1-score, support
- Global: macro, micro, weighted averages
- Confusion matrices with visualizations

✅ **Deployment Ready**
- ONNX FP32 export
- Dynamic INT8 quantization
- Static INT8 quantization with calibration
- Model artifact comparison

✅ **Organized Results**
- `part_1_results/`: Training metrics and confusion matrices
- `part_2_results/`: Deployment comparison tables

## Project Structure

```
├── configs/
│   ├── model.yaml          # Model architecture configuration
│   └── train.yaml          # Training hyperparameters & dataset config
├── data/                   # Downloaded datasets (auto-created)
├── results/                # Training checkpoints (best_model.pt)
├── part_1_results/         # Part 1 outputs (metrics, confusion matrices)
├── part_2_results/         # Part 2 outputs (ONNX variants, comparison)
├── utils/
│   ├── metrics.py          # Metrics computation (sklearn-based)
│   └── seed.py             # Reproducibility utilities
├── config_loader.py        # YAML configuration loader
├── datasets.py             # DataLoader factory
├── models.py               # Model architecture definitions
├── train.py                # Main training script
├── eval.py                 # Evaluation utilities
├── part2_deploy.py         # ONNX export & quantization
└── requirements.txt        # Python dependencies
```

## Installation

### 1. Clone and navigate

```bash
git clone https://github.com/AbdullahTheScientist/End-to-end-CV-pipeline-with-pytorch.git
cd End-to-end-CV-pipeline-with-pytorch
```

### 2. Create virtual environment (recommended)

```bash
python -m venv cv
# Windows
cv\Scripts\activate
# macOS/Linux
source cv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Core packages:**
- `torch`, `torchvision` — Deep learning framework
- `scikit-learn` — Metrics computation
- `pyyaml` — Configuration loading
- `pandas` — Results formatting
- `matplotlib`, `seaborn` — Visualization
- `onnx`, `onnxruntime` — Model export and inference (optional for Part 2)

## Configuration

### Model Configuration (`configs/model.yaml`)

```yaml
arch: tinycnn                    # Options: tinycnn, customcnn, convnext_tiny, vit, fasterrcnn_backbone
pretrained: True                 # Use pretrained weights (if supported)
num_classes: 10                  # Auto-overridden by dataset
freeze_backbone: False           # Freeze feature extractor for transfer learning
```

### Training Configuration (`configs/train.yaml`)

```yaml
dataset: cifar10                 # Options: cifar10, cifar100, stl10
epochs: 3
batch_size: 16
lr: 0.001
weight_decay: 0.0
use_amp: False                   # Mixed precision (useful on GPU)
use_scheduler: False             # Cosine annealing LR scheduler
use_grad_clip: False             # Gradient clipping
seed: 42
num_workers: 0                   # DataLoader workers (increase on GPU)
results_dir: results
```

## Usage

### Part 1: Train & Evaluate

Run training with your configured model and dataset:

```bash
python train.py
```

**What happens:**
1. Loads `configs/model.yaml` and `configs/train.yaml`
2. Creates or downloads dataset
3. Trains the model for specified epochs
4. Saves best checkpoint to `results/best_model.pt`
5. Evaluates best model and saves:
   - `part_1_results/cn_{model_arch}_{dataset}.png` — Confusion matrix image
   - `part_1_results/summary.txt` — Appends experiment metrics table

**Output example:**
```
Experiment: tinycnn | Dataset: cifar10
         accuracy  precision    recall        f1  support
class 0   0.823      0.801     0.814     0.807      1000
class 1   0.891      0.876     0.889     0.882      1000
...

          precision    recall        f1
macro      0.85      0.85      0.85
micro      0.85      0.85      0.85
weighted   0.85      0.85      0.85
```

### Part 2: ONNX Export & Quantization

After training, compare model variants:

```bash
python part2_deploy.py
```

**What happens:**
1. Loads best checkpoint from `results/best_model.pt`
2. Exports FP32 ONNX model
3. Creates dynamic INT8 quantized ONNX (if `onnxruntime` available)
4. Creates static INT8 quantized ONNX with calibration data
5. Evaluates all variants on validation set
6. Saves `part_2_results/summary.txt` with comparison table

**Output example:**
```
      Model Artifact  Accuracy  Macro F1  Weighted F1
     PyTorch FP32      0.8500    0.8450      0.8500
ONNX INT8 Dynamic      0.8490    0.8440      0.8495
 ONNX INT8 Static      0.8495    0.8445      0.8500
```

## Supported Models

| Architecture | Speed | Pretrained | Notes |
|---|---|---|---|
| `tinycnn` | ⚡ Fast | N/A | Minimal CNN for pipeline testing |
| `customcnn` | ⚡ Fast | N/A | Lightweight custom CNN |
| `convnext_tiny` | 🔄 Medium | ✅ | Modern efficient architecture |
| `vit` | 🐌 Slow | ✅ | Vision Transformer (GPU recommended) |
| `fasterrcnn_backbone` | 🐌 Slow | ✅ | Detection backbone (GPU recommended) |

## Supported Datasets

| Dataset | Classes | Samples | Auto-Download |
|---|---|---|---|
| `cifar10` | 10 | 60,000 | ✅ Yes |
| `cifar100` | 100 | 60,000 | ✅ Yes |
| `stl10` | 10 | 113,000 | ✅ Yes |

## Experiment Examples

### Example 1: Quick Pipeline Test
```yaml
# configs/model.yaml
arch: tinycnn
pretrained: False
freeze_backbone: False

# configs/train.yaml
dataset: cifar10
epochs: 1
batch_size: 16
```
```bash
python train.py          # ~30 seconds
python part2_deploy.py   # ~10 seconds
```

### Example 2: Transfer Learning with ConvNeXt
```yaml
# configs/model.yaml
arch: convnext_tiny
pretrained: True         # Load ImageNet weights
freeze_backbone: True    # Freeze feature extractor

# configs/train.yaml
dataset: cifar10
epochs: 10
batch_size: 64
lr: 0.0001               # Lower LR for fine-tuning
```
```bash
python train.py
python part2_deploy.py
```

### Example 3: Compare Multiple Datasets
Edit `configs/train.yaml` and run multiple times:
```bash
# cifar10
python train.py
# cifar100
sed -i 's/dataset: cifar10/dataset: cifar100/' configs/train.yaml
python train.py
# cifar100 results appended to part_1_results/summary.txt
```

## Results Structure

### Part 1 Results (`part_1_results/`)

```
part_1_results/
├── cn_tinycnn_cifar10.png        # Confusion matrix for experiment
├── cn_customcnn_cifar100.png     # Another experiment
└── summary.txt                   # Aggregated metrics for all experiments
```

**summary.txt format:**
- One section per experiment
- Per-class metrics table (5 columns: accuracy, precision, recall, f1, support)
- Global aggregates (macro, micro, weighted)
- Blank line between experiments for readability

### Part 2 Results (`part_2_results/`)

```
part_2_results/
├── tinycnn_cifar10_fp32.onnx                  # Original model
├── tinycnn_cifar10_int8_dynamic.onnx          # Dynamic quantized
├── tinycnn_cifar10_int8_static.onnx           # Static quantized
└── summary.txt                                # Comparison table
```

## Key Code Components

### Load Configuration
```python
from config_loader import load_configs, print_config

config = load_configs(
    model_config_path="configs/model.yaml",
    train_config_path="configs/train.yaml"
)
print_config(config)  # Pretty-print loaded config
```

### Compute Metrics
```python
from utils.metrics import compute_metrics

metrics = compute_metrics(labels, predictions)
# Returns:
# - metrics["per_class"]: DataFrame (accuracy, precision, recall, f1, support per class)
# - metrics["global"]: DataFrame (macro, micro, weighted aggregates)
# - metrics["accuracy"]: float (overall accuracy)
# - metrics["confusion_matrix"]: numpy array
```

### Train & Evaluate
```python
from train import train

config = load_configs()
train(config)  # Trains model and saves Part 1 results automatically
```

### Export & Quantize
```bash
python part2_deploy.py  # Handles all ONNX operations
```

## Notes

- **Data**: Downloaded to `data/` folder (ignored in git). Redownloads automatically if missing.
- **GPU**: Optional but recommended for larger models (convnext, ViT, FasterRCNN). Set `num_workers > 0` for faster data loading on GPU systems.
- **Reproducibility**: Seed set in config ensures consistent results (default: 42).
- **Quantization**: Requires `onnxruntime` package. If unavailable, FP32 ONNX is still exported.

## Troubleshooting

**Out of Memory**
- Reduce `batch_size` in `configs/train.yaml`
- Use lighter models (`tinycnn`, `customcnn`)
- Set `use_amp: True` for mixed precision (GPU only)

**Slow Training**
- Reduce `epochs` for testing
- Use `fast_dev_run: True` in code (for debugging)
- Switch to `tinycnn` to verify pipeline quickly

**ONNX Quantization Fails**
- Install `onnxruntime`: `pip install onnxruntime`
- FP32 ONNX export still works without it
- Part 2 will print warnings and skip unavailable steps

## License

MIT License — See LICENSE file

## Citation

If you use this project, cite as:

```
@software{cv_pipeline_2026,
  title={End-to-End CV Pipeline with PyTorch},
  author={Abdullah},
  year={2026},
  url={https://github.com/AbdullahTheScientist/End-to-end-CV-pipeline-with-pytorch}
}
```

---

**Questions?** Open an issue on GitHub or check the [PyTorch documentation](https://pytorch.org/docs/).
