# ThaoOCR — Khmer OCR Recognition System

A deep learning OCR system for Khmer script recognition, built with PyTorch.

## Architecture

**ThaoNet** — a modular CNN + Transformer pipeline for text line recognition.

```
Image → [Backbone] → [Neck] → [Head] → Text
         ResNet CNN   Linear   Transformer + CTC
```

| Component | Type | Description |
|-----------|------|-------------|
| **Backbone** | `resnet` | 4-stage CNN with residual blocks |
| | `lightweight` | 3-stage plain CNN (faster) |
| **Neck** | `linear` | Linear + LayerNorm + Dropout |
| **Head** | `transformer_ctc` | Transformer Encoder + CTC |
| | `bilstm_ctc` | BiLSTM + CTC (lighter alternative) |

Components are swappable via `config.yml` — no code changes needed.

### Model Presets

| Preset | Height | Transformer | Layers | ~Params |
|--------|--------|-------------|--------|---------|
| `small` | 32px | d=128 | 2 | ~1M |
| `base` | 48px | d=256 | 4 | ~8M |
| `large` | 48px | d=384 | 6 | ~20M |
| `xlarge` | 64px | d=512 | 8 | ~45M |

## Project Structure

```
thaocr/
├── config.yml               # Training configuration
├── requirements.txt          # Python dependencies
│
├── src/                      # Core library
│   ├── config.py             #   Configuration & presets
│   ├── model.py              #   ThaoNet model + component registry
│   ├── dataset.py            #   Data loading & augmentation
│   ├── vocab.py              #   Character vocabulary (Khmer + Latin)
│   └── utils.py              #   Image processing & metrics
│
├── tools/                    # Runnable scripts
│   ├── train/
│   │   ├── train.py          #   Training script
│   │   └── evaluate.py       #   Evaluation script
│   └── export/
│       ├── export_onnx.py    #   Export to ONNX
│       └── predict.py        #   Single image inference
│
├── checkpoints/              # Saved models (gitignored)
└── khmer-ocr-synth-v1/      # Dataset (gitignored)
```

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/yourusername/thaocr.git
cd thaocr

# Create conda environment
conda create -n thaocr python=3.10 -y
conda activate thaocr

# Install PyTorch (adjust for your CUDA version)
# See: https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### 2. Download the dataset

Download the Khmer OCR synthetic dataset from Kaggle:

```bash
# Option 1: Using Kaggle CLI
kaggle datasets download thareah/khmer-ocr-synth-v1
unzip khmer-ocr-synth-v1.zip -d khmer-ocr-synth-v1

# Option 2: Direct download
# https://www.kaggle.com/api/v1/datasets/download/thareah/khmer-ocr-synth-v1
```

The dataset should be placed in the project root:

```
thaocr/
└── khmer-ocr-synth-v1/
    ├── train.txt          # 339K training samples
    ├── val.txt            # 37K validation samples
    ├── train/             # Training images
    │   ├── img_000.jpg
    │   └── ...
    └── val/               # Validation images
        ├── img_000.jpg
        └── ...
```

## Usage

### Training

```bash
# Train with default config
python tools/train/train.py config.yml

# Train with custom settings
python tools/train/train.py config.yml --epochs 100 --batch-size 64

# Resume from checkpoint
python tools/train/train.py config.yml --resume checkpoints/latest.pt
```

### Configuration

Edit `config.yml` to customize:

```yaml
model:
  name: "base"                 # Preset: small, base, large, xlarge
  backbone:
    type: "resnet"             # Swap backbone type
  head:
    type: "transformer_ctc"    # Swap head type

train:
  epochs: 50
  batch_size: 32
  lr: 3.0e-4
```

### Evaluation

```bash
python tools/train/evaluate.py \
  --checkpoint checkpoints/best.pt \
  --data khmer-ocr-synth-v1/val.txt \
  --data-root khmer-ocr-synth-v1
```

### Inference

```bash
# Single image
python tools/export/predict.py \
  --checkpoint checkpoints/best.pt \
  --image path/to/line.png

# Directory of images
python tools/export/predict.py \
  --checkpoint checkpoints/best.pt \
  --dir path/to/images/
```

### Export to ONNX

```bash
python tools/export/export_onnx.py \
  --checkpoint checkpoints/best.pt \
  --output model.onnx
```

## Extending the Architecture

ThaoNet uses a **plugin registry**. To add a new component, just write a class in `src/model.py`:

```python
@register_backbone("mobilenet")
class MobileNetBackbone(nn.Module):
    h_divisor = 16              # How much H shrinks
    w_divisor = 4               # How much W shrinks

    def __init__(self, channels=(64, 128, 256, 256), **kwargs):
        super().__init__()
        self.out_channels = channels[-1]
        # ... your implementation ...

    def forward(self, x):
        return x
```

Then use it in `config.yml`:

```yaml
backbone:
  type: "mobilenet"   # Automatically available!
```

No need to edit `train.py`, `config.py`, or any other file.

## Checkpoints

The training script saves three types of checkpoints:

| File | When | Purpose |
|------|------|---------|
| `best.pt` | When CER improves | Best accuracy model |
| `latest.pt` | Every epoch | Most recent model |
| `epoch_005.pt` | Every N epochs | Historical snapshots |

## Data Format

Annotation files use tab-separated format:

```
train/img_000.jpg	ថាផ្អែកលើព័ត៌មានដែល
train/img_001.jpg	ទទួលបាន
```

## License

MIT


python tools/train/train.py config.yml --resume checkpoints/best.pt