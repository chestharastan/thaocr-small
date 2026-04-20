# Recognition Configuration (`config_rec`)

This document details the configuration parameters for the **ThaoNet** recognition model. These settings are found in `config.yml`.

## Model Architecture (`model`)

The model consists of three main components: Backbone, Neck, and Head.

### 1. Global Settings
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | "base" | Model preset to use (`small`, `base`, `large`, `xlarge`). |
| `target_h` | int | 48 | **Crucial**: Input image height in pixels. <br>• Use **32** for speed/low-res.<br>• Use **48** for standard printed text.<br>• Use **64** for **handwriting** or dense Khmer script (subscripts). |

### 2. Backbone (Feature Extractor)
Extracts visual features from the input image.

| Parameter | Options | Description |
|-----------|---------|-------------|
| `type` | `resnet` | **Recommended**. 4-stage ResNet. Deep, powerful, handles complex backgrounds well. |
| | `lightweight` | 3-stage plain CNN. 2x faster, uses less memory, but less accurate on noisy data. |
| `channels` | list[int] | e.g. `[64, 128, 256, 256]`. Number of filters in each stage. |

### 3. Neck (Bridge)
Connects the 2D CNN (Backbone) to the 1D Sequence Model (Head).

| Parameter | Options | Description |
|-----------|---------|-------------|
| `type` | `linear` | Standard projection + LayerNorm + Dropout. |
| `dropout` | float | e.g. `0.1`. Increase to `0.2` or `0.3` if overfitting on small datasets. |

### 4. Head (Sequence Modeling)
Decodes the features into text.

| Parameter | Options | Description |
|-----------|---------|-------------|
| `type` | `transformer_ctc` | **State-of-the-art**. Transformer Encoder + CTC Loss. Handles global context best. |
| | `bilstm_ctc` | BiLSTM + CTC. Traditional approach. Good for very long sequences but slower training. |
| `num_layers`| int | Depth of the Transformer/LSTM. <br>• **2 layers**: Best for **handwriting** or small datasets (prevents overfitting).<br>• **4-6 layers**: Best for large datasets (100k+ samples). |
| `d_model` | int | Hidden dimension size (e.g. 256). |
| `nhead` | int | Number of attention heads (Transformer only). |

---

## Recommended Configurations

### A. Handwriting / Dense Script (ThaoNet-HighRes)
*Best for: Handwriting, complex Khmer subscripts, variable stroke widths.*

```yaml
model:
  name: "base"
  target_h: 64            # Higher resolution to see small subscripts
  
  backbone:
    type: "resnet"        # Strong feature extraction
    channels: [64, 128, 256, 256]

  neck:
    type: "linear"
    dropout: 0.1

  head:
    type: "transformer_ctc"
    num_layers: 2         # Reduced depth to prevent overfitting
    d_model: 256
```

### B. Standard Printed Text (ThaoNet-Base)
*Best for: Books, documents, screenshots.*

```yaml
model:
  name: "base"
  target_h: 48
  
  backbone:
    type: "resnet"
    
  head:
    type: "transformer_ctc"
    num_layers: 4
```

### C. Speed / Mobile (ThaoNet-Lite)
*Best for: Real-time inference on CPU or edge devices.*

```yaml
model:
  name: "small"
  target_h: 32
  
  backbone:
    type: "lightweight"   # Faster CNN
    
  head:
    type: "bilstm_ctc"    # LSTM is often faster than Transformer on specific CPUs
    num_layers: 2
```
