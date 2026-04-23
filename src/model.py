"""
model.py
========
A lightweight CRNN (CNN + BiLSTM + CTC) for Khmer printed-text OCR.

Architecture
------------
Backbone  : MobileNetV3-Small truncated at the last feature map  (fast, small)
Neck      : Adaptive average-pool height → 1, keep width sequence
Head      : 2-layer BiLSTM  →  linear projection  →  CTC

The model is intentionally simple and trainable on a single mid-range GPU.
Swap the backbone for ResNet-34 or EfficientNet if you have more compute.
"""

import math
import torch
import torch.nn as nn
import torchvision.models as tvm
from typing import Tuple


# ── BACKBONE ──────────────────────────────────────────────────────────────────

class MobileNetV3Backbone(nn.Module):
    """
    MobileNetV3-Small truncated before the classifier head.
    Output: [B, C, H', W'] where H' is small (≈ target_h / 8).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = tvm.mobilenet_v3_small(
            weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Keep only the feature extractor (remove AdaptiveAvgPool + classifier)
        self.features = base.features      # 16 InvertedResidual blocks
        self.out_channels = 576            # final channel count of MobileNetV3-Small

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)            # [B, 576, H', W']


class TinyResNetBackbone(nn.Module):
    """
    Lighter option: first 3 ResNet-18 layers (stride 8 total).
    Output channels: 128.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = tvm.resnet18(
            weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool
        )
        self.layer1 = base.layer1   # 64 ch
        self.layer2 = base.layer2   # 128 ch
        self.out_channels = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x                           # [B, 128, H', W']


# ── CRNN MODEL ────────────────────────────────────────────────────────────────

class KhmerOCRModel(nn.Module):
    """
    CRNN for variable-width Khmer OCR with CTC loss.

    Parameters
    ----------
    vocab_size   : int   — number of output classes (including CTC blank at 0)
    backbone     : str   — 'mobilenet' | 'resnet_tiny'
    hidden_size  : int   — BiLSTM hidden units per direction
    num_layers   : int   — number of BiLSTM layers
    pretrained   : bool  — load ImageNet weights for backbone
    dropout      : float — dropout between LSTM layers
    """

    def __init__(
        self,
        vocab_size:  int,
        backbone:    str   = "mobilenet",
        hidden_size: int   = 256,
        num_layers:  int   = 2,
        pretrained:  bool  = True,
        dropout:     float = 0.1,
    ):
        super().__init__()

        # ── Backbone
        if backbone == "mobilenet":
            self.cnn = MobileNetV3Backbone(pretrained=pretrained)
        elif backbone == "resnet_tiny":
            self.cnn = TinyResNetBackbone(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone!r}. "
                             "Choose 'mobilenet' or 'resnet_tiny'.")

        c_in = self.cnn.out_channels

        # ── Neck: collapse height → 1  (adaptive pooling)
        self.pool = nn.AdaptiveAvgPool2d((1, None))   # height → 1, width kept

        # ── Head: BiLSTM sequence modelling
        self.rnn = nn.LSTM(
            input_size=c_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── Output projection
        self.head = nn.Linear(hidden_size * 2, vocab_size)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : [B, 1, H, W] or [B, 3, H, W]
        returns: [T, B, V]   — time-first for CTC loss
        """
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)  # grayscale → pseudo-RGB
        feat  = self.cnn(x)               # [B, C, H', W']
        feat  = self.pool(feat)           # [B, C,  1, W']
        feat  = feat.squeeze(2)           # [B, C, W']
        feat  = feat.permute(0, 2, 1)     # [B, W', C]  == [B, T, C]

        out, _ = self.rnn(feat)           # [B, T, 2*H]
        out   = self.head(out)            # [B, T, V]
        out   = out.permute(1, 0, 2)      # [T, B, V]  — time-first
        return out

    # ------------------------------------------------------------------
    def count_params(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total":     total,
            "trainable": trainable,
            "total_M":   total / 1e6,
        }


# ── OCRRecModel (ResNet → Transformer → CTC) ──────────────────────────────────

class _ResBlock(nn.Module):
    """Basic residual block with configurable spatial stride."""

    def __init__(self, in_ch: int, out_ch: int, stride=(1, 1)):
        super().__init__()
        s = stride if isinstance(stride, tuple) else (stride, stride)
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        need_proj = (s != (1, 1)) or (in_ch != out_ch)
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=s, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if need_proj else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.skip(x))


class _ResNetBackbone(nn.Module):
    """4-stage ResNet backbone built for OCR (grayscale input).

    Stride schedule (H × W):
        stem  1×1  channels[0]
        s1    2×1  channels[0]   — halve height, keep width
        s2    2×1  channels[1]   — halve height, keep width
        s3    2×2  channels[2]   — halve height, halve width
        s4    1×1  channels[3]   — deeper features only

    For target_h=48: H goes 48→24→12→6→6; W compressed 2× in s3.
    AdaptiveAvgPool(1, None) after this gives T ≈ W/2 time steps.
    """

    def __init__(self, channels=(64, 128, 256, 256)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.s1 = _ResBlock(channels[0], channels[0], stride=(2, 1))
        self.s2 = _ResBlock(channels[0], channels[1], stride=(2, 1))
        self.s3 = _ResBlock(channels[1], channels[2], stride=(2, 2))
        self.s4 = _ResBlock(channels[2], channels[3], stride=(1, 1))
        self.out_channels = channels[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 3:  # pseudo-RGB → grayscale
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        return x  # [B, C, H', W']


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (time-first layout)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.drop = nn.Dropout(p=dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))  # [max_len, 1, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:x.size(0)])


class OCRRecModel(nn.Module):
    """
    ThaoOCR recognition model.

    Pipeline: [B,1,H,W] → ResNet backbone → height pool → Linear neck
              → Positional encoding → Transformer encoder → CTC head
              → [T,B,V]

    Parameters
    ----------
    vocab_size         : number of output classes (CTC blank = 0)
    backbone_channels  : (C1, C2, C3, C4) for each ResNet stage
    d_model            : Transformer hidden dim
    nhead              : attention heads
    num_layers         : Transformer encoder layers
    dim_feedforward    : FFN hidden dim inside each Transformer layer
    neck_dropout       : dropout in Linear neck
    head_dropout       : dropout inside Transformer layers
    """

    def __init__(
        self,
        vocab_size:        int,
        backbone_channels: tuple = (64, 128, 256, 256),
        d_model:           int   = 256,
        nhead:             int   = 4,
        num_layers:        int   = 4,
        dim_feedforward:   int   = 1024,
        neck_dropout:      float = 0.1,
        head_dropout:      float = 0.1,
    ):
        super().__init__()
        self.backbone = _ResNetBackbone(channels=backbone_channels)
        c_in = self.backbone.out_channels

        self.pool = nn.AdaptiveAvgPool2d((1, None))  # collapse height → 1

        self.neck = nn.Sequential(
            nn.Linear(c_in, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(neck_dropout),
        )

        self.pos_enc = _PositionalEncoding(d_model, dropout=head_dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= dim_feedforward,
            dropout        = head_dropout,
            batch_first    = False,   # time-first [T, B, d_model]
            norm_first     = True,    # pre-norm: more stable training
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 1, H, W]
        returns [T, B, V] — time-first logits for CTC loss
        """
        feat = self.backbone(x)       # [B, C, H', W']
        feat = self.pool(feat)        # [B, C,  1, W']
        feat = feat.squeeze(2)        # [B, C, W']
        feat = feat.permute(0, 2, 1)  # [B, T, C]
        feat = self.neck(feat)        # [B, T, d_model]
        feat = feat.permute(1, 0, 2)  # [T, B, d_model]
        feat = self.pos_enc(feat)
        feat = self.transformer(feat) # [T, B, d_model]
        return self.head(feat)        # [T, B, V]

    @classmethod
    def from_config(cls, cfg, vocab_size: int) -> "OCRRecModel":
        """Instantiate from a ModelConfig object (config.py)."""
        return cls(
            vocab_size        = vocab_size,
            backbone_channels = tuple(cfg.backbone.channels),
            d_model           = cfg.head.d_model,
            nhead             = cfg.head.nhead,
            num_layers        = cfg.head.num_layers,
            dim_feedforward   = cfg.head.dim_feedforward,
            neck_dropout      = cfg.neck.dropout,
            head_dropout      = cfg.head.dropout,
        )

    def count_params(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "total_M": total / 1e6}