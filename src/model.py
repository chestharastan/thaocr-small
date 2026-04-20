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