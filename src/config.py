"""
ThaoOCR — Configuration
All hyperparameters, paths, and settings in one place.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VocabConfig:
    """Character vocabulary settings."""
    # CTC blank is always index 0
    blank_token: str = "[BLANK]"
    pad_token: str = "[PAD]"

    # ─── Character sets ───────────────────────────────────────
    # Latin + digits + punctuation
    latin_chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    digit_chars: str = "0123456789"
    punct_chars: str = ".,!?()[]-+/:;\"' "

    # Khmer consonants
    khmer_consonants: str = (
        "\u1780\u1781\u1782\u1783\u1784"   # ក ខ គ ឃ ង
        "\u1785\u1786\u1787\u1788\u1789"   # ច ឆ ជ ឈ ញ
        "\u178A\u178B\u178C\u178D\u178E"   # ដ ឋ ឌ ឍ ណ
        "\u178F\u1790\u1791\u1792\u1793"   # ត ថ ទ ធ ន
        "\u1794\u1795\u1796\u1797\u1798"   # ប ផ ព ភ ម
        "\u1799\u179A\u179B\u179C\u179D"   # យ រ ល វ ឝ
        "\u179E\u179F\u17A0\u17A1\u17A2"   # ឞ ស ហ ឡ អ
    )

    # Khmer independent vowels
    khmer_indep_vowels: str = (
        "\u17A3\u17A4\u17A5\u17A6\u17A7"   # ឣ ឤ ឥ ឦ ឧ
        "\u17A8\u17A9\u17AA\u17AB\u17AC"   # ឨ ឩ ឪ ឫ ឬ
        "\u17AD\u17AE\u17AF\u17B0\u17B1"   # ឭ ឮ ឯ ឰ ឱ
        "\u17B2\u17B3"                     # ឲ ឳ
    )

    # Khmer dependent vowels (combining)
    khmer_dep_vowels: str = (
        "\u17B6\u17B7\u17B8\u17B9\u17BA"   # ា ិ ី ឹ ឺ
        "\u17BB\u17BC\u17BD\u17BE\u17BF"   # ុ ូ ួ ើ ឿ
        "\u17C0\u17C1\u17C2\u17C3\u17C4"   # ៀ េ ែ ៃ ោ
        "\u17C5"                           # ៅ
    )

    # Khmer signs / diacritics
    khmer_signs: str = (
        "\u17C6\u17C7\u17C8\u17C9\u17CA"   # ំ ះ ៈ ៉ ៊
        "\u17CB\u17CC\u17CD\u17CE\u17CF"   # ់ ៌ ៍ ៎ ៏
        "\u17D0\u17D1\u17D2"               # ័ ៑ ្ (coeng/subscript)
    )

    # Khmer digits
    khmer_digits: str = (
        "\u17E0\u17E1\u17E2\u17E3\u17E4"   # ០ ១ ២ ៣ ៤
        "\u17E5\u17E6\u17E7\u17E8\u17E9"   # ៥ ៦ ៧ ៨ ៩
    )

    # Khmer punctuation / symbols
    khmer_punct: str = (
        "\u17D4\u17D5\u17D6"               # ។ ៕ ៖
        "\u17D7\u17D8\u17D9\u17DA"         # ៗ ៘ ៙ ៚
    )

    # Extra symbols (zero-width, etc.) — add as needed
    extra_chars: str = "\u200B\u200C\u200D"  # ZWSP, ZWNJ, ZWJ (common in Khmer text)


# ─── Model component configs ──────────────────────────────────
@dataclass
class BackboneConfig:
    """CNN feature extractor — extracts visual features from the image."""
    type: str = "resnet"                # "resnet" (CNN with residual blocks)
    channels: List[int] = field(default_factory=lambda: [64, 128, 256, 256])


@dataclass
class NeckConfig:
    """Bridge between backbone and head — projects CNN features to model dim."""
    type: str = "linear"                # "linear" (Linear + LayerNorm + Dropout)
    dropout: float = 0.1


@dataclass
class HeadConfig:
    """Sequence modeler + classifier — reads features and outputs text."""
    type: str = "transformer_ctc"       # "transformer_ctc" (Transformer Encoder + CTC)
    d_model: int = 256                  # Hidden dimension
    nhead: int = 4                      # Attention heads
    num_layers: int = 4                 # Transformer encoder layers
    dim_feedforward: int = 1024         # FFN hidden dim
    dropout: float = 0.1


# ─── Model presets ─────────────────────────────────────────────
# Architecture: ThaoNet = ResNet Backbone → Linear Neck → Transformer+CTC Head
#
# Each preset defines a full architecture. Pick one via model.name in config.yml,
# then override individual fields if needed.
MODEL_PRESETS = {
    "small": {
        "target_h": 32,
        "backbone": {"channels": [32, 64, 128, 128]},
        "neck":     {"dropout": 0.1},
        "head":     {"d_model": 128, "nhead": 4, "num_layers": 2, "dim_feedforward": 512, "dropout": 0.1},
    },
    "base": {
        "target_h": 48,
        "backbone": {"channels": [64, 128, 256, 256]},
        "neck":     {"dropout": 0.1},
        "head":     {"d_model": 256, "nhead": 4, "num_layers": 4, "dim_feedforward": 1024, "dropout": 0.1},
    },
    "large": {
        "target_h": 48,
        "backbone": {"channels": [64, 128, 256, 512]},
        "neck":     {"dropout": 0.1},
        "head":     {"d_model": 384, "nhead": 8, "num_layers": 6, "dim_feedforward": 1536, "dropout": 0.1},
    },
    "xlarge": {
        "target_h": 64,
        "backbone": {"channels": [64, 128, 256, 512]},
        "neck":     {"dropout": 0.1},
        "head":     {"d_model": 512, "nhead": 8, "num_layers": 8, "dim_feedforward": 2048, "dropout": 0.1},
    },
}


@dataclass
class ModelConfig:
    """
    ThaoNet — OCR Recognition Architecture

    Pipeline:  Image → [Backbone] → [Neck] → [Head] → Text
                        ResNet CNN    Linear   Transformer+CTC

    Set `name` to a preset ("small", "base", "large", "xlarge") to auto-fill
    all architecture params. You can then override individual fields.
    """
    name: str = "base"                                          # Preset name
    target_h: int = 48                                          # Input image height
    backbone: BackboneConfig = field(default_factory=BackboneConfig)   # CNN feature extractor
    neck: NeckConfig = field(default_factory=NeckConfig)               # Feature projection
    head: HeadConfig = field(default_factory=HeadConfig)               # Transformer + CTC


@dataclass
class TrainConfig:
    """Training settings."""
    epochs: int = None
    batch_size: int = None
    lr: float = None
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    grad_accum_steps: int = 1          # Gradient accumulation (chunk & merge)
    num_workers: int = 4
    pin_memory: bool = True

    # Data paths
    data_root: str = "khmer-ocr-synth-v1"  # Root dir for image paths in txt files
    train_txt: str = None
    val_txt: str = None

    # Checkpoint
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5               # Save every N epochs
    resume: Optional[str] = None      # Path to checkpoint to resume from

    # Logging
    log_interval: int = 50            # Print loss every N batches

    # Augmentation
    aug_enabled: bool = True


@dataclass
class Config:
    """Master config."""
    vocab: VocabConfig = field(default_factory=VocabConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    device: str = "cuda"              # "cuda" or "cpu"  (auto-detected at runtime)


def _apply_yaml_overrides(cfg_obj, yaml_dict: dict):
    """Recursively apply YAML overrides to a dataclass instance."""
    from dataclasses import fields as dc_fields
    field_names = {f.name for f in dc_fields(cfg_obj)}
    for key, value in yaml_dict.items():
        if key not in field_names:
            print(f"⚠ Unknown config key: '{key}' — skipping")
            continue
        current = getattr(cfg_obj, key)
        # If the current attribute is a dataclass and value is a dict, recurse
        if hasattr(current, '__dataclass_fields__') and isinstance(value, dict):
            _apply_yaml_overrides(current, value)
        else:
            setattr(cfg_obj, key, value)


def get_config(yaml_path: str = None) -> Config:
    """
    Return config with defaults, optionally overridden by a YAML file.

    If model.name is set to a preset, all model params are filled from the
    preset first, then any explicit YAML overrides are applied on top.

    Usage:
        cfg = get_config()                          # defaults only
        cfg = get_config("config.yml")              # defaults + YAML overrides
    """
    cfg = Config()

    if yaml_path is None:
        raise ValueError("Error: You MUST provide a config.yml file! Using defaults is disabled.")
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_dict = yaml.safe_load(f) or {}

    # Resolve model preset FIRST, then let YAML override individual fields
    model_dict = yaml_dict.get("model", {})
    preset_name = model_dict.get("name", cfg.model.name)
    if preset_name in MODEL_PRESETS:
        preset = MODEL_PRESETS[preset_name]
        cfg.model.name = preset_name
        # Apply preset (target_h is flat, backbone/neck/head are nested)
        if "target_h" in preset:
            cfg.model.target_h = preset["target_h"]
        if "backbone" in preset:
            _apply_yaml_overrides(cfg.model.backbone, preset["backbone"])
        if "neck" in preset:
            _apply_yaml_overrides(cfg.model.neck, preset["neck"])
        if "head" in preset:
            _apply_yaml_overrides(cfg.model.head, preset["head"])
    elif preset_name != "base":
        print(f"⚠ Unknown model preset: '{preset_name}' — "
              f"available: {list(MODEL_PRESETS.keys())}")

    _apply_yaml_overrides(cfg, yaml_dict)

    # Validation: Ensure required fields are set
    missing = []
    if cfg.train.epochs is None: missing.append("train.epochs")
    if cfg.train.batch_size is None: missing.append("train.batch_size")
    if cfg.train.lr is None: missing.append("train.lr")
    if cfg.train.train_txt is None: missing.append("train.train_txt")
    if cfg.train.val_txt is None: missing.append("train.val_txt")

    if missing:
        raise ValueError(f"Missing required config fields: {missing}. Please set them in config.yml")

    return cfg
