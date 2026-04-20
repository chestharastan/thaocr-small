"""
ThaoOCR — Export Model to ONNX
Export a trained checkpoint to ONNX format for deployment.

Usage:
    python tools/export/export_onnx.py --checkpoint checkpoints/best.pt --output model.onnx
"""
import os
import sys
import argparse
import torch

# Add project paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import get_config

# Import load_model from the train evaluate module
sys.path.insert(0, os.path.join(ROOT, "tools", "train"))
from evaluate import load_model


def export_onnx(checkpoint_path: str, output_path: str, target_h: int = 48, sample_width: int = 256):
    """
    Export the OCR model to ONNX format.

    Args:
        checkpoint_path: Path to the .pt checkpoint
        output_path: Output .onnx file path
        target_h: Input image height
        sample_width: Sample width for tracing (dynamic axes handle variable widths)
    """
    print(f"Loading model from {checkpoint_path}...")
    model, vocab = load_model(checkpoint_path, device="cpu")
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 1, target_h, sample_width)

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=14,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch", 3: "width"},
            "logits": {0: "time", 1: "batch"},
        },
    )

    # Save vocab alongside for inference
    vocab_path = output_path.replace(".onnx", "_vocab.json")
    import json
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({
            "itos": vocab.itos,
            "blank_id": vocab.blank_id,
        }, f, ensure_ascii=False, indent=2)

    print(f"✓ ONNX model saved: {output_path}")
    print(f"✓ Vocab saved: {vocab_path}")

    # Verify
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model verification passed")
    except ImportError:
        print("⚠ Install `onnx` package to verify the exported model")
    except Exception as e:
        print(f"⚠ ONNX verification warning: {e}")


def main():
    parser = argparse.ArgumentParser(description="ThaoOCR Export to ONNX")
    parser.add_argument("config_pos", nargs="?", default=None, help="Path to YAML config file (positional)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX path")
    parser.add_argument("--width", type=int, default=1024, help="Sample width for tracing")
    args = parser.parse_args()

    config_path = args.config_pos or args.config or "config.yml"
    cfg = get_config(config_path)

    print(f"Using config: {config_path} (target_h={cfg.model.target_h})")
    
    # Pass the same config_path to load_model so it builds the correct architecture
    print(f"Loading model from {args.checkpoint}...")
    from evaluate import load_model
    model, vocab = load_model(args.checkpoint, device="cpu", config_path=config_path)
    model.eval()

    # Create dummy input based on config height
    dummy_input = torch.randn(1, 1, cfg.model.target_h, args.width)

    print(f"Exporting to ONNX: {args.output} (Opset 17)")
    
    # We use a dummy input, but dynamic_axes tells ONNX that the width can change
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        opset_version=17,              # Higher opset is generally better for Transformers
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch", 3: "width"},
            "logits": {0: "time", 1: "batch"},
        },
    )

    # Save vocab alongside for inference
    vocab_path = args.output.replace(".onnx", "_vocab.json")
    import json
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({
            "itos": vocab.itos,
            "blank_id": vocab.blank_id,
        }, f, ensure_ascii=False, indent=2)

    print(f"✓ ONNX model saved: {args.output}")
    print(f"✓ Vocab saved: {vocab_path}")

    # Verify
    try:
        import onnx
        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model verification passed")
    except ImportError:
        print("⚠ Install `onnx` package to verify the exported model")
    except Exception as e:
        print(f"⚠ ONNX verification warning: {e}")


if __name__ == "__main__":
    main()
