"""
ThaoOCR — Single Image Prediction
Run inference on a single image or directory of images.

Usage:
    python tools/export/predict.py --checkpoint checkpoints/best.pt --image path/to/line.png
    python tools/export/predict.py --checkpoint checkpoints/best.pt --dir path/to/images/
"""
import os
import sys
import argparse
import glob
import torch

# Add project paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import get_config
from utils import preprocess_image

# Import load_model from the train evaluate module
sys.path.insert(0, os.path.join(ROOT, "tools", "train"))
from evaluate import load_model


def predict_single(model, vocab, image_path: str, device, cfg, use_beam=False, beam_width=10):
    """Predict text from a single image."""
    x = preprocess_image(image_path, target_h=cfg.model.target_h)
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)  # [T, 1, V]

    if use_beam:
        texts = vocab.ctc_decode_beam(logits, beam_width=beam_width)
    else:
        texts = vocab.ctc_decode_greedy(logits)

    return texts[0]


def main():
    parser = argparse.ArgumentParser(description="ThaoOCR Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to config file")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image")
    parser.add_argument("--dir", type=str, default=None, help="Directory of images to predict")
    parser.add_argument("--beam", action="store_true", help="Use beam search")
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None, help="Output file (optional)")
    args = parser.parse_args()

    if args.image is None and args.dir is None:
        parser.error("Must specify --image or --dir")

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    print(f"Loading model from {args.checkpoint}...")
    # Load config first to pass to load_model and predict_single
    cfg = get_config(args.config)
    model, vocab = load_model(args.checkpoint, device, config_path=args.config)

    # Collect images
    images = []
    if args.image:
        images.append(args.image)
    if args.dir:
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]:
            images.extend(sorted(glob.glob(os.path.join(args.dir, ext))))

    print(f"Predicting {len(images)} images...\n")

    results = []
    for img_path in images:
        try:
            text = predict_single(model, vocab, img_path, device, cfg, args.beam, args.beam_width)
            print(f"{os.path.basename(img_path):>40s}  →  {text}")
            results.append((img_path, text))
        except Exception as e:
            print(f"{os.path.basename(img_path):>40s}  →  ERROR: {e}")
            results.append((img_path, f"ERROR: {e}"))

    # Optionally write results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for path, text in results:
                f.write(f"{path}\t{text}\n")
        print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
