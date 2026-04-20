"""
ThaoOCR — Evaluation Script
Evaluate a trained model on a test/val set with detailed metrics.

Usage:
    python tools/train/evaluate.py --checkpoint checkpoints/best.pt --data data/val.txt
"""
import os
import sys
import argparse
import torch
from collections import defaultdict
from torch.utils.data import DataLoader

# Add project paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import get_config
from vocab import Vocab
from dataset import LineRecDataset, collate_fn
from model import OCRRecModel
from utils import compute_cer, compute_wer


def load_model(checkpoint_path: str, device: str = "cpu", config_path: str = "config.yml"):
    """Load model and vocab from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct vocab from saved itos
    vocab = Vocab()
    if "itos" in ckpt:
        vocab.itos = ckpt["itos"]
        vocab.stoi = {ch: i for i, ch in enumerate(vocab.itos)}

    cfg = get_config(config_path)
    # No dropout at inference
    cfg.model.head.dropout = 0.0
    cfg.model.neck.dropout = 0.0
    model = OCRRecModel.from_config(cfg.model, vocab_size=vocab.size).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, vocab


@torch.no_grad()
def evaluate_full(model, loader, vocab, device, use_beam=False, beam_width=10):
    """
    Full evaluation with per-sample metrics.

    Returns:
        results: list of dicts with pred, target, cer, wer, match
        summary: dict with aggregate metrics
    """
    results = []

    for x, x_lens, y_cat, y_lens, raw_labels in loader:
        x = x.to(device)
        logits = model(x)

        if use_beam:
            preds = vocab.ctc_decode_beam(logits, beam_width=beam_width)
        else:
            preds = vocab.ctc_decode_greedy(logits)

        for pred, gt in zip(preds, raw_labels):
            cer = compute_cer(pred, gt)
            wer = compute_wer(pred, gt)
            results.append({
                "pred": pred,
                "target": gt,
                "cer": cer,
                "wer": wer,
                "match": pred == gt,
            })

    # Aggregate
    n = len(results)
    avg_cer = sum(r["cer"] for r in results) / max(1, n)
    avg_wer = sum(r["wer"] for r in results) / max(1, n)
    exact_match = sum(1 for r in results if r["match"]) / max(1, n)

    # CER breakdown by length bucket
    len_buckets = defaultdict(list)
    for r in results:
        bucket = len(r["target"]) // 10 * 10  # 0-9, 10-19, 20-29, ...
        len_buckets[bucket].append(r["cer"])

    bucket_stats = {}
    for bucket in sorted(len_buckets.keys()):
        cers = len_buckets[bucket]
        bucket_stats[f"{bucket}-{bucket+9}"] = {
            "count": len(cers),
            "avg_cer": sum(cers) / len(cers),
        }

    summary = {
        "num_samples": n,
        "avg_cer": avg_cer,
        "avg_wer": avg_wer,
        "exact_match": exact_match,
        "cer_by_length": bucket_stats,
    }

    return results, summary


def print_report(summary, results, show_errors=20):
    """Print a formatted evaluation report."""
    print(f"\n{'='*60}")
    print(f"  EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"  Samples:      {summary['num_samples']}")
    print(f"  Avg CER:      {summary['avg_cer']:.4f} ({summary['avg_cer']*100:.2f}%)")
    print(f"  Avg WER:      {summary['avg_wer']:.4f} ({summary['avg_wer']*100:.2f}%)")
    print(f"  Exact Match:  {summary['exact_match']:.4f} ({summary['exact_match']*100:.2f}%)")

    print(f"\n  CER by Label Length:")
    for bucket, stats in summary["cer_by_length"].items():
        print(f"    {bucket:>8s} chars: n={stats['count']:>5d}  CER={stats['avg_cer']:.4f}")

    # Show worst predictions
    errors = [r for r in results if not r["match"]]
    errors.sort(key=lambda r: r["cer"], reverse=True)

    if errors:
        print(f"\n  ── Top {min(show_errors, len(errors))} Errors (by CER) ──")
        for r in errors[:show_errors]:
            print(f"  CER={r['cer']:.3f}")
            print(f"    GT  : {r['target']}")
            print(f"    PRED: {r['pred']}")
            print()

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="ThaoOCR Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to test/val .txt file")
    parser.add_argument("--data-root", type=str, default=None, help="Root dir for image paths")
    parser.add_argument("--beam", action="store_true", help="Use beam search decoding")
    parser.add_argument("--beam-width", type=int, default=10, help="Beam width")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--show-errors", type=int, default=20, help="Number of errors to show")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    print(f"Loading model from {args.checkpoint}...")
    model, vocab = load_model(args.checkpoint, device)

    print(f"Loading data from {args.data}...")
    dataset = LineRecDataset(args.data, vocab, augment=False, data_root=args.data_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    print(f"Evaluating... (beam={args.beam}, beam_width={args.beam_width})")
    results, summary = evaluate_full(
        model, loader, vocab, device,
        use_beam=args.beam,
        beam_width=args.beam_width,
    )

    print_report(summary, results, show_errors=args.show_errors)


if __name__ == "__main__":
    main()
