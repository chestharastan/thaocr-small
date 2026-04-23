"""
ThaoOCR — Training Script  (A40-optimized)
==========================================
Architecture : OCRRecModel  (ResNet backbone → Transformer encoder → CTC)
Data reading : --data <folder>  that contains labels.txt and images/
GPU target   : NVIDIA A40 (Ampere sm_86, 48 GB) — runs on any CUDA GPU.

Usage
-----
  # Single or multi-GPU (DataParallel auto-detected)
  python train1.py --data ./ocr_data

  # Larger model, more epochs
  python train1.py --data ./ocr_data --preset large --epochs 50 --batch 96

  # Resume after interruption
  python train1.py --data ./ocr_data --resume ./checkpoints/latest.pt

Dataset folder layout
---------------------
  ocr_data/
      labels.txt        <- "images/0000001.png<TAB>label text"
      images/
          0000001.png  ...

Model presets (--preset)
------------------------
  small   32px  h  ~3 M params   fastest
  base    48px  h  ~8 M params   recommended
  large   48px  h  ~19M params   best quality
  xlarge  64px  h  ~34M params   max quality / most VRAM
"""

import os
import sys
import time
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

# ── Path resolution: find src/ up to two levels above this file ───────────────
_here = Path(__file__).resolve().parent
for _candidate in [
    _here.parent.parent / "src",  # tools/train/../../src  (project layout)
    _here.parent / "src",
    _here / "src",
]:
    if _candidate.exists():
        sys.path.insert(0, str(_candidate))
        break

from vocab   import Vocab
from dataset import build_dataloaders
from model   import OCRRecModel
from metrics import compute_metrics


# ── Model size presets (mirrors config.py MODEL_PRESETS) ──────────────────────
_PRESETS = {
    "small":  {"target_h": 32, "channels": (32, 64, 128, 128),
               "d_model": 128, "nhead": 4, "num_layers": 2,  "dim_ff":  512},
    "base":   {"target_h": 48, "channels": (64, 128, 256, 256),
               "d_model": 256, "nhead": 4, "num_layers": 4,  "dim_ff": 1024},
    "large":  {"target_h": 48, "channels": (64, 128, 256, 512),
               "d_model": 384, "nhead": 8, "num_layers": 6,  "dim_ff": 1536},
    "xlarge": {"target_h": 64, "channels": (64, 128, 256, 512),
               "d_model": 512, "nhead": 8, "num_layers": 8,  "dim_ff": 2048},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_time(seconds: float) -> str:
    s = int(seconds)
    if s < 60:   return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:   return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def fix_dp_logits(logits: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Fix DataParallel gather artifact so CTC always sees [T, B, V]."""
    if logits.dim() != 3:
        raise RuntimeError(f"Expected 3D logits, got {tuple(logits.shape)}")
    if logits.size(1) == batch_size:        # already [T, B, V]
        return logits
    if logits.size(0) == batch_size:        # batch-first [B, T, V]
        return logits.transpose(0, 1).contiguous()
    b_shard = logits.size(1)               # DataParallel gather on dim=0
    if b_shard > 0 and batch_size % b_shard == 0:
        n = batch_size // b_shard
        if n > 1 and logits.size(0) % n == 0:
            return torch.cat(torch.chunk(logits, n, dim=0), dim=1).contiguous()
    raise RuntimeError(
        f"Cannot realign CTC logits: shape={tuple(logits.shape)}, B={batch_size}"
    )


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, vocab, best_cer):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = (model.module.state_dict()
             if isinstance(model, nn.DataParallel) else model.state_dict())
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics":              metrics,
        "itos":                 vocab.itos,
        "stoi":                 vocab.stoi,
        "best_cer":             best_cer,
    }, path)
    print(f"  checkpt -> {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt   = torch.load(path, map_location="cpu", weights_only=False)
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch    = ckpt.get("epoch", 0)
    best_cer = ckpt.get("best_cer", float("inf"))
    print(f"  loaded  epoch={epoch}  best_CER={best_cer:.4f}")
    return epoch, best_cer


# ── Train one epoch ───────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, scheduler, device,
    blank_id, scaler, log_interval,
    epoch, total_epochs, grad_clip, accum_steps,
    amp_dtype=torch.bfloat16,
) -> float:

    model.train()
    ctc_fn     = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    total_loss = 0.0
    n_batches  = len(loader)
    t0         = time.time()

    optimizer.zero_grad(set_to_none=True)

    for step, (images, targets, target_lens, _raw) in enumerate(loader):
        images      = images.to(device, non_blocking=True)
        targets     = targets.to(device, non_blocking=True)
        target_lens = target_lens.to(device, non_blocking=True)
        B           = images.size(0)

        use_autocast = device.type == "cuda"
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_autocast):
            logits    = model(images)
            logits    = fix_dp_logits(logits, B)
            log_probs = F.log_softmax(logits, dim=-1)
            T         = log_probs.size(0)
            in_lens   = torch.full((B,), T, dtype=torch.long, device=device)
            loss      = ctc_fn(log_probs, targets, in_lens, target_lens)
            loss      = loss / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * accum_steps

        is_boundary = (step + 1) % accum_steps == 0
        is_last     = (step + 1) == n_batches

        if is_boundary or is_last:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if (step + 1) % log_interval == 0:
            avg     = total_loss / (step + 1)
            lr      = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            eta_ep  = elapsed / (step + 1) * (n_batches - step - 1)
            rem     = elapsed / (step + 1) * (
                (n_batches - step - 1) + (total_epochs - epoch) * n_batches
            )
            print(f"    [{step+1:>5d}/{n_batches}]  loss={avg:.4f}  lr={lr:.2e}  "
                  f"elapsed={fmt_time(elapsed)}  "
                  f"eta_epoch={fmt_time(eta_ep)}  "
                  f"eta_total={fmt_time(rem)}")

    return total_loss / max(1, n_batches)


# ── Evaluate ──────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate(model, loader, vocab, device) -> dict:
    model.eval()
    preds_all, targets_all = [], []

    for images, _targets, _lens, raw_texts in loader:
        images = images.to(device, non_blocking=True)
        B      = images.size(0)
        logits = model(images)
        logits = fix_dp_logits(logits, B)
        preds_all.extend(vocab.ctc_decode_greedy(logits))
        targets_all.extend(raw_texts)

    metrics = compute_metrics(preds_all, targets_all)

    print("  -- sample predictions --")
    indices = list(range(len(preds_all)))
    random.shuffle(indices)
    shown = 0
    for i in indices:
        if shown >= 8: break
        ok = "OK" if preds_all[i] == targets_all[i] else "XX"
        print(f"  [{ok}]  GT  : {targets_all[i]}")
        print(f"         PRED: {preds_all[i]}")
        print()
        shown += 1

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ThaoOCR Training — OCRRecModel, A40-optimized",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
A40 (48 GB) recommended:
  python train1.py --data ./ocr_data --preset base --batch 128 --epochs 30

Resume after interruption:
  python train1.py --data ./ocr_data --resume ./checkpoints/latest.pt

Batch size guide (per GPU, base preset, BF16):
  A40 48 GB  ->  --batch 128   (safe)
  A40 48 GB  ->  --batch 192   (if images are short)
  RTX 3090   ->  --batch 96
  OOM?       ->  --batch 64
        """
    )

    # Data
    p.add_argument("--data",       required=True,
                   help="Dataset folder containing labels.txt and images/")
    p.add_argument("--val-ratio",  type=float, default=0.05,
                   help="Fraction of data held out for validation. Default: 0.05")

    # Model
    p.add_argument("--preset",     default="base",
                   choices=list(_PRESETS.keys()),
                   help="Model size preset. Default: base")
    p.add_argument("--target-h",   type=int,   default=None,
                   help="Override image height (px) from preset")
    p.add_argument("--d-model",    type=int,   default=None,
                   help="Override Transformer hidden dim from preset")
    p.add_argument("--num-layers", type=int,   default=None,
                   help="Override Transformer layer count from preset")

    # Training — A40 defaults
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch",      type=int,   default=128,
                   help="Per-GPU batch size. Total = batch × num_gpus. Default: 128")
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--wd",         type=float, default=1e-4)
    p.add_argument("--accum",      type=int,   default=1,
                   help="Gradient accumulation steps")
    p.add_argument("--clip",       type=float, default=5.0)
    p.add_argument("--log-every",  type=int,   default=50)
    p.add_argument("--save-every", type=int,   default=5)
    p.add_argument("--workers",    type=int,   default=8,
                   help="DataLoader worker processes. Default: 8")
    p.add_argument("--seed",       type=int,   default=42)

    # Output / resume
    p.add_argument("--output",  default="./checkpoints")
    p.add_argument("--resume",  default=None,
                   help="Path to checkpoint to resume training from")
    p.add_argument("--device",  default=None,
                   help="'cuda' or 'cpu'. Auto-detected if omitted.")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── A40 / Ampere performance flags ────────────────────────────────────────
    torch.backends.cudnn.benchmark        = True   # auto-tune convolution kernels
    torch.backends.cudnn.allow_tf32       = True   # TF32 for convolutions
    torch.backends.cuda.matmul.allow_tf32 = True   # TF32 for matmuls
    torch.set_float32_matmul_precision("high")

    # ── Device + precision ────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    use_amp  = device.type == "cuda"
    # A40 is Ampere (sm_86): BF16 is natively fast and needs no loss scaler
    use_bf16 = (
        use_amp
        and torch.cuda.get_device_capability(0)[0] >= 8
        and hasattr(torch, "bfloat16")
    )
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    n_gpus    = torch.cuda.device_count() if device.type == "cuda" else 1

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_dir   = Path(args.data)
    label_file = data_dir / "labels.txt"
    if not label_file.exists():
        raise FileNotFoundError(
            f"labels.txt not found in {data_dir}\n"
            "--data must point to the dataset root folder."
        )

    # Build (or reload) vocabulary
    vocab_path = Path(args.output) / "vocab.json"
    if vocab_path.exists() and args.resume:
        print(f"Loading vocab from {vocab_path}")
        vocab = Vocab.from_file(str(vocab_path))
    else:
        print("Building vocabulary from labels.txt ...")
        vocab = Vocab.build_from_labels([str(label_file)])
        os.makedirs(args.output, exist_ok=True)
        vocab.save(str(vocab_path))
    print(f"  vocab size: {vocab.size}  (saved to {vocab_path})")

    # ── Model preset (with per-arg overrides) ─────────────────────────────────
    preset    = _PRESETS[args.preset]
    target_h  = args.target_h   if args.target_h   is not None else preset["target_h"]
    d_model   = args.d_model    if args.d_model    is not None else preset["d_model"]
    n_layers  = args.num_layers if args.num_layers is not None else preset["num_layers"]

    # ── Data loaders ──────────────────────────────────────────────────────────
    # Total batch is split across GPUs automatically by DataParallel
    total_batch = args.batch * n_gpus
    print(f"\nBuilding data loaders  (total_batch={total_batch}) ...")
    train_loader, val_loader = build_dataloaders(
        vocab       = vocab,
        label_file  = str(label_file),
        data_root   = str(data_dir),
        target_h    = target_h,
        batch_size  = total_batch,
        num_workers = args.workers,
        val_ratio   = args.val_ratio,
        pin_memory  = use_amp,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = OCRRecModel(
        vocab_size        = vocab.size,
        backbone_channels = preset["channels"],
        d_model           = d_model,
        nhead             = preset["nhead"],
        num_layers        = n_layers,
        dim_feedforward   = preset["dim_ff"],
        neck_dropout      = 0.1,
        head_dropout      = 0.1,
    ).to(device)

    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"  DataParallel: using {n_gpus} GPUs")

    # torch.compile fuses ops for ~20-30% throughput gain on Ampere
    if hasattr(torch, "compile") and device.type == "cuda":
        print("  torch.compile: ON (first epoch will be slow for kernel tracing)")
        model = torch.compile(model, mode="reduce-overhead")

    # Unwrap compiled/parallel wrappers to call count_params
    raw = getattr(model, "_orig_mod", model)
    raw = raw.module if isinstance(raw, nn.DataParallel) else raw
    p_info = raw.count_params()

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    accum_steps     = args.accum
    steps_per_epoch = -(-len(train_loader) // accum_steps)   # ceil division
    total_steps     = args.epochs * steps_per_epoch
    scheduler = OneCycleLR(
        optimizer,
        max_lr          = args.lr,
        total_steps     = total_steps,
        pct_start       = 0.05,
        anneal_strategy = "cos",
    )

    # BF16 does not need a loss scaler; FP16 does
    scaler = torch.amp.GradScaler("cuda") if (use_amp and not use_bf16) else None

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_cer    = float("inf")
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"  checkpoint not found: {args.resume} — starting fresh")
        else:
            start_epoch, best_cer = load_checkpoint(
                args.resume, model, optimizer, scheduler
            )
            start_epoch += 1

    # ── Banner ────────────────────────────────────────────────────────────────
    gpu_names = ""
    if device.type == "cuda":
        gpu_names = " | ".join(torch.cuda.get_device_name(i) for i in range(n_gpus))

    eff_batch = args.batch * n_gpus * accum_steps
    amp_label = ("BF16" if use_bf16 else "FP16") if use_amp else "OFF"
    compiled  = hasattr(model, "_orig_mod")

    print(f"\n{'='*64}")
    print(f"  ThaoOCR Training  —  OCRRecModel")
    print(f"  Device     : {device}  ({n_gpus} GPU{'s' if n_gpus > 1 else ''})")
    if gpu_names:
        print(f"  GPUs       : {gpu_names}")
    print(f"  AMP        : {amp_label}  |  compile: {'ON' if compiled else 'OFF'}")
    print(f"  Model      : OCRRecModel / {args.preset}  ({p_info['total_M']:.2f}M params)")
    print(f"  Vocab size : {vocab.size}")
    print(f"  target_h   : {target_h}  |  d_model: {d_model}  |  layers: {n_layers}")
    print(f"  Epochs     : {start_epoch} → {args.epochs}")
    print(f"  Batch/GPU  : {args.batch} × {n_gpus} GPU"
          + ("s" if n_gpus > 1 else "")
          + (f" × {accum_steps} accum" if accum_steps > 1 else "")
          + f" = {eff_batch} effective")
    print(f"  LR max     : {args.lr}")
    print(f"  Train batches/epoch : {len(train_loader)}")
    print(f"  Val   batches/epoch : {len(val_loader)}")
    print(f"  Checkpoints  -> {args.output}")
    print(f"{'='*64}\n")

    training_start = time.time()
    epoch_times    = []

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            print(f"\n-- Epoch {epoch}/{args.epochs} " + "-"*46)
            t0 = time.time()

            avg_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, device,
                blank_id     = vocab.blank_id,
                scaler       = scaler,
                log_interval = args.log_every,
                epoch        = epoch,
                total_epochs = args.epochs,
                grad_clip    = args.clip,
                accum_steps  = accum_steps,
                amp_dtype    = amp_dtype,
            )

            print(f"\n  Evaluating ...")
            metrics = evaluate(model, val_loader, vocab, device)

            elapsed = time.time() - t0
            epoch_times.append(elapsed)
            eta      = (sum(epoch_times) / len(epoch_times)) * (args.epochs - epoch)
            total_el = time.time() - training_start

            print(
                f"\nEpoch {epoch:03d}/{args.epochs}  "
                f"loss={avg_loss:.4f}  "
                f"CER={metrics['avg_cer']:.4f}  "
                f"WER={metrics['avg_wer']:.4f}  "
                f"exact={metrics['exact_match']:.3f}  "
                f"epoch={fmt_time(elapsed)}  "
                f"total={fmt_time(total_el)}  "
                f"ETA={fmt_time(eta)}"
            )

            # Always save latest so a crash never loses more than one epoch
            save_checkpoint(
                os.path.join(args.output, "latest.pt"),
                model, optimizer, scheduler, epoch, metrics, vocab, best_cer,
            )

            if metrics["avg_cer"] < best_cer:
                best_cer = metrics["avg_cer"]
                save_checkpoint(
                    os.path.join(args.output, "best.pt"),
                    model, optimizer, scheduler, epoch, metrics, vocab, best_cer,
                )
                print(f"  * new best CER: {best_cer:.4f}")

            if epoch % args.save_every == 0:
                save_checkpoint(
                    os.path.join(args.output, f"epoch_{epoch:03d}.pt"),
                    model, optimizer, scheduler, epoch, metrics, vocab, best_cer,
                )

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        print(f"  Resume: python train1.py --data {args.data} --resume {args.output}/latest.pt")
        sys.exit(0)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nCUDA OOM — current --batch {args.batch} per GPU")
            print(f"  Try: --batch {max(16, args.batch // 2)}")
            torch.cuda.empty_cache()
            sys.exit(1)
        raise

    total = time.time() - training_start
    print(f"\nDone!  Best CER: {best_cer:.4f}  |  Time: {fmt_time(total)}")


if __name__ == "__main__":
    main()
