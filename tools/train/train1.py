"""
train.py
========
Training script for Khmer printed-text OCR.
Optimized for Kaggle T4 ×2 (2× NVIDIA T4, 16 GB each).

Reads the dataset produced by khmer_ocr_generator.py:

    <dataset_dir>/
        images/
            0000001.png  ...
        labels.txt          <- "images/0000001.png\t{text}"
        labels_detail.txt
        metadata.json

Usage
-----
  # Recommended for Kaggle T4 x2  (defaults already tuned)
  python train.py --data ./ocr_data

  # More epochs / bigger batch
  python train.py --data ./ocr_data --epochs 50 --batch 64

  # Resume after Kaggle session timeout
  python train.py --data ./ocr_data --resume ./checkpoints/latest.pt
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

# Support multiple common project layouts:
#   Layout A (flat):   train.py + src/ in same folder
#   Layout B (nested): tools/train/train.py, src/ at project root
#   Layout C (nested): tools/train/train.py, src/ two levels up
_here = Path(__file__).resolve().parent
_candidates = [
    _here / "src",           # Layout A — src next to train.py
    _here.parent / "src",    # Layout B — one level up
    _here.parent.parent / "src",  # Layout C — two levels up
    _here,                   # Layout D — dataset.py in same folder as train.py
]
for _p in _candidates:
    if _p.exists():
        sys.path.insert(0, str(_p))
        ROOT = _p.parent
        break
else:
    # Fallback: add all candidate parents so Python can search
    for _p in _candidates:
        sys.path.insert(0, str(_p))
    ROOT = _here

from dataset import Vocab, build_dataloaders
from model   import KhmerOCRModel
from metrics import compute_metrics


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def fmt_time(seconds: float) -> str:
    s = int(seconds)
    if s < 60:    return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:    return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def fix_dp_logits(logits: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Fix DataParallel gather artifact for CTC loss.

    DataParallel gathers outputs along dim=0. For time-first tensors [T, B, V],
    this produces [n_gpus*T, B//n_gpus, V] instead of [T, B, V].

    This function detects and corrects all three possible layouts:
      1. [T, B, V]         -> already correct
      2. [B, T, V]         -> transpose
      3. [n*T, B/n, V]     -> split on T, cat on B  (DataParallel case)
    """
    if logits.dim() != 3:
        raise RuntimeError(f"Expected 3D logits, got shape {tuple(logits.shape)}")

    # Case 1: already correct
    if logits.size(1) == batch_size:
        return logits

    # Case 2: batch-first [B, T, V]
    if logits.size(0) == batch_size:
        return logits.transpose(0, 1).contiguous()

    # Case 3: DataParallel gather — split on T axis, join on B axis
    b_shard = logits.size(1)
    if b_shard > 0 and batch_size % b_shard == 0:
        n_gpus = batch_size // b_shard
        if n_gpus > 1 and logits.size(0) % n_gpus == 0:
            chunks = torch.chunk(logits, n_gpus, dim=0)
            return torch.cat(chunks, dim=1).contiguous()

    raise RuntimeError(
        f"Cannot align logits for CTC. "
        f"logits shape={tuple(logits.shape)}, expected batch_size={batch_size}"
    )


def save_checkpoint(path, model, optimizer, scheduler,
                    epoch, metrics, vocab, best_cer):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = (model.module.state_dict()
             if isinstance(model, nn.DataParallel)
             else model.state_dict())
    payload = {
        "epoch":                epoch,
        "model_state_dict":     state,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics":              metrics,
        "vocab_itos":           vocab.itos,
        "vocab_stoi":           vocab.stoi,
        "best_cer":             best_cer,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)
    print(f"  checkpt -> {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt   = torch.load(path, map_location="cpu", weights_only=False)
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch    = ckpt.get("epoch", 0)
    best_cer = ckpt.get("best_cer", float("inf"))
    print(f"  loaded checkpoint  epoch={epoch}  best_CER={best_cer:.4f}")
    return epoch, best_cer


# ─── TRAIN ONE EPOCH ──────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, scheduler, device,
    blank_id, scaler, log_interval,
    epoch, total_epochs, grad_clip, accum_steps,
) -> float:

    model.train()
    ctc_fn     = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    total_loss = 0.0
    n_batches  = len(loader)
    t0         = time.time()

    optimizer.zero_grad()

    for step, (images, targets, target_lens, _raw) in enumerate(loader):
        images      = images.to(device, non_blocking=True)
        targets     = targets.to(device, non_blocking=True)
        target_lens = target_lens.to(device, non_blocking=True)
        B           = images.size(0)   # real batch size this step

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logits    = model(images)
            logits    = fix_dp_logits(logits, B)        # [T, B, V] guaranteed
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % log_interval == 0:
            avg     = total_loss / (step + 1)
            lr      = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            eta_ep  = elapsed / (step + 1) * (n_batches - step - 1)
            avg_bt  = elapsed / (step + 1)
            rem     = avg_bt * (
                (n_batches - step - 1) + (total_epochs - epoch) * n_batches
            )
            print(f"    [{step+1:>5d}/{n_batches}]  loss={avg:.4f}  "
                  f"lr={lr:.2e}  "
                  f"elapsed={fmt_time(elapsed)}  "
                  f"eta_epoch={fmt_time(eta_ep)}  "
                  f"eta_total={fmt_time(rem)}")

    return total_loss / max(1, n_batches)


# ─── EVALUATE ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, vocab, device) -> dict:
    model.eval()
    preds_all   = []
    targets_all = []

    for images, _targets, _lens, raw_texts in loader:
        images = images.to(device)
        B      = images.size(0)
        logits = model(images)
        logits = fix_dp_logits(logits, B)              # fix before decode
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


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Khmer OCR Training -- Kaggle T4 x2 ready",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kaggle T4 x2 recommended command
----------------------------------
  python train.py --data ./ocr_data

Batch size guide (per GPU)
--------------------------
  T4  16 GB  ->  --batch 64   (default, safe)
  T4  16 GB  ->  --batch 96   (if images are short, try this)
  OOM?       ->  --batch 32

Resume after Kaggle timeout
----------------------------
  python train.py --data ./ocr_data --resume ./checkpoints/latest.pt
        """
    )

    # Dataset
    p.add_argument("--data",       required=True,  help="Dataset folder (has images/ and labels.txt)")
    p.add_argument("--val_ratio",  type=float, default=0.05, help="Val split fraction. Default: 0.05")

    # Model
    p.add_argument("--backbone",   default="mobilenet", choices=["mobilenet", "resnet_tiny"])
    p.add_argument("--hidden",     type=int,   default=256,  help="BiLSTM hidden size. Default: 256")
    p.add_argument("--layers",     type=int,   default=2,    help="BiLSTM layers. Default: 2")
    p.add_argument("--target_h",   type=int,   default=48,   help="Image height px. Default: 48")
    p.add_argument("--no_pretrain",action="store_true", help="Skip ImageNet pretrained weights")

    # Training  (T4 x2 defaults)
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch",      type=int,   default=64,
                   help="Per-GPU batch size. Total = batch x num_gpus. Default: 64")
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--wd",         type=float, default=1e-4)
    p.add_argument("--accum",      type=int,   default=1,    help="Gradient accumulation steps")
    p.add_argument("--clip",       type=float, default=5.0)
    p.add_argument("--log_every",  type=int,   default=50)
    p.add_argument("--save_every", type=int,   default=5)
    p.add_argument("--workers",    type=int,   default=4,
                   help="Total DataLoader workers. Default: 4")
    p.add_argument("--seed",       type=int,   default=42)

    # Output / resume
    p.add_argument("--output",  default="./checkpoints")
    p.add_argument("--resume",  default=None, help="Path to checkpoint to resume from")
    p.add_argument("--device",  default=None, help="'cuda' or 'cpu'. Auto-detected if omitted")

    return p.parse_args()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    use_amp = device.type == "cuda"
    n_gpus  = torch.cuda.device_count() if device.type == "cuda" else 1

    # Dataset paths
    data_dir   = Path(args.data)
    label_file = data_dir / "labels.txt"
    if not label_file.exists():
        raise FileNotFoundError(
            f"labels.txt not found in {data_dir}\n"
            f"--data must point to the output folder of khmer_ocr_generator.py"
        )

    # Vocabulary
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

    # Data loaders
    # Total batch passed to loader = per-GPU batch x num_gpus
    # DataParallel then slices each GPU's share automatically
    total_batch   = args.batch * n_gpus
    total_workers = args.workers  # treat as total, not per-GPU
    print(f"\nBuilding data loaders (total batch={total_batch}) ...")
    train_loader, val_loader = build_dataloaders(
        label_file  = str(label_file),
        vocab       = vocab,
        data_root   = str(data_dir),
        target_h    = args.target_h,
        batch_size  = total_batch,
        num_workers = total_workers,
        val_ratio   = args.val_ratio,
        pin_memory  = use_amp,
    )

    # Model
    model = KhmerOCRModel(
        vocab_size  = vocab.size,
        backbone    = args.backbone,
        hidden_size = args.hidden,
        num_layers  = args.layers,
        pretrained  = not args.no_pretrain,
    ).to(device)

    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"  DataParallel: {n_gpus} GPUs detected and used")

    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    p_info    = raw_model.count_params()

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    accum_steps     = args.accum
    steps_per_epoch = -(-len(train_loader) // accum_steps)
    total_steps     = args.epochs * steps_per_epoch
    scheduler = OneCycleLR(
        optimizer,
        max_lr          = args.lr,
        total_steps     = total_steps,
        pct_start       = 0.05,
        anneal_strategy = "cos",
    )

    # AMP GradScaler — T4 supports FP16, ~1.5-2x speedup
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Resume
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

    # GPU names
    gpu_names = ""
    if device.type == "cuda":
        names     = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
        gpu_names = " | ".join(names)

    # Banner
    eff_batch = args.batch * n_gpus * accum_steps
    print(f"\n{'='*64}")
    print(f"  Khmer OCR  --  Training")
    print(f"  Device     : {device}  ({n_gpus} GPU{'s' if n_gpus > 1 else ''})")
    if gpu_names:
        print(f"  GPUs       : {gpu_names}")
    print(f"  AMP FP16   : {'ON' if use_amp else 'OFF'}")
    print(f"  Backbone   : {args.backbone}  ({p_info['total_M']:.2f}M params)")
    print(f"  Vocab size : {vocab.size}")
    print(f"  Epochs     : {start_epoch} -> {args.epochs}")
    print(f"  Batch/GPU  : {args.batch} x {n_gpus} GPUs"
          + (f" x {accum_steps} accum" if accum_steps > 1 else "")
          + f" = {eff_batch} effective")
    print(f"  LR max     : {args.lr}")
    print(f"  Train batches/epoch : {len(train_loader)}")
    print(f"  Val   batches/epoch : {len(val_loader)}")
    print(f"  Checkpoints -> {args.output}")
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

            # Save latest (always — so Kaggle timeout never loses progress)
            save_checkpoint(
                os.path.join(args.output, "latest.pt"),
                model, optimizer, scheduler, epoch, metrics, vocab, best_cer,
            )

            # Save best
            if metrics["avg_cer"] < best_cer:
                best_cer = metrics["avg_cer"]
                save_checkpoint(
                    os.path.join(args.output, "best.pt"),
                    model, optimizer, scheduler, epoch, metrics, vocab, best_cer,
                )
                print(f"  * new best CER: {best_cer:.4f}")

            # Periodic snapshot
            if epoch % args.save_every == 0:
                save_checkpoint(
                    os.path.join(args.output, f"epoch_{epoch:03d}.pt"),
                    model, optimizer, scheduler, epoch, metrics, vocab, best_cer,
                )

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        print("  Resume: python train.py --data ./ocr_data --resume checkpoints/latest.pt")
        sys.exit(0)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n\nCUDA OUT OF MEMORY")
            print(f"  Current --batch {args.batch} per GPU")
            print(f"  Try:    --batch {max(8, args.batch // 2)}")
            print("  Then:   --resume checkpoints/latest.pt")
            torch.cuda.empty_cache()
            sys.exit(1)
        raise

    total = time.time() - training_start
    print(f"\nDone!  Best CER: {best_cer:.4f}  |  Time: {fmt_time(total)}")


if __name__ == "__main__":
    main()