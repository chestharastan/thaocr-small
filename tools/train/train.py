import os
import sys
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

# Add project paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import get_config
from vocab import Vocab
from dataset import build_dataloaders
from model import OCRRecModel
from utils import compute_accuracy, preprocess_image

def _fmt_time(seconds: float) -> str:
    """Format seconds into a human-readable string like '1h 23m 45s'."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _normalize_ctc_logits(logits: torch.Tensor, expected_batch: int) -> torch.Tensor:
    """
    Ensure logits are in CTC layout [T, B, V].

    Handles:
    - standard single-device output [T, B, V]
    - accidental [B, T, V] layout
    - DataParallel gather artifact for time-first tensors:
      [num_devices * T, B_per_device, V] -> [T, B, V]
    """
    if logits.dim() != 3:
        raise RuntimeError(f"Expected 3D logits [T, B, V], got shape={tuple(logits.shape)}")

    # Expected layout already
    if logits.size(1) == expected_batch:
        return logits

    # Batch-first [B, T, V] -> [T, B, V]
    if logits.size(0) == expected_batch:
        return logits.transpose(0, 1).contiguous()

    # DataParallel gathered time-first tensors along dim=0.
    # Rebuild by splitting time into per-device chunks and concatenating on batch dim.
    b_shard = logits.size(1)
    if b_shard > 0 and expected_batch % b_shard == 0:
        num_chunks = expected_batch // b_shard
        if num_chunks > 1 and logits.size(0) % num_chunks == 0:
            chunks = torch.chunk(logits, num_chunks, dim=0)
            return torch.cat(chunks, dim=1).contiguous()

    raise RuntimeError(
        "Unable to align logits for CTC loss. "
        f"logits shape={tuple(logits.shape)}, expected_batch={expected_batch}."
    )


def train_one_epoch(model, loader, optimizer, scheduler, device, blank_id,
                    log_interval=50, scaler=None, total_epochs=1, current_epoch=1,
                    grad_accum_steps=1, grad_clip=5.0):
    """Train for one epoch with chunk-and-merge gradient accumulation.

    When ``grad_accum_steps`` > 1 the optimizer step is deferred: gradients
    from *N* consecutive mini-batches (chunks) are accumulated (merged) and a
    single optimiser step is taken once every *N* batches.  This gives an
    effective batch size of ``batch_size × grad_accum_steps`` without needing
    the GPU memory for the larger batch.

    Returns average loss (per mini-batch, *not* per accumulation step).
    """
    model.train()
    ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    total_loss = 0.0
    num_batches = 0
    total_batches = len(loader)
    epoch_start = time.time()
    use_amp = scaler is not None

    optimizer.zero_grad()  # zero once at the start

    for batch_idx, (x, x_lens, y_cat, y_lens, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y_cat = y_cat.to(device, non_blocking=True)
        y_lens = y_lens.to(device, non_blocking=True)

        # Forward pass with optional mixed precision
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)                         # [T, B, V]
            logits = _normalize_ctc_logits(logits, expected_batch=y_lens.size(0))
            log_probs = F.log_softmax(logits, dim=-1)
            T, B = log_probs.size(0), log_probs.size(1)
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
            loss = ctc_loss_fn(log_probs, y_cat, input_lengths, y_lens)
            # Scale loss by accumulation steps so merged gradients are averaged
            loss = loss / grad_accum_steps

        # Backward — gradients accumulate in .grad buffers
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * grad_accum_steps  # track unscaled loss
        num_batches += 1

        # --- Merge step: update weights every grad_accum_steps batches ---
        is_accum_boundary = (batch_idx + 1) % grad_accum_steps == 0
        is_last_batch = (batch_idx + 1) == total_batches

        if is_accum_boundary or is_last_batch:
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

            optimizer.zero_grad()  # reset for next accumulation window

        if (batch_idx + 1) % log_interval == 0:
            avg = total_loss / num_batches
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - epoch_start
            batches_done = batch_idx + 1
            eta_epoch = (elapsed / batches_done) * (total_batches - batches_done)

            avg_batch_time = elapsed / batches_done

            # --- ETA Total (All Epochs) ---
            batches_per_epoch = total_batches
            remaining_batches_current_epoch = batches_per_epoch - batches_done
            future_epochs = total_epochs - current_epoch
            total_remaining_batches = remaining_batches_current_epoch + (future_epochs * batches_per_epoch)

            eta_total = avg_batch_time * total_remaining_batches

            print(f"  [batch {batches_done:>5d}/{total_batches}] "
                  f"loss={avg:.4f}  lr={lr:.2e}  "
                  f"elapsed={_fmt_time(elapsed)}  eta_epoch={_fmt_time(eta_epoch)}  eta_total={_fmt_time(eta_total)}")

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(model, loader, vocab, device):
    """Evaluate on validation set. Returns metrics dict."""
    model.eval()
    all_preds = []
    all_targets = []

    for x, x_lens, y_cat, y_lens, raw_labels in loader:
        x = x.to(device)
        logits = model(x)                             # [T, B, V]
        logits = _normalize_ctc_logits(logits, expected_batch=x.size(0))
        preds = vocab.ctc_decode_greedy(logits)
        all_preds.extend(preds)
        all_targets.extend(raw_labels)

    metrics = compute_accuracy(all_preds, all_targets)

    # Print random sample predictions (mix of correct & wrong)
    print("  ── Sample predictions ──")
    n_samples = min(8, len(all_preds))
    indices = list(range(len(all_preds)))
    random.shuffle(indices)
    shown = 0
    for i in indices:
        if shown >= n_samples:
            break
        status = "✓" if all_preds[i] == all_targets[i] else "✗"
        print(f"  {status} GT : {all_targets[i]}")
        print(f"    PRED: {all_preds[i]}")
        print()
        shown += 1

    return metrics


def test_custom_image(model, vocab, image_path, device, target_h=48):
    """Run inference on a custom image during training."""
    if not os.path.exists(image_path):
        print(f"  ⚠ Test image not found: {image_path}")
        return

    try:
        # Preprocess using the shared utility
        x = preprocess_image(image_path, target_h=target_h)
        x = x.to(device)

        model.eval()
        with torch.no_grad():
            logits = model(x)
            preds = vocab.ctc_decode_greedy(logits)

        print(f"  ── Test Image: {os.path.basename(image_path)} ──")
        print(f"    PRED: {preds[0]}")
        print()

    except Exception as e:
        print(f"  ⚠ Failed to predict on test image: {e}")


def save_checkpoint(model, optimizer, scheduler, vocab, epoch, metrics, path, best_cer=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()

    payload = {
        "epoch": epoch,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "itos": vocab.itos,
        "stoi": vocab.stoi,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if best_cer is not None:
        payload["best_cer"] = best_cer

    torch.save(payload, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load a checkpoint. Returns (epoch, best_cer)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    target.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch = ckpt.get("epoch", 0)
    best_cer = ckpt.get("best_cer", float('inf'))
    print(f"  ✓ Loaded checkpoint from epoch {epoch}: {path}")
    if best_cer < float('inf'):
        print(f"    Best CER so far: {best_cer:.4f}")
    return epoch, best_cer


def main():
    parser = argparse.ArgumentParser(description="ThaoOCR Training")
    parser.add_argument("config_pos", nargs="?", default=None, help="Path to YAML config file (positional)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--train-txt", type=str, default=None, help="Training data file")
    parser.add_argument("--val-txt", type=str, default=None, help="Validation data file")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--test-image", type=str, default=None, help="Path to a real image to test after every epoch")
    args = parser.parse_args()

    # ─── Config ───────────────────────────────────────────────
    config_path = args.config_pos or args.config   # positional takes priority
    cfg = get_config(config_path)
    if args.train_txt:
        cfg.train.train_txt = args.train_txt
    if args.val_txt:
        cfg.train.val_txt = args.val_txt
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.lr:
        cfg.train.lr = args.lr
    if args.resume:
        cfg.train.resume = args.resume
    if args.device:
        cfg.device = args.device

    # Auto-detect device
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        cfg.device = "cpu"
    device = torch.device(cfg.device)

    # ─── Vocab ────────────────────────────────────────────────
    vocab = Vocab(cfg.vocab)
    print(f"Vocabulary size: {vocab.size}")

    # ─── Data ─────────────────────────────────────────────────
    print(f"Loading data...")
    train_loader, val_loader = build_dataloaders(
        train_txt=cfg.train.train_txt,
        val_txt=cfg.train.val_txt,
        vocab=vocab,
        target_h=cfg.model.target_h,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        data_root=cfg.train.data_root,
    )

    # ─── Model ────────────────────────────────────────────────
    model = OCRRecModel.from_config(cfg.model, vocab_size=vocab.size).to(device)

    # Use both GPUs (simple multi-GPU)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    real_model = model.module if hasattr(model, "module") else model
    params = real_model.count_params()
    print(f"Model: ThaoNet-{cfg.model.name}")
    print(f"  Backbone: {cfg.model.backbone.type} {cfg.model.backbone.channels}")
    print(f"  Neck:     {cfg.model.neck.type}")
    print(f"  Head:     {cfg.model.head.type} (d={cfg.model.head.d_model}, "
          f"layers={cfg.model.head.num_layers}, heads={cfg.model.head.nhead})")
    print(f"  Params:   {params['total_M']:.2f}M ({params['trainable']} trainable)")

    # ─── Optimizer + Scheduler ────────────────────────────────
    accum_steps = cfg.train.grad_accum_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    # total_steps = optimizer steps (not mini-batches) for the scheduler
    steps_per_epoch = -(-len(train_loader) // accum_steps)   # ceil division
    total_steps = cfg.train.epochs * steps_per_epoch
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.train.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
    )

    # ─── Resume ───────────────────────────────────────────────
    start_epoch = 1
    best_cer = float('inf')
    if cfg.train.resume:
        epoch_loaded, best_cer = load_checkpoint(
            cfg.train.resume, model, optimizer, scheduler
        )
        start_epoch = epoch_loaded + 1
        print(f"  Resuming from epoch {start_epoch}")
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)

    # ─── Mixed Precision (AMP) ────────────────────────────────
    use_amp = cfg.device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    total_epochs = cfg.train.epochs
    num_remaining = total_epochs - start_epoch + 1

    print(f"\n{'='*60}")
    print(f"  ThaoOCR Training")
    print(f"  Device: {device}")
    print(f"  AMP:    {'ON (FP16)' if use_amp else 'OFF'}")
    print(f"  Epochs: {start_epoch} → {total_epochs}")
    print(f"  Batch size: {cfg.train.batch_size}")
    if accum_steps > 1:
        eff_bs = cfg.train.batch_size * accum_steps
        print(f"  Grad accumulation: {accum_steps} steps  (effective batch size: {eff_bs})")
    print(f"  LR: {cfg.train.lr}")
    print(f"  Batches/epoch: {len(train_loader)}")
    print(f"  Optimizer steps/epoch: {steps_per_epoch}")
    print(f"  Total Optimizer Steps: {total_steps}")
    
    # --- Benchmark Speed ---
    benchmark_batches = min(50, len(train_loader))
    print(f"  Benchmarking speed ({benchmark_batches} batches)...", end="", flush=True)
    model.train()
    t0 = time.time()
    bench_steps = 0
    with torch.no_grad():
        for i, (x, *_) in enumerate(train_loader):
            if i >= benchmark_batches:
                break
            x = x.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                model(x)
            bench_steps += 1
    
    dt = time.time() - t0
    avg_batch_time = dt / max(1, bench_steps)
    estimated_total_seconds = avg_batch_time * (cfg.train.epochs * len(train_loader))
    print(f" Done! ({avg_batch_time*1000:.1f}ms/batch)")
    print(f"  Est. Total Time: {_fmt_time(estimated_total_seconds)}")
    print(f"{'='*60}\n")

    training_start = time.time()
    epoch_times = []  # track elapsed time per epoch for ETA

    try:
        for epoch in range(start_epoch, total_epochs + 1):
            t0 = time.time()

            avg_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, device,
                blank_id=vocab.blank_id,
                log_interval=cfg.train.log_interval,
                scaler=scaler,
                total_epochs=total_epochs,
                current_epoch=epoch,
                grad_accum_steps=accum_steps,
                grad_clip=cfg.train.grad_clip,
            )

            # Evaluate
            metrics = evaluate(model, val_loader, vocab, device)
            elapsed = time.time() - t0
            epoch_times.append(elapsed)

            # Compute ETA
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            epochs_left = total_epochs - epoch
            eta_remaining = avg_epoch_time * epochs_left
            total_elapsed = time.time() - training_start

            # Print epoch summary
            print(f"Epoch {epoch:03d}/{total_epochs} "
                  f"| loss={avg_loss:.4f} "
                  f"| CER={metrics['avg_cer']:.4f} "
                  f"| WER={metrics['avg_wer']:.4f} "
                  f"| exact={metrics['exact_match']:.3f} "
                  f"| epoch={_fmt_time(elapsed)} "
                  f"| total={_fmt_time(total_elapsed)} "
                  f"| ETA={_fmt_time(eta_remaining)}")
            print()

            # Test on custom image if provided
            if args.test_image:
                test_custom_image(model, vocab, args.test_image, device, cfg.model.target_h)

            # Save best checkpoint
            if metrics['avg_cer'] < best_cer:
                best_cer = metrics['avg_cer']
                save_checkpoint(
                    model, optimizer, scheduler, vocab, epoch, metrics,
                    os.path.join(cfg.train.checkpoint_dir, "best.pt"),
                    best_cer=best_cer,
                )
                print(f"  ★ New best CER: {best_cer:.4f}")

            # Always save latest (so you never lose more than 1 epoch)
            save_checkpoint(
                model, optimizer, scheduler, vocab, epoch, metrics,
                os.path.join(cfg.train.checkpoint_dir, "latest.pt"),
                best_cer=best_cer,
            )

            # Periodic checkpoint
            if epoch % cfg.train.save_every == 0:
                save_checkpoint(
                    model, optimizer, scheduler, vocab, epoch, metrics,
                    os.path.join(cfg.train.checkpoint_dir, f"epoch_{epoch:03d}.pt"),
                    best_cer=best_cer,
                )

        total_time = time.time() - training_start
        print(f"\n✓ Training complete! Best CER: {best_cer:.4f} | Total time: {_fmt_time(total_time)}")

    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user!")
        print("  Tips: Resume later using '--resume checkpoints/latest.pt'")
        sys.exit(0)

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n\nCUDA OUT OF MEMORY ERROR")
            print(f"   Your GPU ({torch.cuda.get_device_name(0)}) ran out of memory.")
            print(f"   Current batch size: {cfg.train.batch_size}")
            print("   ACTION: Reduce 'batch_size' in config.yml (try {cfg.train.batch_size // 2})")
            print("   You can resume training after changing the config.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            sys.exit(1)
        else:
            raise e


if __name__ == "__main__":
    main()
