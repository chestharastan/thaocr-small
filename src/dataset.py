"""
ThaoOCR — Dataset and DataLoader utilities
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional

from utils import read_grayscale, resize_keep_ratio, normalize_img, ImageAugmentor
from vocab import Vocab


class LineRecDataset(Dataset):
    """
    OCR line recognition dataset.

    Expects a text file with lines:
        /path/to/image.png<TAB>label text

    Returns (image_tensor, label_ids, raw_label_str).
    """

    def __init__(
        self,
        txt_path: str,
        vocab: Vocab,
        target_h: int = 48,
        augment: bool = False,
        max_label_len: Optional[int] = None,
        data_root: Optional[str] = None,
    ):
        self.vocab = vocab
        self.target_h = target_h
        self.augmentor = ImageAugmentor() if augment else None
        self.max_label_len = max_label_len
        self.data_root = data_root

        self.items: List[Tuple[str, str]] = []
        self._load(txt_path)

    def _load(self, txt_path: str):
        """Parse the annotation file."""
        skipped = 0
        with open(txt_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip("\n").strip("\r")
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    skipped += 1
                    continue
                path, label = parts
                if self.max_label_len and len(label) > self.max_label_len:
                    skipped += 1
                    continue
                path = path.strip()
                # Resolve image path relative to data_root if provided
                if self.data_root and not os.path.isabs(path):
                    path = os.path.join(self.data_root, path)
                self.items.append((path, label))

        if skipped > 0:
            print(f"[Dataset] Skipped {skipped} entries from {txt_path}")
        print(f"[Dataset] Loaded {len(self.items)} samples from {txt_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]

        # Read and preprocess image
        img = read_grayscale(path)

        # Apply augmentation (training only)
        if self.augmentor is not None:
            img = self.augmentor(img)

        img = resize_keep_ratio(img, self.target_h)
        x = normalize_img(img)           # [H, W]
        x = x.unsqueeze(0)               # [1, H, W]

        # Encode label
        y = self.vocab.encode(label)      # [L]

        return x, y, label


def collate_fn(batch):
    """
    Custom collation for variable-width images.

    Returns:
        x_pad:    [B, 1, H, max_W]  padded image batch
        x_lens:   [B]               original widths (for CTC input lengths)
        y_cat:    [sum(L)]           concatenated labels (CTC format)
        y_lens:   [B]               individual label lengths
        raw:      tuple of str       original label strings
    """
    xs, ys, raw = zip(*batch)

    # Verify all have same height
    heights = set(x.shape[1] for x in xs)
    assert len(heights) == 1, f"All images must have same height, got {heights}"

    H = xs[0].shape[1]
    max_w = max(x.shape[2] for x in xs)

    # Pad images (pad with 0 = black / neutral after normalization)
    x_pad = torch.zeros(len(xs), 1, H, max_w, dtype=torch.float32)
    x_lens = torch.zeros(len(xs), dtype=torch.long)

    for i, x in enumerate(xs):
        w = x.shape[2]
        x_pad[i, :, :, :w] = x
        x_lens[i] = w

    # Concatenate labels (CTC format)
    y_lens = torch.tensor([len(y) for y in ys], dtype=torch.long)
    y_cat = torch.cat(ys, dim=0) if any(len(y) > 0 for y in ys) else torch.tensor([], dtype=torch.long)

    return x_pad, y_cat, y_lens, raw


def build_dataloaders(
    vocab: Vocab,
    train_txt: Optional[str] = None,
    val_txt: Optional[str] = None,
    label_file: Optional[str] = None,
    val_ratio: float = 0.1,
    target_h: int = 48,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    data_root: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders.

    Accepts either:
      - ``train_txt`` + ``val_txt``: pre-split label files
      - ``label_file`` + ``val_ratio``: single label file split randomly
    """
    import random

    if label_file is not None:
        # Single-file mode: split into train/val by val_ratio
        with open(label_file, encoding="utf-8") as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        random.shuffle(lines)
        n_val = max(1, int(len(lines) * val_ratio))
        val_lines   = lines[:n_val]
        train_lines = lines[n_val:]

        import tempfile, os
        def _write_tmp(line_list):
            fd, path = tempfile.mkstemp(suffix=".txt", text=True)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write("\n".join(line_list))
            return path

        train_txt = _write_tmp(train_lines)
        val_txt   = _write_tmp(val_lines)
        _cleanup   = [train_txt, val_txt]
    else:
        if train_txt is None or val_txt is None:
            raise ValueError("Provide either label_file or both train_txt and val_txt.")
        _cleanup = []

    train_ds = LineRecDataset(
        train_txt, vocab, target_h=target_h, augment=True, data_root=data_root
    )
    val_ds = LineRecDataset(
        val_txt, vocab, target_h=target_h, augment=False, data_root=data_root
    )

    for p in _cleanup:
        try:
            import os; os.unlink(p)
        except OSError:
            pass

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
