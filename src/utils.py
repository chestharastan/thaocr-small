"""
ThaoOCR — Image utilities and helpers
"""
import cv2
import numpy as np
import torch


# ─── Image loading / preprocessing ────────────────────────────────

def read_grayscale(path: str) -> np.ndarray:
    """Read an image as grayscale. Raises FileNotFoundError if missing."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def resize_keep_ratio(img: np.ndarray, target_h: int = 48) -> np.ndarray:
    """Resize image to target height while keeping aspect ratio."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"Invalid image dimensions: {h}x{w}")
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
    return img


def normalize_img(img: np.ndarray) -> torch.Tensor:
    """
    Convert uint8 grayscale [H,W] → float tensor [H,W], normalized to [-1, 1].
    """
    x = torch.from_numpy(img).float() / 255.0
    x = (x - 0.5) / 0.5  # [-1, 1]
    return x


def preprocess_image(path: str, target_h: int = 48) -> torch.Tensor:
    """
    Full pipeline: read → resize → normalize → [1, 1, H, W] tensor.
    Ready for model input (single image, no batching).
    """
    img = read_grayscale(path)
    img = resize_keep_ratio(img, target_h)
    x = normalize_img(img)        # [H, W]
    x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return x


# ─── Augmentation ─────────────────────────────────────────────────

class ImageAugmentor:
    """
    Simple augmentations for training OCR.
    All operations work on uint8 grayscale [H, W] numpy arrays.
    """

    def __init__(self, seed: int = None):
        self.rng = np.random.RandomState(seed)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply random augmentations."""
        if self.rng.random() < 0.3:
            img = self._adjust_brightness(img)
        if self.rng.random() < 0.3:
            img = self._adjust_contrast(img)
        if self.rng.random() < 0.2:
            img = self._gaussian_noise(img)
        if self.rng.random() < 0.2:
            img = self._gaussian_blur(img)
        if self.rng.random() < 0.15:
            img = self._erosion_dilation(img)
        return img

    def _adjust_brightness(self, img: np.ndarray) -> np.ndarray:
        delta = self.rng.randint(-30, 31)
        img = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        return img

    def _adjust_contrast(self, img: np.ndarray) -> np.ndarray:
        factor = self.rng.uniform(0.7, 1.3)
        mean = img.mean()
        img = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
        return img

    def _gaussian_noise(self, img: np.ndarray) -> np.ndarray:
        sigma = self.rng.uniform(1, 10)
        noise = self.rng.randn(*img.shape) * sigma
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img

    def _gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        k = self.rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
        return img

    def _erosion_dilation(self, img: np.ndarray) -> np.ndarray:
        k = self.rng.choice([2, 3])
        kernel = np.ones((k, k), np.uint8)
        if self.rng.random() < 0.5:
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)
        return img


# ─── Metrics ──────────────────────────────────────────────────────

def edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_cer(pred: str, target: str) -> float:
    """Character Error Rate = edit_distance / len(target)."""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return edit_distance(pred, target) / len(target)


def compute_wer(pred: str, target: str) -> float:
    """Word Error Rate = edit_distance(pred_words, target_words) / len(target_words)."""
    pred_words = pred.split()
    target_words = target.split()
    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    # Standard WER compares lists of words, not the joint string
    return edit_distance(pred_words, target_words) / len(target_words)


def compute_accuracy(preds: list, targets: list) -> dict:
    """
    Compute batch-level metrics.

    Returns dict with:
        exact_match, avg_cer, avg_wer
    """
    assert len(preds) == len(targets)
    total = len(preds)
    exact = sum(1 for p, t in zip(preds, targets) if p == t)
    cers = [compute_cer(p, t) for p, t in zip(preds, targets)]
    wers = [compute_wer(p, t) for p, t in zip(preds, targets)]

    return {
        "exact_match": exact / max(1, total),
        "avg_cer": sum(cers) / max(1, total),
        "avg_wer": sum(wers) / max(1, total),
    }
