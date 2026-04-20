"""
metrics.py
==========
OCR evaluation metrics: CER, WER, and exact match.
"""
from typing import List


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_metrics(preds: List[str], targets: List[str]) -> dict:
    """
    Compute CER, WER, and exact match over a list of prediction/target pairs.

    Returns:
        {
            "avg_cer":     float,   # mean character error rate
            "avg_wer":     float,   # mean word error rate
            "exact_match": float,   # fraction of exactly correct predictions
        }
    """
    if not preds:
        return {"avg_cer": 0.0, "avg_wer": 0.0, "exact_match": 0.0}

    total_cer = 0.0
    total_wer = 0.0
    exact = 0

    for pred, target in zip(preds, targets):
        # CER: edit distance at character level, normalised by target length
        t_len = max(len(target), 1)
        total_cer += _edit_distance(pred, target) / t_len

        # WER: edit distance at word level, normalised by target word count
        pred_words   = pred.split()
        target_words = target.split()
        w_len = max(len(target_words), 1)
        total_wer += _edit_distance(pred_words, target_words) / w_len

        if pred == target:
            exact += 1

    n = len(preds)
    return {
        "avg_cer":     total_cer / n,
        "avg_wer":     total_wer / n,
        "exact_match": exact / n,
    }
