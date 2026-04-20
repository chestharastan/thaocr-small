"""
ThaoOCR — Vocabulary / Tokenizer
Handles encoding text → tensor IDs and decoding IDs → text.
"""
import json
import torch
from typing import List, Optional
from config import VocabConfig


class Vocab:
    """
    Character-level vocabulary for CTC-based OCR.

    Index layout:
        0  = [BLANK] (CTC blank)
        1  = [PAD]
        2… = actual characters
    """

    def __init__(self, cfg: Optional[VocabConfig] = None):
        if cfg is None:
            cfg = VocabConfig()

        self.blank_token = cfg.blank_token
        self.pad_token = cfg.pad_token

        # Build character list
        all_chars = (
            cfg.latin_chars
            + cfg.digit_chars
            + cfg.punct_chars
            + cfg.khmer_consonants
            + cfg.khmer_indep_vowels
            + cfg.khmer_dep_vowels
            + cfg.khmer_signs
            + cfg.khmer_digits
            + cfg.khmer_punct
            + cfg.extra_chars
        )

        # Deduplicate while preserving order
        seen = set()
        chars = []
        for ch in all_chars:
            if ch not in seen:
                chars.append(ch)
                seen.add(ch)

        # Build the full index-to-string / string-to-index maps
        self.itos: List[str] = [cfg.blank_token, cfg.pad_token] + chars
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

        self.blank_id: int = self.stoi[cfg.blank_token]
        self.pad_id: int = self.stoi[cfg.pad_token]

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> torch.Tensor:
        """Encode a string into a tensor of token IDs. Unknown chars are skipped."""
        ids = []
        for ch in text:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
            # else: skip unknown chars (warn in debug mode)
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids) -> str:
        """Decode a list/tensor of IDs back to a string (raw, no CTC collapse)."""
        chars = []
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        for i in ids:
            token = self.itos[i]
            if token not in (self.blank_token, self.pad_token):
                chars.append(token)
        return "".join(chars)

    def ctc_decode_greedy(self, logits_TBV: torch.Tensor) -> List[str]:
        """
        Greedy CTC decoding.

        Args:
            logits_TBV: [T, B, V] raw logits from the model.

        Returns:
            List of decoded strings, one per batch element.
        """
        pred = logits_TBV.argmax(dim=-1)  # [T, B]
        T, B = pred.shape
        results = []

        for b in range(B):
            seq = pred[:, b].tolist()
            prev = None
            chars = []
            for t in seq:
                if t != self.blank_id and t != prev:
                    token = self.itos[t]
                    if token != self.pad_token:
                        chars.append(token)
                prev = t
            results.append("".join(chars))

        return results

    def ctc_decode_beam(self, logits_TBV: torch.Tensor, beam_width: int = 10) -> List[str]:
        """
        Beam search CTC decoding (prefix beam search).

        Args:
            logits_TBV: [T, B, V] raw logits
            beam_width: number of beams

        Returns:
            List of decoded strings, one per batch element.
        """
        import torch.nn.functional as F

        log_probs = F.log_softmax(logits_TBV, dim=-1)  # [T, B, V]
        T, B, V = log_probs.shape
        results = []

        for b in range(B):
            # beams: list of (prefix_tuple, log_prob_blank, log_prob_non_blank)
            beams = {(): (0.0, float('-inf'))}  # prefix → (p_blank, p_non_blank)

            for t in range(T):
                new_beams = {}

                for prefix, (p_b, p_nb) in beams.items():
                    p_total = _log_add(p_b, p_nb)

                    for c in range(V):
                        lp = log_probs[t, b, c].item()

                        if c == self.blank_id:
                            # Extend with blank
                            _beam_update(new_beams, prefix, p_total + lp, is_blank=True)
                        else:
                            new_prefix = prefix + (c,)
                            if len(prefix) > 0 and c == prefix[-1]:
                                # Same char as last: only extend if previous was blank
                                _beam_update(new_beams, new_prefix, p_b + lp, is_blank=False)
                                _beam_update(new_beams, prefix, p_nb + lp, is_blank=False)
                            else:
                                _beam_update(new_beams, new_prefix, p_total + lp, is_blank=False)

                # Prune to top-k beams
                scored = [(pref, _log_add(pb, pnb)) for pref, (pb, pnb) in new_beams.items()]
                scored.sort(key=lambda x: x[1], reverse=True)
                beams = {}
                for pref, _ in scored[:beam_width]:
                    beams[pref] = new_beams[pref]

            # Best beam
            best_prefix = max(beams, key=lambda p: _log_add(*beams[p]))
            chars = [self.itos[c] for c in best_prefix if self.itos[c] not in (self.blank_token, self.pad_token)]
            results.append("".join(chars))

        return results

    @classmethod
    def build_from_labels(cls, label_files: List[str]) -> "Vocab":
        """
        Build a Vocab by scanning label files for all unique characters.

        Each line in a label file is expected to be:
            /path/to/image.png<TAB>label text

        Starts from the base VocabConfig and extends with any chars found
        in the label files that are not already present.
        """
        vocab = cls()  # build default vocab from VocabConfig
        all_chars: set = set()
        for path in label_files:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if "\t" in line:
                        text = line.split("\t", 1)[1]
                    else:
                        text = line  # fallback: treat whole line as text
                    all_chars.update(text)
        new_chars = "".join(ch for ch in sorted(all_chars) if ch not in vocab.stoi)
        if new_chars:
            vocab.extend_vocab(new_chars)
        return vocab

    def save(self, path: str) -> None:
        """Save vocabulary to a JSON file."""
        data = {
            "blank_token": self.blank_token,
            "pad_token": self.pad_token,
            "itos": self.itos,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_file(cls, path: str) -> "Vocab":
        """Load a previously saved vocabulary from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls.__new__(cls)
        vocab.blank_token = data["blank_token"]
        vocab.pad_token = data["pad_token"]
        vocab.itos = data["itos"]
        vocab.stoi = {ch: i for i, ch in enumerate(vocab.itos)}
        vocab.blank_id = vocab.stoi[vocab.blank_token]
        vocab.pad_id = vocab.stoi[vocab.pad_token]
        return vocab

    def extend_vocab(self, new_chars: str):
        """Add new characters to the vocabulary dynamically."""
        for ch in new_chars:
            if ch not in self.stoi:
                idx = len(self.itos)
                self.itos.append(ch)
                self.stoi[ch] = idx

    def coverage_report(self, texts: List[str]) -> dict:
        """Check how many characters in the given texts are covered by the vocab."""
        total = 0
        covered = 0
        missing = set()
        for text in texts:
            for ch in text:
                total += 1
                if ch in self.stoi:
                    covered += 1
                else:
                    missing.add(ch)
        return {
            "total_chars": total,
            "covered": covered,
            "coverage": covered / max(1, total),
            "missing_chars": sorted(missing),
            "num_missing_unique": len(missing),
        }


# ─── Helper functions for beam search ─────────────────────────────

import math

def _log_add(a: float, b: float) -> float:
    """Numerically stable log-add-exp."""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))


def _beam_update(beams: dict, prefix: tuple, score: float, is_blank: bool):
    """Update beam with additive log probabilities."""
    if prefix not in beams:
        beams[prefix] = (float('-inf'), float('-inf'))

    p_b, p_nb = beams[prefix]
    if is_blank:
        p_b = _log_add(p_b, score)
    else:
        p_nb = _log_add(p_nb, score)
    beams[prefix] = (p_b, p_nb)
