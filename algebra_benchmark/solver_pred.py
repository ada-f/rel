#
# Minimal prediction helpers for algebra_benchmark evaluation.
# Parse model output to get answer index (0-7); majority vote; guard answer.
# Compatible with the rpmllm mamba environment (Python 3.10).
#

from __future__ import annotations

import re
from collections import Counter


def text2num(
    text: str,
    n_attr: int,
    n_return: int = 1,
    answer_queue: str = "",
) -> list[int]:
    """
    Extract answer index (0-7) from model output text.
    Prefer ``Answer N`` / ``Answer #N`` with N in 0–7 (label matches index).
    ``Answer 8`` is accepted as legacy (old eighth 1-based label → index 7).
    For a bare digit, prefers 0-7 (0-based index) first, then 1-8 (1-based label) for backward
    compatibility (e.g. ``8`` → index 7).
    n_attr and n_return are kept for compatibility with parent API; we return
    a list of length n_return (repeating the parsed index if n_return > 1).
    """
    if not text or not text.strip():
        return [0] * max(1, n_return)
    # Look for "Answer 3", "Answer #3", "3", "# 3", etc.
    text = text.strip()
    # Prefer explicit "Answer N" or "Answer #N"
    m = re.search(r"Answer\s*#?\s*(\d+)", text, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        if 0 <= num <= 7:
            idx = num
        elif num == 8:
            idx = 7  # legacy: old prompts used Answer 1..8
        else:
            idx = 0
        idx = max(0, min(7, idx))
        return [idx] * max(1, n_return)
    # Bare digit: prefer 0-based index (0-7), then 1-based option label (1-8)
    m = re.search(r"\b([0-7])\b", text)
    if m:
        idx = int(m.group(1))
        return [idx] * max(1, n_return)
    m = re.search(r"\b([1-8])\b", text)
    if m:
        num = int(m.group(1))
        idx = num - 1
        return [idx] * max(1, n_return)
    return [0] * max(1, n_return)


def majority_vote(predictions: list[int]) -> int:
    """Return the most common value; tie-break by first occurrence."""
    if not predictions:
        return 0
    counts = Counter(predictions)
    best = max(counts.keys(), key=lambda k: (counts[k], -predictions.index(k)))
    return best


def guard_answer(x: int | list[int]) -> int | list[int]:
    """Clamp answer index to 0-7. Accept int or list of ints."""
    if isinstance(x, list):
        return [max(0, min(7, int(v))) for v in x]
    return max(0, min(7, int(x)))
