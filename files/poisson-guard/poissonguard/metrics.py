"""Lightweight ranking metrics (no scikit-learn dependency).

Scores are oriented so that *higher = more anomalous*.
"""

from __future__ import annotations

import numpy as np


def roc_auc(labels, scores) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney U) identity."""
    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = _average_ranks(s[order])
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_ranks(sorted_vals: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged, for an already-sorted array."""
    n = len(sorted_vals)
    ranks = np.arange(1, n + 1, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def average_precision(labels, scores) -> float:
    """Area under the precision-recall curve (average precision)."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / y.sum()
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def threshold_at_fpr(labels, scores, target_fpr: float) -> float:
    """Smallest score threshold whose FPR <= target on the negatives."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    neg = np.sort(s[y == 0])[::-1]
    if neg.size == 0:
        return float("inf")
    k = int(np.floor(target_fpr * neg.size))
    k = min(max(k, 0), neg.size - 1)
    return float(neg[k])


def detection_rate_at_fpr(labels, scores, target_fpr: float = 0.05) -> float:
    """True-positive rate when the threshold is set to the target FPR."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    thr = threshold_at_fpr(y, s, target_fpr)
    pos = s[y == 1]
    if pos.size == 0:
        return float("nan")
    return float((pos >= thr).mean())
