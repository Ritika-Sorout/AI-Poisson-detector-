import numpy as np

from poissonguard.metrics import (
    average_precision,
    detection_rate_at_fpr,
    roc_auc,
    threshold_at_fpr,
)


def test_perfect_separation_auc_one():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert np.isclose(roc_auc(labels, scores), 1.0)
    assert np.isclose(average_precision(labels, scores), 1.0)


def test_random_auc_half():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=4000)
    scores = rng.random(4000)
    assert abs(roc_auc(labels, scores) - 0.5) < 0.05


def test_inverted_scores_auc_zero():
    labels = [0, 0, 1, 1]
    scores = [1.0, 0.9, 0.2, 0.1]
    assert np.isclose(roc_auc(labels, scores), 0.0)


def test_auc_handles_ties():
    labels = [0, 1, 0, 1]
    scores = [0.5, 0.5, 0.5, 0.5]
    assert np.isclose(roc_auc(labels, scores), 0.5)


def test_detection_rate_at_fpr():
    labels = [0] * 100 + [1] * 100
    scores = list(np.linspace(0, 1, 100)) + list(np.linspace(2, 3, 100))
    # positives entirely above negatives -> full detection at low FPR
    assert detection_rate_at_fpr(labels, scores, 0.05) == 1.0


def test_threshold_monotone_in_fpr():
    rng = np.random.default_rng(1)
    labels = [0] * 200 + [1] * 200
    scores = list(rng.normal(0, 1, 200)) + list(rng.normal(2, 1, 200))
    t_low = threshold_at_fpr(labels, scores, 0.01)
    t_high = threshold_at_fpr(labels, scores, 0.20)
    assert t_low >= t_high  # stricter FPR => higher threshold


def test_single_class_returns_nan():
    assert np.isnan(roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))
    assert np.isnan(average_precision([0, 0], [0.1, 0.2]))
