import numpy as np

from poissonguard.bucketing import BucketingConfig
from poissonguard.evaluation import (
    build_trained_pair,
    evaluate_detection,
    evaluate_poisoning,
)
from poissonguard.generators import generate_eval_windows

CFG = BucketingConfig()


def _pair():
    return build_trained_pair(weeks=4, n_entities=8, seed=0, bucketing=CFG)


def test_full_system_beats_legacy_auc():
    detector, legacy, population = _pair()
    windows = generate_eval_windows(population, CFG, seed=100)
    rep = evaluate_detection(detector, legacy, windows)
    assert rep.auc_full > 0.85
    assert rep.auc_full >= rep.auc_legacy


def test_shape_tests_add_value_over_rate_only():
    detector, legacy, population = _pair()
    windows = generate_eval_windows(population, CFG, seed=101)
    rep = evaluate_detection(detector, legacy, windows)
    # Shape-bearing attacks (bot/bursty) lift the full AUC above rate-only.
    assert rep.auc_full >= rep.auc_rate_only


def test_legacy_blind_to_shape_attacks():
    # A metronomic bot at *normal* daily volume: identical count, different shape.
    # Only the shape tests can catch it; the count-only legacy cannot.
    from poissonguard.evaluation import score_full, score_legacy
    from poissonguard.generators import LabeledWindow, make_attack, DAY, WEEK

    detector, legacy, population = _pair()
    profile = population[0]
    base = 500 * WEEK
    ts = make_attack("regular_bot", profile, base, CFG, np.random.default_rng(0), severity=1.0)
    lw = LabeledWindow(profile.entity, profile.event_type, ts, base, base + DAY, 1, "regular_bot")

    assert score_full(detector, lw) > score_legacy(legacy, lw)
    # legacy barely reacts (count looks normal), full system fires on shape.
    assert score_legacy(legacy, lw) < 3.0
    assert score_full(detector, lw) > 6.0


def test_poisoning_is_defended():
    detector, legacy, population = _pair()
    pr = evaluate_poisoning(detector, legacy, population[0], days=40, end_mult=3.0)
    assert pr.guard_frozen
    assert pr.guard_anchor < pr.target_rate          # never dragged to target
    assert pr.full_detects_target                    # target still flagged
    assert pr.legacy_lambda_final > pr.legacy_lambda_initial  # legacy poisoned upward
