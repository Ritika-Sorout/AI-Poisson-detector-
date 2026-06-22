"""Evaluation + ablation engine.

Produces three kinds of evidence:

1. **Detection AUC / AP** on labeled windows, overall and per attack family.
2. **Ablation**: full system vs rate-only (shape tests removed) vs the legacy
   detector -- isolating what each component contributes.
3. **Poisoning defense**: replay a boil-the-frog ramp through both detectors
   with online baseline updates and check whether the attacker's target rate is
   still detected afterwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import bayes_rate
from .bucketing import bucketize
from .detector import Detector
from .generators import (
    ATTACK_TYPES,
    EntityProfile,
    generate_poisoning_sequence,
    generate_training,
    make_population,
)
from .legacy import LegacyDetector
from .metrics import average_precision, detection_rate_at_fpr, roc_auc
from .schemas import Bucket, DriftDecision, SECONDS_PER_HOUR

_BIG = 12.0  # -log10(p) cap for p == 1e-12


def _nlog10(p: float) -> float:
    return min(-math.log10(max(p, 1e-12)), _BIG)


def score_full(detector: Detector, lw) -> float:
    """Anomaly score from the full system: max over buckets of -log10(fused_p)."""
    results = detector.detect(lw.entity, lw.event_type, lw.timestamps, lw.start, lw.end)
    if not results:
        return 0.0
    return max(_nlog10(r.fused_p_value) for r in results)


def score_rate_only(detector: Detector, lw) -> float:
    """Ablation: rate p-value only (shape tests and drift disabled)."""
    buckets = bucketize(lw.entity, lw.event_type, lw.timestamps, lw.start, lw.end,
                        detector.config.bucketing)
    best = 0.0
    for bucket, window in buckets.items():
        key = f"{lw.entity}|{lw.event_type}|{bucket.value}"
        baseline = detector.baselines.get(key)
        if baseline is None:
            continue
        p, _ = bayes_rate.predictive_pvalue(baseline, window.count, window.duration_hours,
                                            tail=detector.config.tail)
        best = max(best, _nlog10(p))
    return best


def score_legacy(legacy: LegacyDetector, lw) -> float:
    p, _ = legacy.detect(lw.entity, lw.event_type, lw.timestamps, lw.start, lw.end)
    return _nlog10(p)


@dataclass
class DetectionReport:
    auc_full: float
    auc_rate_only: float
    auc_legacy: float
    ap_full: float
    ap_legacy: float
    per_attack_full: dict   # attack_type -> detection rate @ FPR
    per_attack_legacy: dict


def evaluate_detection(detector: Detector, legacy: LegacyDetector, windows,
                       target_fpr: float = 0.05) -> DetectionReport:
    labels = np.array([w.label for w in windows])
    s_full = np.array([score_full(detector, w) for w in windows])
    s_rate = np.array([score_rate_only(detector, w) for w in windows])
    s_leg = np.array([score_legacy(legacy, w) for w in windows])

    per_attack_full, per_attack_legacy = {}, {}
    neg_mask = labels == 0
    for atk in ATTACK_TYPES:
        mask = neg_mask | np.array([w.attack_type == atk for w in windows])
        sub_labels = labels[mask]
        per_attack_full[atk] = detection_rate_at_fpr(sub_labels, s_full[mask], target_fpr)
        per_attack_legacy[atk] = detection_rate_at_fpr(sub_labels, s_leg[mask], target_fpr)

    return DetectionReport(
        auc_full=roc_auc(labels, s_full),
        auc_rate_only=roc_auc(labels, s_rate),
        auc_legacy=roc_auc(labels, s_leg),
        ap_full=average_precision(labels, s_full),
        ap_legacy=average_precision(labels, s_leg),
        per_attack_full=per_attack_full,
        per_attack_legacy=per_attack_legacy,
    )


@dataclass
class PoisoningReport:
    target_rate: float
    guard_frozen: bool
    guard_anchor: float
    legacy_lambda_initial: float
    legacy_lambda_final: float
    full_detects_target: bool
    legacy_detects_target: bool
    freeze_day: int


def evaluate_poisoning(detector: Detector, legacy: LegacyDetector, profile: EntityProfile,
                       days: int = 40, end_mult: float = 3.0) -> PoisoningReport:
    cfg = detector.config.bucketing
    seq = generate_poisoning_sequence(profile, cfg, days=days, end_rate_mult=end_mult)
    key = f"{profile.entity}|{profile.event_type}|business"

    # Start from a clean gate so a prior detection pass cannot contaminate it.
    from .drift_guard import DriftGuard
    detector.guards[key] = DriftGuard(anchor=detector.baselines[key].posterior_mean_rate,
                                      config=detector.config.drift)

    legacy_lambda_initial = legacy.lambda_for(profile.entity, profile.event_type)
    freeze_day = -1
    for d, (ts, s, e, _mult) in enumerate(seq):
        results = detector.detect(profile.entity, profile.event_type, ts, s, e, update_baseline=True)
        legacy.update(profile.entity, profile.event_type, ts, s, e)
        biz = [r for r in results if r.bucket == Bucket.BUSINESS]
        if freeze_day < 0 and biz and biz[0].drift_decision == DriftDecision.FREEZE:
            freeze_day = d

    guard = detector.guards[key]
    target_rate = profile.business_rate * end_mult

    # Would each detector flag a full day at the attacker's target rate?
    full_p = _target_full_pvalue(detector, profile, target_rate)
    legacy_lambda_final = legacy.lambda_for(profile.entity, profile.event_type)
    legacy_target_p = legacy.score(profile.entity, profile.event_type,
                                   count=int(target_rate * 8), window_hours=24.0)

    return PoisoningReport(
        target_rate=target_rate,
        guard_frozen=guard.frozen,
        guard_anchor=guard.anchor,
        legacy_lambda_initial=legacy_lambda_initial,
        legacy_lambda_final=legacy_lambda_final,
        full_detects_target=full_p < detector.config.anomaly_threshold,
        legacy_detects_target=legacy_target_p < legacy.anomaly_threshold,
        freeze_day=freeze_day,
    )


def poisoning_trace(detector: Detector, legacy: LegacyDetector, profile: EntityProfile,
                    days: int = 40, end_mult: float = 3.0) -> list[dict]:
    """Per-day trace of the boil-the-frog ramp, for dashboard plotting.

    Each row reports the attacker's target rate that day, the observed business
    rate, the gate's held anchor and CUSUM, and its decision -- so a chart can
    show the anchor flat-lining at the freeze point while the attack ramps on.
    """
    cfg = detector.config.bucketing
    seq = generate_poisoning_sequence(profile, cfg, days=days, end_rate_mult=end_mult)
    key = f"{profile.entity}|{profile.event_type}|business"

    from .drift_guard import DriftGuard
    detector.guards[key] = DriftGuard(anchor=detector.baselines[key].posterior_mean_rate,
                                      config=detector.config.drift)

    trace = []
    for d, (ts, s, e, mult) in enumerate(seq):
        results = detector.detect(profile.entity, profile.event_type, ts, s, e, update_baseline=True)
        legacy.update(profile.entity, profile.event_type, ts, s, e)
        biz = [r for r in results if r.bucket == Bucket.BUSINESS]
        guard = detector.guards[key]
        observed = biz[0].observed_count if biz else 0
        decision = biz[0].drift_decision.value if biz else "accept"
        trace.append({
            "day": d,
            "attacker_rate": round(profile.business_rate * mult, 3),
            "observed_rate": round(observed / 8.0, 3),
            "anchor": round(guard.anchor, 3),
            "cusum": round(guard.cusum, 3),
            "frozen": guard.frozen,
            "decision": decision,
            "legacy_lambda": round(legacy.lambda_for(profile.entity, profile.event_type), 3),
        })
    return trace


def _target_full_pvalue(detector: Detector, profile: EntityProfile, target_rate: float) -> float:
    """Rate p-value for a full business day at ``target_rate`` under current baseline."""
    key = f"{profile.entity}|{profile.event_type}|business"
    baseline = detector.baselines[key]
    business_hours = 8.0
    count = int(target_rate * business_hours)
    p, _ = bayes_rate.predictive_pvalue(baseline, count, business_hours, tail=detector.config.tail)
    return p


def build_trained_pair(weeks: int = 4, n_entities: int = 12, seed: int = 0, bucketing=None):
    """Train a fresh full Detector and a legacy detector on the same data."""
    from .training import train_detector
    population = make_population(n_entities, seed=seed)
    streams, start, end = generate_training(population, weeks=weeks, cfg=bucketing, seed=seed + 1)
    detector = train_detector(streams, start, end, bucketing=bucketing)
    legacy = LegacyDetector()
    for (entity, event_type), ts in streams.items():
        legacy.fit_stream(entity, event_type, ts, start, end)
    return detector, legacy, population
