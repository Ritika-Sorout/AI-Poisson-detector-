"""Detector orchestrator: composes every module into a verdict.

Pipeline for one ``(entity, event_type)`` raw event stream:

    bucketize ->  per bucket:
        rate p-value      (bayes_rate)
        Fano + expo tests (shape_tests)
        fuse              (fusion: Fisher)
        drift gate        (drift_guard: accept / freeze / regime)
    -> DetectionResult per active bucket

The detector owns the persistent state -- the learned baselines and the
drift-guard state per ``(entity, event_type, bucket)`` -- and can serialize it
to / from JSON for the train and serve pipelines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import bayes_rate, shape_tests
from .bayes_rate import Prior
from .bucketing import BucketingConfig, bucketize
from .drift_guard import DriftGuard, DriftGuardConfig
from .fusion import fisher_combine
from .schemas import (
    Baseline,
    Bucket,
    DetectionResult,
    DriftDecision,
    Severity,
    Window,
)

_SEVERITY_ORDER = [Severity.NONE, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]


def _max_severity(a: Severity, b: Severity) -> Severity:
    return a if _SEVERITY_ORDER.index(a) >= _SEVERITY_ORDER.index(b) else b


@dataclass
class DetectorConfig:
    prior: Prior = field(default_factory=Prior)
    drift: DriftGuardConfig = field(default_factory=DriftGuardConfig)
    bucketing: BucketingConfig = field(default_factory=BucketingConfig)
    anomaly_threshold: float = 1e-3   # fused p below this => anomaly
    tail: str = "greater"
    min_events_to_score: int = 1


class Detector:
    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self.baselines: dict[str, Baseline] = {}
        self.guards: dict[str, DriftGuard] = {}

    # ------------------------------------------------------------------ state
    def set_baseline(self, baseline: Baseline) -> None:
        key = Baseline.key(baseline.entity, baseline.event_type, baseline.bucket)
        self.baselines[key] = baseline
        self.guards[key] = DriftGuard(anchor=baseline.posterior_mean_rate, config=self.config.drift)

    def _cold_baseline(self, entity: str, event_type: str, bucket: Bucket) -> Baseline:
        b = bayes_rate.fit_baseline(entity, event_type, bucket, count=0, exposure_hours=0.0,
                                    prior=self.config.prior)
        self.set_baseline(b)
        return b

    # ------------------------------------------------------------------ scoring
    def _score_bucket(self, baseline: Baseline, window: Window, update: bool) -> DetectionResult:
        key = Baseline.key(baseline.entity, baseline.event_type, baseline.bucket)
        exposure = window.duration_hours
        count = window.count

        sub_scores = []
        rate_p, rate_stat = bayes_rate.predictive_pvalue(baseline, count, exposure, tail=self.config.tail)
        expected = bayes_rate.predictive_mean(baseline, exposure)
        from .schemas import SubScore
        sub_scores.append(SubScore("rate", rate_p, rate_stat,
                                   f"obs={count}, exp={expected:.2f}/win"))
        sub_scores.extend(shape_tests.run_shape_tests(window))

        fusion = fisher_combine(sub_scores)

        # Drift gate governs *baseline updates*, so it only runs (and mutates
        # state) when we are actually updating. For pure scoring we report the
        # current frozen status without advancing the gate.
        guard = self.guards.get(key) or DriftGuard(anchor=baseline.posterior_mean_rate,
                                                   config=self.config.drift)
        self.guards[key] = guard
        observed_rate = window.empirical_rate_per_hour()
        if update and count > 0:
            drift_decision = guard.observe(observed_rate)
            if drift_decision == DriftDecision.ACCEPT:
                self._fold_in(baseline, count, exposure, observed_rate)
        else:
            drift_decision = DriftDecision.FREEZE if guard.frozen else DriftDecision.ACCEPT

        severity = Severity.from_pvalue(fusion.fused_p_value)
        if drift_decision == DriftDecision.FREEZE:
            severity = _max_severity(severity, Severity.HIGH)
        elif drift_decision == DriftDecision.REGIME_CHANGE:
            severity = _max_severity(severity, Severity.MEDIUM)

        is_anomaly = (
            fusion.fused_p_value < self.config.anomaly_threshold
            or drift_decision in (DriftDecision.FREEZE, DriftDecision.REGIME_CHANGE)
        )

        detail = f"dominant={fusion.dominant}; drift={drift_decision.value}"
        return DetectionResult(
            entity=baseline.entity,
            event_type=baseline.event_type,
            bucket=baseline.bucket,
            observed_count=count,
            expected_count=expected,
            sub_scores=sub_scores,
            fused_p_value=fusion.fused_p_value,
            severity=severity,
            drift_decision=drift_decision,
            is_anomaly=is_anomaly,
            detail=detail,
        )

    @staticmethod
    def _fold_in(baseline: Baseline, count: int, exposure: float, rate: float) -> None:
        baseline.alpha, baseline.beta = bayes_rate.posterior(baseline.alpha, baseline.beta, count, exposure)
        baseline.n_events += int(count)
        baseline.exposure_hours += float(exposure)
        baseline.rate_history.append(rate)

    # ------------------------------------------------------------------ public
    def detect(
        self,
        entity: str,
        event_type: str,
        timestamps,
        start: float,
        end: float,
        update_baseline: bool = False,
    ) -> list[DetectionResult]:
        buckets = bucketize(entity, event_type, timestamps, start, end, self.config.bucketing)
        results: list[DetectionResult] = []
        for bucket, window in buckets.items():
            key = Baseline.key(entity, event_type, bucket)
            baseline = self.baselines.get(key) or self._cold_baseline(entity, event_type, bucket)
            if window.count < self.config.min_events_to_score and baseline.n_events == 0:
                continue  # nothing to learn from or score
            results.append(self._score_bucket(baseline, window, update_baseline))
        return results

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "baselines": {k: b.to_dict() for k, b in self.baselines.items()},
            "guards": {k: g.state_dict() for k, g in self.guards.items()},
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str, config: DetectorConfig | None = None) -> "Detector":
        with open(path) as f:
            data = json.load(f)
        det = cls(config)
        for k, bd in data.get("baselines", {}).items():
            det.baselines[k] = Baseline.from_dict(bd)
        for k, gs in data.get("guards", {}).items():
            det.guards[k] = DriftGuard.from_state(gs, det.config.drift)
        # ensure every baseline has a guard
        for k, b in det.baselines.items():
            det.guards.setdefault(k, DriftGuard(anchor=b.posterior_mean_rate, config=det.config.drift))
        return det
