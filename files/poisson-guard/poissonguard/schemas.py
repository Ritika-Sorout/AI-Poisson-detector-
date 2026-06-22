"""Core data structures shared across every PoissonGuard module.

These are deliberately dependency-light (stdlib + numpy) so the schema layer can
be imported by anything without pulling in scipy/fastapi. Pydantic request/
response models live in ``serving.py`` instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

SECONDS_PER_HOUR = 3600.0


class Bucket(str, Enum):
    """Diurnal time-of-day buckets (see ``bucketing.py``)."""

    BUSINESS = "business"
    OFFHOURS = "offhours"


class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_pvalue(cls, p: float) -> "Severity":
        if p >= 0.05:
            return cls.NONE
        if p >= 1e-2:
            return cls.LOW
        if p >= 1e-3:
            return cls.MEDIUM
        if p >= 1e-5:
            return cls.HIGH
        return cls.CRITICAL


class DriftDecision(str, Enum):
    """Verdict of the drift-integrity gate for a candidate baseline update."""

    ACCEPT = "accept"           # benign: fold the observation into the baseline
    FREEZE = "freeze"           # suspicious slow drift: refuse to update
    REGIME_CHANGE = "regime"    # large abrupt shift: flag, reset baseline


@dataclass
class Event:
    """A single timestamped event for an entity."""

    entity: str
    event_type: str
    timestamp: float  # epoch seconds


@dataclass
class Window:
    """A batch of events for one ``(entity, event_type)`` over ``[start, end]``.

    ``start``/``end`` define the *exposure* (observation interval in seconds),
    which is what the rate model divides by. They default to the span of the
    timestamps but should be set explicitly when the true observation window is
    known (it usually is, and assuming otherwise biases the rate).
    """

    entity: str
    event_type: str
    timestamps: np.ndarray
    start: float
    end: float

    @classmethod
    def from_timestamps(
        cls,
        entity: str,
        event_type: str,
        timestamps,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> "Window":
        ts = np.asarray(sorted(float(t) for t in timestamps), dtype=float)
        if start is None:
            start = float(ts[0]) if ts.size else 0.0
        if end is None:
            end = float(ts[-1]) if ts.size else start
        return cls(entity=entity, event_type=event_type, timestamps=ts, start=start, end=end)

    @property
    def count(self) -> int:
        return int(self.timestamps.size)

    @property
    def duration_seconds(self) -> float:
        return max(self.end - self.start, 0.0)

    @property
    def duration_hours(self) -> float:
        return self.duration_seconds / SECONDS_PER_HOUR

    def interarrival_seconds(self) -> np.ndarray:
        """Gaps between consecutive events; empty if fewer than 2 events."""
        if self.count < 2:
            return np.empty(0, dtype=float)
        return np.diff(self.timestamps)

    def empirical_rate_per_hour(self) -> float:
        h = self.duration_hours
        return self.count / h if h > 0 else 0.0


@dataclass
class Baseline:
    """Learned per-(entity, event_type, bucket) model of the event rate.

    The rate ``lambda`` (events/hour) is given a Gamma(alpha, beta) prior/
    posterior. ``alpha``/``beta`` are the posterior shape/rate after observing
    training exposure. ``rate_history`` retains recent accepted rate estimates so
    the drift gate can reason about slow poisoning over time.
    """

    entity: str
    event_type: str
    bucket: Bucket
    alpha: float           # Gamma posterior shape  (>= prior)
    beta: float            # Gamma posterior rate (in 1/hour units of exposure)
    n_events: int          # total events folded in
    exposure_hours: float  # total exposure folded in
    rate_history: list = field(default_factory=list)

    @property
    def posterior_mean_rate(self) -> float:
        """Posterior mean of lambda (events/hour)."""
        return self.alpha / self.beta if self.beta > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "entity": self.entity,
            "event_type": self.event_type,
            "bucket": self.bucket.value,
            "alpha": self.alpha,
            "beta": self.beta,
            "n_events": self.n_events,
            "exposure_hours": self.exposure_hours,
            "rate_history": list(self.rate_history),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Baseline":
        return cls(
            entity=d["entity"],
            event_type=d["event_type"],
            bucket=Bucket(d["bucket"]),
            alpha=float(d["alpha"]),
            beta=float(d["beta"]),
            n_events=int(d["n_events"]),
            exposure_hours=float(d["exposure_hours"]),
            rate_history=list(d.get("rate_history", [])),
        )

    @staticmethod
    def key(entity: str, event_type: str, bucket: Bucket) -> str:
        return f"{entity}|{event_type}|{bucket.value}"


@dataclass
class SubScore:
    """Output of one sub-detector (rate / Fano / exponentiality)."""

    name: str
    p_value: float
    statistic: float
    detail: str = ""

    def __post_init__(self):
        # Clamp to a sane open interval; Fisher's method needs p in (0, 1].
        if not math.isfinite(self.p_value):
            self.p_value = 1.0
        self.p_value = min(max(self.p_value, 1e-12), 1.0)


@dataclass
class DetectionResult:
    """Final per-window verdict returned by the detector and serving layer."""

    entity: str
    event_type: str
    bucket: Bucket
    observed_count: int
    expected_count: float
    sub_scores: list  # list[SubScore]
    fused_p_value: float
    severity: Severity
    drift_decision: DriftDecision
    is_anomaly: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "entity": self.entity,
            "event_type": self.event_type,
            "bucket": self.bucket.value,
            "observed_count": self.observed_count,
            "expected_count": self.expected_count,
            "sub_scores": [
                {"name": s.name, "p_value": s.p_value, "statistic": s.statistic, "detail": s.detail}
                for s in self.sub_scores
            ],
            "fused_p_value": self.fused_p_value,
            "severity": self.severity.value,
            "drift_decision": self.drift_decision.value,
            "is_anomaly": self.is_anomaly,
            "detail": self.detail,
        }
