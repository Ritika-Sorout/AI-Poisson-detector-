"""Legacy baseline detector -- faithful reimplementation of the original.

This is intentionally *not* improved. It reproduces the original design so the
evaluation can measure exactly what each PoissonGuard component buys:

* rate = ``total_events / full_calendar_span`` -- the **calendar-span dilution
  bug**: idle nights/weekends inflate the denominator, under-estimating the
  true active-hours rate;
* a **plug-in Poisson** tail probability (no posterior uncertainty);
* **no shape tests** (count only -> blind to bots/bursts at normal volume);
* **no diurnal bucketing** (one rate for all hours);
* a baseline that simply re-learns from all observed data -> **poisonable** by a
  slow ramp.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .schemas import SECONDS_PER_HOUR


@dataclass
class _LegacyBaseline:
    total_events: int
    span_hours: float

    @property
    def lambda_per_hour(self) -> float:
        return self.total_events / self.span_hours if self.span_hours > 0 else 0.0


@dataclass
class LegacyDetector:
    anomaly_threshold: float = 1e-3
    baselines: dict = field(default_factory=dict)

    @staticmethod
    def _key(entity: str, event_type: str) -> str:
        return f"{entity}|{event_type}"

    def fit_stream(self, entity: str, event_type: str, timestamps, start: float, end: float) -> None:
        span_hours = max(end - start, 0.0) / SECONDS_PER_HOUR  # full calendar span (the bug)
        self.baselines[self._key(entity, event_type)] = _LegacyBaseline(
            total_events=int(np.asarray(timestamps).size), span_hours=span_hours
        )

    def lambda_for(self, entity: str, event_type: str) -> float:
        b = self.baselines.get(self._key(entity, event_type))
        return b.lambda_per_hour if b else 0.0

    def score(self, entity: str, event_type: str, count: int, window_hours: float) -> float:
        """Plug-in Poisson upper-tail p-value P(N >= count)."""
        lam = self.lambda_for(entity, event_type)
        mu = lam * max(window_hours, 0.0)
        if mu <= 0:
            return 1.0
        return float(stats.poisson.sf(count - 1, mu))

    def detect(self, entity: str, event_type: str, timestamps, start: float, end: float):
        """Return ``(p_value, is_anomaly)`` for a window (whole window, one bucket)."""
        count = int(np.asarray(timestamps).size)
        window_hours = max(end - start, 0.0) / SECONDS_PER_HOUR
        p = self.score(entity, event_type, count, window_hours)
        return p, p < self.anomaly_threshold

    def update(self, entity: str, event_type: str, timestamps, start: float, end: float) -> None:
        """Re-learn by folding the window into the baseline (poisonable)."""
        key = self._key(entity, event_type)
        b = self.baselines.get(key)
        add_events = int(np.asarray(timestamps).size)
        add_hours = max(end - start, 0.0) / SECONDS_PER_HOUR
        if b is None:
            self.baselines[key] = _LegacyBaseline(add_events, add_hours)
        else:
            b.total_events += add_events
            b.span_hours += add_hours
