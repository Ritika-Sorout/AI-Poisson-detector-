"""Synthetic population, training history, and labeled attack windows.

Everything here is reproducible given a seed. The population mimics entities
with strong diurnal behaviour (busy in business hours, near-silent otherwise) so
that the bucketing module's calendar-span correction actually matters.

Attack families
---------------
* ``volume_spike``   -- business-hours rate multiplied (loud, easy).
* ``bursty``         -- tight bursts with silence between (over-dispersed).
* ``regular_bot``    -- metronomic equal spacing (under-dispersed, non-expo).
* ``offhours``       -- elevated activity when the entity is normally idle.
* ``boil_the_frog``  -- a slow multi-window ramp used for the drift defense.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bucketing import BucketingConfig, active_intervals
from .schemas import Bucket, SECONDS_PER_HOUR

HOUR = SECONDS_PER_HOUR
DAY = 24 * HOUR
WEEK = 7 * DAY

ATTACK_TYPES = ("volume_spike", "bursty", "regular_bot", "offhours")


@dataclass
class EntityProfile:
    entity: str
    event_type: str
    business_rate: float   # events/hour during business hours
    offhours_rate: float   # events/hour otherwise (near zero)


@dataclass
class LabeledWindow:
    entity: str
    event_type: str
    timestamps: np.ndarray
    start: float
    end: float
    label: int           # 1 = attack, 0 = normal
    attack_type: str     # "" for normal


def make_population(n_entities: int = 12, seed: int = 0) -> list[EntityProfile]:
    rng = np.random.default_rng(seed)
    event_types = ["login", "api_call", "download"]
    pop = []
    for i in range(n_entities):
        et = event_types[i % len(event_types)]
        biz = float(rng.uniform(2.0, 8.0))
        off = float(biz * rng.uniform(0.0, 0.04))
        pop.append(EntityProfile(f"entity_{i:02d}", et, biz, off))
    return pop


def _poisson_on_intervals(intervals, rate_per_hour, rng) -> np.ndarray:
    ts = []
    for a, b in intervals:
        hours = (b - a) / HOUR
        n = rng.poisson(max(rate_per_hour, 0.0) * hours)
        if n > 0:
            ts.append(a + rng.uniform(0, b - a, size=n))
    if not ts:
        return np.empty(0, dtype=float)
    return np.sort(np.concatenate(ts))


def generate_normal(profile: EntityProfile, start: float, end: float,
                    cfg: BucketingConfig, rng) -> np.ndarray:
    """Diurnal Poisson stream over ``[start, end]``."""
    biz = active_intervals(start, end, Bucket.BUSINESS, cfg)
    off = active_intervals(start, end, Bucket.OFFHOURS, cfg)
    return np.sort(np.concatenate([
        _poisson_on_intervals(biz, profile.business_rate, rng),
        _poisson_on_intervals(off, profile.offhours_rate, rng),
    ]))


def make_attack(attack_type: str, profile: EntityProfile, day_start: float,
                cfg: BucketingConfig, rng, severity: float = 6.0) -> np.ndarray:
    """Generate one day's worth of malicious events."""
    end = day_start + DAY
    biz = active_intervals(day_start, end, Bucket.BUSINESS, cfg)
    off = active_intervals(day_start, end, Bucket.OFFHOURS, cfg)

    if attack_type == "volume_spike":
        return _poisson_on_intervals(biz, profile.business_rate * severity, rng)

    if attack_type == "bursty":
        ts = []
        for a, b in biz:
            n_bursts = 4
            for _ in range(n_bursts):
                center = rng.uniform(a, b)
                size = int(profile.business_rate * (b - a) / HOUR * severity / n_bursts)
                ts.append(center + rng.uniform(0, 30, size=max(size, 1)))
        return np.sort(np.concatenate(ts)) if ts else np.empty(0)

    if attack_type == "regular_bot":
        ts = []
        for a, b in biz:
            rate = profile.business_rate * severity
            period = HOUR / max(rate, 1e-6)
            ts.append(np.arange(a, b, period))
        return np.sort(np.concatenate(ts)) if ts else np.empty(0)

    if attack_type == "offhours":
        # Activity when the entity is normally idle.
        return _poisson_on_intervals(off, max(profile.business_rate, 1.0) * 0.5 * severity, rng)

    raise ValueError(f"unknown attack_type: {attack_type!r}")


def generate_training(population, weeks: int, cfg: BucketingConfig, seed: int = 1):
    """Return ``(streams, start, end)`` where streams[(entity, event_type)] = timestamps."""
    rng = np.random.default_rng(seed)
    start = 0.0
    end = weeks * WEEK
    streams = {}
    for p in population:
        streams[(p.entity, p.event_type)] = generate_normal(p, start, end, cfg, rng)
    return streams, start, end


def generate_eval_windows(population, cfg: BucketingConfig, seed: int = 2,
                          normal_days: int = 5, severity: float = 6.0):
    """Build a labeled set of daily windows (normal + every attack family).

    Attack windows are placed on guaranteed business days so business-hour
    attacks are never accidentally generated on a weekend (which would be empty).
    """
    rng = np.random.default_rng(seed)
    base = 100 * WEEK  # eval period well after training
    windows: list[LabeledWindow] = []
    for p in population:
        # normal days: consecutive calendar days (weekends are legitimately quiet)
        for d in range(normal_days):
            s = base + d * DAY
            ts = generate_normal(p, s, s + DAY, cfg, rng)
            windows.append(LabeledWindow(p.entity, p.event_type, ts, s, s + DAY, 0, ""))

        biz_days = _business_day_starts(base + normal_days * DAY, cfg,
                                        n=len(ATTACK_TYPES) + 2)
        # one loud window per attack family
        for atk, s in zip(ATTACK_TYPES, biz_days):
            ts = make_attack(atk, p, s, cfg, rng, severity)
            windows.append(LabeledWindow(p.entity, p.event_type, ts, s, s + DAY, 1, atk))
        # stealthy near-normal-volume shape attacks: same count, anomalous shape.
        for atk, s in zip(("regular_bot", "bursty"), biz_days[len(ATTACK_TYPES):]):
            ts = make_attack(atk, p, s, cfg, rng, severity=1.3)
            windows.append(LabeledWindow(p.entity, p.event_type, ts, s, s + DAY, 1, atk))
    return windows


def _business_day_starts(after: float, cfg: BucketingConfig, n: int) -> list[float]:
    """Day-start epochs (>= ``after``) that contain business hours, ``n`` of them."""
    out: list[float] = []
    day = float(after)
    while len(out) < n:
        if active_intervals(day, day + DAY, Bucket.BUSINESS, cfg):
            out.append(day)
        day += DAY
    return out


def generate_poisoning_sequence(profile: EntityProfile, cfg: BucketingConfig,
                                days: int = 30, start_rate_mult: float = 1.0,
                                end_rate_mult: float = 3.0, seed: int = 3):
    """Daily windows whose business rate slowly ramps (boil-the-frog)."""
    rng = np.random.default_rng(seed)
    starts = _business_day_starts(200 * WEEK, cfg, n=days)  # ramp over business days only
    mults = np.linspace(start_rate_mult, end_rate_mult, days)
    out = []
    for s, mult in zip(starts, mults):
        biz = active_intervals(s, s + DAY, Bucket.BUSINESS, cfg)
        ts = _poisson_on_intervals(biz, profile.business_rate * mult, rng)
        out.append((ts, s, s + DAY, float(mult)))
    return out
