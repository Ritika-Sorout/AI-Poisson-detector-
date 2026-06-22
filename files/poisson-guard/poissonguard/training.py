"""Training pipeline: fit empirical-Bayes priors and per-key baselines.

Steps:

1. Bucketize every entity's training history into BUSINESS / OFFHOURS active
   time (correct exposure -- no calendar-span dilution).
2. For each bucket, pool ``(count, exposure)`` across all entities and fit a
   population Gamma prior via empirical Bayes (``fusion.empirical_bayes_prior``).
3. Fit each ``(entity, event_type, bucket)`` baseline as the posterior of that
   shared prior given the entity's own counts -> sparse entities shrink toward
   the population, dense entities dominate their own data.
4. Return a ready-to-serve :class:`Detector`.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from .bayes_rate import Prior, fit_baseline
from .bucketing import BucketingConfig, bucketize
from .detector import Detector, DetectorConfig
from .drift_guard import DriftGuardConfig
from .fusion import empirical_bayes_prior
from .generators import EntityProfile, generate_training, make_population
from .schemas import Bucket


def train_detector(
    streams: dict,
    start: float,
    end: float,
    bucketing: BucketingConfig | None = None,
    drift: DriftGuardConfig | None = None,
    anomaly_threshold: float = 1e-3,
) -> Detector:
    """Fit a Detector from ``streams[(entity, event_type)] = timestamps``."""
    bucketing = bucketing or BucketingConfig()

    # 1+2: bucketize and collect per-bucket (count, exposure) for EB priors.
    per_key_buckets: dict = {}
    pooled: dict = {Bucket.BUSINESS: ([], []), Bucket.OFFHOURS: ([], [])}
    for (entity, event_type), ts in streams.items():
        buckets = bucketize(entity, event_type, ts, start, end, bucketing)
        per_key_buckets[(entity, event_type)] = buckets
        for bucket, window in buckets.items():
            pooled[bucket][0].append(window.count)
            pooled[bucket][1].append(window.duration_hours)

    priors = {
        bucket: empirical_bayes_prior(counts, exposures)
        for bucket, (counts, exposures) in pooled.items()
    }

    # 3+4: fit shrinkage baselines.
    config = DetectorConfig(
        prior=priors[Bucket.BUSINESS],
        drift=drift or DriftGuardConfig(),
        bucketing=bucketing,
        anomaly_threshold=anomaly_threshold,
    )
    detector = Detector(config)
    for (entity, event_type), buckets in per_key_buckets.items():
        for bucket, window in buckets.items():
            baseline = fit_baseline(
                entity, event_type, bucket,
                count=window.count, exposure_hours=window.duration_hours,
                prior=priors[bucket],
            )
            detector.set_baseline(baseline)
    return detector


def train_synthetic(weeks: int = 4, n_entities: int = 12, seed: int = 0,
                    bucketing: BucketingConfig | None = None):
    """Convenience: build a synthetic population and train on it."""
    bucketing = bucketing or BucketingConfig()
    population = make_population(n_entities, seed=seed)
    streams, start, end = generate_training(population, weeks=weeks, cfg=bucketing, seed=seed + 1)
    detector = train_detector(streams, start, end, bucketing=bucketing)
    return detector, population


def save_population(population, path: str) -> None:
    with open(path, "w") as f:
        json.dump([asdict(p) for p in population], f, indent=2)


def load_population(path: str):
    with open(path) as f:
        return [EntityProfile(**d) for d in json.load(f)]
