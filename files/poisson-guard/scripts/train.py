#!/usr/bin/env python3
"""Train PoissonGuard on a synthetic population and persist the detector.

Usage:
    python scripts/train.py --weeks 4 --entities 12 --seed 0 \
        --out artifacts/baselines.json --population artifacts/population.json
"""

from __future__ import annotations

import argparse
import os

from poissonguard.bucketing import BucketingConfig
from poissonguard.schemas import Bucket
from poissonguard.training import save_population, train_synthetic


def main() -> None:
    ap = argparse.ArgumentParser(description="Train PoissonGuard baselines.")
    ap.add_argument("--weeks", type=int, default=4)
    ap.add_argument("--entities", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/baselines.json")
    ap.add_argument("--population", default="artifacts/population.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cfg = BucketingConfig()
    detector, population = train_synthetic(
        weeks=args.weeks, n_entities=args.entities, seed=args.seed, bucketing=cfg
    )
    detector.save(args.out)
    save_population(population, args.population)

    n_keys = len(detector.baselines)
    biz_prior = detector.config.prior
    print(f"Trained {n_keys} baselines across {args.entities} entities "
          f"({args.weeks} weeks).")
    print(f"Business EB prior: mean={biz_prior.mean_rate_per_hour:.3f}/hr, "
          f"strength={biz_prior.strength_hours:.2f}")
    sample = next(iter(detector.baselines.values()))
    print(f"Example baseline [{sample.entity}/{sample.event_type}/{sample.bucket.value}]: "
          f"rate={sample.posterior_mean_rate:.3f}/hr, n_events={sample.n_events}")
    print(f"Saved detector -> {args.out}")
    print(f"Saved population -> {args.population}")


if __name__ == "__main__":
    main()
