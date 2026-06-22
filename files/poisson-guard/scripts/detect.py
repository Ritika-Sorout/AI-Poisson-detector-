#!/usr/bin/env python3
"""Run a trained PoissonGuard detector on a window of events.

Two modes:

* ``--events FILE.json`` -- score a custom window. JSON schema::

      {"entity": "...", "event_type": "...",
       "start": <epoch>, "end": <epoch>, "timestamps": [<epoch>, ...]}

* demo (default) -- pick an entity from the trained population and score a clean
  day plus one window of each attack family, side by side.

Usage:
    python scripts/detect.py --baselines artifacts/baselines.json \
        --population artifacts/population.json [--attack volume_spike]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from poissonguard.detector import Detector
from poissonguard.generators import ATTACK_TYPES, generate_normal, make_attack, DAY, WEEK
from poissonguard.schemas import DetectionResult
from poissonguard.training import load_population


def _fmt(results: list[DetectionResult]) -> str:
    lines = []
    for r in results:
        flag = "ANOMALY" if r.is_anomaly else "ok"
        subs = ", ".join(f"{s.name}={s.p_value:.2e}" for s in r.sub_scores)
        lines.append(
            f"  [{r.bucket.value:8s}] {flag:7s} "
            f"obs={r.observed_count:<4d} exp={r.expected_count:6.1f} "
            f"fused_p={r.fused_p_value:.2e} sev={r.severity.value:8s} "
            f"drift={r.drift_decision.value:6s} | {subs}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PoissonGuard detection.")
    ap.add_argument("--baselines", default="artifacts/baselines.json")
    ap.add_argument("--population", default="artifacts/population.json")
    ap.add_argument("--events", default=None, help="JSON window to score")
    ap.add_argument("--attack", choices=ATTACK_TYPES, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    detector = Detector.load(args.baselines)

    if args.events:
        with open(args.events) as f:
            w = json.load(f)
        results = detector.detect(
            w["entity"], w["event_type"], w["timestamps"], w["start"], w["end"]
        )
        print(f"Window {w['entity']}/{w['event_type']}:")
        print(_fmt(results))
        return

    # demo mode
    population = load_population(args.population)
    profile = population[0]
    rng = np.random.default_rng(args.seed)
    base = 300 * WEEK

    print(f"Entity {profile.entity}/{profile.event_type} "
          f"(business_rate={profile.business_rate:.2f}/hr)\n")

    normal = generate_normal(profile, base, base + DAY, detector.config.bucketing, rng)
    print("NORMAL DAY:")
    print(_fmt(detector.detect(profile.entity, profile.event_type, normal, base, base + DAY)))

    attacks = [args.attack] if args.attack else list(ATTACK_TYPES)
    for atk in attacks:
        ts = make_attack(atk, profile, base, detector.config.bucketing, rng, severity=6.0)
        print(f"\nATTACK [{atk}]:")
        print(_fmt(detector.detect(profile.entity, profile.event_type, ts, base, base + DAY)))


if __name__ == "__main__":
    main()
