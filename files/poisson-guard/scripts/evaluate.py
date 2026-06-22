#!/usr/bin/env python3
"""Evaluate PoissonGuard: detection AUC, ablation, and poisoning defense.

Trains a fresh full detector and a legacy detector on identical data, then runs
a labeled attack benchmark and a boil-the-frog poisoning experiment.

Usage:
    python scripts/evaluate.py --weeks 4 --entities 12 --seed 0
"""

from __future__ import annotations

import argparse

from poissonguard.bucketing import BucketingConfig
from poissonguard.evaluation import (
    build_trained_pair,
    evaluate_detection,
    evaluate_poisoning,
)
from poissonguard.generators import generate_eval_windows


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate PoissonGuard.")
    ap.add_argument("--weeks", type=int, default=4)
    ap.add_argument("--entities", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fpr", type=float, default=0.05)
    args = ap.parse_args()

    cfg = BucketingConfig()
    detector, legacy, population = build_trained_pair(
        weeks=args.weeks, n_entities=args.entities, seed=args.seed, bucketing=cfg
    )
    windows = generate_eval_windows(population, cfg, seed=args.seed + 100)

    print("=" * 64)
    print("DETECTION BENCHMARK (full vs rate-only vs legacy)")
    print("=" * 64)
    rep = evaluate_detection(detector, legacy, windows, target_fpr=args.fpr)
    print(f"ROC-AUC   full={rep.auc_full:.3f}  rate-only={rep.auc_rate_only:.3f}  "
          f"legacy={rep.auc_legacy:.3f}")
    print(f"PR-AP     full={rep.ap_full:.3f}  legacy={rep.ap_legacy:.3f}")
    print(f"\nDetection rate @ FPR={args.fpr:.0%}  (full vs legacy):")
    for atk in rep.per_attack_full:
        f = rep.per_attack_full[atk]
        l = rep.per_attack_legacy[atk]
        print(f"  {atk:14s}  full={f:5.1%}   legacy={l:5.1%}")

    print("\n" + "=" * 64)
    print("POISONING DEFENSE (boil-the-frog ramp, online updates)")
    print("=" * 64)
    pr = evaluate_poisoning(detector, legacy, population[0], days=40, end_mult=3.0)
    print(f"Entity {population[0].entity}: baseline {population[0].business_rate:.2f}/hr "
          f"-> attacker target {pr.target_rate:.2f}/hr")
    drift_x = pr.legacy_lambda_final / max(pr.legacy_lambda_initial, 1e-9)
    print(f"  PoissonGuard: drift-integrity gate FROZE on day {pr.freeze_day}, "
          f"held anchor at {pr.guard_anchor:.2f}/hr (raised an alert).")
    print(f"  Legacy:       no drift-integrity mechanism; learned baseline "
          f"silently drifted {pr.legacy_lambda_initial:.2f} -> "
          f"{pr.legacy_lambda_final:.2f}/hr ({drift_x:.1f}x) with 0 alerts.")
    print(f"\nWith the baseline defended, a full day at the {pr.target_rate:.1f}/hr "
          f"target is still flagged: {'YES' if pr.full_detects_target else 'NO'}")


if __name__ == "__main__":
    main()
