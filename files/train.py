"""
train.py — Train the Poisson Anomaly Detector on historical SIEM data
======================================================================
Generates a realistic synthetic dataset across 4 users and 4 event types,
trains the model, prints a profile summary, and saves profiles to disk.

Usage:
    python train.py
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ai_poisson_detector import PoissonAnomalyDetector

random.seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Generate synthetic historical events (30-day window)
# ──────────────────────────────────────────────────────────────────────────────
BASE_DATE = datetime(2024, 1, 1, 8, 0, 0)   # training window start

# Normal rates (events per hour, during business hours 08:00-18:00)
NORMAL_RATES = {
    "user_alice": {
        "login":       0.5,    # ~1 login every 2 h
        "file_access": 8.0,    # heavy file user
        "api_call":    3.0,
        "db_query":    5.0,
    },
    "user_bob": {
        "login":       0.3,
        "file_access": 2.0,
        "api_call":    10.0,   # API-heavy developer
        "db_query":    1.0,
    },
    "user_carol": {
        "login":       0.4,
        "file_access": 4.0,
        "api_call":    1.0,
        "db_query":    12.0,   # DBA
    },
    "user_dave": {
        "login":       0.2,
        "file_access": 1.0,
        "api_call":    0.5,
        "db_query":    0.8,
    },
}

BUSINESS_HOURS = range(8, 18)   # 08:00 – 17:59
TRAINING_DAYS  = 30


def generate_training_events(
    rates: dict,
    days: int = TRAINING_DAYS,
) -> dict:
    """
    Simulate Poisson arrivals for each user/event pair over `days` days.
    Only during business hours (10 active hours per day).
    """
    user_events = {}

    for user_id, event_rates in rates.items():
        user_events[user_id] = {}

        for event_type, rate_per_hour in event_rates.items():
            timestamps = []

            for day_offset in range(days):
                day_start = BASE_DATE + timedelta(days=day_offset)

                for hour in BUSINESS_HOURS:
                    hour_start = day_start.replace(hour=hour, minute=0, second=0)

                    # Number of events this hour ~ Poisson(rate)
                    n_events = np.random.poisson(rate_per_hour)

                    for _ in range(n_events):
                        # Random minute/second within this hour
                        offset_sec = random.randint(0, 3599)
                        timestamps.append(hour_start + timedelta(seconds=offset_sec))

            user_events[user_id][event_type] = sorted(timestamps)
            print(
                f"  {user_id}/{event_type}: {len(timestamps)} events "
                f"(target rate={rate_per_hour}/h)"
            )

    return user_events


# ──────────────────────────────────────────────────────────────────────────────
# 2. Train
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  AI Poisson Detector — Training")
    print("=" * 60)

    print("\n[1] Generating 30-day synthetic training data …")
    user_events = generate_training_events(NORMAL_RATES)

    print("\n[2] Training detector …")
    detector = PoissonAnomalyDetector(
        threshold=0.05,
        alpha=0.1,
        window_hours=1.0,
        min_events=5,
    )
    detector.train(user_events)

    print("\n[3] Lambda Profile Summary:")
    print("-" * 50)
    summary = detector.profile_summary()
    print(summary.to_string(index=False))

    print("\n[4] Saving profiles to profiles.json …")
    detector.save_profiles("profiles.json")

    # ──────────────────────────────────────────────────────────────────────
    # 3. Quick sanity check — does the detector agree with expected rates?
    # ──────────────────────────────────────────────────────────────────────
    print("\n[5] Sanity check (comparing trained λ vs target rate):")
    print(f"  {'User':<15} {'Event':<15} {'Target':>10} {'Trained λ':>12} {'Δ%':>8}")
    print("  " + "-" * 62)

    for uid, etypes in NORMAL_RATES.items():
        for etype, target in etypes.items():
            trained = detector.lambda_profiles[uid].get(etype, float("nan"))
            delta_pct = 100 * abs(trained - target) / target if target > 0 else float("nan")
            print(f"  {uid:<15} {etype:<15} {target:>10.2f} {trained:>12.4f} {delta_pct:>7.1f}%")

    print("\n✅  Training complete. Profiles saved to profiles.json")
