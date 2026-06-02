"""
detect.py — Real-time anomaly detection demo
=============================================
Simulates a live event stream mixing normal and attack scenarios,
then runs the Poisson detector and prints a SIEM-style alert report.

Attack scenarios modelled:
  A1 — Brute-force login spike (user_bob,  login, 40×  normal)
  A2 — Data exfiltration burst  (user_alice, file_access, 20× normal)
  A3 — API key abuse             (user_bob,  api_call,    15× normal)
  A4 — Credential stuffing       (user_dave, login,       30× normal)

Usage:
    python detect.py
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ai_poisson_detector import PoissonAnomalyDetector

random.seed(7)
np.random.seed(7)

# ──────────────────────────────────────────────────────────────────────────────
PROFILES_PATH = "profiles.json"
DETECTION_TS  = datetime(2024, 2, 15, 14, 30, 0)    # simulated "now"
WINDOW_HOURS  = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Helper: generate a burst of events within the last `window_hours`
# ──────────────────────────────────────────────────────────────────────────────
def burst_events(n: int, anchor: datetime, hours_back: float = 1.0) -> list[datetime]:
    start = anchor - timedelta(hours=hours_back)
    return sorted(
        start + timedelta(seconds=random.randint(0, int(hours_back * 3600)))
        for _ in range(n)
    )


def normal_events(rate_per_hour: float, anchor: datetime, hours_back: float = 1.0) -> list[datetime]:
    n = np.random.poisson(rate_per_hour * hours_back)
    return burst_events(n, anchor, hours_back)


# ──────────────────────────────────────────────────────────────────────────────
# Detection scenarios
# ──────────────────────────────────────────────────────────────────────────────
SCENARIOS = [
    # ── Normal behaviour ─────────────────────────────────────────────────────
    dict(label="Normal",         user="user_alice", etype="login",       rate=0.5),
    dict(label="Normal",         user="user_alice", etype="file_access", rate=8.0),
    dict(label="Normal",         user="user_alice", etype="api_call",    rate=3.0),
    dict(label="Normal",         user="user_bob",   etype="login",       rate=0.3),
    dict(label="Normal",         user="user_bob",   etype="api_call",    rate=10.0),
    dict(label="Normal",         user="user_carol", etype="db_query",    rate=12.0),
    dict(label="Normal",         user="user_dave",  etype="file_access", rate=1.0),

    # ── Attack scenarios ──────────────────────────────────────────────────────
    # A1: Brute-force login
    dict(label="A1-BruteForce",  user="user_bob",   etype="login",       count=48),
    # A2: Data exfiltration
    dict(label="A2-Exfiltration",user="user_alice", etype="file_access", count=180),
    # A3: API key abuse
    dict(label="A3-APIAbuse",    user="user_bob",   etype="api_call",    count=220),
    # A4: Credential stuffing (unknown user)
    dict(label="A4-CredStuff",   user="user_dave",  etype="login",       count=35),
    # A5: New user with no profile
    dict(label="A5-NewUser",     user="user_eve",   etype="api_call",    count=50),
]


def run_detection(detector: PoissonAnomalyDetector) -> list:
    results = []

    for sc in SCENARIOS:
        user, etype = sc["user"], sc["etype"]

        if "count" in sc:
            recent = burst_events(sc["count"], DETECTION_TS, WINDOW_HOURS)
        else:
            recent = normal_events(sc["rate"], DETECTION_TS, WINDOW_HOURS)

        is_anomaly, result = detector.detect_anomaly(user, etype, DETECTION_TS, recent)
        results.append((sc["label"], result))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Pretty report
# ──────────────────────────────────────────────────────────────────────────────
SEVERITY_EMOJI = {
    "normal":   "✅",
    "low":      "🟡",
    "medium":   "🟠",
    "high":     "🔴",
    "critical": "🚨",
}


def print_report(results: list) -> None:
    print("\n" + "=" * 75)
    print("  SIEM ANOMALY DETECTION REPORT")
    print(f"  Detection timestamp : {DETECTION_TS.isoformat()}")
    print(f"  Window              : {WINDOW_HOURS}h")
    print("=" * 75)

    header = f"  {'Scenario':<20} {'User':<14} {'Event':<14} {'Obs':>5} {'Exp':>7} {'p-val':>8} {'Sev':<10}"
    print(header)
    print("  " + "-" * 73)

    anomalies = []
    for label, res in results:
        emoji = SEVERITY_EMOJI.get(res.severity, "?")
        flag  = " ← ALERT" if res.is_anomaly else ""
        print(
            f"  {label:<20} {res.user_id:<14} {res.event_type:<14} "
            f"{res.observed_count:>5} {res.expected_count:>7.2f} "
            f"{res.p_value:>8.4f} {emoji} {res.severity:<8}{flag}"
        )
        if res.is_anomaly:
            anomalies.append(res)

    print("  " + "-" * 73)
    print(f"\n  Total events checked : {len(results)}")
    print(f"  Anomalies detected   : {len(anomalies)}")

    if anomalies:
        print("\n  ── Alert Detail ──────────────────────────────────────────────")
        for res in anomalies:
            fold = res.observed_count / res.expected_count if res.expected_count > 0 else float("inf")
            print(
                f"\n  🚨 [{res.severity.upper()}] {res.user_id} | {res.event_type}\n"
                f"     Observed: {res.observed_count} events  "
                f"Expected: {res.expected_count:.2f}  "
                f"({fold:.1f}× normal)\n"
                f"     p-value: {res.p_value:.6f}  "
                f"Timestamp: {res.timestamp.isoformat()}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Online-learning demo: adapt lambda after a false-positive window
# ──────────────────────────────────────────────────────────────────────────────
def demo_online_update(detector: PoissonAnomalyDetector) -> None:
    print("\n\n" + "=" * 75)
    print("  ONLINE LEARNING DEMO — lambda adaptation via exponential smoothing")
    print("=" * 75)

    uid, etype = "user_carol", "db_query"
    old_lam = detector.lambda_profiles[uid][etype]

    # Suppose carol's normal DB query rate increased to 18/h after a new project
    new_observed = 18.0
    print(f"\n  Scenario : {uid} | {etype}")
    print(f"  λ before update : {old_lam:.4f} events/h")
    print(f"  New observed rate: {new_observed:.1f} events/h")

    for step in range(1, 11):
        updated = detector.update_lambda_online(uid, etype, new_observed)
        print(f"  Step {step:>2} : λ = {updated:.4f}")

    print(f"\n  λ after 10 updates : {detector.lambda_profiles[uid][etype]:.4f}")
    print("  (converging toward new observed rate, not a hard reset)")


# ──────────────────────────────────────────────────────────────────────────────
# Batch detection demo
# ──────────────────────────────────────────────────────────────────────────────
def demo_batch_detection(detector: PoissonAnomalyDetector) -> None:
    print("\n\n" + "=" * 75)
    print("  BATCH DETECTION — DataFrame API")
    print("=" * 75)

    rows = []
    base = DETECTION_TS - timedelta(hours=2)

    # Normal events for alice
    for i in range(10):
        rows.append(dict(
            user_id="user_alice", event_type="api_call",
            timestamp=base + timedelta(minutes=i * 12),
        ))

    # Attack burst — 40 api_calls in the last hour
    for i in range(40):
        rows.append(dict(
            user_id="user_alice", event_type="api_call",
            timestamp=base + timedelta(hours=1, minutes=random.randint(0, 59)),
        ))

    df = pd.DataFrame(rows)
    result_df = detector.detect_batch(df)

    anomaly_rows = result_df[result_df["is_anomaly"]]
    print(f"\n  Rows processed : {len(result_df)}")
    print(f"  Anomaly rows   : {len(anomaly_rows)}")
    if not anomaly_rows.empty:
        print("\n  Anomalous records:")
        print(
            anomaly_rows[["user_id","event_type","observed_count","expected_count","p_value","severity"]]
            .drop_duplicates()
            .to_string(index=False)
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not Path(PROFILES_PATH).exists():
        print("profiles.json not found — run train.py first.\n")
        raise SystemExit(1)

    detector = PoissonAnomalyDetector(threshold=0.05, alpha=0.1, window_hours=WINDOW_HOURS)
    detector.load_profiles(PROFILES_PATH)

    results = run_detection(detector)
    print_report(results)

    demo_online_update(detector)
    demo_batch_detection(detector)

    print("\n\n✅  Detection demo complete.")
