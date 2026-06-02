# AI-Powered Poisson Anomaly Detector for SIEM

A statistical anomaly detection system for **Security Information and Event Management (SIEM)** that models normal user behaviour using the **Poisson distribution** and flags statistically unusual event rates in real-time.

---

## Overview

User actions in an enterprise environment — logins, file accesses, API calls, database queries — occur at roughly constant average rates per user. Deviations from this baseline (e.g. 40 logins in an hour instead of 0.3) are strong indicators of account compromise, brute-force attempts, or insider threats.

This system:

1. **Trains** a per-user, per-event-type Poisson lambda (λ = events/hour) from 30 days of historical data.
2. **Detects** anomalies in real-time via a right-tailed p-value test: flag if `P(X ≥ observed | λ) < 0.05`.
3. **Adapts** online using exponential smoothing so legitimate behaviour changes don't cause persistent false positives.

---

## Mathematical Foundation

### Poisson Model

Events are assumed to follow a Poisson process with rate λ (events per hour).

```
λ = total_events / span_hours       (training phase)
λ_scaled = λ × window_hours         (scaled to detection window)
```

### Anomaly Test (right-tailed)

```
p_value = P(X ≥ k | λ_scaled)
        = 1 − CDF_Poisson(k−1, λ_scaled)

is_anomaly = (p_value < threshold)   # default threshold = 0.05
```

### Online Learning (exponential smoothing)

```
λ_new = (1 − α) × λ_old + α × new_rate     # α = 0.1
```

### Severity Classification

| p-value range     | Severity  |
|-------------------|-----------|
| p ≥ 0.05          | normal    |
| 0.025 ≤ p < 0.05  | low       |
| 0.01 ≤ p < 0.025  | medium    |
| 0.001 ≤ p < 0.01  | high      |
| p < 0.001         | critical  |

---

## Project Structure

```
ai_poisson_detector/
├── ai_poisson_detector.py   # Core PoissonAnomalyDetector class
├── train.py                 # Training script (generates profiles.json)
├── detect.py                # Real-time & batch detection demo
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1 — Train

```bash
python train.py
```

Generates `profiles.json` with trained λ per (user, event_type).

### 2 — Detect

```bash
python detect.py
```

Runs five attack simulations and prints a SIEM-style alert report.

---

## API Reference

### `PoissonAnomalyDetector(threshold, alpha, window_hours, min_events)`

| Parameter      | Default | Description                                      |
|----------------|---------|--------------------------------------------------|
| `threshold`    | 0.05    | p-value threshold for flagging anomalies         |
| `alpha`        | 0.1     | Smoothing factor for online lambda updates       |
| `window_hours` | 1.0     | Detection window size in hours                   |
| `min_events`   | 5       | Minimum events needed to build a profile         |

---

### `train(user_events)`

```python
user_events = {
    'user_alice': {
        'login':       [datetime(2024,1,1,9,0), datetime(2024,1,1,11,30), ...],
        'file_access': [datetime(2024,1,1,9,5), ...],
    },
    'user_bob': { ... }
}
detector.train(user_events)
```

---

### `calculate_p_value(user_id, event_type, observed_count) → (p_value, lambda_scaled)`

```python
p_val, lam = detector.calculate_p_value("user_alice", "login", 15)
# p_val  = P(X >= 15 | lambda_scaled)
# lam    = lambda × window_hours
```

---

### `detect_anomaly(user_id, event_type, timestamp, recent_events) → (bool, AnomalyResult)`

```python
is_anomaly, result = detector.detect_anomaly(
    user_id      = "user_alice",
    event_type   = "file_access",
    timestamp    = datetime.now(),
    recent_events= [dt1, dt2, ..., dt180],  # last hour
)
print(result)
# 🚨 ANOMALY | user=user_alice event=file_access observed=180 expected=8.00 p=0.0000 severity=critical
```

---

### `update_lambda_online(user_id, event_type, new_observed_rate) → float`

```python
# Carol's DB query rate increased legitimately — adapt over time
new_lam = detector.update_lambda_online("user_carol", "db_query", 18.0)
```

---

### `detect_batch(event_df) → pd.DataFrame`

```python
import pandas as pd

df = pd.DataFrame([
    {"user_id": "user_alice", "event_type": "api_call", "timestamp": dt1},
    {"user_id": "user_alice", "event_type": "api_call", "timestamp": dt2},
    ...
])
result_df = detector.detect_batch(df)
# Returns df with added columns: observed_count, expected_count, p_value, is_anomaly, severity
```

---

## SIEM Integration

The detector is designed to slot into any SIEM pipeline:

```
[Log Aggregator]  →  [Event Parser]  →  PoissonAnomalyDetector.detect_anomaly()
                                              ↓
                                     [AnomalyResult]
                                              ↓
                              [Alert Router / Ticket System]
```

### Splunk / Kafka Example

```python
# Consuming from Kafka
for message in consumer:
    event = json.loads(message.value)
    is_anomaly, result = detector.detect_anomaly(
        user_id      = event["user"],
        event_type   = event["action"],
        timestamp    = datetime.fromisoformat(event["ts"]),
        recent_events= recent_cache.get(event["user"], event["action"]),
    )
    if is_anomaly:
        send_to_siem_alert_queue(result.to_dict())
```

---

## Attack Scenarios Detected

| Scenario              | Event Type   | Signal                        |
|-----------------------|--------------|-------------------------------|
| Brute-force login     | login        | 40+ logins/h vs 0.3 baseline  |
| Data exfiltration     | file_access  | 180 accesses/h vs 8 baseline  |
| API key abuse         | api_call     | 220 calls/h vs 10 baseline    |
| Credential stuffing   | login        | 35 logins/h vs 0.2 baseline   |
| Unknown user spike    | any          | Falls back to global mean λ   |

---

## Limitations & Extensions

- Poisson assumes a stationary rate — use **time-of-day bucketing** for users with strong diurnal patterns.
- Low-frequency events (< 5/day) may need a **negative-binomial** model for overdispersion.
- Extend with a supervised attack-type classifier head (see `classifier.py` in the repo) for finer labelling.
