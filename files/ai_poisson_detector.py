"""
AI-Powered Poisson-Based Anomaly Detector for SIEM Systems
===========================================================
Models normal user event rates using Poisson distribution,
then flags statistically unusual behaviour via right-tailed p-value tests.

Architecture mirrors the repo's pipeline pattern:
  train → model user baselines
  detect_anomaly → real-time flagging
  update_lambda_online → exponential-smoothing adaptation

Author : Ritika-Sorout/AI-Poisson-detector-
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("PoissonAnomalyDetector")


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────
@dataclass
class AnomalyResult:
    """Structured output for a single detection call."""
    user_id: str
    event_type: str
    observed_count: int
    expected_count: float          # lambda (scaled to window)
    p_value: float
    is_anomaly: bool
    timestamp: datetime
    severity: str = "normal"       # normal | low | medium | high | critical
    window_hours: float = 1.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    def __str__(self) -> str:
        flag = "🚨 ANOMALY" if self.is_anomaly else "✅ normal"
        return (
            f"{flag} | user={self.user_id} event={self.event_type} "
            f"observed={self.observed_count} expected={self.expected_count:.2f} "
            f"p={self.p_value:.4f} severity={self.severity}"
        )


# ──────────────────────────────────────────────
# Core Detector
# ──────────────────────────────────────────────
class PoissonAnomalyDetector:
    """
    Poisson-distribution anomaly detector for SIEM / user-behaviour analytics.

    Parameters
    ----------
    threshold      : p-value below which an event is flagged (default 0.05)
    alpha          : smoothing factor for online lambda updates (default 0.1)
    window_hours   : observation window for real-time counts (default 1.0)
    min_events     : minimum historical events to build a profile (default 5)
    """

    SEVERITY_BANDS = [
        (0.001, "critical"),
        (0.01,  "high"),
        (0.025, "medium"),
        (0.05,  "low"),
    ]

    def __init__(
        self,
        threshold: float = 0.05,
        alpha: float = 0.1,
        window_hours: float = 1.0,
        min_events: int = 5,
    ):
        self.threshold = threshold
        self.alpha = alpha
        self.window_hours = window_hours
        self.min_events = min_events

        # lambda_profiles[user_id][event_type] = events-per-hour
        self.lambda_profiles: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Raw training stats for audit / explainability
        self._training_stats: Dict[str, Dict[str, dict]] = defaultdict(dict)

        self._trained = False

    # ──────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────
    def train(self, user_events: Dict[str, Dict[str, List[datetime]]]) -> None:
        """
        Compute per-user, per-event-type Poisson lambda from historical data.

        Parameters
        ----------
        user_events : {
            'user_1': {
                'login':       [dt1, dt2, ...],
                'file_access': [dt1, dt2, ...],
            },
            ...
        }
        """
        logger.info("Training on %d users …", len(user_events))

        for user_id, events_by_type in user_events.items():
            for event_type, timestamps in events_by_type.items():
                if len(timestamps) < self.min_events:
                    logger.warning(
                        "Skipping %s/%s — only %d events (min=%d)",
                        user_id, event_type, len(timestamps), self.min_events,
                    )
                    continue

                timestamps_sorted = sorted(timestamps)
                lam = self._compute_lambda(timestamps_sorted)
                self.lambda_profiles[user_id][event_type] = lam

                # Store stats for later inspection
                self._training_stats[user_id][event_type] = {
                    "event_count": len(timestamps),
                    "span_hours": (timestamps_sorted[-1] - timestamps_sorted[0]).total_seconds() / 3600,
                    "lambda_per_hour": lam,
                }

        user_count = len(self.lambda_profiles)
        profile_count = sum(len(v) for v in self.lambda_profiles.values())
        logger.info("Training complete — %d user profiles, %d (user, event) pairs.", user_count, profile_count)
        self._trained = True

    def _compute_lambda(self, timestamps: List[datetime]) -> float:
        """Events per hour over the full observation span."""
        span_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
        if span_seconds <= 0:
            return float(len(timestamps))          # all at same instant → burst
        span_hours = span_seconds / 3600.0
        return len(timestamps) / span_hours

    # ──────────────────────────────────────────
    # P-Value Calculation
    # ──────────────────────────────────────────
    def calculate_p_value(
        self,
        user_id: str,
        event_type: str,
        observed_count: int,
    ) -> Tuple[float, float]:
        """
        Right-tailed Poisson test: P(X ≥ observed | λ_scaled).

        Returns
        -------
        (p_value, lambda_scaled)
        """
        lam = self._get_lambda(user_id, event_type)
        lam_scaled = lam * self.window_hours          # scale λ to the detection window

        if lam_scaled <= 0:
            return 1.0, lam_scaled                   # no baseline → not anomalous

        # P(X >= k) = 1 - P(X <= k-1) = 1 - CDF(k-1)
        p_value = 1.0 - poisson.cdf(observed_count - 1, lam_scaled)
        return float(p_value), float(lam_scaled)

    def _get_lambda(self, user_id: str, event_type: str) -> float:
        """Return trained lambda, or a global fallback mean."""
        if user_id in self.lambda_profiles and event_type in self.lambda_profiles[user_id]:
            return self.lambda_profiles[user_id][event_type]

        # Fallback: mean across all users for that event type
        vals = [
            self.lambda_profiles[u][event_type]
            for u in self.lambda_profiles
            if event_type in self.lambda_profiles[u]
        ]
        if vals:
            logger.debug("No profile for %s/%s — using global mean λ.", user_id, event_type)
            return float(np.mean(vals))

        logger.warning("No lambda available for %s/%s.", user_id, event_type)
        return 0.0

    # ──────────────────────────────────────────
    # Real-Time Detection
    # ──────────────────────────────────────────
    def detect_anomaly(
        self,
        user_id: str,
        event_type: str,
        timestamp: datetime,
        recent_events: List[datetime],
    ) -> Tuple[bool, AnomalyResult]:
        """
        Detect whether the event rate in `recent_events` is anomalous.

        Parameters
        ----------
        user_id       : identifier of the user generating events
        event_type    : category string, e.g. 'login', 'file_access', 'api_call'
        timestamp     : current wall-clock time
        recent_events : list of event timestamps within the detection window

        Returns
        -------
        (is_anomaly, AnomalyResult)
        """
        window_start = timestamp - timedelta(hours=self.window_hours)
        count_in_window = sum(1 for t in recent_events if window_start <= t <= timestamp)

        p_value, lam_scaled = self.calculate_p_value(user_id, event_type, count_in_window)
        is_anomaly = p_value < self.threshold
        severity = self._classify_severity(p_value, is_anomaly)

        result = AnomalyResult(
            user_id=user_id,
            event_type=event_type,
            observed_count=count_in_window,
            expected_count=lam_scaled,
            p_value=p_value,
            is_anomaly=is_anomaly,
            timestamp=timestamp,
            severity=severity,
            window_hours=self.window_hours,
        )

        if is_anomaly:
            logger.warning(str(result))
        else:
            logger.debug(str(result))

        return is_anomaly, result

    def _classify_severity(self, p_value: float, is_anomaly: bool) -> str:
        if not is_anomaly:
            return "normal"
        for cutoff, label in self.SEVERITY_BANDS:
            if p_value < cutoff:
                return label
        return "low"

    # ──────────────────────────────────────────
    # Online Learning
    # ──────────────────────────────────────────
    def update_lambda_online(
        self,
        user_id: str,
        event_type: str,
        new_observed_rate: float,
    ) -> float:
        """
        Exponential smoothing update:  λ_new = (1-α)·λ_old + α·new_rate

        Parameters
        ----------
        new_observed_rate : recently measured events-per-hour for this user/type

        Returns
        -------
        Updated lambda value.
        """
        old_lam = self._get_lambda(user_id, event_type)
        updated_lam = (1 - self.alpha) * old_lam + self.alpha * new_observed_rate

        if user_id not in self.lambda_profiles:
            self.lambda_profiles[user_id] = {}
        self.lambda_profiles[user_id][event_type] = updated_lam

        logger.debug(
            "λ update %s/%s: %.4f → %.4f (new_rate=%.4f)",
            user_id, event_type, old_lam, updated_lam, new_observed_rate,
        )
        return updated_lam

    # ──────────────────────────────────────────
    # Batch Detection (DataFrame API)
    # ──────────────────────────────────────────
    def detect_batch(self, event_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a DataFrame of events and return anomaly results.

        Expected columns: user_id, event_type, timestamp
        Optional         : any extra columns are preserved.

        Returns the input DataFrame augmented with:
        observed_count, expected_count, p_value, is_anomaly, severity
        """
        required = {"user_id", "event_type", "timestamp"}
        if not required.issubset(event_df.columns):
            raise ValueError(f"DataFrame must contain columns: {required}")

        df = event_df.copy().sort_values("timestamp")
        results = []

        for _, row in df.iterrows():
            uid, etype, ts = row["user_id"], row["event_type"], row["timestamp"]
            window_start = ts - timedelta(hours=self.window_hours)
            recent = df[
                (df["user_id"] == uid) &
                (df["event_type"] == etype) &
                (df["timestamp"] >= window_start) &
                (df["timestamp"] <= ts)
            ]["timestamp"].tolist()

            _, res = self.detect_anomaly(uid, etype, ts, recent)
            results.append(res.to_dict())

        result_df = pd.DataFrame(results)
        return result_df

    # ──────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────
    def save_profiles(self, path: str) -> None:
        """Serialise lambda profiles to JSON."""
        with open(path, "w") as f:
            json.dump(dict(self.lambda_profiles), f, indent=2)
        logger.info("Profiles saved to %s", path)

    def load_profiles(self, path: str) -> None:
        """Load lambda profiles from JSON."""
        with open(path) as f:
            data = json.load(f)
        self.lambda_profiles = defaultdict(dict, {u: dict(v) for u, v in data.items()})
        self._trained = True
        logger.info("Profiles loaded from %s (%d users)", path, len(self.lambda_profiles))

    # ──────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────
    def profile_summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising all trained lambda values."""
        rows = []
        for user_id, etypes in self.lambda_profiles.items():
            for etype, lam in etypes.items():
                rows.append({"user_id": user_id, "event_type": etype, "lambda_per_hour": lam})
        return pd.DataFrame(rows).sort_values(["user_id", "event_type"])
