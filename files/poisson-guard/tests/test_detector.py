import numpy as np
from datetime import datetime, timezone

from poissonguard.bayes_rate import Prior, fit_baseline
from poissonguard.detector import Detector, DetectorConfig
from poissonguard.schemas import Bucket, DriftDecision

MONDAY = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
HOUR = 3600.0
RNG = np.random.default_rng(7)


def business_day_events(rate_per_hour, day=0):
    """Homogeneous Poisson events within Mon-Fri 9-17 of the given weekday index."""
    base = MONDAY + day * 24 * HOUR + 9 * HOUR
    gaps = RNG.exponential(HOUR / rate_per_hour, size=2000)
    ts = base + np.cumsum(gaps)
    return ts[ts < base + 8 * HOUR]


def make_detector(baseline_rate=5.0):
    det = Detector(DetectorConfig(prior=Prior(1.0, 1.0), anomaly_threshold=1e-3))
    # ~baseline_rate/hr over a 40h business-hour week.
    b = fit_baseline("u", "login", Bucket.BUSINESS, count=int(baseline_rate * 40),
                     exposure_hours=40.0, prior=Prior(1.0, 1.0))
    det.set_baseline(b)
    return det


def test_normal_day_not_anomalous():
    det = make_detector(5.0)
    ts = business_day_events(5.0)
    results = det.detect("u", "login", ts, MONDAY, MONDAY + 24 * HOUR)
    biz = [r for r in results if r.bucket == Bucket.BUSINESS][0]
    assert not biz.is_anomaly
    assert biz.fused_p_value > 0.01


def test_volume_spike_is_anomalous():
    det = make_detector(5.0)
    ts = business_day_events(30.0)  # 6x normal
    results = det.detect("u", "login", ts, MONDAY, MONDAY + 24 * HOUR)
    biz = [r for r in results if r.bucket == Bucket.BUSINESS][0]
    assert biz.is_anomaly
    assert biz.fused_p_value < 1e-3
    assert biz.sub_scores[0].name == "rate"


def test_cold_start_creates_baseline():
    det = Detector(DetectorConfig(prior=Prior(2.0, 1.0)))
    ts = business_day_events(2.0)
    results = det.detect("new", "api", ts, MONDAY, MONDAY + 24 * HOUR)
    assert any(r.bucket == Bucket.BUSINESS for r in results)
    assert "new|api|business" in det.baselines


def test_update_baseline_folds_in_on_accept():
    det = make_detector(5.0)
    key = "u|login|business"
    before = det.baselines[key].n_events
    ts = business_day_events(5.0)
    det.detect("u", "login", ts, MONDAY, MONDAY + 24 * HOUR, update_baseline=True)
    assert det.baselines[key].n_events > before


def test_persistence_roundtrip(tmp_path):
    det = make_detector(5.0)
    ts = business_day_events(5.0)
    det.detect("u", "login", ts, MONDAY, MONDAY + 24 * HOUR, update_baseline=True)
    path = tmp_path / "baselines.json"
    det.save(str(path))

    det2 = Detector.load(str(path), det.config)
    assert det2.baselines.keys() == det.baselines.keys()
    b1 = det.baselines["u|login|business"]
    b2 = det2.baselines["u|login|business"]
    assert np.isclose(b1.alpha, b2.alpha) and np.isclose(b1.beta, b2.beta)
    assert "u|login|business" in det2.guards


def test_drift_freeze_marks_anomaly():
    det = make_detector(5.0)
    # Replay a poisoning ramp through update_baseline; gate should freeze.
    decisions = []
    for day, rate in enumerate(np.linspace(5.0, 15.0, 5)):
        ts = business_day_events(rate, day=day % 5)
        res = det.detect("u", "login", ts, MONDAY + day * 24 * HOUR,
                         MONDAY + (day + 1) * 24 * HOUR, update_baseline=True)
        biz = [r for r in res if r.bucket == Bucket.BUSINESS][0]
        decisions.append(biz.drift_decision)
    assert DriftDecision.ACCEPT in decisions
    # The baseline anchor must not have been dragged to the attacker's target.
    assert det.guards["u|login|business"].anchor < 13.0
