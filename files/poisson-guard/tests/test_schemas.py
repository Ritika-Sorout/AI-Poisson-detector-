import numpy as np

from poissonguard.schemas import (
    Baseline,
    Bucket,
    DetectionResult,
    DriftDecision,
    Event,
    Severity,
    SubScore,
    Window,
)


def test_window_derivations():
    w = Window.from_timestamps("u1", "login", [0.0, 3600.0, 7200.0], start=0.0, end=7200.0)
    assert w.count == 3
    assert w.duration_hours == 2.0
    assert w.empirical_rate_per_hour() == 1.5
    np.testing.assert_allclose(w.interarrival_seconds(), [3600.0, 3600.0])


def test_window_sorts_and_defaults_span():
    w = Window.from_timestamps("u1", "login", [10.0, 0.0, 5.0])
    assert list(w.timestamps) == [0.0, 5.0, 10.0]
    assert w.start == 0.0 and w.end == 10.0


def test_window_empty():
    w = Window.from_timestamps("u1", "login", [])
    assert w.count == 0
    assert w.empirical_rate_per_hour() == 0.0
    assert w.interarrival_seconds().size == 0


def test_severity_thresholds():
    assert Severity.from_pvalue(0.2) == Severity.NONE
    assert Severity.from_pvalue(0.02) == Severity.LOW
    assert Severity.from_pvalue(5e-3) == Severity.MEDIUM
    assert Severity.from_pvalue(5e-4) == Severity.HIGH
    assert Severity.from_pvalue(1e-9) == Severity.CRITICAL


def test_subscore_clamps_pvalue():
    assert SubScore("x", 0.0, 1.0).p_value == 1e-12
    assert SubScore("x", float("nan"), 1.0).p_value == 1.0
    assert SubScore("x", 2.0, 1.0).p_value == 1.0


def test_baseline_roundtrip_and_mean():
    b = Baseline("u1", "login", Bucket.BUSINESS, alpha=10.0, beta=2.0, n_events=8, exposure_hours=4.0, rate_history=[5.0])
    assert b.posterior_mean_rate == 5.0
    b2 = Baseline.from_dict(b.to_dict())
    assert b2 == b
    assert Baseline.key("u1", "login", Bucket.BUSINESS) == "u1|login|business"


def test_detection_result_to_dict():
    r = DetectionResult(
        entity="u1", event_type="login", bucket=Bucket.OFFHOURS,
        observed_count=20, expected_count=3.0,
        sub_scores=[SubScore("rate", 1e-4, 5.0)],
        fused_p_value=1e-4, severity=Severity.HIGH,
        drift_decision=DriftDecision.FREEZE, is_anomaly=True,
    )
    d = r.to_dict()
    assert d["is_anomaly"] is True
    assert d["sub_scores"][0]["name"] == "rate"
    assert d["drift_decision"] == "freeze"


def test_event_dataclass():
    e = Event("u1", "login", 123.0)
    assert e.entity == "u1" and e.timestamp == 123.0
