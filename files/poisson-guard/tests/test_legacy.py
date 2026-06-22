import numpy as np

from poissonguard.bucketing import BucketingConfig, bucketize
from poissonguard.generators import EntityProfile, generate_normal, make_attack, DAY, WEEK
from poissonguard.legacy import LegacyDetector
from poissonguard.schemas import Bucket

CFG = BucketingConfig()


def test_calendar_span_dilution_underestimates_business_rate():
    p = EntityProfile("e", "login", business_rate=5.0, offhours_rate=0.0)
    rng = np.random.default_rng(0)
    ts = generate_normal(p, 0.0, 4 * WEEK, CFG, rng)

    legacy = LegacyDetector()
    legacy.fit_stream("e", "login", ts, 0.0, 4 * WEEK)
    diluted = legacy.lambda_for("e", "login")

    true_business = bucketize("e", "login", ts, 0.0, 4 * WEEK, CFG)[Bucket.BUSINESS].empirical_rate_per_hour()
    # Diluted estimate is far below the real business-hours rate (~40/168 factor).
    assert diluted < true_business * 0.5


def test_legacy_detects_large_volume_spike():
    p = EntityProfile("e", "login", 5.0, 0.0)
    rng = np.random.default_rng(0)
    train = generate_normal(p, 0.0, 4 * WEEK, CFG, rng)
    legacy = LegacyDetector()
    legacy.fit_stream("e", "login", train, 0.0, 4 * WEEK)

    spike = make_attack("volume_spike", p, 0.0, CFG, rng, severity=8.0)
    _, is_anom = legacy.detect("e", "login", spike, 0.0, DAY)
    assert is_anom


def test_legacy_is_blind_to_shape_at_normal_volume():
    p = EntityProfile("e", "login", 5.0, 0.0)
    rng = np.random.default_rng(0)
    train = generate_normal(p, 0.0, 4 * WEEK, CFG, rng)
    legacy = LegacyDetector()
    legacy.fit_stream("e", "login", train, 0.0, 4 * WEEK)

    # A metronomic bot at ~normal daily volume: same count, different shape.
    bot = make_attack("regular_bot", p, 0.0, CFG, rng, severity=1.0)
    p_val, _ = legacy.detect("e", "login", bot, 0.0, DAY)
    assert p_val > 1e-3  # count-only detector does not flag it


def test_legacy_baseline_is_poisonable():
    legacy = LegacyDetector()
    legacy.fit_stream("e", "login", np.arange(0, 100) * 3600.0, 0.0, 100 * 3600.0)
    before = legacy.lambda_for("e", "login")
    # Fold in escalating activity -> baseline rate creeps up (no defense).
    for d in range(20):
        n = 100 + d * 20
        ts = np.linspace(0, DAY, n)
        legacy.update("e", "login", ts, d * DAY, (d + 1) * DAY)
    after = legacy.lambda_for("e", "login")
    assert after > before  # poisoned upward


def test_zero_baseline_safe():
    legacy = LegacyDetector()
    assert legacy.score("x", "y", 5, 1.0) == 1.0
