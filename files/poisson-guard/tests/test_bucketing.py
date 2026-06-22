import numpy as np
from datetime import datetime, timezone

from poissonguard.bucketing import (
    BucketingConfig,
    bucketize,
    exposure_hours,
    active_intervals,
)
from poissonguard.schemas import Bucket

# Monday 2024-01-01 00:00:00 UTC
MONDAY = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
HOUR = 3600.0
DAY = 24 * HOUR
WEEK = 7 * DAY


def test_bucket_for_business_vs_offhours():
    cfg = BucketingConfig()
    assert cfg.bucket_for(MONDAY + 10 * HOUR) == Bucket.BUSINESS   # Mon 10:00
    assert cfg.bucket_for(MONDAY + 22 * HOUR) == Bucket.OFFHOURS   # Mon 22:00
    assert cfg.bucket_for(MONDAY + 5 * DAY + 10 * HOUR) == Bucket.OFFHOURS  # Sat 10:00


def test_business_exposure_one_week():
    cfg = BucketingConfig()
    biz = exposure_hours(MONDAY, MONDAY + WEEK, Bucket.BUSINESS, cfg)
    off = exposure_hours(MONDAY, MONDAY + WEEK, Bucket.OFFHOURS, cfg)
    assert np.isclose(biz, 5 * 8)          # 5 days * 8 business hrs = 40
    assert np.isclose(biz + off, 7 * 24)   # buckets partition the span


def test_calendar_span_dilution_is_corrected():
    cfg = BucketingConfig()
    # 200 events spread only across business hours of one week.
    rng = np.random.default_rng(0)
    ts = []
    for day in range(5):  # Mon-Fri
        base = MONDAY + day * DAY + 9 * HOUR
        ts.extend(base + rng.uniform(0, 8 * HOUR, size=40))
    ts = np.array(sorted(ts))
    start, end = MONDAY, MONDAY + WEEK

    naive_rate = len(ts) / ((end - start) / HOUR)            # diluted
    buckets = bucketize("u", "e", ts, start, end, cfg)
    biz = buckets[Bucket.BUSINESS]
    bucketed_rate = biz.empirical_rate_per_hour()

    assert biz.count == 200
    assert buckets[Bucket.OFFHOURS].count == 0
    # business-only rate should be ~4.2x the diluted one (168/40)
    assert bucketed_rate > naive_rate * 3.5


def test_compressed_window_has_active_exposure():
    cfg = BucketingConfig()
    ts = [MONDAY + 10 * HOUR, MONDAY + 11 * HOUR, MONDAY + DAY + 10 * HOUR]
    buckets = bucketize("u", "e", ts, MONDAY, MONDAY + 2 * DAY, cfg)
    biz = buckets[Bucket.BUSINESS]
    # Two business days => 16 active hours of exposure regardless of overnight gap.
    assert np.isclose(biz.duration_hours, 16.0)
    # Events compressed: gap between day-1 and day-2 events is < 24h on active clock.
    assert biz.timestamps.max() < 16 * HOUR


def test_active_intervals_partition():
    cfg = BucketingConfig()
    biz = active_intervals(MONDAY, MONDAY + DAY, Bucket.BUSINESS, cfg)
    off = active_intervals(MONDAY, MONDAY + DAY, Bucket.OFFHOURS, cfg)
    total = sum(b - a for a, b in biz) + sum(b - a for a, b in off)
    assert np.isclose(total, DAY)


def test_empty_events():
    buckets = bucketize("u", "e", [], MONDAY, MONDAY + DAY)
    assert buckets[Bucket.BUSINESS].count == 0
    assert buckets[Bucket.OFFHOURS].count == 0
