import numpy as np

from poissonguard.bucketing import BucketingConfig, bucketize
from poissonguard.generators import (
    ATTACK_TYPES,
    EntityProfile,
    generate_eval_windows,
    generate_normal,
    generate_poisoning_sequence,
    generate_training,
    make_attack,
    make_population,
    DAY,
    WEEK,
)
from poissonguard.schemas import Bucket

CFG = BucketingConfig()


def test_population_is_reproducible():
    a = make_population(8, seed=0)
    b = make_population(8, seed=0)
    assert [p.entity for p in a] == [p.entity for p in b]
    assert np.isclose(a[0].business_rate, b[0].business_rate)
    assert all(p.offhours_rate < p.business_rate for p in a)


def test_normal_business_rate_matches_profile():
    p = EntityProfile("e", "login", business_rate=5.0, offhours_rate=0.05)
    rng = np.random.default_rng(0)
    ts = generate_normal(p, 0.0, 4 * WEEK, CFG, rng)
    biz = bucketize("e", "login", ts, 0.0, 4 * WEEK, CFG)[Bucket.BUSINESS]
    assert 4.0 < biz.empirical_rate_per_hour() < 6.0  # recovers ~5/hr


def test_training_streams_cover_population():
    pop = make_population(6, seed=0)
    streams, start, end = generate_training(pop, weeks=2, cfg=CFG)
    assert len(streams) == 6
    assert end == 2 * WEEK
    assert all(len(ts) > 0 for ts in streams.values())


def test_volume_spike_has_more_events_than_normal():
    p = EntityProfile("e", "login", 5.0, 0.05)
    rng = np.random.default_rng(0)
    normal = make_attack("volume_spike", p, 0.0, CFG, rng, severity=1.0)
    spike = make_attack("volume_spike", p, 0.0, CFG, rng, severity=6.0)
    assert spike.size > normal.size * 3


def test_regular_bot_has_constant_gaps():
    p = EntityProfile("e", "login", 5.0, 0.05)
    ts = make_attack("regular_bot", p, 0.0, CFG, np.random.default_rng(0))
    biz = bucketize("e", "login", ts, 0.0, DAY, CFG)[Bucket.BUSINESS]
    gaps = biz.interarrival_seconds()
    # near-constant spacing within a business block
    assert gaps.std() / gaps.mean() < 0.2


def test_offhours_attack_lands_in_offhours_bucket():
    p = EntityProfile("e", "login", 5.0, 0.0)
    ts = make_attack("offhours", p, 0.0, CFG, np.random.default_rng(0), severity=6.0)
    buckets = bucketize("e", "login", ts, 0.0, DAY, CFG)
    assert buckets[Bucket.OFFHOURS].count > buckets[Bucket.BUSINESS].count


def test_eval_windows_labeled():
    pop = make_population(3, seed=0)
    windows = generate_eval_windows(pop, CFG, normal_days=3)
    normals = [w for w in windows if w.label == 0]
    attacks = [w for w in windows if w.label == 1]
    assert len(normals) == 3 * 3
    assert {w.attack_type for w in attacks} == set(ATTACK_TYPES)


def test_poisoning_sequence_ramps():
    p = EntityProfile("e", "login", 5.0, 0.05)
    seq = generate_poisoning_sequence(p, CFG, days=10)
    mults = [m for *_, m in seq]
    assert mults[0] < mults[-1]
    assert len(seq) == 10
