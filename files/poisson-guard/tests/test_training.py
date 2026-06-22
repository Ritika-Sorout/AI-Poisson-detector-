import numpy as np

from poissonguard.bucketing import BucketingConfig
from poissonguard.schemas import Bucket
from poissonguard.training import (
    load_population,
    save_population,
    train_detector,
    train_synthetic,
)
from poissonguard.generators import make_population, generate_training

CFG = BucketingConfig()


def test_train_detector_fits_business_rates_near_truth():
    pop = make_population(8, seed=0)
    streams, start, end = generate_training(pop, weeks=4, cfg=CFG, seed=1)
    det = train_detector(streams, start, end, bucketing=CFG)

    # Every (entity, event_type) should have both bucket baselines.
    assert len(det.baselines) == 8 * 2
    # Business baselines should recover roughly the profile rate.
    for p in pop:
        b = det.baselines[f"{p.entity}|{p.event_type}|business"]
        assert abs(b.posterior_mean_rate - p.business_rate) < 2.0


def test_train_creates_guards_with_anchor():
    det, pop = train_synthetic(weeks=2, n_entities=4, seed=0)
    for key, b in det.baselines.items():
        assert key in det.guards
        assert np.isclose(det.guards[key].anchor, b.posterior_mean_rate)


def test_population_roundtrip(tmp_path):
    pop = make_population(5, seed=3)
    path = tmp_path / "pop.json"
    save_population(pop, str(path))
    loaded = load_population(str(path))
    assert [p.entity for p in loaded] == [p.entity for p in pop]
    assert np.isclose(loaded[0].business_rate, pop[0].business_rate)


def test_eb_prior_is_set_on_config():
    det, _ = train_synthetic(weeks=2, n_entities=6, seed=0)
    assert det.config.prior.mean_rate_per_hour > 0
    assert det.config.prior.strength_hours > 0
