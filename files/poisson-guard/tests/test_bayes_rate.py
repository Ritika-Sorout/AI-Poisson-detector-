import numpy as np
from scipy import stats

from poissonguard.bayes_rate import (
    GammaPoissonModel,
    Prior,
    fit_baseline,
    posterior,
    predictive_mean,
    predictive_pvalue,
)
from poissonguard.schemas import Bucket


def test_prior_encodes_mean():
    a, b = Prior(mean_rate_per_hour=3.0, strength_hours=2.0).as_gamma()
    assert np.isclose(a / b, 3.0)
    assert np.isclose(b, 2.0)


def test_posterior_update():
    a, b = posterior(2.0, 1.0, count=10, exposure_hours=5.0)
    assert a == 12.0 and b == 6.0


def test_fit_baseline_mean_between_prior_and_data():
    # data rate = 100/10 = 10/hr, prior mean 1/hr; posterior mean in (1, 10)
    b = fit_baseline("u", "e", Bucket.BUSINESS, count=100, exposure_hours=10.0,
                     prior=Prior(1.0, 1.0))
    assert 1.0 < b.posterior_mean_rate < 10.0
    assert b.n_events == 100


def test_predictive_pvalue_normal_count_not_significant():
    base = fit_baseline("u", "e", Bucket.BUSINESS, count=100, exposure_hours=100.0,
                        prior=Prior(1.0, 1.0))  # ~1/hr
    p, stat = predictive_pvalue(base, count=1, exposure_hours=1.0, tail="greater")
    assert p > 0.2
    assert abs(stat) < 2


def test_predictive_pvalue_spike_is_significant():
    base = fit_baseline("u", "e", Bucket.BUSINESS, count=100, exposure_hours=100.0,
                        prior=Prior(1.0, 1.0))  # ~1/hr
    p, stat = predictive_pvalue(base, count=50, exposure_hours=1.0, tail="greater")
    assert p < 1e-6
    assert stat > 5


def test_predictive_matches_scipy_nbinom():
    base = fit_baseline("u", "e", Bucket.BUSINESS, count=20, exposure_hours=10.0,
                        prior=Prior(1.0, 1.0))
    t = 3.0
    n = base.alpha
    p = base.beta / (base.beta + t)
    expected = float(stats.nbinom(n, p).sf(7 - 1))
    pv, _ = predictive_pvalue(base, count=7, exposure_hours=t, tail="greater")
    assert np.isclose(pv, expected)


def test_two_sided_and_less_tail():
    base = fit_baseline("u", "e", Bucket.BUSINESS, count=100, exposure_hours=100.0,
                        prior=Prior(1.0, 1.0))
    p_less, _ = predictive_pvalue(base, count=0, exposure_hours=10.0, tail="less")
    p_two, _ = predictive_pvalue(base, count=0, exposure_hours=10.0, tail="two-sided")
    assert p_less < 0.05          # 0 events when ~10 expected is suppressed
    assert p_two <= 1.0


def test_sparse_entity_has_wide_tolerance():
    # Tiny exposure -> uncertain rate -> a moderate count should NOT be critical.
    sparse = fit_baseline("u", "e", Bucket.BUSINESS, count=2, exposure_hours=2.0,
                          prior=Prior(1.0, 0.5))
    dense = fit_baseline("u", "e", Bucket.BUSINESS, count=200, exposure_hours=200.0,
                         prior=Prior(1.0, 0.5))
    p_sparse, _ = predictive_pvalue(sparse, count=8, exposure_hours=1.0)
    p_dense, _ = predictive_pvalue(dense, count=8, exposure_hours=1.0)
    assert p_sparse > p_dense  # sparse model is less surprised


def test_zero_exposure_safe():
    base = fit_baseline("u", "e", Bucket.BUSINESS, 10, 10.0)
    assert predictive_pvalue(base, 5, 0.0) == (1.0, 0.0)
    assert predictive_mean(base, 0.0) == 0.0


def test_model_wrapper():
    m = GammaPoissonModel(Prior(2.0, 1.0))
    b = m.fit("u", "e", Bucket.OFFHOURS, 20, 10.0)
    p, _ = m.score(b, 3, 1.0)
    assert 0 < p <= 1
    assert m.expected(b, 2.0) > 0
