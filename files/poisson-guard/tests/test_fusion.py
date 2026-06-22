import numpy as np
from scipy import stats

from poissonguard.fusion import empirical_bayes_prior, fisher_combine
from poissonguard.schemas import SubScore

RNG = np.random.default_rng(2)


def test_fisher_all_abstain_returns_one():
    r = fisher_combine([SubScore("a", 1.0, 0.0), SubScore("b", 1.0, 0.0)])
    assert r.fused_p_value == 1.0
    assert r.n_active == 0


def test_fisher_single_active_equals_itself():
    r = fisher_combine([SubScore("rate", 0.01, 3.0), SubScore("fano", 1.0, 0.0)])
    assert np.isclose(r.fused_p_value, 0.01, rtol=1e-6)
    assert r.n_active == 1
    assert r.dominant == "rate"


def test_fisher_combines_to_stronger_signal():
    r = fisher_combine([SubScore("a", 0.04, 1.0), SubScore("b", 0.04, 1.0)])
    assert r.fused_p_value < 0.04  # joint evidence is stronger than either alone


def test_fisher_matches_chi2_formula():
    ps = [0.2, 0.03, 0.5]
    r = fisher_combine([SubScore(str(i), p, 0.0) for i, p in enumerate(ps)])
    x = -2 * sum(np.log(ps))
    assert np.isclose(r.fused_p_value, stats.chi2.sf(x, 2 * len(ps)))


def test_dominant_is_smallest_p():
    r = fisher_combine([SubScore("a", 0.2, 0), SubScore("b", 1e-5, 0), SubScore("c", 0.4, 0)])
    assert r.dominant == "b"


def test_empirical_bayes_recovers_population_mean():
    # Entities with rate ~5/hr, varying exposures.
    true_rate = 5.0
    exposures = RNG.uniform(20, 200, size=40)
    counts = RNG.poisson(true_rate * exposures)
    prior = empirical_bayes_prior(counts, exposures)
    assert 4.0 < prior.mean_rate_per_hour < 6.0
    assert prior.strength_hours > 0


def test_empirical_bayes_overdispersed_gives_weaker_strength():
    exposures = np.full(40, 50.0)
    # Heterogeneous entity rates -> larger between-group variance -> weaker prior.
    hetero_rates = RNG.uniform(1, 20, size=40)
    homo_rates = np.full(40, 5.0)
    counts_h = RNG.poisson(hetero_rates * exposures)
    counts_o = RNG.poisson(homo_rates * exposures)
    p_h = empirical_bayes_prior(counts_h, exposures)
    p_o = empirical_bayes_prior(counts_o, exposures)
    assert p_h.strength_hours < p_o.strength_hours


def test_empirical_bayes_too_few_groups_fallback():
    prior = empirical_bayes_prior([10], [5.0])
    assert prior.strength_hours == 1.0
    assert np.isclose(prior.mean_rate_per_hour, 2.0)
