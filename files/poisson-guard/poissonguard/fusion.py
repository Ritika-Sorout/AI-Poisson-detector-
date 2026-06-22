"""Fusion / scoring layer.

Two responsibilities:

1. **Combine sub-tests** -- the rate model and the two shape tests each emit a
   p-value. Under their nulls these are ~Uniform(0, 1) and approximately
   independent, so **Fisher's method** combines them::

       X = -2 * sum_i ln(p_i)  ~  chi2_{2m}   (m = number of *active* tests)

   Tests that abstain (``p == 1``, e.g. too little data) are dropped from the
   combination rather than inflating the degrees of freedom, which would
   otherwise wash out a real signal.

2. **Hierarchical empirical-Bayes prior** -- pool counts/exposures across all
   entities to estimate a population Gamma prior on the rate (Clayton-Kaldor
   moment estimator). Individual baselines then shrink toward the population
   mean, so a sparsely-observed entity inherits sane expectations instead of
   producing knee-jerk "critical" alerts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .bayes_rate import Prior
from .schemas import SubScore

_ABSTAIN = 1.0


@dataclass
class FusionResult:
    fused_p_value: float
    statistic: float
    n_active: int
    dominant: str  # name of the most significant active sub-test


def fisher_combine(sub_scores: list[SubScore]) -> FusionResult:
    """Combine sub-test p-values via Fisher's method, ignoring abstentions."""
    active = [s for s in sub_scores if s.p_value < _ABSTAIN]
    if not active:
        return FusionResult(fused_p_value=1.0, statistic=0.0, n_active=0, dominant="")

    x = -2.0 * float(np.sum([np.log(s.p_value) for s in active]))
    df = 2 * len(active)
    fused = float(stats.chi2.sf(x, df))
    dominant = min(active, key=lambda s: s.p_value).name
    return FusionResult(
        fused_p_value=min(max(fused, 1e-12), 1.0),
        statistic=x,
        n_active=len(active),
        dominant=dominant,
    )


def empirical_bayes_prior(
    counts,
    exposures_hours,
    min_groups: int = 3,
    fallback_strength: float = 1.0,
) -> Prior:
    """Estimate a population Gamma prior from per-entity (count, exposure) data.

    Uses the Clayton-Kaldor moment estimator for the between-group variance of
    the underlying rates (removing the Poisson sampling component). Returns a
    :class:`Prior` whose mean is the population rate and whose strength reflects
    how concentrated entity rates are around that mean.
    """
    k = np.asarray(counts, dtype=float)
    t = np.asarray(exposures_hours, dtype=float)
    mask = t > 0
    k, t = k[mask], t[mask]

    total_t = t.sum()
    if k.size < min_groups or total_t <= 0:
        # Not enough information to pool -> weak prior at the grand mean.
        m = (k.sum() / total_t) if total_t > 0 else 1.0
        return Prior(mean_rate_per_hour=max(m, 1e-6), strength_hours=fallback_strength)

    m = k.sum() / total_t                      # population mean rate (events/hr)
    rates = k / t
    w = t                                      # weight groups by exposure
    # Clayton-Kaldor between-group variance estimate.
    num = float(np.sum(w * (rates - m) ** 2) - (k.size - 1) * m)
    denom = float(total_t - np.sum(w ** 2) / total_t)
    var_between = num / denom if denom > 0 else 0.0

    if var_between <= 0:
        # Under-dispersed: entities are tighter than Poisson -> strong prior.
        strength = max(total_t / k.size, fallback_strength)
        return Prior(mean_rate_per_hour=max(m, 1e-6), strength_hours=strength)

    # Gamma(alpha, beta): mean = m = alpha/beta, var = alpha/beta^2 = var_between
    # => beta (== strength) = m / var_between
    strength = max(m / var_between, 1e-3)
    return Prior(mean_rate_per_hour=max(m, 1e-6), strength_hours=strength)
