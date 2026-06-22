"""Gamma-Poisson Bayesian rate model.

Model
-----
For a given (entity, event_type, bucket) the event rate ``lambda`` (events/hour)
is unknown and given a conjugate prior::

    lambda ~ Gamma(alpha, beta)            # beta is a *rate* parameter

Over an exposure of ``t`` hours the observed count is::

    k | lambda ~ Poisson(lambda * t)

Conjugacy gives the posterior in closed form::

    lambda | k, t ~ Gamma(alpha + k, beta + t)

Crucially the *posterior-predictive* distribution of a future count over
exposure ``t`` is Negative-Binomial::

    N ~ NegBin(r = alpha, p = beta / (beta + t)),   E[N] = alpha * t / beta

so the predictive p-value of an observed count needs no sampling. Using the
predictive (rather than a plug-in Poisson at the posterior mean) is what keeps
sparse entities from producing spurious "critical" alerts: small training
exposure -> small ``alpha`` -> heavy-tailed predictive -> wide tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import stats

from .schemas import Baseline, Bucket


@dataclass(frozen=True)
class Prior:
    """Weakly-informative Gamma prior centred on a rate guess.

    ``strength`` is the prior's pseudo-exposure in hours: larger -> more
    confident prior. A prior mean rate ``m`` is encoded as
    ``alpha = m * strength``, ``beta = strength`` (so prior mean = alpha/beta = m).
    """

    mean_rate_per_hour: float = 1.0
    strength_hours: float = 1.0

    def as_gamma(self) -> Tuple[float, float]:
        beta = max(self.strength_hours, 1e-6)
        alpha = max(self.mean_rate_per_hour * beta, 1e-6)
        return alpha, beta


def posterior(alpha: float, beta: float, count: int, exposure_hours: float) -> Tuple[float, float]:
    """Conjugate Gamma update: returns updated ``(alpha, beta)``."""
    return alpha + float(count), beta + max(exposure_hours, 0.0)


def fit_baseline(
    entity: str,
    event_type: str,
    bucket: Bucket,
    count: int,
    exposure_hours: float,
    prior: Prior | None = None,
) -> Baseline:
    """Fit a :class:`Baseline` from aggregate training counts/exposure."""
    prior = prior or Prior()
    a0, b0 = prior.as_gamma()
    a, b = posterior(a0, b0, count, exposure_hours)
    rate = a / b if b > 0 else 0.0
    return Baseline(
        entity=entity,
        event_type=event_type,
        bucket=bucket,
        alpha=a,
        beta=b,
        n_events=int(count),
        exposure_hours=float(exposure_hours),
        rate_history=[rate],
    )


def _nbinom_params(baseline: Baseline, exposure_hours: float) -> Tuple[float, float]:
    """Map the Gamma posterior + exposure to scipy NegBin ``(n, p)``."""
    n = baseline.alpha
    p = baseline.beta / (baseline.beta + max(exposure_hours, 1e-9))
    return n, p


def predictive_mean(baseline: Baseline, exposure_hours: float) -> float:
    """Expected count over the given exposure under the posterior-predictive."""
    return baseline.posterior_mean_rate * max(exposure_hours, 0.0)


def predictive_pvalue(
    baseline: Baseline,
    count: int,
    exposure_hours: float,
    tail: str = "greater",
) -> Tuple[float, float]:
    """Posterior-predictive p-value and a signed z-like statistic.

    ``tail``:
      - ``"greater"`` : P(N >= count)  -> detects elevated rates (default)
      - ``"less"``    : P(N <= count)  -> detects suppressed rates / silence
      - ``"two-sided"``: 2 * min(greater, less), clamped to 1

    The statistic is ``(count - mean) / std`` of the predictive, useful for
    ranking and human-readable detail. Returns ``(p_value, statistic)``.
    """
    if exposure_hours <= 0:
        return 1.0, 0.0

    n, p = _nbinom_params(baseline, exposure_hours)
    dist = stats.nbinom(n, p)
    k = int(count)

    mean = dist.mean()
    std = dist.std()
    statistic = float((k - mean) / std) if std > 0 else 0.0

    p_greater = float(dist.sf(k - 1))   # P(N >= k)
    p_less = float(dist.cdf(k))         # P(N <= k)

    if tail == "greater":
        pv = p_greater
    elif tail == "less":
        pv = p_less
    elif tail == "two-sided":
        pv = min(1.0, 2.0 * min(p_greater, p_less))
    else:
        raise ValueError(f"unknown tail: {tail!r}")

    return float(min(max(pv, 0.0), 1.0)), statistic


class GammaPoissonModel:
    """Convenience wrapper bundling prior + fit + scoring."""

    def __init__(self, prior: Prior | None = None):
        self.prior = prior or Prior()

    def fit(self, entity, event_type, bucket, count, exposure_hours) -> Baseline:
        return fit_baseline(entity, event_type, bucket, count, exposure_hours, self.prior)

    def score(self, baseline: Baseline, count: int, exposure_hours: float, tail: str = "greater"):
        return predictive_pvalue(baseline, count, exposure_hours, tail)

    def expected(self, baseline: Baseline, exposure_hours: float) -> float:
        return predictive_mean(baseline, exposure_hours)
