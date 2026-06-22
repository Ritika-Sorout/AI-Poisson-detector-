"""Process-shape tests: is the arrival pattern actually Poisson?

The rate model (``bayes_rate``) only looks at *how many* events occurred. These
tests look at *how they are distributed in time*. An attacker (or a buggy bot)
can reproduce a normal count while leaving a non-Poisson fingerprint:

* **Fano factor / index of dispersion** -- bin the window and compare the count
  variance across bins to the mean. A homogeneous Poisson process has Fano = 1.
  Metronomic bots are under-dispersed (Fano << 1); bursty floods are
  over-dispersed (Fano >> 1). Under H0, ``(b - 1) * Fano ~ chi2_{b-1}``.

* **Inter-arrival exponentiality** -- for a Poisson process the gaps between
  events are i.i.d. Exponential. A one-sample KS test against the MLE-fitted
  Exponential rejects equal-spacing / clustering.

Both return a :class:`SubScore` with a p-value in (0, 1]. When there is too
little data to judge shape, they return ``p = 1.0`` (abstain) rather than
fabricating significance.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .schemas import SubScore, Window


def fano_factor_test(
    window: Window,
    n_bins: int | None = None,
    min_events: int = 10,
    min_bins: int = 4,
) -> SubScore:
    """Index-of-dispersion test. Two-sided chi-square on the Fano factor."""
    count = window.count
    duration = window.duration_seconds
    if count < min_events or duration <= 0:
        return SubScore("fano", 1.0, float("nan"), "insufficient data for dispersion test")

    if n_bins is None:
        # Aim for a healthy expected count per bin (>= ~5), bounded to [min_bins, 30].
        n_bins = int(np.clip(count // 5, min_bins, 30))
    n_bins = max(n_bins, 2)

    edges = np.linspace(window.start, window.end, n_bins + 1)
    counts, _ = np.histogram(window.timestamps, bins=edges)
    mean = counts.mean()
    if mean <= 0:
        return SubScore("fano", 1.0, float("nan"), "empty bins")

    fano = counts.var(ddof=1) / mean
    df = n_bins - 1
    d_stat = df * fano  # ~ chi2_df under Poisson
    p_over = float(stats.chi2.sf(d_stat, df))   # over-dispersed (bursty)
    p_under = float(stats.chi2.cdf(d_stat, df))  # under-dispersed (regular)
    p = min(1.0, 2.0 * min(p_over, p_under))

    shape = "bursty" if fano > 1 else "regular"
    return SubScore("fano", p, float(fano), f"Fano={fano:.3f} ({shape}, {n_bins} bins)")


def exponentiality_test(window: Window, min_gaps: int = 8) -> SubScore:
    """KS test of inter-arrival times against a fitted Exponential."""
    gaps = window.interarrival_seconds()
    if gaps.size < min_gaps:
        return SubScore("exponentiality", 1.0, float("nan"), "insufficient gaps for KS test")

    scale = float(gaps.mean())
    if scale <= 0:
        return SubScore("exponentiality", 1.0, float("nan"), "degenerate (zero mean gap)")

    ks_stat, p = stats.kstest(gaps, "expon", args=(0.0, scale))
    cv = float(gaps.std() / scale)  # coefficient of variation; ~1 for exponential
    return SubScore("exponentiality", float(p), float(ks_stat), f"KS={ks_stat:.3f}, CV={cv:.3f}")


def run_shape_tests(window: Window) -> list[SubScore]:
    """Convenience: run both shape tests and return their sub-scores."""
    return [fano_factor_test(window), exponentiality_test(window)]
