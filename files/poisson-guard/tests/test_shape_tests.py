import numpy as np

from poissonguard.schemas import Window
from poissonguard.shape_tests import (
    exponentiality_test,
    fano_factor_test,
    run_shape_tests,
)

RNG = np.random.default_rng(0)


def poisson_window(rate_per_sec=0.05, duration=4000.0):
    """Homogeneous Poisson process via exponential gaps."""
    gaps = RNG.exponential(1.0 / rate_per_sec, size=5000)
    ts = np.cumsum(gaps)
    ts = ts[ts < duration]
    return Window.from_timestamps("u", "e", ts, start=0.0, end=duration)


def regular_window(period=20.0, duration=4000.0):
    ts = np.arange(0.0, duration, period)
    return Window.from_timestamps("u", "e", ts, start=0.0, end=duration)


def bursty_window(duration=4000.0):
    # Three tight bursts separated by long silence -> over-dispersed.
    bursts = []
    for center in (200.0, 2000.0, 3800.0):
        bursts.append(center + RNG.uniform(0, 5, size=40))
    ts = np.sort(np.concatenate(bursts))
    return Window.from_timestamps("u", "e", ts, start=0.0, end=duration)


def test_poisson_passes_both_tests():
    w = poisson_window()
    fano = fano_factor_test(w)
    expo = exponentiality_test(w)
    assert fano.p_value > 0.05
    assert expo.p_value > 0.05


def test_regular_bot_fails_fano_and_exponentiality():
    w = regular_window()
    fano = fano_factor_test(w)
    expo = exponentiality_test(w)
    assert fano.statistic < 1.0       # under-dispersed
    assert fano.p_value < 0.05
    assert expo.p_value < 0.05        # gaps not exponential


def test_bursty_attack_overdispersed():
    w = bursty_window()
    fano = fano_factor_test(w)
    assert fano.statistic > 1.0
    assert fano.p_value < 0.05


def test_insufficient_data_abstains():
    w = Window.from_timestamps("u", "e", [1.0, 2.0, 3.0], start=0.0, end=10.0)
    assert fano_factor_test(w).p_value == 1.0
    assert exponentiality_test(w).p_value == 1.0


def test_run_shape_tests_returns_two():
    scores = run_shape_tests(poisson_window())
    assert [s.name for s in scores] == ["fano", "exponentiality"]
