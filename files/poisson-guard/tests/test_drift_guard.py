import numpy as np

from poissonguard.drift_guard import DriftGuard, DriftGuardConfig, evaluate_rate_stream
from poissonguard.schemas import DriftDecision

RNG = np.random.default_rng(1)


def test_stable_traffic_is_accepted():
    rates = list(5.0 + RNG.normal(0, 0.2, size=50))
    decisions, anchor, _ = evaluate_rate_stream(rates, anchor=5.0)
    assert all(d == DriftDecision.ACCEPT for d in decisions)
    assert 4.5 < anchor < 5.5  # anchor stays put


def test_boil_the_frog_is_frozen_below_target():
    # Attacker ramps 5 -> 15 in small steps; gate must NOT reach the target.
    ramp = list(np.linspace(5.0, 15.0, 60))
    decisions, anchor, guard = evaluate_rate_stream(ramp, anchor=5.0)
    assert DriftDecision.FREEZE in decisions
    assert guard.frozen
    assert anchor < 12.0          # never dragged up to the attacker's 15
    # once frozen, stays frozen
    assert guard.observe(15.0) == DriftDecision.FREEZE


def test_abrupt_jump_is_regime_change_not_freeze():
    rates = [5.0] * 10 + [12.0] * 10  # sudden doubling
    decisions, anchor, guard = evaluate_rate_stream(rates, anchor=5.0)
    assert DriftDecision.REGIME_CHANGE in decisions
    # re-anchored to the new level, not frozen
    assert not guard.frozen
    assert anchor > 10.0


def test_freeze_stops_adaptation_at_freeze_point():
    ramp = list(np.linspace(5.0, 20.0, 80))
    _, anchor, guard = evaluate_rate_stream(ramp, anchor=5.0)
    assert guard.frozen
    assert anchor < 15.0


def test_reset_clears_freeze():
    ramp = list(np.linspace(5.0, 15.0, 60))
    _, _, guard = evaluate_rate_stream(ramp, anchor=5.0)
    assert guard.frozen
    guard.reset(anchor=5.0)
    assert not guard.frozen
    assert guard.observe(5.1) == DriftDecision.ACCEPT


def test_zero_anchor_establishes_on_first_rate():
    g = DriftGuard(anchor=0.0)
    assert g.observe(3.0) == DriftDecision.ACCEPT
    assert g.anchor == 3.0


def test_state_roundtrip():
    g = DriftGuard(anchor=5.0)
    for r in [5.1, 5.2, 5.0, 4.9]:
        g.observe(r)
    g2 = DriftGuard.from_state(g.state_dict())
    assert np.isclose(g2.anchor, g.anchor)
    assert g2.cusum == g.cusum
    assert g2.frozen == g.frozen


def test_sensitive_config_freezes_faster():
    ramp = list(np.linspace(5.0, 10.0, 60))
    strict = DriftGuardConfig(cusum_slack=0.01, cusum_budget=0.2)
    decisions, _, guard = evaluate_rate_stream(ramp, anchor=5.0, config=strict)
    assert guard.frozen
    assert decisions.index(DriftDecision.FREEZE) < 40  # trips early
