"""Drift-integrity gate -- defense against adversarial baseline poisoning.

Threat model
------------
Rate-based detectors typically re-estimate their own baseline from recent
traffic. An attacker who increases activity slowly enough ("boil the frog")
keeps every single step below the per-update alarm, dragging the learned
baseline up until the attacker's target rate looks normal. The baseline is
*poisoned*.

Defense
-------
We never trust an unbounded re-learn. The gate keeps an **anchor** -- the rate
established from trusted training data -- and runs two change detectors on the
stream of observed rates, both referenced to that anchor:

* **Slow-drift CUSUM** (scale-free): accumulate the normalized upward deviation
  ``d_t = (rate - anchor) / anchor`` minus a per-step slack ``k``, where
  ``anchor`` is the **fixed trusted baseline** -- it is *not* re-learned from
  recent traffic. Individually tiny steps still accumulate against the trusted
  reference, so the cumulative budget ``h`` trips even when no single step is
  suspicious. Referencing a *moving* anchor here would let the detector follow
  the attacker and defeat the whole defense, so we deliberately do not.

* **Page-Hinkley**: a classic sequential change detector on the rate mean, used
  as corroboration / faster reaction to sharper shifts.

Decision per observed rate:

* a large single-step jump -> ``REGIME_CHANGE`` (a legitimate abrupt shift such
  as a role change): flag for review and re-anchor under supervision;
* slow accumulation tripping CUSUM/PH -> ``FREEZE``: refuse to fold the
  observation into the baseline and raise an integrity alert; the anchor is
  held, so adaptation stops *below* the attacker's target;
* otherwise -> ``ACCEPT``: fold in with bounded EWMA adaptation of the anchor.

Once frozen the gate stays frozen (returning ``FREEZE``) until an operator
calls :meth:`DriftGuard.reset`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import DriftDecision

_EPS = 1e-9


@dataclass
class DriftGuardConfig:
    cusum_slack: float = 0.05      # k: per-step normalized drift tolerated (5%)
    cusum_budget: float = 0.75     # h: cumulative normalized drift before freeze
    ph_delta: float = 0.05         # Page-Hinkley tolerance (relative)
    ph_threshold: float = 1.5      # Page-Hinkley alarm threshold (relative units)
    regime_jump: float = 1.0       # single-step jump ratio => abrupt regime change


@dataclass
class DriftGuard:
    """Stateful, serializable drift-integrity gate for one baseline stream."""

    anchor: float                  # fixed trusted reference (events/hour)
    config: DriftGuardConfig = field(default_factory=DriftGuardConfig)
    # internal running state
    n: int = 0
    cusum: float = 0.0
    ph_sum: float = 0.0
    ph_min: float = 0.0
    frozen: bool = False

    def observe(self, rate: float) -> DriftDecision:
        """Process one observed rate; return the gate's verdict and update state."""
        rate = float(max(rate, 0.0))

        # Establish anchor if we never had a trustworthy one.
        if self.anchor <= _EPS:
            if rate > _EPS:
                self._reanchor(rate)
            return DriftDecision.ACCEPT

        if self.frozen:
            return DriftDecision.FREEZE

        self.n += 1
        # Normalized deviation from the FIXED trusted anchor (not a moving mean).
        step = (rate - self.anchor) / self.anchor

        # --- abrupt regime change: a single large jump (up or down) ---
        if abs(step) >= self.config.regime_jump:
            self._reanchor(rate)
            return DriftDecision.REGIME_CHANGE

        # --- slow-drift CUSUM (upward, anchored at the trusted baseline) ---
        self.cusum = max(0.0, self.cusum + step - self.config.cusum_slack)
        cusum_alarm = self.cusum > self.config.cusum_budget

        # --- Page-Hinkley on the relative deviation from the trusted anchor ---
        self.ph_sum += step - self.config.ph_delta
        self.ph_min = min(self.ph_min, self.ph_sum)
        ph_alarm = (self.ph_sum - self.ph_min) > self.config.ph_threshold

        if cusum_alarm or ph_alarm:
            self.frozen = True
            return DriftDecision.FREEZE

        # --- benign fluctuation around the trusted anchor: accept, hold anchor ---
        return DriftDecision.ACCEPT

    def _reanchor(self, rate: float) -> None:
        self.anchor = rate
        self.cusum = 0.0
        self.ph_sum = 0.0
        self.ph_min = 0.0
        self.n = 0
        self.frozen = False

    def reset(self, anchor: float | None = None) -> None:
        """Operator-driven reset after reviewing a freeze/regime alert."""
        self._reanchor(self.anchor if anchor is None else float(anchor))

    def state_dict(self) -> dict:
        return {
            "anchor": self.anchor, "n": self.n,
            "cusum": self.cusum, "ph_sum": self.ph_sum, "ph_min": self.ph_min,
            "frozen": self.frozen,
        }

    @classmethod
    def from_state(cls, state: dict, config: DriftGuardConfig | None = None) -> "DriftGuard":
        g = cls(anchor=float(state["anchor"]), config=config or DriftGuardConfig())
        g.n = int(state.get("n", 0))
        g.cusum = float(state.get("cusum", 0.0))
        g.ph_sum = float(state.get("ph_sum", 0.0))
        g.ph_min = float(state.get("ph_min", 0.0))
        g.frozen = bool(state.get("frozen", False))
        return g


def evaluate_rate_stream(
    rates, anchor: float, config: DriftGuardConfig | None = None
):
    """Run a guard over a sequence of rates.

    Returns ``(decisions, final_anchor, guard)`` -- handy for tests, evaluation
    and replaying a baseline's ``rate_history``.
    """
    guard = DriftGuard(anchor=anchor, config=config or DriftGuardConfig())
    decisions = [guard.observe(r) for r in rates]
    return decisions, guard.anchor, guard
