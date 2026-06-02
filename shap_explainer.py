"""
SHAP Explainability — Per-sample Forensic Trace
==================================================
Wraps SHAP's KernelExplainer (model-agnostic) around the ensemble
classifier to produce per-sample feature importance scores.

For each poisoned sample the explainer outputs:
  • shap_values  : contribution of each forensic feature to the anomaly score
  • top_features : human-readable list of the most influential signals
  • plot()       : waterfall / beeswarm visualisation helper
"""

import numpy as np
import shap
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


# Feature names correspond to the fingerprint layout produced by Stage 1.
# The last two features are always grad_norm and hessian_curvature.
def _build_feature_names(activation_dim: int) -> List[str]:
    names = [f"act_{i}" for i in range(activation_dim)]
    names += ["grad_norm", "hessian_curvature"]
    return names


@dataclass
class ShapTrace:
    """SHAP explanation for a single sample."""
    sample_index: int
    shap_values: np.ndarray            # (D,)  per-feature contributions
    base_value: float                  # expected model output
    feature_names: List[str]
    top_features: List[Tuple[str, float]] = field(default_factory=list)

    def __post_init__(self):
        # Build top-5 most impactful features (by |shap|)
        order = np.argsort(np.abs(self.shap_values))[::-1][:5]
        self.top_features = [
            (self.feature_names[i], float(self.shap_values[i]))
            for i in order
        ]

    def summary(self) -> str:
        lines = [f"[Sample {self.sample_index}] SHAP Trace — top drivers:"]
        for name, val in self.top_features:
            direction = "↑" if val > 0 else "↓"
            lines.append(f"  {direction} {name:30s}  {val:+.4f}")
        return "\n".join(lines)


class ShapExplainer:
    """
    SHAP-based explainability wrapper for the ensemble anomaly scorer.

    Args:
        score_fn        : Callable (N, D) → (N,) returning anomaly scores.
                          Typically EnsembleClassifier.predict_scores.
        background_data : (K, D) representative background samples for
                          KernelExplainer (K ≈ 50–200 recommended).
        activation_dim  : Number of activation features in the fingerprint.
        nsamples        : Number of SHAP samples (higher = more accurate,
                          but slower). Default 512.
    """

    def __init__(
        self,
        score_fn,
        background_data: np.ndarray,
        activation_dim: int,
        nsamples: int = 512,
    ):
        self.feature_names = _build_feature_names(activation_dim)
        self.nsamples = nsamples

        # KernelExplainer is model-agnostic: works with any scoring function
        self.explainer = shap.KernelExplainer(score_fn, background_data)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        fingerprints: np.ndarray,
        sample_indices: Optional[List[int]] = None,
    ) -> List[ShapTrace]:
        """
        Compute SHAP values for the provided fingerprints.

        Args:
            fingerprints   : (N, D) array.
            sample_indices : Original dataset indices (optional).

        Returns:
            List of ShapTrace, one per sample.
        """
        N = fingerprints.shape[0]
        if sample_indices is None:
            sample_indices = list(range(N))

        shap_values = self.explainer.shap_values(
            fingerprints, nsamples=self.nsamples, silent=True
        )  # (N, D)

        base_value = float(self.explainer.expected_value)

        traces = []
        for i in range(N):
            traces.append(
                ShapTrace(
                    sample_index=sample_indices[i],
                    shap_values=shap_values[i],
                    base_value=base_value,
                    feature_names=self.feature_names,
                )
            )
        return traces

    def plot_waterfall(self, trace: ShapTrace, show: bool = True):
        """Render a SHAP waterfall plot for a single sample."""
        explanation = shap.Explanation(
            values=trace.shap_values,
            base_values=trace.base_value,
            feature_names=trace.feature_names,
        )
        shap.plots.waterfall(explanation, show=show)

    def plot_beeswarm(
        self,
        fingerprints: np.ndarray,
        sample_indices: Optional[List[int]] = None,
        show: bool = True,
    ):
        """Render a SHAP beeswarm plot across multiple samples."""
        traces = self.explain(fingerprints, sample_indices)
        all_shap = np.stack([t.shap_values for t in traces])
        explanation = shap.Explanation(
            values=all_shap,
            data=fingerprints,
            feature_names=self.feature_names,
        )
        shap.plots.beeswarm(explanation, show=show)
