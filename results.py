"""
Detection result returned by UnifiedShield.scan().
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np
from .ensemble.classifier import SamplePrediction
from .explainability.shap_explainer import ShapTrace


@dataclass
class DetectionResult:
    """
    Aggregated output of a UnifiedShield scan.

    Attributes:
        predictions   : Per-sample SamplePrediction objects.
        shap_traces   : Per-sample SHAP explanations (if explain=True).
        n_total       : Total samples scanned.
        n_poisoned    : Flagged poisoned samples.
        attack_counts : Breakdown by attack type.
    """

    predictions: List[SamplePrediction]
    shap_traces: Optional[List[ShapTrace]] = None

    @property
    def n_total(self) -> int:
        return len(self.predictions)

    @property
    def n_poisoned(self) -> int:
        return sum(1 for p in self.predictions if p.is_poisoned)

    @property
    def attack_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for p in self.predictions:
            counts[p.attack_type] = counts.get(p.attack_type, 0) + 1
        return counts

    def poisoned_indices(self) -> List[int]:
        """Return dataset indices of samples flagged as poisoned."""
        return [p.sample_index for p in self.predictions if p.is_poisoned]

    def summary(self) -> str:
        lines = [
            "=" * 52,
            "  UnifiedShield — Detection Summary",
            "=" * 52,
            f"  Total samples scanned : {self.n_total}",
            f"  Poisoned (flagged)    : {self.n_poisoned}  "
            f"({100 * self.n_poisoned / max(self.n_total, 1):.1f}%)",
            "",
            "  Attack type breakdown:",
        ]
        for attack, count in sorted(self.attack_counts.items()):
            lines.append(f"    {attack:<20s} {count}")
        lines.append("=" * 52)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DetectionResult(n_total={self.n_total}, "
            f"n_poisoned={self.n_poisoned}, "
            f"attacks={self.attack_counts})"
        )
