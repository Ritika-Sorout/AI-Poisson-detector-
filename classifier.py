"""
Stage 3 — Ensemble Classifier
================================
Three anomaly detectors vote on each sample using the anomaly scores
produced by the Stage 2 Transformer:

  1. Isolation Forest    — tree-based outlier isolation
  2. Local Outlier Factor (LOF) — density-based scoring
  3. One-Class SVM       — hyperplane boundary detection

Voting produces a calibrated confidence score and maps to attack type.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# Attack type labels produced by the ensemble
ATTACK_TYPES = {
    0: "clean",
    1: "label_flip",
    2: "backdoor",
    3: "clean_label",
}


@dataclass
class SamplePrediction:
    """Prediction output for a single training sample."""
    sample_index: int
    is_poisoned: bool
    attack_type: str
    confidence: float                  # 0.0 – 1.0
    votes: Dict[str, int] = field(default_factory=dict)  # detector → 0/1


class EnsembleClassifier:
    """
    Fits three anomaly detectors on clean (or mixed) fingerprint data,
    then scores new samples by majority vote.

    Args:
        contamination   : Expected fraction of poisoned samples (0.0–0.5).
                          Passed to detectors that accept it.
        iforest_trees   : Number of trees in Isolation Forest.
        lof_neighbors   : Number of neighbours for LOF.
        svm_nu          : Nu parameter for One-Class SVM (≈ contamination).
        random_state    : Seed for reproducibility.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        iforest_trees: int = 200,
        lof_neighbors: int = 20,
        svm_nu: float = 0.1,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.scaler = StandardScaler()

        self.isolation_forest = IsolationForest(
            n_estimators=iforest_trees,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.lof = LocalOutlierFactor(
            n_neighbors=lof_neighbors,
            contamination=contamination,
            novelty=True,              # novelty=True → predict() on new data
            n_jobs=-1,
        )
        self.ocsvm = OneClassSVM(
            nu=svm_nu,
            kernel="rbf",
            gamma="scale",
        )

        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, fingerprints: np.ndarray) -> "EnsembleClassifier":
        """
        Fit all three detectors on the provided fingerprints.

        Args:
            fingerprints : (N, D) array of forensic fingerprints.
        """
        X = self.scaler.fit_transform(fingerprints)
        self.isolation_forest.fit(X)
        self.lof.fit(X)
        self.ocsvm.fit(X)
        self._fitted = True
        return self

    def predict(
        self,
        fingerprints: np.ndarray,
        sample_indices: Optional[List[int]] = None,
    ) -> List[SamplePrediction]:
        """
        Predict poisoning status for each sample.

        Args:
            fingerprints   : (N, D) array of forensic fingerprints.
            sample_indices : Original dataset indices (optional, for tracking).

        Returns:
            List of SamplePrediction, one per sample.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        N = fingerprints.shape[0]
        if sample_indices is None:
            sample_indices = list(range(N))

        X = self.scaler.transform(fingerprints)

        # Each detector returns +1 (normal) or -1 (anomaly)
        iforest_preds = self.isolation_forest.predict(X)   # (N,)
        lof_preds     = self.lof.predict(X)                # (N,)
        ocsvm_preds   = self.ocsvm.predict(X)              # (N,)

        # Convert to binary: 1 = anomalous, 0 = clean
        def to_bin(preds):
            return (preds == -1).astype(int)

        iforest_bin = to_bin(iforest_preds)
        lof_bin     = to_bin(lof_preds)
        ocsvm_bin   = to_bin(ocsvm_preds)

        # Anomaly scores (for confidence calibration)
        iforest_scores = -self.isolation_forest.score_samples(X)   # higher = more anomalous
        lof_scores     = -self.lof.score_samples(X)
        ocsvm_scores   = -self.ocsvm.score_samples(X)

        results = []
        for i in range(N):
            votes = {
                "isolation_forest": int(iforest_bin[i]),
                "lof":              int(lof_bin[i]),
                "ocsvm":            int(ocsvm_bin[i]),
            }
            vote_sum = sum(votes.values())
            is_poisoned = vote_sum >= 2  # majority vote

            # Calibrated confidence: normalised mean of anomaly scores
            raw_scores = np.array([iforest_scores[i], lof_scores[i], ocsvm_scores[i]])
            confidence = float(np.clip(np.mean(raw_scores), 0.0, 1.0))

            # Attack type classification (heuristic — extend with supervised head)
            attack_type = self._classify_attack_type(
                fingerprints[i], is_poisoned, confidence
            )

            results.append(
                SamplePrediction(
                    sample_index=sample_indices[i],
                    is_poisoned=is_poisoned,
                    attack_type=attack_type,
                    confidence=confidence,
                    votes=votes,
                )
            )

        return results

    def predict_scores(self, fingerprints: np.ndarray) -> np.ndarray:
        """Return raw ensemble anomaly scores (N,) without thresholding."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_scores().")
        X = self.scaler.transform(fingerprints)
        scores = (
            -self.isolation_forest.score_samples(X)
            - self.lof.score_samples(X)
            - self.ocsvm.score_samples(X)
        ) / 3.0
        return scores

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_attack_type(
        self,
        fingerprint: np.ndarray,
        is_poisoned: bool,
        confidence: float,
    ) -> str:
        """
        Heuristic attack-type classification based on fingerprint statistics.
        Replace with a supervised classifier once labelled data is available.

        Heuristics:
          - Very high gradient norm  → likely label-flip
          - High activation deviation + low curvature → likely backdoor
          - Low gradient norm + moderate curvature → likely clean-label
        """
        if not is_poisoned:
            return ATTACK_TYPES[0]  # "clean"

        # Fingerprint layout: [...activations..., grad_norm, hessian_curvature]
        grad_norm   = float(fingerprint[-2])
        h_curvature = float(fingerprint[-1])
        act_std     = float(np.std(fingerprint[:-2]))

        if grad_norm > 2.0:
            return ATTACK_TYPES[1]  # label_flip
        elif act_std > 1.5 and h_curvature < 0.5:
            return ATTACK_TYPES[2]  # backdoor
        else:
            return ATTACK_TYPES[3]  # clean_label
