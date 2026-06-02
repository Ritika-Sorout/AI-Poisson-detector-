"""
UnifiedShield — Main Pipeline
==============================
Orchestrates Stage 1 → Stage 2 → Stage 3 and optional SHAP explanation.

Usage::

    from unifiedshield import UnifiedShield

    shield = UnifiedShield(device="cuda", contamination=0.1)
    result = shield.scan(model, train_loader, criterion, epochs=10)
    print(result.summary())

    # Inspect per-sample SHAP trace
    for trace in result.shap_traces:
        print(trace.summary())
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from .features.extractor import ForensicFeatureExtractor
from .ensemble.classifier import EnsembleClassifier
from .explainability.shap_explainer import ShapExplainer
from .results import DetectionResult


class UnifiedShield:
    """
    End-to-end data poisoning detector.

    Args:
        device          : 'cpu' or 'cuda' (default: auto-detect).
        layer_name      : Penultimate layer name for activation hooking.
                          None = auto-detect last-but-one Linear layer.
        contamination   : Expected poison fraction for ensemble detectors.
        d_model         : Transformer embedding dimension.
        nhead           : Number of attention heads.
        num_tf_layers   : Number of transformer encoder layers.
        iforest_trees   : Isolation Forest estimators.
        lof_neighbors   : LOF neighbours.
        svm_nu          : One-Class SVM nu.
        shap_nsamples   : SHAP kernel samples (higher = slower but more accurate).
        random_state    : Global random seed.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        layer_name: Optional[str] = None,
        contamination: float = 0.1,
        d_model: int = 128,
        nhead: int = 4,
        num_tf_layers: int = 2,
        iforest_trees: int = 200,
        lof_neighbors: int = 20,
        svm_nu: float = 0.1,
        shap_nsamples: int = 256,
        random_state: int = 42,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.layer_name = layer_name
        self.contamination = contamination
        self.d_model = d_model
        self.nhead = nhead
        self.num_tf_layers = num_tf_layers
        self.iforest_trees = iforest_trees
        self.lof_neighbors = lof_neighbors
        self.svm_nu = svm_nu
        self.shap_nsamples = shap_nsamples
        self.random_state = random_state

        torch.manual_seed(random_state)
        np.random.seed(random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module = None,
        epochs: int = 5,
        explain: bool = True,
        verbose: bool = True,
    ) -> DetectionResult:
        """
        Run a full UnifiedShield scan over multiple training epochs.

        Args:
            model        : Target PyTorch model (can be mid-training or pretrained).
            train_loader : DataLoader for the (potentially poisoned) training set.
            criterion    : Loss function. Defaults to CrossEntropyLoss if None.
            epochs       : Number of epochs to observe fingerprints over.
            explain      : Whether to run SHAP explainability on flagged samples.
            verbose      : Show tqdm progress bars.

        Returns:
            DetectionResult with predictions and optional SHAP traces.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        model = model.to(self.device)
        model.eval()  # inference-only observation; does not alter model weights

        extractor = ForensicFeatureExtractor(
            model, device=self.device, layer_name=self.layer_name
        )

        # -----------------------------------------------------------
        # Stage 1: Collect forensic fingerprints over T epochs
        # fingerprint_sequences[sample_idx] = list of T fingerprints
        # -----------------------------------------------------------
        if verbose:
            print(f"[UnifiedShield] Stage 1 — Collecting forensic fingerprints "
                  f"({epochs} epochs)...")

        # First pass: determine number of samples & feature dim
        all_epoch_fingerprints = []  # list of (N, D) arrays, one per epoch

        for epoch in range(epochs):
            epoch_fps = []
            epoch_bar = tqdm(
                train_loader,
                desc=f"  Epoch {epoch + 1}/{epochs}",
                disable=not verbose,
                leave=False,
            )
            for inputs, targets in epoch_bar:
                fp = extractor.extract(inputs, targets, criterion)  # (B, D)
                epoch_fps.append(fp)

            all_epoch_fingerprints.append(np.vstack(epoch_fps))  # (N, D)

        # Stack → (N, T, D)
        fingerprint_sequences = np.stack(all_epoch_fingerprints, axis=1)
        N, T, D = fingerprint_sequences.shape

        if verbose:
            print(f"  → Collected fingerprints: {N} samples × {T} epochs × {D} features")

        # -----------------------------------------------------------
        # Stage 2: Transformer anomaly scores
        # (Note: Stage 2 is optional at inference if skipping fine-tuning)
        # For the initial release we use the mean fingerprint across epochs
        # as input to Stage 3; the transformer is available for fine-tuning.
        # -----------------------------------------------------------
        if verbose:
            print("[UnifiedShield] Stage 2 — Computing temporal anomaly scores...")

        # Aggregate fingerprints across epochs: mean + std → richer representation
        fp_mean = fingerprint_sequences.mean(axis=1)   # (N, D)
        fp_std  = fingerprint_sequences.std(axis=1)    # (N, D)
        fp_agg  = np.concatenate([fp_mean, fp_std], axis=1)  # (N, 2D)

        # -----------------------------------------------------------
        # Stage 3: Ensemble classifier
        # -----------------------------------------------------------
        if verbose:
            print("[UnifiedShield] Stage 3 — Fitting ensemble detectors...")

        ensemble = EnsembleClassifier(
            contamination=self.contamination,
            iforest_trees=self.iforest_trees,
            lof_neighbors=self.lof_neighbors,
            svm_nu=self.svm_nu,
            random_state=self.random_state,
        )
        ensemble.fit(fp_agg)
        predictions = ensemble.predict(fp_agg, sample_indices=list(range(N)))

        n_poisoned = sum(1 for p in predictions if p.is_poisoned)
        if verbose:
            print(f"  → {n_poisoned}/{N} samples flagged as poisoned "
                  f"({100 * n_poisoned / N:.1f}%)")

        # -----------------------------------------------------------
        # SHAP Explainability
        # -----------------------------------------------------------
        shap_traces = None
        if explain and n_poisoned > 0:
            if verbose:
                print("[UnifiedShield] Computing SHAP explanations for flagged samples...")

            poisoned_indices = [p.sample_index for p in predictions if p.is_poisoned]
            poisoned_fps = fp_agg[poisoned_indices]

            # Use a random subset of clean samples as SHAP background
            clean_indices = [p.sample_index for p in predictions if not p.is_poisoned]
            bg_size = min(100, len(clean_indices))
            bg_idx = np.random.choice(clean_indices, size=bg_size, replace=False)
            background = fp_agg[bg_idx]

            explainer = ShapExplainer(
                score_fn=ensemble.predict_scores,
                background_data=background,
                activation_dim=D,
                nsamples=self.shap_nsamples,
            )
            shap_traces = explainer.explain(poisoned_fps, sample_indices=poisoned_indices)

            if verbose:
                print(f"  → SHAP traces computed for {len(shap_traces)} samples.")

        return DetectionResult(predictions=predictions, shap_traces=shap_traces)
