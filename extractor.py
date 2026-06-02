"""
Stage 1 — Forensic Feature Extractor
======================================
Extracts three complementary signals per sample per epoch:
  1. Activation vectors  — penultimate-layer representations
  2. Gradient norms      — per-sample loss gradient magnitude
  3. Hessian curvature   — loss landscape curvature estimate (Hutchinson diagonal)

These are concatenated into a unified forensic fingerprint fed into Stage 2.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional


class ForensicFeatureExtractor:
    """
    Hooks into a PyTorch model to extract per-sample forensic signals
    during a training epoch.

    Args:
        model      : The target PyTorch model being monitored.
        device     : Torch device ('cpu' or 'cuda').
        layer_name : Name of the penultimate layer to hook activations from.
                     If None, the last-but-one Linear layer is used automatically.

    Example::
        extractor = ForensicFeatureExtractor(model, device=torch.device("cuda"))
        fingerprints = extractor.extract(inputs, targets, criterion)
        # fingerprints.shape == (batch_size, activation_dim + 2)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cpu"),
        layer_name: Optional[str] = None,
    ):
        self.model = model
        self.device = device
        self.layer_name = layer_name
        self._activation_cache: Dict[str, torch.Tensor] = {}
        self._hooks: List = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
    ) -> np.ndarray:
        """
        Run a forward + backward pass and return the forensic fingerprint
        for every sample in the batch.

        Args:
            inputs   : Input tensor of shape (B, *).
            targets  : Ground-truth label tensor of shape (B,).
            criterion: Loss function (must support reduction='none').

        Returns:
            fingerprints : np.ndarray of shape (B, activation_dim + 2)
        """
        self._register_hooks()
        inputs = inputs.to(self.device).requires_grad_(True)
        targets = targets.to(self.device)

        # Forward pass — hooks fire here
        outputs = self.model(inputs)
        loss_per_sample = self._per_sample_loss(outputs, targets)

        # --- Signal 2: gradient norms ---
        grad_norms = self._compute_gradient_norms(loss_per_sample, inputs)

        # --- Signal 3: Hessian diagonal estimate ---
        hessian_diag = self._estimate_hessian_diagonal(loss_per_sample, inputs)

        # --- Signal 1: activations captured by hook ---
        activations = self._activation_cache.get("penultimate")
        if activations is None:
            raise RuntimeError(
                "Activation hook did not fire. "
                "Verify that `layer_name` matches an existing layer in your model."
            )

        self._remove_hooks()

        # Concatenate all signals → unified forensic fingerprint
        fingerprints = np.concatenate(
            [
                activations.detach().cpu().numpy(),   # (B, D_act)
                grad_norms.detach().cpu().numpy(),     # (B, 1)
                hessian_diag.detach().cpu().numpy(),   # (B, 1)
            ],
            axis=1,
        )
        return fingerprints  # (B, D_act + 2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _per_sample_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Return un-reduced cross-entropy loss per sample: shape (B,)."""
        return nn.CrossEntropyLoss(reduction="none")(outputs, targets)

    def _compute_gradient_norms(
        self,
        loss_per_sample: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """L2 norm of ∂loss_i/∂x_i for each sample i: shape (B, 1)."""
        norms = []
        for i in range(loss_per_sample.shape[0]):
            grad = torch.autograd.grad(
                loss_per_sample[i],
                inputs,
                retain_graph=True,
                create_graph=False,
            )[0][i]
            norms.append(grad.norm(p=2).unsqueeze(0))
        return torch.cat(norms).unsqueeze(1)  # (B, 1)

    def _estimate_hessian_diagonal(
        self,
        loss_per_sample: torch.Tensor,
        inputs: torch.Tensor,
        n_probes: int = 5,
    ) -> torch.Tensor:
        """
        Hutchinson's trace estimator for the diagonal of the input Hessian.
            diag(H) ≈ (1/K) Σ_k  v_k ⊙ (H v_k),   v_k ~ Rademacher{±1}

        Returns a scalar curvature estimate per sample: shape (B, 1).
        """
        batch_size = loss_per_sample.shape[0]
        curvature = torch.zeros(batch_size, device=self.device)

        for _ in range(n_probes):
            v = (torch.randint(0, 2, inputs.shape, device=self.device).float() * 2 - 1)
            for i in range(batch_size):
                grad = torch.autograd.grad(
                    loss_per_sample[i], inputs,
                    retain_graph=True, create_graph=True,
                )[0][i]
                hv = torch.autograd.grad(
                    (grad * v[i]).sum(), inputs,
                    retain_graph=True, create_graph=False,
                )[0][i]
                curvature[i] += (v[i] * hv).sum()

        curvature /= n_probes
        return curvature.unsqueeze(1)  # (B, 1)

    def _register_hooks(self):
        """Attach a forward hook to the resolved penultimate layer."""
        target_layer = self._resolve_layer()

        def hook_fn(module, input, output):
            act = output
            if act.dim() > 2:
                act = act.flatten(start_dim=1)   # flatten CNN spatial dims
            self._activation_cache["penultimate"] = act

        self._hooks.append(target_layer.register_forward_hook(hook_fn))

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._activation_cache.clear()

    def _resolve_layer(self) -> nn.Module:
        """Return the target layer by name, or auto-detect the penultimate Linear layer."""
        if self.layer_name:
            for name, module in self.model.named_modules():
                if name == self.layer_name:
                    return module
            raise ValueError(f"Layer '{self.layer_name}' not found in model.")

        # Auto-detect: second-to-last Linear layer
        linear_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if isinstance(mod, nn.Linear)
        ]
        if len(linear_layers) < 2:
            raise RuntimeError(
                "Model has fewer than 2 Linear layers. "
                "Specify `layer_name` manually."
            )
        return linear_layers[-2][1]
