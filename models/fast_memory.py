"""
CMS-like (Continuum Memory System) fast memory module.

Design, adapted from nested_learning:
  - nested_learning/src/nested_learning/cms.py  →  CMSBlock architecture
  - nested_learning/src/nested_learning/titan/memory.py  →  update() pattern

Two operating modes
───────────────────
1. Normal forward (teach_signal=None):
     y = x + MLP(LayerNorm(x))          residual pass-through, no weight change

2. Update forward (teach_signal provided):
     y = x + MLP(LayerNorm(x))          same output (from BEFORE the update)
     then: update fast weights locally   gradient step INSIDE forward, not via
                                         the main optimizer (mirrors Titan.update())

Key invariant (tested by test_fast_memory.py::test_meta_params_unchanged_after_update):
  - Calling forward with teach_signal ONLY changes self.net parameters.
  - Backbone and classifier parameters are never touched here.

The teach_signal is the negative gradient of the CE loss w.r.t. the input features,
computed by compute_teach_signal() in continual_model.py.  It tells fast_memory
"in which direction to move its output to reduce the classification error".
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FastMemory(nn.Module):
    """
    Minimal CMS-inspired fast memory module.

    Architecture (matches nested_learning/cms.py CMSBlock):
        LayerNorm → Linear(dim, hidden) → GELU → Linear(hidden, dim)
        residual: y = x + clipped(MLP(norm(x)))

    Fast weight update rule (matches nested_learning/titan/memory.py update()):
        loss_fast = -mean(teach_signal · delta)   # dot-product alignment
        grads     = autograd(loss_fast, net.params)
        net.params += -fast_lr * grads
    """

    def __init__(
        self,
        dim: int,
        hidden_multiplier: int = 4,
        fast_lr: float = 1e-2,
        grad_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.fast_lr = fast_lr
        self.grad_clip = grad_clip
        hidden = dim * hidden_multiplier

        # Shared norm (matches CMSBlock: norm is inside the residual branch)
        self.norm = nn.LayerNorm(dim)

        # Fast parameters — these are updated by teach_signal, NOT by the main optimizer.
        # The main optimizer's param groups must EXCLUDE these.
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

        # Sentinel so the trainer can identify and exclude these from meta optimizer
        self._is_fast_memory = True

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        teach_signal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x            : (B, dim) input features from backbone
        teach_signal : (B, dim) error signal, or None for normal forward

        Returns y : (B, dim) — always the output computed BEFORE any weight update,
        so that the main loss.backward() graph is valid regardless of update order.
        """
        # ── Residual forward (same in both modes) ────────────────────────
        x_norm = self.norm(x)
        delta = self.net(x_norm)  # (B, dim)

        # Gradient clipping on the output (matches nested_learning/cms.py CMSBlock.forward())
        if self.training and self.grad_clip > 0:
            with torch.no_grad():
                norm_val = delta.norm(dim=-1, keepdim=True)
                scale = torch.clamp(norm_val / self.grad_clip, min=1.0)
            delta = delta / scale

        y = x + delta  # residual

        # ── Fast weight update (only when teach_signal is provided) ──────
        # This runs AFTER computing y so that the output tensor is clean.
        # The update is completely separate from the main autograd graph.
        if teach_signal is not None and self.training:
            self._update_fast_weights(x.detach(), teach_signal.detach())

        return y

    # ------------------------------------------------------------------
    # Internal fast weight update — mirrors Titan Memory update()
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_fast_weights(
        self,
        x: torch.Tensor,          # (B, dim) — detached input
        teach_signal: torch.Tensor,  # (B, dim) — detached error signal
    ) -> None:
        """
        Local gradient step on fast memory weights.

        Adapted from nested_learning/src/nested_learning/titan/memory.py update():
            with torch.enable_grad():
                prediction = self.forward(key_detached)
                loss = mean(error_signal * prediction)   # error_signal mode
            grads = autograd(loss, net.parameters())
            param += -lr * grad

        The dot-product loss `mean(teach_signal · delta)` maximises alignment
        between fast memory output and the teach_signal (improvement direction).
        Negated for minimisation: loss = -mean(teach_signal · delta).

        This entire function runs under @torch.no_grad(), with torch.enable_grad()
        used only for the local autograd computation — exactly as in Titan.update().
        Meta parameters (backbone, classifier) are never touched.
        """
        params = list(self.net.parameters())

        with torch.enable_grad():
            # Re-compute delta with gradients enabled for net params only
            x_norm = self.norm(x)              # norm has no params affected here
            delta_pred = self.net(x_norm)      # (B, dim) — gradients w.r.t. params

            # Loss: align delta with teach_signal direction
            # Matches Titan error_signal mode: loss = mean(error_signal * prediction)
            loss_fast = -(teach_signal * delta_pred).mean()

        grads = torch.autograd.grad(loss_fast, params, allow_unused=True)

        # Manual gradient step (no optimizer, matches Titan: param.add_(grad, alpha=-lr))
        for param, grad in zip(params, grads):
            if grad is None:
                continue
            # Per-parameter gradient clipping for stability
            if self.grad_clip > 0:
                g_norm = grad.norm()
                if g_norm > self.grad_clip:
                    grad = grad * (self.grad_clip / (g_norm + 1e-8))
            param.add_(grad, alpha=-self.fast_lr)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def fast_params(self) -> list[nn.Parameter]:
        """Return all fast memory parameters (for exclusion from meta optimizer).

        Includes self.norm because the norm is part of the fast residual branch
        and should NOT be updated by the meta optimizer — only by the teach signal
        path (indirectly via _update_fast_weights inputs).
        Previously only returned self.net.parameters(), which left norm.weight/bias
        in the meta optimizer — a design inconsistency now corrected.
        """
        return list(self.net.parameters()) + list(self.norm.parameters())

    def reset_fast_weights(self) -> None:
        """Reinitialise fast memory to zero (useful at task boundary experiments)."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
