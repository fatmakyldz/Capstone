"""
Muon-style hybrid meta optimizer for capstone3.

Adapted from nested_learning/src/nested_learning/optim/m3.py.

Two-path update rule (mirrors nested_learning ablation §4 "Muon hybrid"):
  ≥2D params (Linear.weight, Conv2d.weight) → Newton-Schulz orthogonalized gradient
  <2D params (bias, norm.weight/bias)       → standard AdamW

Newton-Schulz orthogonalization makes successive gradient updates more
orthogonal to each other, reducing the interference between tasks that
causes catastrophic forgetting.  This is the dominant effect observed in
nested_learning ablations: AdamW → continual CE ≈50, Muon → CE ≈11 (4.5×).

Reference: nested_learning/src/nested_learning/optim/m3.py _newton_schulz()
           Identical algorithm, adapted for image classification param groups.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Newton-Schulz orthogonalization (direct port of m3.py)
# ──────────────────────────────────────────────────────────────────────────────

def _newton_schulz(M: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
    """
    Newton-Schulz iteration to orthogonalize a 2D matrix.

    Algorithm (same as nested_learning/optim/m3.py _newton_schulz()):
        X  = M / (‖M‖ + eps)
        for _ in range(steps):
            X = 0.5 * X @ (3I − XᵀX)
        return X

    Converges so that XᵀX → I (orthonormal columns when m ≥ n).
    For wide matrices (m < n), rows become approximately orthonormal.
    """
    assert M.ndim == 2, f"Newton-Schulz expects 2D matrix, got {M.ndim}D"
    m, n = M.shape
    X = M / (torch.linalg.norm(M) + eps)
    eye = torch.eye(n, device=M.device, dtype=M.dtype)
    for _ in range(steps):
        X = 0.5 * X @ (3.0 * eye - X.T @ X)
    return X


def _orthogonalize(tensor: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
    """
    Orthogonalize a weight gradient of any dimension ≥ 2.

    Conv kernels are 4D (out, in, kH, kW) → flattened to (out, in·kH·kW),
    orthogonalized, then reshaped back.  Mirrors _orthogonalize() in m3.py.
    """
    if tensor.ndim < 2:
        return tensor
    orig_shape = tensor.shape
    mat = tensor.reshape(orig_shape[0], -1)          # (out, rest)
    ortho = _newton_schulz(mat, steps=steps, eps=eps)
    return ortho.reshape(orig_shape)


# ──────────────────────────────────────────────────────────────────────────────
# MuonMeta: hybrid optimizer
# ──────────────────────────────────────────────────────────────────────────────

class MuonMeta(torch.optim.Optimizer):
    """
    Hybrid Muon meta optimizer.

    Per-parameter routing:
      p.ndim >= 2  →  Muon path: orthogonalized gradient + Adam second moment
      p.ndim <  2  →  AdamW path: standard update with decoupled weight decay

    Works with param_groups that carry per-group 'lr' (e.g., different lr for
    backbone vs classifier), so it can replace Adam/AdamW in build_optimizer()
    without changing the rest of the pipeline.

    Args:
        params:       param groups (list of dicts) or iterable of tensors
        lr:           default learning rate (overridden per group if set)
        betas:        (β₁, β₂) momentum coefficients for Adam moments
        eps:          numerical stability for Adam denominator
        weight_decay: decoupled weight decay (AdamW style, applied to all params)
        ns_steps:     Newton-Schulz iterations (5 is sufficient; more → slower)
        ns_eps:       epsilon for Newton-Schulz normalisation
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        ns_steps: int = 5,
        ns_eps: float = 1e-6,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
        )
        super().__init__(params, defaults)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr          = group["lr"]
            beta1, beta2 = group["betas"]
            eps         = group["eps"]
            wd          = group["weight_decay"]
            ns_steps    = group["ns_steps"]
            ns_eps      = group["ns_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Lazy initialisation of state
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"]    = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                exp_avg    = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                step       = state["step"]

                # Bias correction terms
                bc1 = 1.0 - beta1 ** step
                bc2 = 1.0 - beta2 ** step

                if p.ndim >= 2:
                    # ── Muon path ─────────────────────────────────────────
                    # Faithfully mirrors m3.py:
                    #   m1.add_(grad, alpha=beta1)      ← raw grad → EMA first
                    #   o1 = _orthogonalize(m1, ...)    ← then orthogonalize EMA
                    # NS operates on the SMOOTHED momentum buffer, not the noisy
                    # raw gradient.  This gives a stable update direction over time.

                    # 1. Accumulate raw gradient into EMA buffers
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    # 2. Bias-corrected momentum estimate
                    m_hat = exp_avg    / bc1
                    v_hat = exp_avg_sq / bc2

                    # 3. Orthogonalize the bias-corrected momentum (not raw grad)
                    ortho_m = _orthogonalize(m_hat, steps=ns_steps, eps=ns_eps)

                    update = ortho_m / (v_hat.sqrt().add_(eps))

                    # Decoupled weight decay (AdamW style)
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)

                    p.add_(update, alpha=-lr)

                else:
                    # ── AdamW path ────────────────────────────────────────
                    # Standard Adam update for 1D tensors (bias, norm params).
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    m_hat = exp_avg    / bc1
                    v_hat = exp_avg_sq / bc2

                    update = m_hat / (v_hat.sqrt().add_(eps))

                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)

                    p.add_(update, alpha=-lr)

        return loss


# ──────────────────────────────────────────────────────────────────────────────
# Public helper — expose orthogonalization for tests
# ──────────────────────────────────────────────────────────────────────────────

def orthogonalize_matrix(M: torch.Tensor, steps: int = 5, eps: float = 1e-6) -> torch.Tensor:
    """
    Public wrapper around Newton-Schulz orthogonalization.
    Accepts 2D matrix; returns orthogonalized matrix of same shape.
    Used in tests to verify correctness independently of the optimizer.
    """
    if M.ndim != 2:
        raise ValueError(f"orthogonalize_matrix expects 2D input, got {M.ndim}D")
    return _newton_schulz(M, steps=steps, eps=eps)
