"""
ContinualModel: assembles backbone + FastMemory + classifier.

Two-pass interface (mirrors nested_learning memorize_tokens() pattern):
  Pass-1 (meta forward):   features = backbone(x)
                           fast_out  = fast_memory(features)          # no update
                           logits    = classifier(fast_out)
                           meta_loss = CE(logits, labels)

  Teach-signal:            teach = compute_teach_signal(features, logits, labels, classifier)
                           # Adapted from nested_learning/training.py compute_teach_signal()

  Pass-2 (fast update):    fast_memory(features.detach(), teach_signal=teach)
                           # Internally updates fast_memory.net params, nothing else

  Meta backward:           meta_loss.backward()
                           optimizer.step()    # backbone + classifier only

compute_teach_signal() is adapted from:
  nested_learning/src/nested_learning/training.py  compute_teach_signal()
  Original: gradient of next-token CE loss w.r.t. pre-norm hidden states (B, T, dim)
  Here:     gradient of CE loss w.r.t. feature embeddings (B, dim)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ResNetBackbone
from .fast_memory import FastMemory


# ──────────────────────────────────────────────────────────────────────
# Teach signal computation (adapted from nested_learning/training.py)
# ──────────────────────────────────────────────────────────────────────

def compute_teach_signal(
    features: torch.Tensor,    # (B, dim) — backbone output (shape reference only)
    logits: torch.Tensor,      # (B, num_classes) — classifier output from Pass-1
    labels: torch.Tensor,      # (B,) — ground-truth class indices
    classifier: nn.Module,
) -> torch.Tensor:
    """
    Closed-form CE gradient w.r.t. features. Safe under torch.no_grad().

    Faithfully mirrors nested_learning/src/nested_learning/training.py
    compute_teach_signal() which is PURELY ANALYTICAL — no torch.autograd.grad.

    Derivation (for linear classifier logits = features @ W^T + b):
      grad_logits  = (softmax(logits) - one_hot(labels)) / B   — (B, C)
      grad_features = grad_logits @ W                          — (B, D)
      teach        = -grad_features   (improvement = -gradient)

    Because there is no autograd call, this works correctly inside
    torch.no_grad() (as used in engine.py Pass-2 block).
    """
    with torch.no_grad():
        B = features.size(0)
        p = torch.softmax(logits.detach(), dim=-1)       # (B, C)
        p[torch.arange(B, device=p.device), labels] -= 1.0  # subtract one-hot
        p = p / B                                          # mean reduction
        W = classifier.weight.detach()                    # (C, D)
        teach = -(p @ W)                                  # (B, D) — negated
    return teach


# ──────────────────────────────────────────────────────────────────────
# Full model
# ──────────────────────────────────────────────────────────────────────

class ContinualModel(nn.Module):
    """
    ResNet18 backbone + FastMemory + shared linear classifier.

    Parameter groups
    ─────────────────
    • backbone.net  →  frozen or low-lr meta params
    • fast_memory.net, fast_memory.norm  →  fast params (NO main optimizer)
    • classifier    →  meta params with standard lr

    The model exposes a two-pass training interface via forward() + update_fast().
    """

    def __init__(
        self,
        num_classes: int = 10,
        feature_dim: int = 512,
        freeze_backbone: bool = False,
        freeze_backbone_until: int = 0,
        pretrained_backbone: bool = False,
        fast_hidden_multiplier: int = 4,
        fast_lr: float = 1e-2,
        fast_grad_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # ── Backbone ─────────────────────────────────────────────────────
        self.backbone = ResNetBackbone(
            freeze=freeze_backbone,
            freeze_up_to=freeze_backbone_until,
            pretrained=pretrained_backbone,
        )

        # ── Fast Memory ───────────────────────────────────────────────────
        # Sits between backbone and classifier.
        # Updated only by teach_signal — excluded from meta optimizer.
        self.fast_memory = FastMemory(
            dim=feature_dim,
            hidden_multiplier=fast_hidden_multiplier,
            fast_lr=fast_lr,
            grad_clip=fast_grad_clip,
        )

        # ── Shared classifier (class-incremental: all classes share one head) ──
        self.classifier = nn.Linear(feature_dim, num_classes)

    # ------------------------------------------------------------------
    # Forward pass (Pass-1: meta forward, no fast weight update)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        teach_signal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits, features).

        If teach_signal is provided (Pass-2), fast_memory updates its weights
        internally but the returned logits/features are from the clean forward.

        Returning features here allows the caller to:
          a) compute teach_signal after Pass-1
          b) pass features to update_fast() without a third backbone forward
        """
        features = self.backbone(x)                            # (B, 512)
        fast_out = self.fast_memory(x=features,                # (B, 512)
                                    teach_signal=teach_signal)
        logits = self.classifier(fast_out)                     # (B, num_classes)
        return logits, features

    # ------------------------------------------------------------------
    # Pass-2: standalone fast memory update
    # Called AFTER meta loss.backward() so the main graph is already freed.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_fast(
        self,
        features: torch.Tensor,       # (B, 512) — detached backbone output
        teach_signal: torch.Tensor,   # (B, 512) — from compute_teach_signal()
    ) -> None:
        """
        Trigger fast memory weight update without a full model forward.

        This is more efficient than a second full forward() call because we
        reuse the backbone features computed in Pass-1.

        Mirrors nested_learning/memorize.py memorize_tokens():
          model(token_batch, teach_signal=teach_signal, fast_state=fast_state)
        The teach_signal triggers the internal CMS/Titan update inside forward().
        """
        # We call fast_memory.forward() directly; backbone and classifier are skipped.
        # The teach_signal triggers _update_fast_weights() inside FastMemory.forward().
        self.fast_memory(x=features.detach(), teach_signal=teach_signal)

    # ------------------------------------------------------------------
    # Optimizer helpers
    # ------------------------------------------------------------------

    def meta_parameters(self) -> list[nn.Parameter]:
        """
        Parameters updated by the main optimizer.
        Excludes fast_memory.net params (those are updated via teach_signal).

        Matches nested_learning's distinction:
          meta params = backbone + classifier  (slow, outer-loop updates)
          fast params = CMS/Titan weights      (fast, inner-loop updates)
        """
        fast_ids = {id(p) for p in self.fast_memory.fast_params()}
        return [p for p in self.parameters() if id(p) not in fast_ids]

    def meta_param_groups(
        self,
        backbone_lr: float = 1e-4,
        classifier_lr: float = 1e-3,
    ) -> list[dict]:
        """
        Return parameter groups for the meta optimizer with separate LRs.
        Backbone gets a lower lr to avoid overwriting general representations.
        """
        backbone_params, classifier_params = [], []
        fast_ids = {id(p) for p in self.fast_memory.fast_params()}

        for name, p in self.named_parameters():
            if id(p) in fast_ids:
                continue  # Exclude fast memory entirely
            if "backbone" in name:
                backbone_params.append(p)
            else:
                classifier_params.append(p)

        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        if classifier_params:
            groups.append({"params": classifier_params, "lr": classifier_lr})
        return groups

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def snapshot_meta_state(self) -> dict[str, torch.Tensor]:
        """
        Snapshot meta parameters only.
        Mirrors nested_learning/memorize.py snapshot_state_dict() but excludes
        fast memory weights (those are task-specific fast state).
        """
        fast_ids = {id(p) for p in self.fast_memory.fast_params()}
        return {
            n: p.detach().cpu().clone()
            for n, p in self.named_parameters()
            if id(p) not in fast_ids
        }
