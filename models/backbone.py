"""
ResNet18 backbone adapted for CIFAR-10 (32×32 images).

Design decisions:
- conv1: kernel 3×3, stride 1, no padding reduction   → prevents 32×32 images shrinking too fast
- maxpool → Identity()                                 → same reason
- Final FC removed; output is (B, 512) feature vector

Reference: capstone1/ai_memory_test/src/models.py HopeCNN (same CIFAR adaptation).
Freeze mode matches nested_learning/src/nested_learning/model.py freeze_backbone():
backbone attention blocks are frozen while memory modules stay trainable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    """
    ResNet18 feature extractor.
    Output: (B, 512) after global average pooling + flatten.

    Freezing modes (mutually exclusive; freeze takes priority):
      freeze=True              → freeze ALL backbone params
      freeze_up_to=N  (N > 0) → freeze children[0 .. N-1] of self.net

    backbone.net children (CIFAR-10 variant):
      [0] Conv2d          — stem conv (1,728 params)
      [1] BatchNorm2d     — stem BN   (128 params)
      [2] ReLU            — (0 params)
      [3] Identity        — former maxpool (0 params)
      [4] Sequential      — layer1 / ResBlock×2 (147,968 params)
      [5] Sequential      — layer2 / ResBlock×2 (525,568 params)
      [6] Sequential      — layer3 / ResBlock×2 (2,099,712 params)
      [7] Sequential      — layer4 / ResBlock×2 (8,393,728 params)
      [8] AdaptiveAvgPool2d (0 params)
    """

    def __init__(
        self,
        freeze: bool = False,
        freeze_up_to: int = 0,
        pretrained: bool = False,
    ) -> None:
        super().__init__()

        weights = "IMAGENET1K_V1" if pretrained else None
        base = models.resnet18(weights=weights)

        # ── CIFAR-10 adaptation (32×32 input) ────────────────────────────
        # Original ResNet18 expects 224×224; this adaptation avoids spatial
        # collapse at early layers by using smaller kernel / no maxpool.
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()  # type: ignore[assignment]

        # Remove the final FC layer → use as feature extractor.
        # Output shape after avgpool: (B, 512, 1, 1)
        self.net = nn.Sequential(*list(base.children())[:-1])
        self.feature_dim = 512

        if freeze:
            # Full freeze: all backbone params non-trainable
            for p in self.net.parameters():
                p.requires_grad_(False)
        elif freeze_up_to > 0:
            # Partial freeze: only the first freeze_up_to children
            children = list(self.net.children())
            n_freeze = min(freeze_up_to, len(children))
            for child in children[:n_freeze]:
                for p in child.parameters():
                    p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, 512) feature vectors."""
        feat = self.net(x)       # (B, 512, 1, 1)
        return feat.flatten(1)  # (B, 512)
