"""
SIGReg: Covariance isotropy regularization (leJEPA-inspired).

Tasarım:
  - Projeksiyon kafası: MLP(512 → 512 → 256, GELU) + L2 normalize
  - SIGReg loss:        C = ZᵀZ / B,  loss = ||C − I||²_F
  - Backbone çıktısına uygulanır (classifier ÖNCE, feature space'de)
  - Current + replay örneklerin TÜMÜNE uygulanır

Motivasyon:
  Feature collapse (tüm örneklerin düşük boyutlu alt uzaya yığılması) sürekli
  öğrenmede kritik bir sorun olur; her task backbone'u kendi sınıfları için
  optimize ederken diğer sınıfların feature'larını ezebilir.
  SIGReg, kovaryans matrisinin birim matrise yakın kalmasını zorlayarak:
    • Köşegen ≈ 1  → her boyutun bilgi taşımasını sağlar (collapse önler)
    • Köşegen dışı ≈ 0 → boyutlar arası korelasyonu azaltır (redundancy önler)

leJEPA referansı: feature space isotropy için kovaryans ceza terimi.
VICReg (Bardes et al. 2022) "variance" terimiyle aynı prensip, farklı formülasyon.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SIGRegProjector(nn.Module):
    """
    İki katmanlı MLP projeksiyon kafası, L2-normalize çıktı.

    Mimari: input_dim → hidden_dim (GELU) → output_dim → L2 normalize

    Varsayılanlar (spec'e uygun): 512 → 512 → 256
    Çıktı her satır için L2-normalized: ||z_i||_2 = 1

    NOT: Bu modül meta-optimizer'a dahil edilir (backbone + classifier ile).
    Fast memory optimizer'a dahil EDİLMEZ.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """(B, input_dim) → (B, output_dim), L2 normalize edilmiş."""
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


def sigreg_loss(features: Tensor, projector: SIGRegProjector) -> Tensor:
    """
    Kovaryans izotropi kayıp fonksiyonu.

        C = ZᵀZ / B
        loss = ||C − I||²_F

    Z: projector(features) — (B, output_dim), L2-normalized satırlar.

    Z satırları L2-normalize olduğu için C köşegenindeki ideal değer
    tam olarak 1 değil — ancak B büyüdükçe 1'e yaklaşır.
    Frobenius normu bu sapmaları hem köşegen hem köşegen dışı için penalize eder.

    Args:
        features: (B, feature_dim) — backbone çıktısı, classifier öncesi
        projector: SIGRegProjector instance

    Returns:
        Skaler loss tensörü (gradient akışı projector + backbone'a gider)
    """
    z = projector(features)          # (B, output_dim), L2-normalize
    B = z.size(0)

    # Kovaryans matrisi: (output_dim, output_dim)
    C = z.T @ z / B

    # Birim matris hedef
    I = torch.eye(C.size(0), device=z.device, dtype=z.dtype)

    # Frobenius norm kare: ||C − I||²_F  =  sum((C − I)²)
    diff = C - I
    return (diff * diff).sum()
