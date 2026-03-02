"""
Loss hesaplama modülü.

Bileşenler:
  1. CE loss (current batch)
  2. CE loss (replay batch) — ayrı ağırlıkla
  3. EWC penalty (opsiyonel) — capstone2/train_continual.py ewc_penalty() ile aynı mantık

EWC referansı: capstone2/ai_memory_test111/train_continual.py
  snapshot_ewc_state() → Fisher matris tahmini
  ewc_penalty()        → L2 regularizasyon
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def compute_loss(
    logits: Tensor,                  # (B_total, num_classes)
    labels: Tensor,                  # (B_total,)
    n_current: int,                  # current batch büyüklüğü
    current_weight: float = 1.0,
    replay_weight: float = 1.0,
) -> Tensor:
    """
    Ağırlıklı current + replay CE loss.

    Eğer sadece current varsa (n_current == len(labels)) replay terimi sıfır.
    Current ve replay ayrı CE hesaplanır çünkü ağırlıkları farklı olabilir.

    Capstone2 referansı: train_continual.py satır 270-273:
      cur_loss = criterion(outputs[:cur_count], cur_labels)
      rep_loss = criterion(outputs[cur_count:], rep_labels)
      loss = current_weight * cur_loss + replay_weight * rep_loss
    """
    cur_logits = logits[:n_current]
    cur_labels = labels[:n_current]
    cur_loss = F.cross_entropy(cur_logits, cur_labels)

    total = current_weight * cur_loss

    if n_current < logits.size(0):
        rep_logits = logits[n_current:]
        rep_labels = labels[n_current:]
        rep_loss = F.cross_entropy(rep_logits, rep_labels)
        total = total + replay_weight * rep_loss

    return total


# ── EWC yardımcı fonksiyonlar ─────────────────────────────────────────────────

def compute_ewc_fisher(
    model: nn.Module,
    dataloader,
    device: torch.device,
    n_samples: int = 200,
) -> tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """
    Fisher matrisini ve parametre ortalamalarını hesapla.

    Kaynak: capstone2/train_continual.py snapshot_ewc_state() — birebir uyarlandı.
    Fisher = E[grad^2] → önemli parametreleri büyük Fisher değeriyle penalize eder.

    Sadece meta parametreler (backbone + classifier) için hesaplanır;
    fast_memory parametreleri EWC kapsamı dışında tutulur.
    """
    model.eval()
    params = {n: p for n, p in model.named_parameters()
              if p.requires_grad and "fast_memory" not in n}
    if not params:
        return {}, {}

    fisher: Dict[str, Tensor] = {n: torch.zeros_like(p, device=device)
                                  for n, p in params.items()}
    count = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        model.zero_grad(set_to_none=True)
        logits, _ = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        for n, p in params.items():
            if p.grad is not None:
                fisher[n] += p.grad.detach().pow(2)
        count += 1
        if count * images.size(0) >= n_samples:
            break

    if count > 0:
        fisher = {n: v / count for n, v in fisher.items()}
    means = {n: p.detach().clone() for n, p in params.items()}
    return fisher, means


def ewc_penalty(
    model: nn.Module,
    fisher: Dict[str, Tensor],
    means: Dict[str, Tensor],
) -> Tensor:
    """
    EWC regularizasyon terimi: sum_i F_i * (theta_i - theta_i*)^2

    Capstone2/train_continual.py ewc_penalty() ile özdeş mantık.
    Dönüş: skaler tensor (loss'a eklenir).
    """
    if not fisher:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    device = next(model.parameters()).device
    loss = torch.tensor(0.0, device=device)
    named_params = dict(model.named_parameters())
    for n, f in fisher.items():
        p = named_params.get(n)
        if p is None:
            continue
        loss = loss + (f.to(device) * (p - means[n].to(device)).pow(2)).sum()
    return loss


def mask_old_class_grads(
    classifier: nn.Module,
    current_class_ids: list[int],
    device: torch.device,
) -> None:
    """
    Eski sınıfların classifier gradyanlarını sıfırla.

    Class-incremental setup'ta yeni task'ı öğrenirken SADECE mevcut task sınıflarının
    classifier satırları güncellenmeli; eski sınıf satırlarına dokunmamalıyız.

    Capstone2/train_continual.py mask_old_class_grads() ile aynı mantık.
    """
    w = getattr(classifier, "weight", None)
    b = getattr(classifier, "bias", None)

    if w is None or w.grad is None:
        return

    keep = torch.zeros(w.shape[0], device=device, dtype=torch.bool)
    for c in current_class_ids:
        keep[c] = True

    w.grad[~keep] = 0.0
    if b is not None and b.grad is not None:
        b.grad[~keep] = 0.0
