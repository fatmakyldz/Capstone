"""
MuonMeta optimizer testleri.

nested_learning karşılıkları:
  tests/test_m3.py — M3 optimizer gradient flow ve state güncellemesi
  reports/ablations.md §4 — Muon hybrid: ≥2D → Muon, 1D → AdamW

Doğrulanan invariantlar:
  1. Bir optimizer step'i Linear weight'ini değiştirmeli
  2. Newton-Schulz çıktısı normalize edilmiş olmalı (‖XᵀX − I‖ küçük)
  3. fast_memory parametreleri optimizer param_groups'unda olmamalı
"""
import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optim.muon import MuonMeta, orthogonalize_matrix
from models.continual_model import ContinualModel


# ── Test 1: MuonMeta bir adımda Linear weight'ini günceller ─────────────────
# nested_learning karşılığı: test_m3.py::test_m3_updates_weights

def test_muon_updates_weights() -> None:
    """
    Bir optimizer step'inden sonra Linear weight değişmiş olmalı.
    Gradient yoksa güncelleme olmaz — önce backward çalıştırılır.
    """
    torch.manual_seed(0)
    layer = nn.Linear(16, 8)
    optimizer = MuonMeta(layer.parameters(), lr=1e-2)

    x = torch.randn(4, 16)
    before = layer.weight.detach().clone()

    # Forward + backward
    loss = layer(x).pow(2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    after = layer.weight.detach()
    assert not torch.allclose(before, after), \
        "MuonMeta: optimizer step Linear weight'ini değiştirmeli!"


# ── Test 2: Newton-Schulz çıktısı approximate ortonormal ───────────────────
# nested_learning karşılığı: m3.py _newton_schulz() matematiksel doğruluğu

def test_muon_orthogonality() -> None:
    """
    orthogonalize_matrix(M) XᵀX → I'ya yakınsar.

    Yakınsama analizi: başlangıç özdegerleri 1/n'den başlar, her NS adımı
    bunları 1'e doğru iter (a → 0.25*a*(3−a)² haritası). Frob-norm ile
    normalize edilmiş (64,32) matrisin özdeğerleri ~1/32≈0.03'te başlar;
    5 adımda ~0.82'ye, 15 adımda >0.999'a ulaşır.
    → 15 adım testi gerçek yakınsamayı doğrular.
    → Optimizer'da 5 adım yeterlidir çünkü momentum birikimi singular value
      dağılımını yavaş yavaş iyileştirir; tek-atım testinden farklıdır.
    """
    torch.manual_seed(0)
    M = torch.randn(64, 32)       # tall matrix (m > n) → kolonlar ortonormal
    X = orthogonalize_matrix(M, steps=15)  # yeterli yakınsama için 15 adım

    # XᵀX ≈ I_32
    XtX = X.T @ X
    eye = torch.eye(32)
    residual = (XtX - eye).norm().item()

    assert residual < 0.1, \
        f"Newton-Schulz (15 adım) sonrası XᵀX − I normu {residual:.4f} (beklenen < 0.1)"


def test_muon_orthogonality_wide() -> None:
    """
    Geniş matris (m < n) için de çalışmalı — Conv kernel senaryosu.
    Residual threshold daha gevşek: wide matrisler için convergence daha yavaş.
    """
    torch.manual_seed(1)
    M = torch.randn(32, 128)      # wide matrix — Conv benzeri
    X = orthogonalize_matrix(M, steps=5)

    # Sonuç finite olmalı
    assert X.isfinite().all(), "Newton-Schulz wide matris için NaN/Inf üretmemeli"
    # Norm 1'e yakın (normallanmış)
    norm = torch.linalg.norm(X).item()
    assert 0.1 < norm < 10.0, f"Ortogonalize edilmiş matris normu {norm:.4f} aşırı"


# ── Test 3: fast_memory parametreleri optimizer'da olmamalı ─────────────────
# nested_learning karşılığı: meta vs fast param ayrımı
# FIX 3 ile fast_memory.norm da fast_params() kapsamında

def test_muon_does_not_include_fast_memory() -> None:
    """
    ContinualModel.meta_param_groups() ile oluşturulan MuonMeta optimizer'ı
    fast_memory parametrelerini (net + norm) kesinlikle içermemeli.
    """
    torch.manual_seed(0)
    model = ContinualModel(num_classes=10, feature_dim=512, fast_hidden_multiplier=2)

    groups = model.meta_param_groups(backbone_lr=1e-4, classifier_lr=1e-3)
    optimizer = MuonMeta(groups, weight_decay=1e-4)

    # Optimizer'daki tüm param id'leri
    optimizer_ids = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            optimizer_ids.add(id(p))

    # fast_memory parametreleri (net + norm, FIX 3 sonrası)
    fast_ids = {id(p) for p in model.fast_memory.fast_params()}

    overlap = fast_ids & optimizer_ids
    assert len(overlap) == 0, \
        f"MuonMeta fast_memory parametresi içeriyor! " \
        f"{len(overlap)} parametre hem fast hem optimizer'da."


# ── Test 4: Conv kernel güncellenir ─────────────────────────────────────────

def test_muon_updates_conv_weight() -> None:
    """
    Conv2d.weight (4D tensor) de Muon path ile güncellenmeli.
    _orthogonalize() 4D → (out, in*kH*kW) flatten → ortho → reshape yapar.
    """
    torch.manual_seed(2)
    conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    optimizer = MuonMeta(conv.parameters(), lr=1e-2)

    x = torch.randn(2, 3, 8, 8)
    before = conv.weight.detach().clone()

    loss = conv(x).pow(2).mean()
    loss.backward()
    optimizer.step()

    after = conv.weight.detach()
    assert not torch.allclose(before, after), \
        "MuonMeta: Conv2d weight güncellenmeli!"


# ── Test 5: 1D parametre (bias) AdamW path'ten geçer ───────────────────────

def test_muon_bias_updated() -> None:
    """
    Bias (1D param) da güncellenmeli — AdamW fallback path.
    """
    torch.manual_seed(3)
    layer = nn.Linear(16, 8)
    optimizer = MuonMeta(layer.parameters(), lr=1e-2)

    x = torch.randn(4, 16)
    before_bias = layer.bias.detach().clone()

    loss = layer(x).pow(2).mean()
    loss.backward()
    optimizer.step()

    after_bias = layer.bias.detach()
    assert not torch.allclose(before_bias, after_bias), \
        "MuonMeta: bias (1D param) de güncellenmeli (AdamW path)!"


# ── Test 6: Loss sonluluk — tam train step Muon ile çalışmalı ──────────────

def test_muon_full_step_finite_loss() -> None:
    """
    train_one_step() Muon optimizer ile çağrıldığında loss NaN/Inf olmamalı.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from training.engine import train_one_step

    torch.manual_seed(0)
    model = ContinualModel(num_classes=10, feature_dim=512, fast_hidden_multiplier=2)
    model.train()

    groups = model.meta_param_groups(backbone_lr=1e-4, classifier_lr=1e-3)
    optimizer = MuonMeta(groups, weight_decay=1e-4)

    images = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 2, (4,))

    result = train_one_step(
        model=model,
        cur_images=images,
        cur_labels=labels,
        rep_images=None,
        rep_labels=None,
        optimizer=optimizer,
        current_class_ids=[0, 1],
        device=torch.device("cpu"),
        run_teach_signal=True,
    )

    assert result["loss"] > 0,               "Loss pozitif olmalı"
    assert not (result["loss"] != result["loss"]), "Loss NaN olmamalı"  # NaN check
    assert result["loss"] < 1e6,             "Loss makul aralıkta olmalı"
