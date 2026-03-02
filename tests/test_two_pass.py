"""
İki-pasajlı eğitim adımının bütünleşik testleri.

nested_learning karşılıkları:
  test_fast_state_meta_grads.py
    → "meta gradients flow in fast_state mode"
  test_hope_selfmod_fast_state_meta_unchanged.py
    → "fast updates do NOT touch meta params (backbone + classifier)"
  test_algorithm_mode_grad.py
    → "differentiable vs non-differentiable gradient paths"

Doğrulanan invariantlar:
  1. Pass-2 (fast update) backbone parametrelerini DEĞİŞTİRMEMELİ
  2. Pass-2 classifier parametrelerini DEĞİŞTİRMEMELİ
  3. Meta backward sonrası backbone gradyanları akıyor olmalı
  4. compute_teach_signal doğru şekil döndürmeli
  5. Tam train_one_step loss skalar ve sonlu olmalı
"""
import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.continual_model import ContinualModel, compute_teach_signal


NUM_CLASSES = 10
FEATURE_DIM = 512
BATCH = 4


@pytest.fixture
def model() -> ContinualModel:
    torch.manual_seed(42)
    return ContinualModel(
        num_classes=NUM_CLASSES,
        feature_dim=FEATURE_DIM,
        freeze_backbone=False,
        fast_lr=1e-2,
        fast_hidden_multiplier=2,
    )


def fake_images() -> torch.Tensor:
    return torch.randn(BATCH, 3, 32, 32)


def fake_labels(class_range=(0, 2)) -> torch.Tensor:
    return torch.randint(class_range[0], class_range[1], (BATCH,))


# ── Test 1: compute_teach_signal şekli ───────────────────────────────────────
# nested_learning referansı: compute_teach_signal() → (B, T, dim) için (B, dim) uyarlaması

def test_teach_signal_shape(model: ContinualModel) -> None:
    """compute_teach_signal (B, dim) şeklinde tensör döndürmeli."""
    images = fake_images()
    labels = fake_labels()
    model.eval()
    with torch.no_grad():
        logits, features = model(images)
    # compute_teach_signal kendi içinde enable_grad kullanır
    teach = compute_teach_signal(features, logits, labels, model.classifier)
    assert teach.shape == (BATCH, FEATURE_DIM), \
        f"Beklenen ({BATCH}, {FEATURE_DIM}), alınan {teach.shape}"
    assert not teach.requires_grad, "Teach signal detached olmalı"


# ── Test 2: Pass-2 backbone parametrelerine dokunmamalı ──────────────────────
# nested_learning karşılığı: test_hope_selfmod_fast_state_meta_unchanged.py
# "fast state updates do NOT mutate meta params"

def test_fast_update_does_not_change_backbone(model: ContinualModel) -> None:
    """
    model.update_fast() çağrısından sonra backbone.net parametreleri
    değişmemiş olmalı. Sadece fast_memory.net değişir.
    """
    model.train()
    images = fake_images()
    labels = fake_labels()

    # Backbone ağırlıklarının snapshot'ını al
    backbone_before = {
        n: p.detach().clone() for n, p in model.backbone.named_parameters()
    }

    # Pass-1: normal forward
    logits, features = model(images)

    # Teach signal hesapla
    teach = compute_teach_signal(features, logits, labels, model.classifier)

    # Pass-2: sadece fast_memory güncellenir
    model.update_fast(features=features.detach(), teach_signal=teach)

    # Backbone parametreleri değişmemiş olmalı
    for name, before_val in backbone_before.items():
        after_val = dict(model.backbone.named_parameters())[name].detach()
        assert torch.allclose(before_val, after_val), \
            f"Backbone param '{name}' fast update sonrası değişti!"


# ── Test 3: Pass-2 classifier parametrelerine dokunmamalı ────────────────────

def test_fast_update_does_not_change_classifier(model: ContinualModel) -> None:
    """
    model.update_fast() çağrısından sonra classifier.weight ve classifier.bias
    değişmemiş olmalı.
    """
    model.train()
    images = fake_images()
    labels = fake_labels()

    clf_w_before = model.classifier.weight.detach().clone()
    clf_b_before = model.classifier.bias.detach().clone() if model.classifier.bias is not None else None

    logits, features = model(images)
    teach = compute_teach_signal(features, logits, labels, model.classifier)
    model.update_fast(features=features.detach(), teach_signal=teach)

    assert torch.allclose(clf_w_before, model.classifier.weight.detach()), \
        "Fast update classifier.weight değiştirmemeli!"
    if clf_b_before is not None:
        assert torch.allclose(clf_b_before, model.classifier.bias.detach()), \
            "Fast update classifier.bias değiştirmemeli!"


# ── Test 4: Meta backward sonrası backbone gradyanları akıyor olmalı ─────────
# nested_learning karşılığı: test_fast_state_meta_grads.py
# "at least one memory/backbone param gets gradients in fast_state mode"

def test_meta_backward_flows_to_backbone(model: ContinualModel) -> None:
    """
    loss.backward() çağrısı backbone parametrelerine gradyan sağlamalı.
    """
    model.train()
    images = fake_images()
    labels = fake_labels()

    optimizer = torch.optim.Adam(model.meta_parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)

    logits, features = model(images)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    # En az bir backbone parametresi grad almış olmalı
    backbone_grads = [
        p.grad for p in model.backbone.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert len(backbone_grads) > 0, \
        "Meta backward sonrası backbone parametreleri gradyan almalı!"


# ── Test 5: meta_parameters() fast_memory parametrelerini dışarıda bırakmalı ─

def test_meta_parameters_exclude_fast_memory(model: ContinualModel) -> None:
    """
    model.meta_parameters() fast_memory.net parametrelerini içermemeli.
    Bu, fast memory'nin meta optimizer'dan bağımsız olduğunu garanti eder.
    """
    fast_ids = {id(p) for p in model.fast_memory.fast_params()}
    meta_ids = {id(p) for p in model.meta_parameters()}

    overlap = fast_ids & meta_ids
    assert len(overlap) == 0, \
        f"meta_parameters() {len(overlap)} fast_memory parametresi içeriyor!"


# ── Test 6: Tam train_one_step sonlu ve skalar loss döndürmeli ───────────────

def test_train_one_step_returns_finite_loss(model: ContinualModel) -> None:
    """
    train_one_step() sonlu bir loss değeri döndürmeli (NaN/Inf değil).
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from training.engine import train_one_step

    model.train()
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.meta_parameters(), lr=1e-3)

    images = fake_images()
    labels = fake_labels(class_range=(0, 2))

    result = train_one_step(
        model=model,
        cur_images=images,
        cur_labels=labels,
        rep_images=None,
        rep_labels=None,
        optimizer=optimizer,
        current_class_ids=[0, 1],
        device=device,
        run_teach_signal=True,
    )

    assert "loss" in result
    assert not torch.isnan(torch.tensor(result["loss"])), "Loss NaN olmamalı!"
    assert not torch.isinf(torch.tensor(result["loss"])), "Loss Inf olmamalı!"
    assert result["loss"] > 0, "Loss pozitif olmalı"


# ── Test 7: update_fast yalnızca current örneklerle çağrılmalı ───────────────
# Replay örnekleri meta loss'ta kullanılır ama fast memory'ye yazılmaz.
# nested_learning referansı: memorize_tokens() yalnızca current token'ları
# fast_state'e yazar.

def test_update_fast_only_receives_current_samples() -> None:
    """
    train_one_step() replay batch varken update_fast'ı SADECE current örneklerle
    çağırmalı. features tensörünün ilk boyutu n_current olmalı, B_total değil.

    Test stratejisi: model.update_fast() patch'lenerek gelen argümanların
    shape'i yakalanır; assert n_current × feature_dim.
    """
    from unittest.mock import patch
    from training.engine import train_one_step

    torch.manual_seed(7)
    m = ContinualModel(
        num_classes=NUM_CLASSES,
        feature_dim=FEATURE_DIM,
        freeze_backbone=False,
        fast_lr=1e-2,
        fast_hidden_multiplier=2,
    )
    m.train()
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(m.meta_parameters(), lr=1e-3)

    B_cur = 4
    B_rep = 4
    cur_images = torch.randn(B_cur, 3, 32, 32)
    cur_labels = torch.randint(0, 2, (B_cur,))
    rep_images = torch.randn(B_rep, 3, 32, 32)
    rep_labels = torch.randint(0, 2, (B_rep,))

    captured: dict = {}
    original_update_fast = m.update_fast

    def mock_update_fast(features, teach_signal):
        captured["features_shape"] = tuple(features.shape)
        captured["teach_shape"]    = tuple(teach_signal.shape)
        return original_update_fast(features, teach_signal)

    with patch.object(m, "update_fast", side_effect=mock_update_fast):
        train_one_step(
            model=m,
            cur_images=cur_images,
            cur_labels=cur_labels,
            rep_images=rep_images,
            rep_labels=rep_labels,
            optimizer=optimizer,
            current_class_ids=[0, 1],
            device=device,
            run_teach_signal=True,
        )

    assert "features_shape" in captured, "update_fast hiç çağrılmadı!"
    assert captured["features_shape"] == (B_cur, FEATURE_DIM), (
        f"update_fast B_cur={B_cur} boyutunda features bekliyordu, "
        f"alınan: {captured['features_shape']}"
    )
    assert captured["teach_shape"] == (B_cur, FEATURE_DIM), (
        f"teach_signal B_cur={B_cur} boyutunda bekliyordu, "
        f"alınan: {captured['teach_shape']}"
    )
