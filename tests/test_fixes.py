"""
FIX 1-4 davranış kilitleri.

Her test, daha önce tespit edilen ve düzeltilen bir hataya karşılık gelir.
"""
import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.continual_model import ContinualModel, compute_teach_signal
from training.engine import train_one_step

NUM_CLASSES = 10
FEATURE_DIM = 512
BATCH = 4


@pytest.fixture
def model() -> ContinualModel:
    torch.manual_seed(0)
    return ContinualModel(
        num_classes=NUM_CLASSES,
        feature_dim=FEATURE_DIM,
        freeze_backbone=False,
        fast_lr=1e-2,
        fast_hidden_multiplier=2,
    )


# ── Test 1 (FIX 1): compute_teach_signal torch.no_grad() altında çalışmalı ───
# Önceki hata: torch.autograd.grad() no_grad context içinde RuntimeError veriyordu.

def test_teach_signal_runs_under_no_grad(model: ContinualModel) -> None:
    """
    compute_teach_signal(), torch.no_grad() bloğu içinden çağrıldığında
    RuntimeError fırlatmamalı ve doğru şekle sahip tensör döndürmeli.

    engine.py Pass-2 bloğu tam olarak bu şekilde çağırır:
        with torch.no_grad():
            teach = compute_teach_signal(features, logits, labels, model.classifier)
    """
    model.eval()
    images = torch.randn(BATCH, 3, 32, 32)
    labels = torch.randint(0, 2, (BATCH,))

    with torch.no_grad():
        logits, features = model(images)

    # Bu satır önceden RuntimeError veriyordu; şimdi çalışmalı.
    with torch.no_grad():
        teach = compute_teach_signal(features, logits, labels, model.classifier)

    assert teach.shape == (BATCH, FEATURE_DIM), \
        f"Teach signal şekli ({BATCH}, {FEATURE_DIM}) olmalı, alınan: {teach.shape}"
    assert not teach.requires_grad, "Teach signal gradient taşımamalı"
    assert teach.isfinite().all(), "Teach signal sonlu değerler içermeli"


# ── Test 2 (FIX 3): fast_memory.norm meta optimizer'ın dışında olmalı ─────────
# Önceki hata: fast_params() yalnızca net parametrelerini döndürüyordu,
# norm.weight/bias meta optimizer tarafından güncelleniyordu.

def test_fast_norm_excluded_from_meta_optimizer(model: ContinualModel) -> None:
    """
    model.meta_parameters() fast_memory.norm parametrelerini içermemeli.

    fast_memory.norm da fast_params() tarafından döndürülmeli, böylece
    meta_parameters() onu dışarıda bırakır.
    """
    fast_ids = {id(p) for p in model.fast_memory.fast_params()}
    meta_ids = {id(p) for p in model.meta_parameters()}

    # norm parametrelerinin hiçbiri meta optimizer'da olmamalı
    for p in model.fast_memory.norm.parameters():
        assert id(p) in fast_ids, \
            "fast_memory.norm parametresi fast_params() tarafından döndürülmeli"
        assert id(p) not in meta_ids, \
            "fast_memory.norm parametresi meta optimizer'a dahil edilmemeli"


# ── Test 3 (FIX 4): fast param grads meta backward sonrası temizlenmeli ───────
# Önceki hata: fast params optimizer'dan dışlandığı için optimizer.zero_grad()
# onları temizlemiyordu; eski gradyanlar birikiyordu.

def test_fast_params_grads_cleared_after_step(model: ContinualModel) -> None:
    """
    train_one_step() çağrısından sonra fast_memory parametrelerinin
    .grad alanı None olmalı — stale gradyan birikmemeli.
    """
    model.train()
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.meta_parameters(), lr=1e-3)

    images = torch.randn(BATCH, 3, 32, 32)
    labels = torch.randint(0, 2, (BATCH,))

    train_one_step(
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

    # fast_memory parametrelerinin hiçbirinde stale grad kalmamalı
    for name, p in model.fast_memory.named_parameters():
        assert p.grad is None, \
            f"fast_memory.{name} grad'ı step sonrası None olmalı, " \
            f"stale grad birikmemeli (shape={p.grad.shape if p.grad is not None else 'N/A'})"
