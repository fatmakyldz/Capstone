"""
FastMemory birim testleri.

nested_learning karşılıkları:
  test_hope_selfmod_fast_state_meta_unchanged.py
    → "fast state updates do NOT mutate meta params"
  test_hope_selfmod_update_pass.py
    → "updates happen ONLY when teach_signal is provided"
  test_hope_block.py
    → "forward pass shape check"
"""
import copy

import pytest
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.fast_memory import FastMemory


DIM = 32
BATCH = 4


@pytest.fixture
def fast_mem() -> FastMemory:
    torch.manual_seed(0)
    return FastMemory(dim=DIM, hidden_multiplier=2, fast_lr=1e-2)


# ── Test 1: Forward pass şekil kontrolü ──────────────────────────────────────
# nested_learning karşılığı: test_hope_block.py::test_hope_block_forward

def test_forward_shape(fast_mem: FastMemory) -> None:
    """Normal forward (teach_signal=None) çıktı şeklini doğrular."""
    x = torch.randn(BATCH, DIM)
    out = fast_mem(x)
    assert out.shape == (BATCH, DIM), f"Beklenen ({BATCH}, {DIM}), alınan {out.shape}"


# ── Test 2: Teach signal olmadan ağırlık değişmemeli ─────────────────────────
# nested_learning karşılığı: test_hope_selfmod_update_pass.py
# "Updates happen ONLY in update pass (teach_signal provided)"

def test_no_update_without_teach_signal(fast_mem: FastMemory) -> None:
    """
    Teach signal olmadan forward sonrası fast_memory.net ağırlıkları
    değişmemiş olmalı.
    """
    fast_mem.train()
    before = [p.detach().clone() for p in fast_mem.net.parameters()]

    x = torch.randn(BATCH, DIM)
    _ = fast_mem(x)  # teach_signal=None → update yok

    after = list(fast_mem.net.parameters())
    for b, a in zip(before, after):
        assert torch.allclose(b, a), "Teach signal olmadan ağırlık değişmemeli!"


# ── Test 3: Teach signal İLE ağırlık değişmeli ───────────────────────────────
# nested_learning karşılığı: test_hope_selfmod_integration.py
# "SelfMod fast_state parameters ARE updated after update pass"

def test_update_with_teach_signal(fast_mem: FastMemory) -> None:
    """
    Teach signal verildiğinde fast_memory.net ağırlıkları güncellenmiş olmalı.
    """
    fast_mem.train()
    before = [p.detach().clone() for p in fast_mem.net.parameters()]

    x = torch.randn(BATCH, DIM)
    teach = torch.randn(BATCH, DIM)
    _ = fast_mem(x, teach_signal=teach)

    after = list(fast_mem.net.parameters())
    changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
    assert changed, "Teach signal verildiğinde fast ağırlıklar değişmeli!"


# ── Test 4: META parametreler (norm) asla değişmemeli ────────────────────────
# nested_learning karşılığı: test_hope_selfmod_fast_state_meta_unchanged.py
# "fast state updates do NOT mutate meta params"
# Burada 'meta' = norm parametreleri (fast_memory.norm.weight/bias)

def test_norm_params_unchanged_after_update(fast_mem: FastMemory) -> None:
    """
    Teach signal update'i sadece fast_mem.net'i değiştirir.
    fast_mem.norm parametreleri (weight, bias) dokunulmaz kalır.
    """
    fast_mem.train()
    norm_before = [p.detach().clone() for p in fast_mem.norm.parameters()]

    x = torch.randn(BATCH, DIM)
    teach = torch.randn(BATCH, DIM)
    _ = fast_mem(x, teach_signal=teach)

    norm_after = list(fast_mem.norm.parameters())
    for b, a in zip(norm_before, norm_after):
        assert torch.allclose(b, a), "Teach signal norm parametrelerine dokunmamalı!"


# ── Test 5: Eval modunda güncelleme olmamalı ─────────────────────────────────

def test_no_update_in_eval_mode(fast_mem: FastMemory) -> None:
    """model.eval() modunda teach_signal verilse bile güncelleme olmamalı."""
    fast_mem.eval()  # ← eval modu
    before = [p.detach().clone() for p in fast_mem.net.parameters()]

    x = torch.randn(BATCH, DIM)
    teach = torch.randn(BATCH, DIM)
    _ = fast_mem(x, teach_signal=teach)

    after = list(fast_mem.net.parameters())
    for b, a in zip(before, after):
        assert torch.allclose(b, a), "Eval modunda ağırlık değişmemeli!"


# ── Test 6: reset_fast_weights sıfırlama ─────────────────────────────────────

def test_reset_fast_weights(fast_mem: FastMemory) -> None:
    """reset_fast_weights() tüm net parametrelerini sıfırlamalı."""
    fast_mem.train()
    # Önce bir güncelleme yap
    x = torch.randn(BATCH, DIM)
    teach = torch.randn(BATCH, DIM)
    _ = fast_mem(x, teach_signal=teach)

    fast_mem.reset_fast_weights()

    for p in fast_mem.net.parameters():
        assert torch.all(p == 0), "reset sonrası tüm ağırlıklar 0 olmalı"
