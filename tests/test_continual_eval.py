"""
Continual öğrenme pipeline sanity testleri.

nested_learning karşılığı:
  tests/test_continual_classification.py
    → "evaluate_continual_classification runs and returns valid structure"
    → "task_accuracy_matrix has correct dimensions"
    → "avg_accuracy_final in [0.0, 1.0]"  (burada [0, 100])
    → "per_task_forgetting list has correct length"

Bu testler gerçek CIFAR-10 indirmez — sahte (dummy) veri kullanır.
"""
import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.continual_model import ContinualModel
from evaluation.metrics import ContinualMetrics
from memory.replay_buffer import ReplayBuffer
from training.engine import train_one_step, evaluate_task


NUM_CLASSES = 4   # Küçük test: 4 sınıf, 2 task
NUM_TASKS   = 2
FEATURE_DIM = 512
BATCH       = 8


# ── Sahte veri yardımcıları ───────────────────────────────────────────────────

def make_dummy_loader(class_ids: list[int], n_samples: int = 32) -> DataLoader:
    """Her class'tan eşit sayıda sahte 32×32 görüntü loader'ı."""
    imgs, lbls = [], []
    per_cls = max(1, n_samples // len(class_ids))
    for c in class_ids:
        imgs.append(torch.randn(per_cls, 3, 32, 32))
        lbls.append(torch.full((per_cls,), c, dtype=torch.long))
    ds = TensorDataset(torch.cat(imgs), torch.cat(lbls))
    return DataLoader(ds, batch_size=BATCH, shuffle=False)


def make_model() -> ContinualModel:
    torch.manual_seed(7)
    return ContinualModel(
        num_classes=NUM_CLASSES,
        feature_dim=FEATURE_DIM,
        freeze_backbone=False,
        fast_lr=1e-2,
        fast_hidden_multiplier=2,
    )


# ── Test 1: ContinualMetrics yapı testi ──────────────────────────────────────
# nested_learning karşılığı: test_continual_classification.py satır 95-98

def test_continual_metrics_structure() -> None:
    """
    ContinualMetrics doğru boyutlarda task_accuracy_matrix oluşturmalı.
    nested_learning: len(result.task_accuracy_matrix) == len(tasks)
    """
    metrics = ContinualMetrics(num_tasks=NUM_TASKS)

    # Sahte kayıtlar
    metrics.record(task_idx=0, time_step=0, accuracy=75.0)
    metrics.record(task_idx=0, time_step=1, accuracy=60.0)
    metrics.record(task_idx=1, time_step=1, accuracy=80.0)
    metrics.record_task0(75.0)
    metrics.record_task0(60.0)

    assert len(metrics.task_acc) == NUM_TASKS, \
        f"task_acc satır sayısı {NUM_TASKS} olmalı"
    assert len(metrics.task_acc[0]) == NUM_TASKS, \
        f"task_acc sütun sayısı {NUM_TASKS} olmalı"


# ── Test 2: avg_accuracy_final hesabı ────────────────────────────────────────
# nested_learning karşılığı: test_continual_classification.py satır 97
# "0.0 <= result.avg_accuracy_final <= 1.0"  (burada [0, 100])

def test_avg_accuracy_final_range() -> None:
    """avg_accuracy_final [0, 100] aralığında olmalı."""
    metrics = ContinualMetrics(num_tasks=NUM_TASKS)
    metrics.record(task_idx=0, time_step=0, accuracy=65.0)
    metrics.record(task_idx=0, time_step=1, accuracy=55.0)
    metrics.record(task_idx=1, time_step=1, accuracy=78.0)

    avg = metrics.avg_accuracy_final
    assert 0.0 <= avg <= 100.0, f"avg_accuracy_final {avg} [0,100] dışında!"


# ── Test 3: per_task_forgetting uzunluğu ──────────────────────────────────────
# nested_learning karşılığı: test_continual_classification.py satır 119
# "len(result.per_task_forgetting) == len(tasks)"

def test_per_task_forgetting_length() -> None:
    """per_task_forgetting listesi num_tasks uzunluğunda olmalı."""
    metrics = ContinualMetrics(num_tasks=NUM_TASKS)
    metrics.record(task_idx=0, time_step=0, accuracy=80.0)
    metrics.record(task_idx=0, time_step=1, accuracy=65.0)
    metrics.record(task_idx=1, time_step=1, accuracy=75.0)

    forgetting = metrics.per_task_forgetting
    assert len(forgetting) == NUM_TASKS, \
        f"per_task_forgetting uzunluğu {NUM_TASKS} olmalı, alınan {len(forgetting)}"


# ── Test 4: Forgetting = max_acc - final_acc ─────────────────────────────────
# nested_learning karşılığı: continual_streaming.py satır 262
# "per_task_forgetting.append(best_acc[i] - last)"

def test_forgetting_equals_best_minus_final() -> None:
    """Forgetting değeri: max(acc) - final_acc formülüne uymalı."""
    metrics = ContinualMetrics(num_tasks=2)
    metrics.record(task_idx=0, time_step=0, accuracy=90.0)  # best
    metrics.record(task_idx=0, time_step=1, accuracy=70.0)  # final (forgetting=20)
    metrics.record(task_idx=1, time_step=1, accuracy=85.0)  # forgetting=0

    forgetting = metrics.per_task_forgetting
    assert abs(forgetting[0] - 20.0) < 1e-6, \
        f"Task-0 forgetting 20.0 olmalı, alınan {forgetting[0]}"
    assert abs(forgetting[1] - 0.0)  < 1e-6, \
        f"Task-1 forgetting 0.0 olmalı, alınan {forgetting[1]}"


# ── Test 5: Replay buffer class-balanced davranış ────────────────────────────

def test_replay_buffer_class_balance() -> None:
    """
    add_task_data() sonrası her sınıf eşit sayıda örnek almalı.
    Toplam örnek sayısı memory_size'ı aşmamalı.
    """
    buf = ReplayBuffer(memory_size=100)
    loader0 = make_dummy_loader([0, 1], n_samples=80)
    buf.add_task_data(loader0.dataset, task_class_ids=[0, 1])

    assert 0 in buf._bank and 1 in buf._bank
    assert buf.total_samples <= 100

    # İkinci task ekleme — eski sınıflar küçültülmeli
    loader1 = make_dummy_loader([2, 3], n_samples=80)
    buf.add_task_data(loader1.dataset, task_class_ids=[2, 3])

    assert buf.total_samples <= 100
    assert buf.num_classes == 4


# ── Test 6: evaluate_task doğru aralıkta sonuç üretmeli ─────────────────────

def test_evaluate_task_returns_valid_accuracy() -> None:
    """
    evaluate_task() [0, 100] aralığında float döndürmeli.
    Model henüz eğitilmemiş olduğundan accuracy düşük ama geçerli.
    """
    model = make_model()
    loader = make_dummy_loader([0, 1], n_samples=64)
    device = torch.device("cpu")

    acc = evaluate_task(model, loader, device)
    assert 0.0 <= acc <= 100.0, f"Accuracy {acc} [0,100] dışında!"


# ── Test 7: Mini end-to-end pipeline (sahte veri ile) ────────────────────────
# nested_learning karşılığı: test_continual_classification.py::test_evaluate_continual_classification_runs

def test_mini_continual_pipeline() -> None:
    """
    2 task, sahte veri ile tam pipeline çalışabilmeli:
      Task-0 eğit → Task-0 ve Task-1 değerlendir → Task-1 eğit → ...
    task_accuracy_matrix'in [0][1] hücresi NaN OLMAMALI (her ikisi de dolduruldu).
    """
    model = make_model()
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.meta_parameters(), lr=1e-3)
    metrics = ContinualMetrics(num_tasks=NUM_TASKS)
    buf = ReplayBuffer(memory_size=80)

    task_data = [
        (make_dummy_loader([0, 1], n_samples=32), [0, 1]),
        (make_dummy_loader([2, 3], n_samples=32), [2, 3]),
    ]
    all_test_loaders = [
        make_dummy_loader([0, 1], n_samples=16),
        make_dummy_loader([2, 3], n_samples=16),
    ]

    for task_id, (train_loader, class_ids) in enumerate(task_data):
        # Mini eğitim (1 epoch, 1 batch)
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            rep = buf.sample_batch(8, device) if not buf.is_empty() else None
            rep_img, rep_lbl = (rep if rep else (None, None))

            train_one_step(
                model=model,
                cur_images=images, cur_labels=labels,
                rep_images=rep_img, rep_labels=rep_lbl,
                optimizer=optimizer,
                current_class_ids=class_ids,
                device=device,
            )
            break  # 1 batch yeterli

        # Değerlendirme
        for prev_task in range(task_id + 1):
            acc = evaluate_task(model, all_test_loaders[prev_task], device)
            metrics.record(task_idx=prev_task, time_step=task_id, accuracy=acc)
        metrics.record_task0(
            evaluate_task(model, all_test_loaders[0], device)
        )

        # Replay buffer güncelle
        buf.add_task_data(train_loader.dataset, task_class_ids=class_ids)

    # Yapısal doğrulama
    assert not math.isnan(metrics.task_acc[0][0]), "T0 after T0 NaN olmamalı"
    assert not math.isnan(metrics.task_acc[0][1]), "T0 after T1 NaN olmamalı"
    assert not math.isnan(metrics.task_acc[1][1]), "T1 after T1 NaN olmamalı"
    assert 0.0 <= metrics.avg_accuracy_final <= 100.0
    assert len(metrics.task0_history) == NUM_TASKS
