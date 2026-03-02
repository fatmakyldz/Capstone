"""
Class-balanced replay buffer with reservoir sampling.

Design: capstone2/ai_memory_test111/train_continual.py update_memory_bank() +
        global memory budget enforcement (per-class shrink when new class arrives).

Behaviour
─────────
- memory_size  : toplam maksimum örnek sayısı
- Her sınıf eşit bütçe alır: budget_per_class = memory_size // num_seen_classes
- Yeni sınıf geldiğinde ESKİ sınıflar küçültülür (balanced reservoir)
- Replay batch DataLoader gibi örneklenir; current batch ile TorchTensor olarak döner
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class ReplayBuffer:
    """
    Class-balanced memory bank.

    Dahili format: { class_id -> [(img_tensor, label), ...] }
    img_tensor: CPU'da saklanır, eğitimde device'a taşınır.
    """

    def __init__(self, memory_size: int = 2000, seed: int = 42) -> None:
        self.memory_size = memory_size
        self._bank: Dict[int, List[Tuple[Tensor, int]]] = {}
        self._rng = random.Random(seed)

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def num_classes(self) -> int:
        return len(self._bank)

    @property
    def total_samples(self) -> int:
        return sum(len(v) for v in self._bank.values())

    def is_empty(self) -> bool:
        return self.total_samples == 0

    def add_task_data(
        self,
        dataset,                     # torch Dataset  [(img, label), ...]
        task_class_ids: List[int],   # sınıf kimlikleri bu task için
    ) -> None:
        """
        Task bittikten sonra çağrılır.
        - Tüm eski sınıfları yeni per-class bütçeye küçültür
        - Bu task'ın sınıflarından örnekler ekler
        """
        # Sonraki total sınıf sayısına göre bütçe hesapla
        new_classes = [c for c in task_class_ids if c not in self._bank]
        all_classes_after = list(self._bank.keys()) + new_classes
        per_class = max(1, self.memory_size // len(all_classes_after))

        # Mevcut sınıfları küçült (global budget korunur)
        for cls in list(self._bank.keys()):
            exemplars = self._bank[cls]
            self._rng.shuffle(exemplars)
            self._bank[cls] = exemplars[:per_class]

        # Dataset'i sınıf bazlı gruplara ayır
        class_to_items: Dict[int, List[Tuple[Tensor, int]]] = {c: [] for c in task_class_ids}
        for img, label in dataset:
            lbl = int(label)
            if lbl in class_to_items:
                class_to_items[lbl].append((img.detach().cpu(), lbl))

        # Her yeni sınıftan per_class kadar ekle
        for cls in task_class_ids:
            items = class_to_items.get(cls, [])
            self._rng.shuffle(items)
            self._bank[cls] = items[:per_class]

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """
        Buffer'dan **sıkı class-balanced** batch döndür.
        Buffer boşsa None döner (Task-0 öncesi).

        Önceki uygulama: tüm örnekleri havuza at, rastgele çek.
        Problem: 26 sample, 4 sınıf → stokastik dağılım (10-5-8-3 olabilir).
        Küçük temsil altında kalan sınıf gradient'ı eziliyor.

        Yeni uygulama: her sınıftan tam olarak ⌊batch_size / n_classes⌋ örnek al.
        Kalan (remainder) örnekler ilk sınıflara dağıtılır.
        Bu, Task1 gibi eski sınıfların her batch'te dengeli temsil edilmesini garanti eder.

        Dönüş: (images, labels) - her ikisi de device üzerinde.
        """
        if self.is_empty():
            return None

        classes = list(self._bank.keys())
        n_cls   = len(classes)
        if n_cls == 0:
            return None

        # Her sınıftan kaç örnek? Kalan örnekler ilk sınıflara birer birer eklenir.
        per_cls  = max(1, batch_size // n_cls)
        remainder = batch_size - per_cls * n_cls   # 0 ≤ remainder < n_cls

        sampled: List[Tuple[Tensor, int]] = []
        for i, cls in enumerate(classes):
            items = self._bank[cls]
            k = per_cls + (1 if i < remainder else 0)
            k = min(k, len(items))               # buffer küçükse sınırla
            sampled.extend(self._rng.sample(items, k))

        # Sıralama bias'ını önle
        self._rng.shuffle(sampled)

        imgs = torch.stack([s[0] for s in sampled]).to(device)
        lbls = torch.tensor([s[1] for s in sampled], dtype=torch.long, device=device)
        return imgs, lbls

    def build_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Optional[DataLoader]:
        """
        Tüm buffer'ı DataLoader olarak döndür (EWC Fisher hesabı için).
        Buffer boşsa None.
        """
        if self.is_empty():
            return None

        all_items: List[Tuple[Tensor, int]] = []
        for items in self._bank.values():
            all_items.extend(items)

        imgs = torch.stack([s[0] for s in all_items])
        lbls = torch.tensor([s[1] for s in all_items], dtype=torch.long)
        ds = TensorDataset(imgs, lbls)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def seen_classes(self) -> List[int]:
        """Şimdiye kadar görülen sınıf id'leri (sıralı)."""
        return sorted(self._bank.keys())

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(memory_size={self.memory_size}, "
            f"classes={self.num_classes}, total={self.total_samples})"
        )
