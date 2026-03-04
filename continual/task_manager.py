"""
Task definition ve yönetimi — class-incremental continual learning.

Desteklenen veri setleri:
  cifar10  → 10 sınıf,  5 task × 2 sınıf  (varsayılan)
  cifar100 → 100 sınıf, 5 task × 20 sınıf

Dinamik bölümleme:
  classes_per_task = num_classes // num_tasks
  Task-k: sınıflar [k*cpt, (k+1)*cpt)

Tasarım: nested_learning/src/nested_learning/continual_streaming.py build_streaming_tasks()
ile aynı prensip: etiket listesi task_size büyüklüğünde gruplara bölünür,
her grup için train/test veri seti ayrılır.

Class-incremental değerlendirme: her task sonunda TÜM görülen task'lar test edilir.
Bu, nested_learning/continual_streaming.py evaluate_continual_classification()'ın
task_accuracy_matrix yapısını birebir karşılar.
"""
from __future__ import annotations

import ssl
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

# macOS SSL fix
ssl._create_default_https_context = ssl._create_unverified_context


# ── Dataset metadata ──────────────────────────────────────────────────────────

DATASET_CONFIGS: Dict[str, dict] = {
    "cifar10": {
        "num_classes":     10,
        "normalize_mean":  (0.4914, 0.4822, 0.4465),
        "normalize_std":   (0.247,  0.243,  0.261),
        "torchvision_cls": "CIFAR10",
    },
    "cifar100": {
        "num_classes":     100,
        "normalize_mean":  (0.5071, 0.4867, 0.4408),
        "normalize_std":   (0.2675, 0.2565, 0.2761),
        "torchvision_cls": "CIFAR100",
    },
}


# ── Task veri yapısı ──────────────────────────────────────────────────────────

@dataclass
class TaskDef:
    task_id:       int
    class_ids:     List[int]      # Bu task'a ait sınıf indeksleri
    train_dataset: Subset
    test_dataset:  Subset


# ── Generic task builder ──────────────────────────────────────────────────────

def build_tasks(
    dataset_name:    str  = "cifar10",
    data_dir:        str  = "./data",
    num_tasks:       int  = 5,
    classes_per_task: Optional[int] = None,   # None → auto: num_classes // num_tasks
    resize:          Optional[Tuple[int, int]] = None,
) -> List[TaskDef]:
    """
    Desteklenen herhangi bir veri setini class-incremental task listesine dönüştürür.

    Args:
        dataset_name:     "cifar10" veya "cifar100"
        data_dir:         Ham veri kök dizini
        num_tasks:        Toplam task sayısı
        classes_per_task: Her task'taki sınıf sayısı.
                          None → num_classes // num_tasks (otomatik)
        resize:           İsteğe bağlı (H, W) yeniden boyutlandırma

    Örnekler:
        CIFAR-10  + 5 task → 2 sınıf/task  (0-1, 2-3, 4-5, 6-7, 8-9)
        CIFAR-100 + 5 task → 20 sınıf/task (0-19, 20-39, ...)
        CIFAR-100 + 10 task → 10 sınıf/task

    Dönüş: List[TaskDef] — her biri train/test Subset'i barındırır.
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Bilinmeyen veri seti: '{dataset_name}'. "
            f"Desteklenenler: {list(DATASET_CONFIGS.keys())}"
        )

    cfg = DATASET_CONFIGS[dataset_name]
    num_classes: int = cfg["num_classes"]

    if classes_per_task is None:
        classes_per_task = num_classes // num_tasks

    if classes_per_task * num_tasks > num_classes:
        raise ValueError(
            f"{dataset_name}: {num_tasks} task × {classes_per_task} sınıf = "
            f"{num_tasks * classes_per_task} > {num_classes} toplam sınıf"
        )

    # ── Dönüşümler ───────────────────────────────────────────────────────────
    transform_list: List = [
        T.ToTensor(),
        T.Normalize(cfg["normalize_mean"], cfg["normalize_std"]),
    ]
    if resize is not None:
        transform_list.insert(0, T.Resize(resize))
    transform = T.Compose(transform_list)

    # ── Tam veri setini yükle (her task için Subset oluşturulacak) ────────────
    ds_cls = getattr(torchvision.datasets, cfg["torchvision_cls"])
    full_train = ds_cls(root=data_dir, train=True,  download=True, transform=transform)
    full_test  = ds_cls(root=data_dir, train=False, download=True, transform=transform)

    # Hız optimizasyonu: tüm etiketleri bir kez ön-bellekle (set lookup O(1))
    train_labels = [int(lbl) for _, lbl in full_train]
    test_labels  = [int(lbl) for _, lbl in full_test]

    # ── Task listesini oluştur ────────────────────────────────────────────────
    tasks: List[TaskDef] = []
    for task_id in range(num_tasks):
        start = task_id * classes_per_task
        end   = start + classes_per_task
        if start >= num_classes:
            break
        class_ids = list(range(start, min(end, num_classes)))
        class_set = set(class_ids)

        train_indices = [i for i, lbl in enumerate(train_labels) if lbl in class_set]
        test_indices  = [i for i, lbl in enumerate(test_labels)  if lbl in class_set]

        tasks.append(TaskDef(
            task_id=task_id,
            class_ids=class_ids,
            train_dataset=Subset(full_train, train_indices),
            test_dataset=Subset(full_test,  test_indices),
        ))

    return tasks


# ── Backward-compatible CIFAR-10 wrapper ──────────────────────────────────────

def build_cifar10_tasks(
    data_dir:        str  = "./data",
    num_tasks:       int  = 5,
    classes_per_task: int = 2,
    resize:          Optional[Tuple[int, int]] = None,
) -> List[TaskDef]:
    """CIFAR-10 task listesi (geriye dönük uyumluluk için korundu)."""
    return build_tasks(
        dataset_name="cifar10",
        data_dir=data_dir,
        num_tasks=num_tasks,
        classes_per_task=classes_per_task,
        resize=resize,
    )


# ── TaskManager: task geçiş olaylarını yönetir ───────────────────────────────

class TaskManager:
    """
    Task sırası boyunca state'i takip eder.

    Sorumluluklar:
    - Aktif task'ı ve şimdiye kadar görülen sınıfları takip etme
    - Classifier gradient masking için current_classes sağlama
    - Task geçişinde replay buffer güncelleme zamanını belirleme

    nested_learning referansı:
      continual_streaming.py'deki for current_task, task in enumerate(tasks): döngüsünün
      Python tarafındaki karşılığı — state tracking burada yapılır.
    """

    def __init__(self, tasks: List[TaskDef]) -> None:
        self.tasks = tasks
        self.current_task_id = -1
        self._seen_class_ids: List[int] = []

    def advance(self) -> TaskDef:
        """Bir sonraki task'a geç, state güncelle."""
        self.current_task_id += 1
        task = self.tasks[self.current_task_id]
        for c in task.class_ids:
            if c not in self._seen_class_ids:
                self._seen_class_ids.append(c)
        return task

    @property
    def current_task(self) -> TaskDef:
        return self.tasks[self.current_task_id]

    @property
    def seen_class_ids(self) -> List[int]:
        return list(self._seen_class_ids)

    @property
    def is_first_task(self) -> bool:
        return self.current_task_id == 0

    def num_tasks_done(self) -> int:
        return self.current_task_id + 1

    def get_train_loader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.current_task.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
        )

    def get_test_loaders(self) -> Dict[int, DataLoader]:
        """Şimdiye kadar görülen tüm task'ların test loader'ları."""
        return {
            t.task_id: DataLoader(
                t.test_dataset,
                batch_size=256,
                shuffle=False,
                num_workers=0,
            )
            for t in self.tasks[: self.current_task_id + 1]
        }
