"""
Task definition ve yönetimi — CIFAR-10 class-incremental setup.

CIFAR-10 → 5 task, her task'ta 2 sınıf:
  Task-0: [0,1]  (airplane, automobile)
  Task-1: [2,3]  (bird, cat)
  Task-2: [4,5]  (deer, dog)
  Task-3: [6,7]  (frog, horse)
  Task-4: [8,9]  (ship, truck)

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
from typing import List

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

# macOS SSL fix (capstone1/models.py'deki yöntemle aynı)
ssl._create_default_https_context = ssl._create_unverified_context


# ── Task veri yapısı ─────────────────────────────────────────────────────────

@dataclass
class TaskDef:
    task_id: int
    class_ids: List[int]          # Bu task'a ait sınıf indeksleri
    train_dataset: Subset
    test_dataset: Subset


# ── CIFAR-10 task inşası ─────────────────────────────────────────────────────

def build_cifar10_tasks(
    data_dir: str = "./data",
    num_tasks: int = 5,
    classes_per_task: int = 2,
    resize: tuple[int, int] | None = None,
) -> List[TaskDef]:
    """
    CIFAR-10'u class-incremental task listesine dönüştürür.

    Sınıf bölümü (5 task × 2 sınıf = 10 sınıf):
      Task 0: airplane(0), automobile(1)
      Task 1: bird(2),     cat(3)
      Task 2: deer(4),     dog(5)
      Task 3: frog(6),     horse(7)
      Task 4: ship(8),     truck(9)

    Dönüş: List[TaskDef] — her biri train/test Subset'i barındırır.
    """
    transform_list = [T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465),
                                                  (0.247,  0.243,  0.261))]
    if resize is not None:
        transform_list.insert(0, T.Resize(resize))
    transform = T.Compose(transform_list)

    full_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    full_test = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    tasks: List[TaskDef] = []
    for task_id in range(num_tasks):
        start = task_id * classes_per_task
        end = start + classes_per_task
        if start >= 10:
            break
        class_ids = list(range(start, min(end, 10)))

        train_indices = [i for i, (_, lbl) in enumerate(full_train) if lbl in class_ids]
        test_indices  = [i for i, (_, lbl) in enumerate(full_test)  if lbl in class_ids]

        tasks.append(TaskDef(
            task_id=task_id,
            class_ids=class_ids,
            train_dataset=Subset(full_train, train_indices),
            test_dataset=Subset(full_test,  test_indices),
        ))

    return tasks


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

    def get_test_loaders(self) -> dict[int, DataLoader]:
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
