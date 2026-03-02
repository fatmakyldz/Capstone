"""
Continual learning metrik takip sistemi.

nested_learning/continual_streaming.py ContinualEvalResult ile özdeş yapı:
  task_accuracy_matrix : List[List[float]]  — task_acc[task_idx][time_step]
  per_task_forgetting  : List[float]        — max_acc[i] - final_acc[i]
  avg_accuracy_final   : float              — son anda tüm taskların ortalama doğruluğu
  avg_forgetting       : float              — ortalama forgetting

Ek metrikler (capstone özgü):
  task0_history        : List[float]        — her task sonunda Task-0 doğruluğu
  per_task_final_acc   : List[float]        — son anda her task'ın doğruluğu
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ContinualMetrics:
    """
    N×N task doğruluk matrisi ve forgetting metriklerini tutar.

    Kullanım:
        metrics = ContinualMetrics(num_tasks=5)
        # Her task sonunda:
        metrics.record(task_idx=0, time_step=1, accuracy=72.3)
        # Tüm tasklar bitince:
        result = metrics.finalize()
    """
    num_tasks: int
    # task_acc[i][j] = Task-i'nin Task-j öğrenildikten sonraki doğruluğu
    task_acc: List[List[float]] = field(init=False)
    _best_acc: List[float]      = field(init=False)
    task0_history: List[float]  = field(default_factory=list)

    def __post_init__(self) -> None:
        self.task_acc  = [[float("nan")] * self.num_tasks for _ in range(self.num_tasks)]
        self._best_acc = [0.0] * self.num_tasks

    def record(self, task_idx: int, time_step: int, accuracy: float) -> None:
        """
        task_idx  : hangi task test edildi (satır)
        time_step : kaçıncı task öğrenildikten sonra (sütun)
        accuracy  : doğruluk %
        """
        self.task_acc[task_idx][time_step] = accuracy
        if not math.isnan(accuracy):
            self._best_acc[task_idx] = max(self._best_acc[task_idx], accuracy)

    def record_task0(self, accuracy: float) -> None:
        """Her task sonunda Task-0 doğruluğunu kaydet (unutma eğrisi için)."""
        self.task0_history.append(accuracy)

    # ── Türetilmiş metrikler ──────────────────────────────────────────────────

    @property
    def avg_accuracy_final(self) -> float:
        """
        Tüm tasklar bittikten sonra tüm görülen task'ların ortalama doğruluğu.

        Kaynak: nested_learning/continual_streaming.py ContinualEvalResult.avg_accuracy_final
          final_accs = [task_acc[i][-1] for i if not nan]
          avg = mean(final_accs)
        """
        last_step = self.num_tasks - 1
        vals = [
            self.task_acc[i][last_step]
            for i in range(self.num_tasks)
            if not math.isnan(self.task_acc[i][last_step])
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    @property
    def per_task_forgetting(self) -> List[float]:
        """
        Her task için: best_acc - final_acc.

        Kaynak: nested_learning/continual_streaming.py satır 257-262:
          per_task_forgetting.append(best_acc[i] - last)
        """
        last_step = self.num_tasks - 1
        result = []
        for i in range(self.num_tasks):
            last = self.task_acc[i][last_step]
            if math.isnan(last):
                result.append(float("nan"))
            else:
                result.append(self._best_acc[i] - last)
        return result

    @property
    def avg_forgetting(self) -> float:
        vals = [f for f in self.per_task_forgetting if not math.isnan(f)]
        return sum(vals) / len(vals) if vals else float("nan")

    @property
    def per_task_final_acc(self) -> List[float]:
        """Son adımdaki her task'ın doğruluğu."""
        last_step = self.num_tasks - 1
        return [self.task_acc[i][last_step] for i in range(self.num_tasks)]

    # ── Raporlama ─────────────────────────────────────────────────────────────

    def print_matrix(self) -> None:
        """
        N×N task doğruluk matrisini yazdır.
        Satır = öğrenme adımı sonrası, Sütun = test edilen task.

        nested_learning/scripts/eval/plot_forgetting.py mantığıyla aynı yapı.
        """
        header = "          " + "  ".join(f"T{j:>5}" for j in range(self.num_tasks))
        print(header)
        for step in range(self.num_tasks):
            cells = []
            for task_idx in range(self.num_tasks):
                v = self.task_acc[task_idx][step]
                cells.append(f"{v:>6.1f}" if not math.isnan(v) else "   NaN")
            print(f"After T{step}: " + "  ".join(cells))

    def print_summary(self) -> None:
        print("\n" + "=" * 55)
        print("  Continual Learning Sonuçları")
        print("=" * 55)
        print(f"  avg_accuracy_final : {self.avg_accuracy_final:6.2f}%")
        print(f"  avg_forgetting     : {self.avg_forgetting:6.2f}%")
        print()
        print("  Task Bazlı Forgetting:")
        for i, f in enumerate(self.per_task_forgetting):
            tag = "NaN" if math.isnan(f) else f"{f:6.2f}%"
            print(f"    Task {i}: {tag}")
        print()
        print("  Task-0 Hatırlama Eğrisi:")
        for step, acc in enumerate(self.task0_history):
            print(f"    After Task {step}: {acc:6.2f}%")
        print("=" * 55)

    def to_dict(self) -> dict:
        return {
            "task_accuracy_matrix":   self.task_acc,
            "per_task_forgetting":    self.per_task_forgetting,
            "avg_accuracy_final":     self.avg_accuracy_final,
            "avg_forgetting":         self.avg_forgetting,
            "per_task_final_acc":     self.per_task_final_acc,
            "task0_history":          self.task0_history,
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[Metrics] Kaydedildi: {path}")
