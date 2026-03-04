"""
CSV-based performance metrics logger for PyTorch DDP training.

Two separate CSV files:
  1. {output_path}  — step-level rows (one per log_step) + epoch-end rows
  2. {summary_path} — task-level summary rows (one per task, written by log_task_summary)

DDP-safe: only rank 0 writes; all other ranks are silent no-ops.
Non-blocking: pynvml/psutil sampled every hw_sample_interval steps.

Step-level CSV columns:
  task_id, epoch, global_step, row_type
  train_loss, val_accuracy
  total_elapsed_s, task_elapsed_s, epoch_time_s, step_time_s, convergence_time_s
  throughput_img_s, fps
  gpu_util_pct, gpu_mem_used_mb, gpu_mem_peak_mb, cpu_util_pct, ram_used_mb

Task-level summary CSV columns (metrics.csv):
  task_id, epoch
  task_train_time_seconds, cumulative_train_time_seconds
  accuracy_per_task, avg_accuracy_final, avg_forgetting
  fps, gpu_utilization_percent, cpu_utilization_percent, memory_usage_mb

Quick start (non-DDP)::

    from training.metrics_logger import MetricsLogger

    logger = MetricsLogger(
        output_path="results/perf.csv",
        summary_path="results/metrics.csv",
        convergence_threshold=80.0,
    )
    logger.start_training()
    for task_id in range(num_tasks):
        logger.start_task(task_id)
        for epoch in range(epochs_per_task):
            logger.start_epoch(epoch)
            for step, batch in enumerate(loader):
                logger.step_start()
                result = train_one_step(...)
                logger.log_step(task_id, epoch, global_step,
                                result["loss"], batch_size)
            logger.log_epoch_end(task_id, epoch, global_step=global_step)
        logger.log_task_summary(task_id, last_epoch, task_accs,
                                avg_acc, avg_forgetting, device)
    logger.close()
"""
from __future__ import annotations

import csv
import json
import os
import time
from typing import Dict, List, Optional

import torch

# ── Optional hardware libs ────────────────────────────────────────────────────

try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    _PYNVML_OK = True
except Exception:
    _PYNVML_OK = False

try:
    import psutil as _psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False

# ── CSV schemas ───────────────────────────────────────────────────────────────

_STEP_FIELDS = [
    "task_id", "epoch", "global_step", "row_type",
    "train_loss", "val_accuracy",
    "total_elapsed_s", "task_elapsed_s", "epoch_time_s",
    "step_time_s", "convergence_time_s",
    "throughput_img_s", "fps",
    "gpu_util_pct", "gpu_mem_used_mb", "gpu_mem_peak_mb",
    "cpu_util_pct", "ram_used_mb",
]

_SUMMARY_FIELDS = [
    "task_id", "epoch",
    "task_train_time_seconds", "cumulative_train_time_seconds",
    "accuracy_per_task", "avg_accuracy_final", "avg_forgetting",
    "fps",
    "gpu_utilization_percent", "cpu_utilization_percent", "memory_usage_mb",
]

_NAN = float("nan")


def _fmt(v: Optional[float], decimals: int = 3) -> str:
    """Format a float for CSV; empty string for None or NaN."""
    if v is None or v != v:   # NaN != NaN
        return ""
    return f"{v:.{decimals}f}"


class MetricsLogger:
    """
    Dual-CSV metrics logger for continual learning experiments.

    Writes:
      • Step-level CSV  → one row per training step + one row per epoch end.
      • Task-level CSV  → one row per task (call log_task_summary() after eval).

    DDP: pass rank=dist.get_rank(); only rank 0 writes. All other ranks no-op.

    Args:
        output_path:            Path for the step-level CSV.
        summary_path:           Path for the task-level metrics.csv.
                                None → no task-level file.
        rank:                   DDP rank. 0 = write; others = silent.
        convergence_threshold:  Val accuracy (%) that stamps convergence_time_s.
        hw_sample_interval:     Sample GPU/CPU every N steps (default 20).
        gpu_index:              NVML GPU device index (0 for single-GPU).
    """

    def __init__(
        self,
        output_path: str,
        summary_path: Optional[str] = None,
        rank: int = 0,
        convergence_threshold: Optional[float] = None,
        hw_sample_interval: int = 20,
        gpu_index: int = 0,
    ) -> None:
        self.rank = rank
        self._active = rank == 0
        self.convergence_threshold = convergence_threshold
        self.hw_sample_interval = hw_sample_interval

        # ── Timing ────────────────────────────────────────────────────────────
        self._t_train: float = 0.0
        self._t_task:  float = 0.0
        self._t_epoch: float = 0.0
        self._t_step:  float = 0.0

        # ── Convergence ───────────────────────────────────────────────────────
        self._converged: bool = False
        self._conv_time: Optional[float] = None

        # ── FPS / image counter (reset per task) ─────────────────────────────
        self._task_images: int = 0

        # ── Hardware cache ────────────────────────────────────────────────────
        self._hw_cache: Dict[str, Optional[float]] = {
            "gpu_util_pct":    None,
            "gpu_mem_used_mb": None,
            "gpu_mem_peak_mb": None,
            "cpu_util_pct":    None,
            "ram_used_mb":     None,
        }
        self._hw_counter: int = 0

        # ── NVML handle ───────────────────────────────────────────────────────
        self._nvml_handle = None
        if _PYNVML_OK and torch.cuda.is_available():
            try:
                self._nvml_handle = _pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                pass

        # ── Step-level CSV ────────────────────────────────────────────────────
        self._step_file = None
        self._step_writer: Optional[csv.DictWriter] = None

        # ── Task-level summary CSV ────────────────────────────────────────────
        self._summary_file = None
        self._summary_writer: Optional[csv.DictWriter] = None

        if self._active:
            self._step_file = self._open_csv(output_path, _STEP_FIELDS)
            self._step_writer = csv.DictWriter(self._step_file, fieldnames=_STEP_FIELDS)
            self._step_writer.writeheader()

            if summary_path is not None:
                self._summary_file = self._open_csv(summary_path, _SUMMARY_FIELDS)
                self._summary_writer = csv.DictWriter(
                    self._summary_file, fieldnames=_SUMMARY_FIELDS
                )
                self._summary_writer.writeheader()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start_training(self) -> None:
        """Call once before the outer task loop."""
        self._t_train = time.perf_counter()

    def start_task(self, task_id: int) -> None:
        """
        Call at the start of each task.
        Resets task timer, convergence state, image counter, and peak GPU memory.
        """
        self._t_task = time.perf_counter()
        self._converged = False
        self._conv_time = None
        self._task_images = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def start_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch."""
        self._t_epoch = time.perf_counter()

    def step_start(self) -> None:
        """
        Call immediately BEFORE train_one_step().
        Stamps the step start so step_time_s excludes data-loading latency.

        Example::

            logger.step_start()
            result = train_one_step(model, ...)
            logger.log_step(task_id, epoch, global_step, result["loss"], batch_size)
        """
        self._t_step = time.perf_counter()

    # ── Row writers ───────────────────────────────────────────────────────────

    def log_step(
        self,
        task_id: int,
        epoch: int,
        global_step: int,
        train_loss: float,
        batch_size: int,
    ) -> None:
        """
        Write one step-level row. Call after train_one_step().

        Hardware metrics are refreshed from cache every hw_sample_interval steps.
        batch_size is accumulated for per-task FPS calculation.
        """
        if not self._active:
            return

        now = time.perf_counter()
        step_time = now - self._t_step if self._t_step > 0.0 else _NAN
        throughput = batch_size / step_time if step_time > 0.0 else _NAN

        self._task_images += batch_size

        self._hw_counter += 1
        if self._hw_counter % self.hw_sample_interval == 1:
            self._refresh_hw()

        row = self._base_step_row(task_id, epoch, global_step, now, "step")
        row.update({
            "train_loss":       _fmt(train_loss, 6),
            "step_time_s":      _fmt(step_time, 4),
            "throughput_img_s": _fmt(throughput, 1),
            "fps":              _fmt(throughput, 1),
        })
        row.update(self._fmt_hw())
        self._step_writer.writerow(row)

    def log_epoch_end(
        self,
        task_id: int,
        epoch: int,
        val_accuracy: Optional[float] = None,
        global_step: Optional[int] = None,
    ) -> None:
        """
        Write one epoch-summary row.
        Call at the end of each epoch after all steps have run.
        epoch_time_s = time from start_epoch() to this call.
        """
        if not self._active:
            return

        now = time.perf_counter()
        epoch_time = now - self._t_epoch if self._t_epoch > 0.0 else _NAN

        self._check_convergence(val_accuracy, now)
        conv_s = self._conv_time

        self._refresh_hw()

        row = self._base_step_row(task_id, epoch, global_step, now, "epoch_end")
        row.update({
            "val_accuracy":       _fmt(val_accuracy, 4),
            "epoch_time_s":       _fmt(epoch_time, 3),
            "convergence_time_s": _fmt(conv_s, 3),
        })
        row.update(self._fmt_hw())
        self._step_writer.writerow(row)

    def log_task_summary(
        self,
        task_id: int,
        epoch: int,
        task_accuracies: List[float],    # acc for each seen task at this time step
        avg_accuracy_final: float,
        avg_forgetting: float,
        device: "torch.device",
    ) -> None:
        """
        Write one task-level summary row to metrics.csv.
        Call once after evaluation at the end of each task.

        Args:
            task_id:            Current task index.
            epoch:              Last epoch trained for this task (0-based).
            task_accuracies:    List of val accuracies for each seen task, ordered
                                by task_id. E.g. [82.5, 61.3] after task 1.
            avg_accuracy_final: Mean accuracy over all seen tasks at this step.
            avg_forgetting:     Mean forgetting over all seen tasks at this step.
            device:             Training device (determines GPU/CPU memory logging).
        """
        if not self._active or self._summary_writer is None:
            return

        now = time.perf_counter()
        task_time = now - self._t_task if self._t_task > 0.0 else _NAN
        cumul_time = now - self._t_train if self._t_train > 0.0 else _NAN
        fps = self._task_images / task_time if task_time > 0.0 else _NAN

        # Hardware metrics (fresh sample at task end)
        self._refresh_hw()

        # GPU utilization: pynvml (CUDA only)
        gpu_util = _fmt(self._hw_cache["gpu_util_pct"], 2) if device.type == "cuda" else ""

        # Memory: torch.cuda.memory_allocated() for CUDA, psutil RAM for others
        if device.type == "cuda":
            mem_mb: Optional[float] = torch.cuda.memory_allocated() / 1_048_576
        elif _PSUTIL_OK:
            try:
                mem_mb = _psutil.virtual_memory().used / 1_048_576
            except Exception:
                mem_mb = None
        else:
            mem_mb = None

        # CPU utilization
        cpu_util: Optional[float] = None
        if _PSUTIL_OK:
            try:
                cpu_util = _psutil.cpu_percent(interval=None)
            except Exception:
                pass

        row = {
            "task_id":                     task_id,
            "epoch":                       epoch,
            "task_train_time_seconds":     _fmt(task_time, 2),
            "cumulative_train_time_seconds": _fmt(cumul_time, 2),
            "accuracy_per_task":           json.dumps([round(a, 2) for a in task_accuracies]),
            "avg_accuracy_final":          _fmt(avg_accuracy_final, 4),
            "avg_forgetting":              _fmt(avg_forgetting, 4),
            "fps":                         _fmt(fps, 1),
            "gpu_utilization_percent":     gpu_util,
            "cpu_utilization_percent":     _fmt(cpu_util, 2),
            "memory_usage_mb":             _fmt(mem_mb, 1),
        }
        self._summary_writer.writerow(row)
        if self._summary_file is not None:
            self._summary_file.flush()

    # ── Context-manager support ───────────────────────────────────────────────

    def close(self) -> None:
        """Flush and close all CSV files."""
        for f in (self._step_file, self._summary_file):
            if f is not None:
                f.flush()
                f.close()
        self._step_file = None
        self._summary_file = None

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _open_csv(path: str, fields: List[str]):
        """Create parent dirs and open a line-buffered CSV file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        return open(path, "w", newline="", buffering=1)

    def _base_step_row(
        self,
        task_id: int,
        epoch: int,
        global_step: Optional[int],
        now: float,
        row_type: str,
    ) -> dict:
        """Empty skeleton row for the step-level CSV."""
        return {
            "task_id":            task_id,
            "epoch":              epoch,
            "global_step":        global_step if global_step is not None else "",
            "row_type":           row_type,
            "train_loss":         "",
            "val_accuracy":       "",
            "total_elapsed_s":    _fmt(now - self._t_train, 3),
            "task_elapsed_s":     _fmt(now - self._t_task,  3),
            "epoch_time_s":       "",
            "step_time_s":        "",
            "convergence_time_s": "",
            "throughput_img_s":   "",
            "fps":                "",
            "gpu_util_pct":       "",
            "gpu_mem_used_mb":    "",
            "gpu_mem_peak_mb":    "",
            "cpu_util_pct":       "",
            "ram_used_mb":        "",
        }

    def _refresh_hw(self) -> None:
        """
        Sample GPU (pynvml) and CPU/RAM (psutil) metrics into self._hw_cache.
        Called at hw_sample_interval cadence inside log_step(), and always at
        log_epoch_end() / log_task_summary().
        """
        if self._nvml_handle is not None:
            try:
                util = _pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                self._hw_cache["gpu_util_pct"] = float(util.gpu)
                mem = _pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                self._hw_cache["gpu_mem_used_mb"] = mem.used / 1_048_576
            except Exception:
                pass

        if torch.cuda.is_available():
            self._hw_cache["gpu_mem_peak_mb"] = (
                torch.cuda.max_memory_allocated() / 1_048_576
            )

        if _PSUTIL_OK:
            try:
                self._hw_cache["cpu_util_pct"] = _psutil.cpu_percent(interval=None)
                self._hw_cache["ram_used_mb"] = _psutil.virtual_memory().used / 1_048_576
            except Exception:
                pass

    def _fmt_hw(self) -> dict:
        """Return hardware cache as formatted strings for the step-level CSV row."""
        return {
            "gpu_util_pct":    _fmt(self._hw_cache["gpu_util_pct"],    2),
            "gpu_mem_used_mb": _fmt(self._hw_cache["gpu_mem_used_mb"], 1),
            "gpu_mem_peak_mb": _fmt(self._hw_cache["gpu_mem_peak_mb"], 1),
            "cpu_util_pct":    _fmt(self._hw_cache["cpu_util_pct"],    2),
            "ram_used_mb":     _fmt(self._hw_cache["ram_used_mb"],     1),
        }

    def _check_convergence(self, val_accuracy: Optional[float], now: float) -> None:
        """Stamp convergence_time_s the first time val_accuracy reaches the threshold."""
        if (
            self.convergence_threshold is not None
            and not self._converged
            and val_accuracy is not None
            and val_accuracy >= self.convergence_threshold
        ):
            self._converged = True
            self._conv_time = now - self._t_task

    def __repr__(self) -> str:
        status = "active" if self._active else f"silent (rank={self.rank})"
        return (
            f"MetricsLogger({status}, "
            f"hw_interval={self.hw_sample_interval}, "
            f"conv_threshold={self.convergence_threshold})"
        )
