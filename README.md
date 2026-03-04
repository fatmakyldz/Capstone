# capstone3 — HOPE-Style Continual Learning

A PyTorch implementation of HOPE-style continual learning on CIFAR-10 and CIFAR-100, inspired by the [nested\_learning](https://github.com/google-deepmind/nested_learning) reference implementation.

---

## Project Overview

Continual learning addresses **catastrophic forgetting**: when a neural network is trained on a sequence of tasks, it tends to overwrite knowledge from earlier tasks as it learns new ones. This project implements a two-pass training loop combining:

- **Fast memory** (inspired by CMS / Titans) — updated only by a teach signal, not backprop
- **Replay buffer** — class-balanced reservoir sampling across all seen tasks
- **Knowledge distillation (LwF)** — previous model acts as a teacher for old classes
- **EWC regularization** — penalises large changes to important parameters
- **SIGReg** — covariance isotropy regularization preventing feature collapse
- **BiC correction** — affine logit correction to remove new-class bias

---

## What is HOPE-Style Training?

HOPE uses a **two-pass** update rule inspired by nested\_learning:

```
Pass 1 (Meta forward):
  features = backbone(x)                       # ResNet18 → 512-dim
  logits   = fast_memory(features)             # fast gate + classifier
  meta_loss = CE(logits, labels) + EWC + KD + SIGReg
  meta_loss.backward()
  optimizer.step()   # updates backbone + classifier only

Teach signal (closed-form, no autograd):
  p = softmax(logits);  p[labels] -= 1;  p /= B
  teach = -(p @ W_classifier)               # ∂CE/∂features, negated

Pass 2 (Fast memory update):
  model.update_fast(features.detach(), teach)  # fast_memory.net only
```

The key insight: **backbone learns slowly via meta-gradient; fast memory adapts instantly via teach signal, acting as a per-step working memory.**

---

## Setup

```bash
cd capstone3
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install torch torchvision psutil pynvml
```

---

## How to Run

### CIFAR-10

```bash
# Quick smoke-test (3 tasks, small buffer):
python train.py --dataset cifar10 --num_tasks 3 --epochs_per_task 5 --memory_size 600

# Full experiment (5 tasks × 2 classes):
python train.py --dataset cifar10 --num_tasks 5 --epochs_per_task 10 --memory_size 2000

# With EMA teacher + learned BiC (best result configuration):
python train.py --dataset cifar10 --num_tasks 5 --epochs_per_task 10 \
    --memory_size 2000 --ema_teacher --learned_bic --no_reset_optim

# With EWC regularization:
python train.py --dataset cifar10 --num_tasks 5 --epochs_per_task 10 \
    --use_ewc --ewc_lambda 0.02
```

### CIFAR-100

```bash
# 5 tasks × 20 classes (num_classes and classes_per_task auto-derived):
python train.py --dataset cifar100 --num_tasks 5 --epochs_per_task 10 --memory_size 20000

# 10 tasks × 10 classes:
python train.py --dataset cifar100 --num_tasks 10 --epochs_per_task 10 --memory_size 20000

# With all regularizers:
python train.py --dataset cifar100 --num_tasks 5 --epochs_per_task 10 \
    --memory_size 20000 --ema_teacher --learned_bic --no_reset_optim \
    --lambda_kd 1.5 --lambda_sig 0.05
```

### Ablations

```bash
# Disable teach signal (fast memory becomes passive):
python train.py --no_teach_signal

# Disable knowledge distillation:
python train.py --lambda_kd 0.0

# Disable SIGReg:
python train.py --lambda_sig 0.0

# Preserve optimizer state across tasks (reduces feature drift):
python train.py --no_reset_optim

# Freeze backbone (only classifier + fast memory train):
python train.py --freeze_backbone
```

---

## Main Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `cifar10` | Dataset: `cifar10` or `cifar100` |
| `--num_tasks` | `5` | Number of tasks |
| `--epochs_per_task` | `5` | Epochs trained per task |
| `--classes_per_task` | auto | Classes per task. Default: `num_classes // num_tasks` |
| `--memory_size` | `600` | Total replay buffer capacity (examples) |
| `--batch_size` | `64` | Total batch size (current + replay) |
| `--replay_ratio` | `0.4` | Fraction of batch reserved for replay |
| `--optimizer` | `adam` | Optimizer: `adam`, `adamw`, `muon` |
| `--backbone_lr` | `1e-4` | Learning rate for backbone |
| `--classifier_lr` | `1e-3` | Learning rate for classifier + projector |
| `--lambda_kd` | `1.5` | Knowledge distillation weight (0 = off) |
| `--lambda_sig` | `0.05` | SIGReg covariance regularization weight (0 = off) |
| `--use_ewc` | off | Enable EWC regularization |
| `--ewc_lambda` | `0.01` | EWC penalty weight |
| `--ema_teacher` | off | Update teacher via EMA each step instead of snapshot |
| `--ema_alpha` | `0.999` | EMA decay: `teacher = α·teacher + (1-α)·student` |
| `--learned_bic` | off | Optimize BiC α,β on replay buffer post-task |
| `--no_reset_optim` | off | Preserve optimizer state across task boundaries |
| `--no_reset_backbone_optim` | off | Preserve backbone state only; reset classifier |
| `--no_teach_signal` | off | Ablation: disable fast memory teach signal |
| `--freeze_backbone` | off | Freeze all backbone weights |
| `--results_dir` | `./results` | Output directory |
| `--convergence_threshold` | None | Val accuracy (%) at which `convergence_time_s` is stamped |
| `--hw_sample_interval` | `20` | Sample GPU/CPU metrics every N steps |
| `--seed` | `42` | Random seed |

---

## Output Files

```
results/
├── metrics.csv                         ← task-level summary (one row per task)
└── {dataset}_t{N}_e{E}_m{M}_{timestamp}/
    ├── config.json                     ← full experiment configuration
    ├── metrics.json                    ← accuracy matrix + forgetting scores
    └── perf_steps.csv                  ← step-level performance (loss, FPS, GPU, ...)
```

### `metrics.csv` columns

| Column | Description |
|---|---|
| `task_id` | Task index (0-based) |
| `epoch` | Last epoch of the task |
| `task_train_time_seconds` | Wall-clock seconds spent training this task |
| `cumulative_train_time_seconds` | Total wall-clock seconds from start |
| `accuracy_per_task` | JSON list of val accuracies for all seen tasks at this point |
| `avg_accuracy_final` | Mean accuracy over all seen tasks |
| `avg_forgetting` | Mean forgetting: `best_acc[i] − current_acc[i]` |
| `fps` | Training throughput: `total_images / task_train_time_seconds` |
| `gpu_utilization_percent` | NVML GPU SM occupancy (CUDA only) |
| `cpu_utilization_percent` | Host CPU utilization (psutil) |
| `memory_usage_mb` | GPU memory allocated (CUDA) or host RAM used (CPU/MPS) |

### `perf_steps.csv` columns

Step-level data: `task_id`, `epoch`, `global_step`, `row_type`, `train_loss`, `val_accuracy`, `total_elapsed_s`, `task_elapsed_s`, `epoch_time_s`, `step_time_s`, `convergence_time_s`, `throughput_img_s`, `fps`, `gpu_util_pct`, `gpu_mem_used_mb`, `gpu_mem_peak_mb`, `cpu_util_pct`, `ram_used_mb`.

Two `row_type` values:
- `step` — one row per training step (train_loss, throughput, HW metrics at interval)
- `epoch_end` — one row per epoch end (epoch_time, val_accuracy when available)

---

## Metrics Explained

### avg\_accuracy\_final
Mean test accuracy across **all tasks** evaluated at the end of training:
```
avg_accuracy_final = mean(acc[task_i] for i in 0..N-1)
```
Higher is better. Reflects the model's ability to retain all learned tasks simultaneously.

### avg\_forgetting
Mean performance drop per task from its peak to the end of training:
```
forgetting[i] = max_acc[i] − final_acc[i]
avg_forgetting = mean(forgetting[i] for i in 0..N-1)
```
Lower is better. A forgetting of 0% means the model perfectly retained all tasks.

### fps (throughput)
Training speed in images per second:
```
fps = total_images_in_task / task_train_time_seconds
```
Measures end-to-end throughput including forward pass, backward pass, fast memory update, and replay sampling.

### gpu\_utilization\_percent
GPU SM (streaming multiprocessor) occupancy sampled via `pynvml`. Values near 100% indicate the GPU is fully utilised. Only available on CUDA.

### cpu\_utilization\_percent
Host CPU utilisation sampled via `psutil.cpu_percent()`. High values during training may indicate data loading bottlenecks.

### memory\_usage\_mb
On CUDA: `torch.cuda.memory_allocated()` — PyTorch-allocated GPU VRAM in MB.
On MPS/CPU: host RAM used (`psutil.virtual_memory().used`).

---

## Project Structure

```
capstone3/
├── train.py                    # Main entry point
├── continual/
│   └── task_manager.py         # Dataset loading + dynamic task splitting
├── models/
│   ├── backbone.py             # ResNet18 (CIFAR-adapted: conv1=3×3, no maxpool)
│   ├── fast_memory.py          # CMS-inspired fast memory (teach signal update)
│   └── continual_model.py      # Backbone + FastMemory + Classifier + 2-pass API
├── training/
│   ├── engine.py               # Two-pass training step + evaluation
│   ├── loss.py                 # CE + EWC loss + gradient masking
│   ├── sigreg.py               # SIGReg projector + covariance isotropy loss
│   └── metrics_logger.py       # CSV performance logger (step-level + task-level)
├── memory/
│   └── replay_buffer.py        # Class-balanced reservoir sampling
├── evaluation/
│   └── metrics.py              # N×N accuracy matrix + forgetting metrics
├── optim/
│   └── muon.py                 # Muon optimizer (Newton-Schulz orthogonalization)
├── configs/
│   └── default.yaml            # Default configuration
└── tests/                      # Unit tests
```

---

## Reference Mapping (nested\_learning → capstone3)

| nested\_learning | capstone3 |
|---|---|
| `compute_teach_signal()` | `models/continual_model.py:compute_teach_signal()` |
| `memorize_tokens()` | `training/engine.py:train_one_step()` pass-2 block |
| `Titan.update()` | `models/fast_memory.py:FastMemory._update_fast_weights()` |
| `CMSBlock` | `models/fast_memory.py:FastMemory` |
| `build_streaming_tasks()` | `continual/task_manager.py:build_tasks()` |
| `ContinualEvalResult` | `evaluation/metrics.py:ContinualMetrics` |
| `m3.py` optimizer | `optim/muon.py:MuonMeta` |
