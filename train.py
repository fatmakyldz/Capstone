"""
capstone3 — Ana eğitim giriş noktası.

Kullanım:
  # Hızlı smoke-test (3 task, küçük buffer):
  cd capstone3
  python train.py

  # Tam deney (5 task, büyük buffer):
  python train.py --num_tasks 5 --epochs_per_task 10 --memory_size 2000

  # Ablation: teach signal kapalı
  python train.py --no_teach_signal

  # Backbone dondurulmuş
  python train.py --freeze_backbone

  # EWC aktif
  python train.py --use_ewc --ewc_lambda 0.02

Pipeline (nested_learning referansları ile):
  1. Task döngüsü             → continual_streaming.py for current_task, task in ...
  2. Train step (2-pass)      → memorize_tokens() + training.py ana döngü
  3. Replay buffer güncelle   → balanced reservoir (capstone2 update_memory_bank)
  4. Evaluate tüm task'lar    → continual_streaming.py _eval_task()
  5. Metrikleri kaydet        → ContinualEvalResult yapısı
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Proje kökünü path'e ekle
sys.path.insert(0, os.path.dirname(__file__))

from continual.task_manager import TaskManager, build_cifar10_tasks
from evaluation.metrics import ContinualMetrics
from memory.replay_buffer import ReplayBuffer
from models.continual_model import ContinualModel
from training.engine import evaluate_task, train_one_step
from training.loss import compute_ewc_fisher, ewc_penalty
from training.sigreg import SIGRegProjector


# ─────────────────────────────────────────────────────────────────────────────
# 1. Argümanlar
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="capstone3 — HOPE-style Continual Learning")

    # Task / Veri
    p.add_argument("--data_dir",         type=str,   default="./data")
    p.add_argument("--num_tasks",        type=int,   default=3,
                   help="Kaç task çalıştırılacak (max 5, her task 2 sınıf)")
    p.add_argument("--classes_per_task", type=int,   default=2)

    # Model
    p.add_argument("--num_classes",           type=int,   default=10)
    p.add_argument("--feature_dim",           type=int,   default=512)
    p.add_argument("--freeze_backbone",       action="store_true")
    p.add_argument("--freeze_backbone_until", type=int, default=0,
                   help="Freeze backbone children[0..N-1]; 0=off, 4=stem only, "
                        "5=+layer1, 6=+layer2, 7=+layer3, 8=all")
    p.add_argument("--pretrained_backbone",   action="store_true")
    p.add_argument("--fast_hidden_multiplier",type=int,   default=4)
    p.add_argument("--fast_lr",               type=float, default=0.01)
    p.add_argument("--fast_grad_clip",        type=float, default=1.0)

    # Optimizer
    p.add_argument("--backbone_lr",   type=float, default=1e-4)
    p.add_argument("--classifier_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=5e-4)
    p.add_argument("--optimizer",     type=str,   default="adam",
                   choices=["adam", "adamw", "muon"])

    # Eğitim
    p.add_argument("--epochs_per_task", type=int,   default=5)
    p.add_argument("--batch_size",      type=int,   default=64)
    p.add_argument("--replay_ratio",    type=float, default=0.4)
    p.add_argument("--current_weight",  type=float, default=1.0)
    p.add_argument("--replay_weight",   type=float, default=1.0)

    # Replay
    p.add_argument("--memory_size", type=int, default=600)

    # EWC
    p.add_argument("--use_ewc",            action="store_true")
    p.add_argument("--ewc_lambda",         type=float, default=0.01)
    p.add_argument("--ewc_fisher_samples", type=int,   default=200)

    # Teach signal
    p.add_argument("--no_teach_signal", action="store_true",
                   help="Ablation: teach signal / fast update kapalı")

    # LwF knowledge distillation
    p.add_argument("--lambda_kd", type=float, default=1.5,
                   help="KD loss ağırlığı; 0.0 → LwF kapalı")
    p.add_argument("--kd_scale_per_task", action="store_true",
                   help="lambda_kd'yi task index ile ölçekle: λ_eff = λ * (task_id+1)")

    # Optimizer reset scope
    p.add_argument("--no_reset_backbone_optim", action="store_true",
                   help="Task sınırında backbone optimizer state'ini KORU, "
                        "sadece classifier+projector state'ini sıfırla. "
                        "Backbone Adam momentum'u korunduğunda ilk adımların "
                        "agresifliği azalır → feature drift düşer.")
    p.add_argument("--no_reset_optim", action="store_true",
                   help="Task sınırında optimizer state'ini HİÇ sıfırlama. "
                        "Backbone + classifier + projector state korunur. "
                        "Eski sınıf karar yüzeyi bozulmaz; forgetting daha düşük "
                        "ama yeni task plasticity azalabilir.")

    # EMA teacher
    p.add_argument("--ema_teacher", action="store_true",
                   help="Teacher'ı her adımda EMA ile güncelle (sabit snapshot yerine)")
    p.add_argument("--ema_alpha",  type=float, default=0.999,
                   help="EMA decay: teacher = α*teacher + (1-α)*student")

    # Learned BiC: optimize α,β post-task on replay buffer
    p.add_argument("--learned_bic", action="store_true",
                   help="BiC parametrelerini replay buffer üzerinde task sonunda öğren")
    p.add_argument("--bic_lr",       type=float, default=0.01)
    p.add_argument("--bic_steps",    type=int,   default=200)

    # SIGReg: covariance isotropy regularization
    p.add_argument("--lambda_sig", type=float, default=0.05,
                   help="SIGReg loss ağırlığı; 0.0 → SIGReg kapalı")

    # Loglama
    p.add_argument("--results_dir",  type=str, default="./results")
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--seed",         type=int, default=42)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Yardımcı fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ema_update(teacher: nn.Module, student: nn.Module, alpha: float) -> None:
    """
    Momentum encoder güncelleme: teacher = α*teacher + (1-α)*student
    Her optimizer step sonrasında çağrılır.

    BYOL/MoCo yaklaşımı: teacher sabit snapshot değil, öğrencinin
    eksponansiyel hareketli ortalaması — bu sayede distillation hedefi
    task boundary'de ani sıçrama yapmaz ve KD sinyali daha stabil kalır.

    alpha=0.999 tipik (her 1000 step'te ~bir kez teacher güncellenir ölçeğinde).
    """
    with torch.no_grad():
        for tp, sp in zip(teacher.parameters(), student.parameters()):
            tp.data.mul_(alpha).add_(sp.data, alpha=1.0 - alpha)


def learn_bic(
    model: nn.Module,
    buf,
    old_class_count: int,
    device: torch.device,
    lr: float = 0.01,
    n_steps: int = 200,
) -> tuple[float, float]:
    """
    BiC (Bias Correction) α,β parametrelerini replay buffer üzerinde öğren.

    Wu et al. 2019: old_logits = α * logits[:, :old] + β
    α ve β küçük val set üzerinde CE'yi minimize ederek bulunur.
    Sadece eski sınıflara ait örnekler kullanılır.

    Dönüş: (alpha, beta) — sabit BiC değerlerini ezer.
    Eğer buffer boş veya eski sınıf yoksa → (1.0, 0.0) döner.
    """
    if buf.is_empty() or old_class_count == 0:
        return 1.0, 0.0

    loader = buf.build_dataloader(batch_size=128)
    if loader is None:
        return 1.0, 0.0

    bic_alpha = nn.Parameter(torch.tensor(1.0, device=device))
    bic_beta  = nn.Parameter(torch.tensor(0.0, device=device))
    opt = torch.optim.SGD([bic_alpha, bic_beta], lr=lr, momentum=0.9)

    model.eval()
    step = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Sadece eski sınıf örnekleri
        mask = labels < old_class_count
        if not mask.any():
            continue
        images  = images[mask]
        labels  = labels[mask]

        with torch.no_grad():
            logits, _ = model(images)

        # evaluate_task() ile birebir eşleşen correction:
        # eski sınıf logitleri → α*logit+β, yeni sınıf logitleri → değişmez.
        # Tam logit vektörü üzerinden CE hesaplanır; bu sayede öğrenilen α,β
        # evaluation-time davranışıyla uyumlu olur.
        corrected_old  = bic_alpha * logits[:, :old_class_count] + bic_beta
        full_logits    = torch.cat([corrected_old, logits[:, old_class_count:]], dim=1)
        loss = F.cross_entropy(full_logits, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        step += 1
        if step >= n_steps:
            break

    return float(bic_alpha.item()), float(bic_beta.item())


def reset_classifier_state(optimizer: torch.optim.Optimizer,
                            backbone_params) -> None:
    """
    Backbone dışındaki parametrelerin optimizer state'ini sıfırla.

    Backbone Adam state'i (exp_avg, exp_avg_sq) korunur → task sınırında
    ilk adımlar yumuşak kalır, feature drift azalır.
    Classifier + projector state sıfırlanır → yeni task sınıfları için
    temiz başlangıç.

    --no_reset_backbone_optim flag'i ile etkinleşir.
    """
    backbone_ids = {id(p) for p in backbone_params}
    for pg in optimizer.param_groups:
        for p in pg["params"]:
            if id(p) not in backbone_ids and p in optimizer.state:
                optimizer.state[p] = {}


def build_optimizer(model: ContinualModel, args, projector=None) -> torch.optim.Optimizer:
    groups = model.meta_param_groups(
        backbone_lr=args.backbone_lr,
        classifier_lr=args.classifier_lr,
    )
    # SIGReg projector trained alongside backbone/classifier at classifier_lr
    if projector is not None:
        groups.append({
            "params": list(projector.parameters()),
            "lr": args.classifier_lr,
        })
    if args.optimizer == "muon":
        from optim.muon import MuonMeta
        return MuonMeta(groups, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    return torch.optim.Adam(groups, weight_decay=args.weight_decay)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ana eğitim döngüsü
# ─────────────────────────────────────────────────────────────────────────────

def run(args) -> ContinualMetrics:
    set_seed(args.seed)

    # Cihaz seçimi
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Setup] Cihaz: {device}")

    # ── Task listesi ──────────────────────────────────────────────────────────
    # nested_learning karşılığı: build_streaming_tasks()
    tasks = build_cifar10_tasks(
        data_dir=args.data_dir,
        num_tasks=args.num_tasks,
        classes_per_task=args.classes_per_task,
    )
    task_manager = TaskManager(tasks)

    # Test loader'ları önceden hazırla (tüm tasklar için)
    all_test_loaders = {
        t.task_id: DataLoader(t.test_dataset, batch_size=256, shuffle=False, num_workers=0)
        for t in tasks
    }

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ContinualModel(
        num_classes=args.num_classes,
        feature_dim=args.feature_dim,
        freeze_backbone=args.freeze_backbone,
        freeze_backbone_until=args.freeze_backbone_until,
        pretrained_backbone=args.pretrained_backbone,
        fast_hidden_multiplier=args.fast_hidden_multiplier,
        fast_lr=args.fast_lr,
        fast_grad_clip=args.fast_grad_clip,
    ).to(device)
    print(f"[Model] Toplam parametre: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Model] Meta parametre  : {sum(p.numel() for p in model.meta_parameters()):,}")
    print(f"[Model] Fast parametre  : {sum(p.numel() for p in model.fast_memory.fast_params()):,}")

    # ── SIGReg projector ──────────────────────────────────────────────────────
    # MLP(512→512→256, GELU) + L2 normalize; None if lambda_sig == 0.
    projector = None
    if args.lambda_sig > 0.0:
        projector = SIGRegProjector(input_dim=args.feature_dim).to(device)
        print(f"[SIGReg] Projector aktif (λ={args.lambda_sig}) | "
              f"Parametre: {sum(p.numel() for p in projector.parameters()):,}")

    # ── Partial freeze summary ─────────────────────────────────────────────────
    frozen_params    = sum(p.numel() for p in model.backbone.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters()          if     p.requires_grad)
    if args.freeze_backbone or args.freeze_backbone_until > 0:
        frozen_layers = (
            "all" if args.freeze_backbone
            else f"children[0..{args.freeze_backbone_until - 1}]"
        )
        print(f"[Freeze] Frozen layers   : {frozen_layers}  ({frozen_params:,} params)")
        print(f"[Freeze] Trainable params: {trainable_params:,}")

    optimizer = build_optimizer(model, args, projector=projector)

    # ── Replay buffer ─────────────────────────────────────────────────────────
    buf = ReplayBuffer(memory_size=args.memory_size, seed=args.seed)

    # ── Metrik takipçisi ──────────────────────────────────────────────────────
    # nested_learning karşılığı: ContinualEvalResult
    metrics = ContinualMetrics(num_tasks=args.num_tasks)

    # EWC state
    ewc_fisher, ewc_means = {}, {}

    # LwF teacher snapshot (None for Task 0; set at end of each task)
    teacher_model = None

    # BiC state: old-class count used during evaluation (updated after each task)
    # Starts at 0 — Task 0 evaluation has no old classes to correct.
    # --learned_bic: α,β optimized on replay buffer post-task (replaces fixed values).
    # --learned_bic OFF: fixed affine correction (shrinks inflated old-class logits).
    bic_old_count = 0
    BIC_A = 1.0    # learned_bic replaces these; fixed fallback if learned_bic off
    BIC_B = 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # Task döngüsü
    # nested_learning karşılığı: continual_streaming.py satır 208
    #   for current_task, task in enumerate(tasks):
    # ══════════════════════════════════════════════════════════════════════════
    for task_id in range(args.num_tasks):
        current_task = task_manager.advance()
        class_ids = current_task.class_ids
        print(f"\n{'='*55}")
        print(f"  Task {task_id}  |  Sınıflar: {class_ids}")
        print(f"{'='*55}")

        # ── Optimizer reset at every task boundary ────────────────────────────
        if task_id == 0:
            # Task 0: her zaman sıfırdan başla (state yok zaten)
            optimizer = build_optimizer(model, args, projector=projector)
            print(f"  [Optimizer] full reset for Task {task_id}")
        elif args.no_reset_optim:
            # Tüm state korunur — backbone + classifier + projector.
            # Eski sınıf karar yüzeyi bozulmaz, forgetting minimum.
            print(f"  [Optimizer] no reset for Task {task_id} "
                  f"(all state preserved)")
        elif args.no_reset_backbone_optim:
            # Backbone state korunur (drift azaltır), sadece classifier/projector sıfırlanır
            reset_classifier_state(optimizer, model.backbone.parameters())
            print(f"  [Optimizer] classifier-only reset for Task {task_id} "
                  f"(backbone state preserved)")
        else:
            # Varsayılan: tam reset (muon momentum bias'ını önler)
            optimizer = build_optimizer(model, args, projector=projector)
            print(f"  [Optimizer] full reset for Task {task_id}")

        # Fast memory intentionally NOT reset at task boundary.
        # nested_learning maintains fast_state across the stream for cumulative adaptation.
        # Resetting here killed cross-task forward transfer — removed.
        # (Ablation: add --reset_fast_memory flag if you want to test reset behavior)

        # Batch boyutu ayrımı: current vs replay
        if task_id == 0 or buf.is_empty():
            cur_bs = args.batch_size
        else:
            cur_bs = max(1, int(args.batch_size * (1.0 - args.replay_ratio)))

        train_loader = DataLoader(
            current_task.train_dataset,
            batch_size=cur_bs,
            shuffle=True,
            num_workers=0,
        )

        # ── Epoch döngüsü ─────────────────────────────────────────────────────
        for epoch in range(args.epochs_per_task):
            model.train()
            running_loss = 0.0
            n_batches = 0

            for batch_i, (cur_imgs, cur_lbls) in enumerate(train_loader):
                cur_imgs = cur_imgs.to(device)
                cur_lbls = cur_lbls.to(device)

                # Replay batch (Task-0'dan sonra)
                rep_imgs, rep_lbls = None, None
                if not buf.is_empty():
                    rep_sample = buf.sample_batch(
                        batch_size=args.batch_size - cur_bs, device=device
                    )
                    if rep_sample is not None:
                        rep_imgs, rep_lbls = rep_sample

                # ── İKİ-PASAJ EĞİTİM ADIMI ────────────────────────────────
                # nested_learning: Pass-1 forward + compute_teach_signal +
                #                  Pass-2 fast update + meta backward
                # Dynamic lambda_kd: artan task sayısıyla KD sinyalini güçlendir
                # task 0→1: λ*1, task 1→2: λ*2, ...  Eski sınıf sayısı arttıkça
                # distillation daha kritik hale gelir.
                effective_kd = (
                    args.lambda_kd * (task_id + 1)
                    if args.kd_scale_per_task
                    else args.lambda_kd
                )

                result = train_one_step(
                    model=model,
                    cur_images=cur_imgs,
                    cur_labels=cur_lbls,
                    rep_images=rep_imgs,
                    rep_labels=rep_lbls,
                    optimizer=optimizer,
                    current_class_ids=class_ids,
                    device=device,
                    ewc_fisher=ewc_fisher if args.use_ewc else None,
                    ewc_means=ewc_means  if args.use_ewc else None,
                    ewc_lambda=args.ewc_lambda,
                    current_weight=args.current_weight,
                    replay_weight=args.replay_weight,
                    run_teach_signal=not args.no_teach_signal,
                    # LwF: teacher is None for task 0, active from task 1 onward
                    teacher_model=teacher_model,
                    old_class_count=task_id * args.classes_per_task,
                    lambda_kd=effective_kd,
                    # SIGReg: covariance isotropy regularization
                    projector=projector,
                    lambda_sig=args.lambda_sig,
                )

                # ── EMA teacher update (per step) ─────────────────────────────
                # teacher = α*teacher + (1-α)*student
                # Bu, task boundary'de teacher'ın ani sıçrama yapmasını engeller.
                # Sabit snapshot yerine öğrencinin hareketli ortalamasını distill eder.
                if args.ema_teacher and teacher_model is not None:
                    ema_update(teacher_model, model, alpha=args.ema_alpha)

                running_loss += result["loss"]
                n_batches += 1

                if (batch_i + 1) % args.log_interval == 0:
                    avg = running_loss / n_batches
                    print(f"  Epoch {epoch+1}/{args.epochs_per_task} "
                          f"| Batch {batch_i+1} | Loss: {avg:.4f}")

            avg_epoch_loss = running_loss / max(n_batches, 1)
            print(f"  ✓ Epoch {epoch+1}/{args.epochs_per_task} tamamlandı "
                  f"| Ortalama Loss: {avg_epoch_loss:.4f}")

        # ── LwF: teacher snapshot — frozen copy of model at end of this task ────
        # Used as the distillation target for the NEXT task.
        # Created here (after all epochs) so the teacher captures the model
        # after it has fully converged on the current task.
        # Teacher is moved to the same device and all gradients disabled.
        teacher_model = deepcopy(model)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        teacher_model.to(device)

        # ── Replay buffer güncelle (task bittikten SONRA) ─────────────────────
        # nested_learning: task öğrenildikten sonra memorize adımı
        buf.add_task_data(
            dataset=current_task.train_dataset,
            task_class_ids=class_ids,
        )
        print(f"  Replay buffer: {buf}")

        # ── Learned BiC: α,β parametrelerini replay buffer üzerinde optimize et ─
        # Wu et al. 2019: val set üzerinde α,β öğrenerek logit bias'ı düzelt.
        # Eski sınıf logitleri sistematik olarak küçülüp büyüdüğünden,
        # sabit bir affine düzeltme yerine veri odaklı optimizasyon daha etkili.
        # Task 0 sonrası: eski sınıf yok → (1.0, 0.0) döner, etkisiz.
        old_class_count_for_bic = task_id * args.classes_per_task
        if args.learned_bic and old_class_count_for_bic > 0:
            BIC_A, BIC_B = learn_bic(
                model=model,
                buf=buf,
                old_class_count=old_class_count_for_bic,
                device=device,
                lr=args.bic_lr,
                n_steps=args.bic_steps,
            )
            print(f"  [LearnedBiC] α={BIC_A:.4f}  β={BIC_B:.4f}")

        # ── EWC: Fisher matrisi güncelle ───────────────────────────────────────
        if args.use_ewc:
            ewc_loader = buf.build_dataloader(batch_size=args.batch_size)
            if ewc_loader is not None:
                ewc_fisher, ewc_means = compute_ewc_fisher(
                    model=model,
                    dataloader=ewc_loader,
                    device=device,
                    n_samples=args.ewc_fisher_samples,
                )
                print(f"  EWC Fisher güncellendi ({len(ewc_fisher)} parametre)")

        # ── Değerlendirme: tüm görülen task'lar ──────────────────────────────
        # nested_learning karşılığı: evaluate_continual_classification()
        #   for task_idx in range(current_task + 1): task_acc[task_idx][current_task] = ...
        # ── BiC: update old-class count for this evaluation round ────────────
        # After training task_id the newest classes are task_id's pair.
        # Old classes = everything before them.  bic_old_count held its
        # previous value (= task_id * classes_per_task) through training;
        # it is updated to include this task's classes AFTER evaluation so
        # the NEXT task's evaluation corrects the right slice.
        print(f"\n  Değerlendirme (Task 0 - {task_id}):")
        for prev_tid in range(task_id + 1):
            acc = evaluate_task(
                model, all_test_loaders[prev_tid], device,
                bic_alpha=BIC_A, bic_beta=BIC_B, bic_old_count=bic_old_count,
            )
            metrics.record(task_idx=prev_tid, time_step=task_id, accuracy=acc)
            print(f"    Task {prev_tid}: {acc:.2f}%")

        # Task-0 hatırlama eğrisi
        metrics.record_task0(
            evaluate_task(
                model, all_test_loaders[0], device,
                bic_alpha=BIC_A, bic_beta=BIC_B, bic_old_count=bic_old_count,
            )
        )

        # bic_old_count: Task_id'nin sınıfları artık "eski" sayılır — sonraki
        # task'tan itibaren bu task'ın logitleri de düzeltme kapsamına girer.
        bic_old_count = (task_id + 1) * args.classes_per_task
        print(f"  [BiC] old_class_count → {bic_old_count}")

        # Apple MPS bellek temizliği
        if device.type == "mps":
            torch.mps.empty_cache()

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sonuçları kaydet
# ─────────────────────────────────────────────────────────────────────────────

def save_results(metrics: ContinualMetrics, args) -> None:
    os.makedirs(args.results_dir, exist_ok=True)

    # JSON
    tag = f"hope_t{args.num_tasks}_e{args.epochs_per_task}_m{args.memory_size}"
    if args.no_teach_signal:
        tag += "_nots"
    if args.use_ewc:
        tag += "_ewc"

    json_path = os.path.join(args.results_dir, f"{tag}_metrics.json")
    metrics.save(json_path)

    # Argümanlar da kaydedilsin
    cfg_path = os.path.join(args.results_dir, f"{tag}_config.json")
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[Kayıt] Config: {cfg_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Giriş noktası
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    print("\n" + "=" * 55)
    print("  capstone3 — HOPE-style Continual Learning")
    print("=" * 55)
    print(f"  Tasks        : {args.num_tasks} × {args.classes_per_task} sınıf")
    print(f"  Epochs/task  : {args.epochs_per_task}")
    print(f"  Memory size  : {args.memory_size}")
    print(f"  Teach signal : {'KAPALI (ablation)' if args.no_teach_signal else 'AÇIK'}")
    print(f"  EWC          : {'AÇIK' if args.use_ewc else 'KAPALI'}")
    print(f"  LwF (KD)     : {'AÇIK λ=' + str(args.lambda_kd) + (' (task-scaled)' if args.kd_scale_per_task else '') if args.lambda_kd > 0 else 'KAPALI'}")
    print(f"  EMA teacher  : {'AÇIK α=' + str(args.ema_alpha) if args.ema_teacher else 'KAPALI'}")
    print(f"  Learned BiC  : {'AÇIK' if args.learned_bic else 'KAPALI (sabit α=1,β=0)'}")
    print(f"  SIGReg       : {'AÇIK λ=' + str(args.lambda_sig) if args.lambda_sig > 0 else 'KAPALI'}")
    print(f"  Freeze bb    : {args.freeze_backbone}")
    print("=" * 55 + "\n")

    metrics = run(args)

    # Özet
    metrics.print_matrix()
    metrics.print_summary()

    # Kaydet
    save_results(metrics, args)
