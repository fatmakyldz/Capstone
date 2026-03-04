"""
İki-pasajlı (two-pass) eğitim adımı — nested_learning pipeline'ının görüntü uyarlaması.

ADIM SIRASI (nested_learning referansları):
─────────────────────────────────────────────────────────────────────────────
Pass-1  →  nested_learning/training.py  ana döngü (satır 918-984)
           model(tokens)  →  logits  (meta forward, grad aktif)

Teach   →  nested_learning/training.py  compute_teach_signal()  (satır 225)
signal     Orijinal: ∂CE_LM/∂pre_norm  (B,T,dim)
           Burada:   ∂CE_img/∂features (B,512)   — aynı prensip, farklı domain

Pass-2  →  nested_learning/memorize.py  memorize_tokens()  (satır 295-307)
           model(tokens, teach_signal=teach_signal, fast_state=fast_state)
           Burada:   model.update_fast(features, teach)
           fast_memory.net ağırlıkları bu adımda güncellenir, başka hiçbir şey değişmez.

Meta    →  nested_learning/training.py  (satır 928, 984)
update     loss.backward()  +  optimizer.step()
           SADECE backbone + classifier (fast_memory.net hariç)

─────────────────────────────────────────────────────────────────────────────

KRİTİK KURAL: Pass-2 ve meta update arasındaki sıra değiştirilebilir.
Burada meta update ÖNCE yapılır (loss grafiği bozulmadan), ardından Pass-2.
Böylece loss.backward() Tihe Pass-1 grafiğini kullanır.

Test invariantları (tests/test_two_pass.py'de doğrulanır):
  - Pass-2 sonrası backbone.net parametreleri değişmemiş olmalı
  - Pass-2 sonrası classifier.weight değişmemiş olmalı
  - Pass-2 yalnızca fast_memory.net parametrelerini değiştirir
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from models.continual_model import ContinualModel, compute_teach_signal
from training.loss import compute_loss, ewc_penalty, mask_old_class_grads
from training.sigreg import SIGRegProjector, sigreg_loss


def train_one_step(
    model: ContinualModel,
    cur_images: Tensor,             # (B_cur, C, H, W) — current task batch
    cur_labels: Tensor,             # (B_cur,)
    rep_images: Optional[Tensor],   # (B_rep, C, H, W) — replay batch veya None
    rep_labels: Optional[Tensor],   # (B_rep,)
    optimizer: Optimizer,
    current_class_ids: List[int],
    device: torch.device,
    # Optional regularization
    ewc_fisher: Optional[Dict[str, Tensor]] = None,
    ewc_means: Optional[Dict[str, Tensor]] = None,
    ewc_lambda: float = 0.01,
    current_weight: float = 1.0,
    replay_weight: float = 1.0,
    # Teach signal control
    run_teach_signal: bool = True,
    # ── SIGReg: covariance isotropy regularization ───────────────────────────
    # projector: SIGRegProjector instance (MLP 512→512→256 + L2 normalize).
    # None → SIGReg disabled.  Applied to ALL features (current + replay).
    projector: Optional[SIGRegProjector] = None,
    lambda_sig: float = 0.05,          # SIGReg loss weight
    # ── LwF knowledge distillation ───────────────────────────────────────────
    # teacher_model: frozen deepcopy of model from end of previous task.
    # None for Task 0 (no distillation).  Teacher stays on the same device
    # as the student and must already have requires_grad=False for all params.
    teacher_model: Optional[nn.Module] = None,
    old_class_count: int = 0,      # number of classes seen before this task
    lambda_kd: float = 1.5,        # KD loss weight
    kd_temperature: float = 2.0,   # softmax temperature for KD
) -> Dict[str, float]:
    """
    Tek bir eğitim adımı — tüm bileşenleri koordine eder.

    Dönüş: {'loss': float, 'cur_loss': float, 'ewc_loss': float, 'kd_loss': float, 'sig_loss': float}
    """
    model.train()

    # ── Batch birleştir ───────────────────────────────────────────────────────
    # current + replay aynı forward pass'tan geçer (verimlilik için)
    if rep_images is not None and rep_labels is not None:
        images = torch.cat([cur_images, rep_images], dim=0)
        labels = torch.cat([cur_labels, rep_labels],  dim=0)
    else:
        images = cur_images
        labels = cur_labels
    n_current = cur_images.size(0)

    # ═══════════════════════════════════════════════════════════════════════════
    # PASS-1: Meta forward (gradyan aktif — backbone + classifier için)
    # ═══════════════════════════════════════════════════════════════════════════
    # Teach signal için features'ı saklıyoruz.
    # model.forward() → (logits, features) döndürür.
    logits, features = model(images)   # features: (B, 512), logits: (B, num_classes)

    # ── LwF knowledge distillation loss ───────────────────────────────────────
    # Applied only when a teacher exists (task ≥ 1) and there are old classes.
    # Teacher sees the SAME combined batch (current + replay) so its logits
    # match the same images the student is being optimised on.
    # KD is masked to old classes only: new-class logits are excluded so we do
    # not force the student's new-task output to match the teacher's random
    # predictions for classes it has never seen.
    kd_loss = torch.tensor(0.0, device=device)
    if teacher_model is not None and old_class_count > 0 and lambda_kd > 0.0:
        with torch.no_grad():
            teacher_logits, _ = teacher_model(images)   # (B, num_classes), no grad

        T = kd_temperature
        logits_old        = logits[:, :old_class_count]          # student — grad flows
        teacher_old       = teacher_logits[:, :old_class_count]  # teacher — detached

        p_teacher         = torch.softmax(teacher_old / T, dim=-1)
        log_p_student     = torch.log_softmax(logits_old / T, dim=-1)

        # KLDiv(input=log_probs, target=probs, reduction="batchmean") * T²
        # T² rescales the gradient magnitude back to the original scale.
        kd_loss = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)

    # ── SIGReg loss ───────────────────────────────────────────────────────────
    # features = backbone avgpool output (512-dim) — NOT fast_memory output.
    # continual_model.forward() returns (logits, backbone_features), so this is safe.
    #
    # Replay ağırlığı: replay örnekleri YENİ backbone'dan geçer ama ESKİ task'ı
    # temsil eder. Eşit ağırlık kovaryans matrisini yanlış dağılıma kilitler.
    # Çözüm: current örnekler tam ağırlık (1.0), replay örnekler yarı ağırlık (0.5).
    # Böylece SIGReg ağırlıklı olarak mevcut task'ın feature space'ini stabilize eder,
    # replay distribution'a overfit olmaz.
    sig_loss = torch.tensor(0.0, device=device)
    if projector is not None and lambda_sig > 0.0:
        feat_cur_sig = features[:n_current]
        feat_rep_sig = features[n_current:]
        sig_loss = sigreg_loss(feat_cur_sig, projector)
        if feat_rep_sig.size(0) > 0:
            sig_loss = sig_loss + 0.5 * sigreg_loss(feat_rep_sig, projector)

    # ── CE + EWC loss ─────────────────────────────────────────────────────────
    ce_loss = compute_loss(
        logits, labels, n_current,
        current_weight=current_weight,
        replay_weight=replay_weight,
    )
    ewc_loss = torch.tensor(0.0, device=device)
    if ewc_fisher and ewc_means and ewc_lambda > 0:
        ewc_loss = ewc_lambda * ewc_penalty(model, ewc_fisher, ewc_means)

    total_loss = ce_loss + ewc_loss + lambda_kd * kd_loss + lambda_sig * sig_loss

    # ── Meta backward + optimizer ─────────────────────────────────────────────
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()

    # Classifier gradient masking: current task + replay sınıfları güncellenir.
    # Sadece current_class_ids kullanmak → eski sınıf head'lerinin ASLA güncellenmemesi
    # demektir (replay gradientı da sıfırlanır) → backbone evrimleşir ama head'ler
    # donup kalır → Task 0 accuracy %0'a düşer. Düzeltme: replay label'larından
    # gelen sınıfları da izin verilenler listesine ekle.
    if rep_labels is not None:
        _allowed = list(set(current_class_ids) | set(rep_labels.tolist()))
    else:
        _allowed = current_class_ids
    mask_old_class_grads(model.classifier, _allowed, device)

    # Gradient clipping (kararlılık için)
    # Projector parametreleri meta-optimizer'a dahil, birlikte clip edilir.
    params_to_clip = list(model.meta_parameters())
    if projector is not None:
        params_to_clip += list(projector.parameters())
    torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)

    optimizer.step()

    # FIX 4: fast_memory params are in the Pass-1 computation graph and receive
    # gradients from total_loss.backward(), but they are NOT in the optimizer's
    # param groups so optimizer.zero_grad() never clears them.  Without this,
    # stale gradients accumulate across steps and corrupt _update_fast_weights.
    for p in model.fast_memory.fast_params():
        p.grad = None

    # ═══════════════════════════════════════════════════════════════════════════
    # PASS-2: Teach signal hesapla → Fast memory güncelle
    # ═══════════════════════════════════════════════════════════════════════════
    # Meta loss.backward() bitti, grafik serbest bırakıldı.
    # Artık features üzerinden teach signal hesaplayabilir,
    # fast_memory.net'i güncelleyebiliriz — backbone/classifier dokunulmaz.
    #
    # Nested_learning referansı:
    #   memorize_tokens() satır 329-359:
    #     teach_signal = compute_teach_signal(model, logits, token_batch)
    #     model(token_batch, teach_signal=teach_signal, fast_state=fast_state)
    if run_teach_signal:
        with torch.no_grad():
            # Teach signal ve fast memory update SADECE current örnekler üzerinde.
            # Replay örnekleri meta loss'ta kullanılır ama fast memory'ye yazılmaz.
            # Gerekçe: fast memory yalnızca yeni görevin anlık özelliklerini
            # memorize etmeli; replay örneklerini yazmak eski task bilgisini
            # bozar (replay = stabilite aracı, fast memory = plastisite aracı).
            # nested_learning referansı: memorize_tokens() yalnızca current
            # token'ları fast_state'e yazar, replay tokens hariç.
            feat_cur   = features[:n_current]
            logits_cur = logits[:n_current]
            labels_cur = labels[:n_current]

            teach = compute_teach_signal(
                features=feat_cur,
                logits=logits_cur,
                labels=labels_cur,
                classifier=model.classifier,
            )
            # Pass-2: fast_memory ağırlıklarını güncelle (backbone, classifier değişmez)
            model.update_fast(features=feat_cur, teach_signal=teach)

    return {
        "loss":     float(total_loss.item()),
        "cur_loss": float(ce_loss.item()),
        "ewc_loss": float(ewc_loss.item()),
        "kd_loss":  float(kd_loss.item()),
        "sig_loss": float(sig_loss.item()),
    }


def evaluate_task(
    model: ContinualModel,
    dataloader,
    device: torch.device,
    bic_alpha: float = 1.0,
    bic_beta: float = 0.0,
    bic_old_count: int = 0,
) -> float:
    """
    Tek bir task'ın test doğruluğunu döndürür (%).

    BiC (Bias Correction) parametreleri:
      bic_old_count > 0 ise eski sınıfların logitlerine affine düzeltme uygulanır:
        logit_old = bic_alpha * logit_old + bic_beta
      Bu, classifier'ın yeni sınıflara olan sistematik logit bias'ını dengeler.
      Sadece evaluation'da uygulanır; eğitime dokunmaz.

    Wu et al. 2019 "Large Scale Incremental Learning" BiC formülü.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)

            # ── BiC: affine correction for old-class logits ───────────────────
            if bic_old_count > 0:
                old_logits = bic_alpha * logits[:, :bic_old_count] + bic_beta
                new_logits = logits[:, bic_old_count:]
                logits = torch.cat([old_logits, new_logits], dim=1)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0
