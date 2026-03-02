"""
3 ablation karşılaştırması:
  A) ResNet + Replay only        (--no_teach_signal)
  B) ResNet + Replay + EWC       (--no_teach_signal --use_ewc)
  C) ResNet + Replay + FastMemory two-pass  (default)

Her deney aynı seed ile çalıştırılır.
Sonuçlar results/ dizinine JSON olarak kaydedilir ve özet tablo yazdırılır.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

PYTHON = sys.executable
TRAIN  = os.path.join(os.path.dirname(__file__), "train.py")

COMMON = [
    "--num_tasks",        "5",
    "--epochs_per_task",  "5",
    "--memory_size",      "1000",
    "--batch_size",       "64",
    "--backbone_lr",      "1e-4",
    "--classifier_lr",    "1e-3",
    "--replay_ratio",     "0.4",
    "--seed",             "42",
    "--results_dir",      "./results",
    "--log_interval",     "999",   # sessiz (log bastırma)
]

EXPERIMENTS = [
    {
        "name": "A_replay_only",
        "label": "ResNet + Replay",
        "extra": ["--no_teach_signal"],
    },
    {
        "name": "B_replay_ewc",
        "label": "ResNet + Replay + EWC",
        "extra": ["--no_teach_signal", "--use_ewc", "--ewc_lambda", "0.01"],
    },
    {
        "name": "C_replay_fastmem",
        "label": "ResNet + Replay + FastMemory",
        "extra": [],   # teach signal aktif (default)
    },
]


def run_one(exp: dict) -> dict:
    cmd = [PYTHON, TRAIN] + COMMON + exp["extra"]
    print(f"\n{'='*60}")
    print(f"  DENEY: {exp['label']}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [HATA] Deney başarısız: {exp['name']}")
        return {"name": exp["name"], "label": exp["label"], "error": True}

    # Sonuç dosyasını oku
    tag = f"hope_t5_e5_m1000"
    if "--no_teach_signal" in exp["extra"]:
        tag += "_nots"
    if "--use_ewc" in exp["extra"]:
        tag += "_ewc"
    json_path = os.path.join("results", f"{tag}_metrics.json")
    if not os.path.exists(json_path):
        print(f"  [UYARI] Sonuç dosyası bulunamadı: {json_path}")
        return {"name": exp["name"], "label": exp["label"], "error": True}

    with open(json_path) as f:
        data = json.load(f)

    return {
        "name":              exp["name"],
        "label":             exp["label"],
        "avg_accuracy":      data.get("avg_accuracy_final", 0.0),
        "avg_forgetting":    data.get("avg_forgetting", 0.0),
        "task0_final":       data.get("task0_final_accuracy", 0.0),
        "elapsed_s":         round(elapsed, 1),
    }


def print_table(results: list[dict]) -> None:
    print("\n" + "="*70)
    print("  KARŞILAŞTIRMA TABLOSU")
    print("="*70)
    header = f"{'Yöntem':<35} {'AvgAcc':>8} {'Forget':>8} {'Task0':>8} {'Süre(s)':>9}"
    print(header)
    print("-"*70)
    for r in results:
        if r.get("error"):
            print(f"  {r['label']:<33}  HATA")
            continue
        print(
            f"  {r['label']:<33}"
            f"  {r['avg_accuracy']:>6.1f}%"
            f"  {r['avg_forgetting']:>6.1f}%"
            f"  {r['task0_final']:>6.1f}%"
            f"  {r['elapsed_s']:>7.0f}s"
        )
    print("="*70)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    all_results = []
    for exp in EXPERIMENTS:
        r = run_one(exp)
        all_results.append(r)

    print_table(all_results)

    # JSON kaydet
    summary_path = "results/comparison.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Özet kaydedildi: {summary_path}")
