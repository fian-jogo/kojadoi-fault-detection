"""
07_baseline_metrics.py
Ringkasan metrik baseline LSTM.
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_JSON = ROOT / "logs" / "lstm_baseline_metrics.json"

print("=" * 60)
print("  07. RINGKASAN BASELINE LSTM")
print("=" * 60)

if not BASELINE_JSON.exists():
    print(f"[ERROR] File tidak ditemukan: {BASELINE_JSON}")
    sys.exit(1)

with open(BASELINE_JSON) as f:
    m = json.load(f)

print(f"\n=== Config ===")
for k, v in m['config'].items():
    print(f"  {k}: {v}")

print(f"\n=== Split ===")
for k, v in m['split'].items():
    print(f"  {k}: {v}")

print(f"\n=== Test Metrics ===")
t = m['test']
print(f"  Accuracy  : {t['accuracy']:.4f}")
print(f"  Precision : {t['precision']:.4f}")
print(f"  Recall    : {t['recall']:.4f}")
print(f"  F1        : {t['f1']:.4f}")
print(f"  TP/FP/FN/TN: {t['tp']}/{t['fp']}/{t['fn']}/{t['tn']}")

print(f"\n=== Localization ===")
loc = m['localization']
print(f"  Top-1 Accuracy : {loc['top1_accuracy']:.4f}")
print(f"  Top-3 Accuracy : {loc['top3_accuracy']:.4f}")
print(f"  Mean IoU       : {loc['mean_iou']:.4f}")
print(f"  N Fault Windows: {loc['n_fault_windows']}")

print(f"\n=== Training History (ringkas) ===")
if 'history' in m:
    h = m['history']
    print(f"  Final train loss : {h['train_loss'][-1]:.4f}")
    print(f"  Final val loss   : {h['val_loss'][-1]:.4f}")
    print(f"  Best val F1      : {max(h['val_f1']):.4f} "
          f"(epoch {h['val_f1'].index(max(h['val_f1']))+1})")
    print(f"  Best val acc     : {max(h['val_acc']):.4f}")