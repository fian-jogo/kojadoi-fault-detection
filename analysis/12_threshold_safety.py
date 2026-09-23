"""
12_threshold_safety.py
Analisis threshold untuk aplikasi keselamatan.
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_JSON = ROOT / "logs" / "final_evaluation.json"

print("=" * 60)
print("  12. ANALISIS THRESHOLD UNTUK KESELAMATAN")
print("=" * 60)

with open(METRICS_JSON) as f:
    m = json.load(f)

print(f"\n{'Threshold':>10}{'F1':>8}{'Prec':>8}{'Rec':>8}{'Acc':>8}  Catatan")
print("-" * 65)
for s in m['threshold_sweep']:
    marker = ""
    if abs(s['threshold'] - 0.5) < 0.001:
        marker = "<- default"
    if s['recall'] >= 0.75 and s['threshold'] in [0.20, 0.25, 0.30]:
        marker += "  <- kandidat keselamatan"
    print(f"  {s['threshold']:>8.2f}{s['f1']:>8.4f}{s['precision']:>8.4f}"
          f"{s['recall']:>8.4f}{s['accuracy']:>8.4f}  {marker}")

print(f"\n=== Rekomendasi Threshold ===")
best_f1 = m['best_threshold']['f1']
print(f"\nBest F1:")
print(f"  Threshold : {best_f1['threshold']:.2f}")
print(f"  F1        : {best_f1['f1']:.4f}")
print(f"  Recall    : {best_f1['recall']:.4f}")
print(f"  Precision : {best_f1['precision']:.4f}")

if m['best_threshold']['high_recall']:
    hr = m['best_threshold']['high_recall']
    print(f"\nHigh Recall (>= 0.75):")
    print(f"  Threshold : {hr['threshold']:.2f}")
    print(f"  F1        : {hr['f1']:.4f}")
    print(f"  Recall    : {hr['recall']:.4f}")
    print(f"  Precision : {hr['precision']:.4f}")

print(f"\n=== Rekomendasi per Skenario Aplikasi ===")
print(f"  Keselamatan (prioritas recall)    : threshold 0.20-0.25")
print(f"  Operasional (prioritas precision) : threshold 0.50-0.65")
print(f"  Seimbang                          : threshold 0.35-0.45")