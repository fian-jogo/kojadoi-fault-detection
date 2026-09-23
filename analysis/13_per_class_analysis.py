"""
13_per_class_analysis.py
Analisis per-kelas fault.
"""
import sys, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS_JSON = ROOT / "logs" / "final_evaluation.json"

print("=" * 60)
print("  13. ANALISIS PER-KELAS FAULT")
print("=" * 60)

with open(METRICS_JSON) as f:
    m = json.load(f)

print(f"\n=== Performa per Kelas ===")
df = pd.DataFrame(m['per_class'])
print(df[['fault_type', 'n_windows', 'n_fault_active',
          'f1', 'precision', 'recall']].to_string(index=False))

print(f"\n=== Breakdown Detail per Kelas ===")
for r in m['per_class']:
    if r['fault_type'] == 'normal':
        continue
    print(f"\n{r['fault_type']}:")
    print(f"  Total window : {r['n_windows']}")
    print(f"  Fault aktif  : {r['n_fault_active']}")
    print(f"  F1           : {r['f1']:.4f}")
    print(f"  Precision    : {r['precision']:.4f}")
    print(f"  Recall       : {r['recall']:.4f}")

print(f"\n=== Ranking Performa ===")
per_class = [r for r in m['per_class'] if r['fault_type'] != 'normal']
per_class_sorted = sorted(per_class, key=lambda x: -x['f1'])
for i, r in enumerate(per_class_sorted, 1):
    print(f"  {i}. {r['fault_type']:<5} F1 = {r['f1']:.4f}")

print(f"\n=== Analisis Jenis Fault ===")
for r in per_class_sorted:
    if r['f1'] > 0.9:
        cat = "Sangat Baik"
    elif r['f1'] > 0.7:
        cat = "Baik"
    elif r['f1'] > 0.5:
        cat = "Cukup"
    else:
        cat = "Perlu Perbaikan"
    print(f"  {r['fault_type']:<5}: {cat}")