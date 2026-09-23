"""
10_latency_analysis.py
Analisis latency deteksi per skenario.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LATENCY_CSV = ROOT / "logs" / "latency_per_scenario.csv"

print("=" * 60)
print("  10. ANALISIS LATENCY DETEKSI")
print("=" * 60)

df = pd.read_csv(LATENCY_CSV)

print(f"\n=== Statistik Umum ===")
print(f"Total skenario fault : {len(df)}")
print(f"Terdeteksi           : {df['detected'].sum()}")
print(f"Tidak terdeteksi     : {(~df['detected']).sum()}")
print(f"Detection rate       : {100*df['detected'].mean():.1f}%")

print(f"\n=== Latency Statistik (hanya yang terdeteksi) ===")
det = df[df['detected']]
if len(det) > 0:
    print(f"  Count   : {len(det)}")
    print(f"  Mean    : {det['latency'].mean():.1f}")
    print(f"  Median  : {det['latency'].median():.1f}")
    print(f"  Std     : {det['latency'].std():.1f}")
    print(f"  Min     : {det['latency'].min()}")
    print(f"  Max     : {det['latency'].max()}")
    print(f"  P25     : {det['latency'].quantile(0.25):.1f}")
    print(f"  P75     : {det['latency'].quantile(0.75):.1f}")
    print(f"  P90     : {det['latency'].quantile(0.90):.1f}")

print(f"\n=== Distribusi per Jenis Fault ===")
for ft in ['LLF', 'LGF', 'PSC']:
    sub = df[df['fault_type'] == ft]
    det_sub = sub[sub['detected']]
    if len(sub) > 0:
        print(f"\n{ft}:")
        print(f"  Total skenario  : {len(sub)}")
        print(f"  Terdeteksi      : {len(det_sub)} ({100*len(det_sub)/len(sub):.1f}%)")
        if len(det_sub) > 0:
            print(f"  Latency median  : {det_sub['latency'].median():.1f}")
            print(f"  Latency mean    : {det_sub['latency'].mean():.1f}")
            print(f"  Latency max     : {det_sub['latency'].max()}")

print(f"\n=== Top-5 Latency Tertinggi ===")
top5 = det.nlargest(5, 'latency')
print(top5[['scenario_id', 'fault_type', 'onset', 'detected_at',
            'latency']].to_string(index=False))

print(f"\n=== Skenario yang Tidak Terdeteksi ===")
missed = df[~df['detected']]
if len(missed) > 0:
    print(missed[['scenario_id', 'fault_type', 'onset']].to_string(index=False))
else:
    print("  Tidak ada (semua terdeteksi)")