"""
03_dataset_statistics.py
Statistik lengkap dataset fault.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = ROOT / "data" / "processed" / "fault_dataset.csv"

print("=" * 60)
print("  03. STATISTIK DATASET FAULT")
print("=" * 60)

if not DATASET_CSV.exists():
    print(f"[ERROR] File tidak ditemukan: {DATASET_CSV}")
    sys.exit(1)

df = pd.read_csv(DATASET_CSV)

print(f"\n=== Ukuran Dataset ===")
print(f"Total baris       : {len(df):,}")
print(f"Total skenario    : {df['scenario_id'].nunique()}")
print(f"Total time steps  : {df['time_step'].nunique()}")
print(f"Total strings     : {df['string'].nunique()}")
print(f"Total node/string : {df['node'].nunique()}")
print(f"Ukuran file       : {DATASET_CSV.stat().st_size / 1e6:.1f} MB")

print(f"\n=== Distribusi Skenario ===")
meta = df.groupby('scenario_id').first()['fault_type_name']
print(meta.value_counts().to_string())

print(f"\n=== Distribusi Baris per Jenis ===")
counts = df['fault_type_name'].value_counts()
for ft, n in counts.items():
    print(f"  {ft:<8}: {n:>7,} ({100*n/len(df):.2f}%)")

print(f"\n=== Baris Fault Aktif ===")
n_fault = int(df['fault_active'].sum())
print(f"Total fault aktif : {n_fault:,} ({100*df['fault_active'].mean():.2f}%)")
fault_rows = df[df['fault_active'] == 1]
print(f"\nDistribusi fault aktif per jenis:")
fault_counts = fault_rows['fault_type_name'].value_counts()
for ft, n in fault_counts.items():
    print(f"  {ft:<8}: {n:>7,}")

print(f"\n=== Range Nilai per Jenis Fault ===")
for ft in ['normal', 'LLF', 'LGF', 'PSC']:
    sub = df[df['fault_type_name'] == ft]
    if len(sub) == 0:
        continue
    print(f"\n{ft}:")
    print(f"  V: [{sub['V'].min():.2f}; {sub['V'].max():.2f}] "
          f"mean={sub['V'].mean():.2f} std={sub['V'].std():.2f}")
    print(f"  I: [{sub['I'].min():.3f}; {sub['I'].max():.3f}] "
          f"mean={sub['I'].mean():.3f} std={sub['I'].std():.3f}")
    print(f"  P: [{sub['P'].min():.2f}; {sub['P'].max():.2f}] "
          f"mean={sub['P'].mean():.2f} std={sub['P'].std():.2f}")

print(f"\n=== Contoh Metadata Tiap Jenis ===")
for ft in ['normal', 'LLF', 'LGF', 'PSC']:
    sub = df[df['fault_type_name'] == ft]
    if len(sub) == 0:
        continue
    row = sub.iloc[0]
    print(f"\n{ft} (scenario {row['scenario_id']}):")
    print(f"  onset          : {row['onset']}")
    print(f"  duration       : {row['duration']}")
    print(f"  affected_nodes : {row['affected_nodes']}")
    print(f"  magnitude      : {row['magnitude']:.4f}")
    print(f"  gradual        : {row['gradual']}")