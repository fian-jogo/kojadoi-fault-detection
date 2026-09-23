"""
14_control_log_verification.py
Verifikasi log state machine kontrol.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG_CSV = ROOT / "data" / "interface" / "control_log.csv"
PRED_JSON = ROOT / "data" / "interface" / "predictions.json"

print("=" * 60)
print("  14. VERIFIKASI LOG STATE MACHINE")
print("=" * 60)

if not LOG_CSV.exists():
    print(f"[ERROR] File tidak ditemukan: {LOG_CSV}")
    sys.exit(1)

df = pd.read_csv(LOG_CSV)

print(f"\n=== Ukuran Log ===")
print(f"Total time step : {len(df)}")

print(f"\n=== Distribusi State ===")
print(df['state_name'].value_counts().to_string())

print(f"\n=== Distribusi Alarm Level ===")
print(df['alarm_level'].value_counts().sort_index().to_string())

print(f"\n=== Waktu dalam Setiap State ===")
for s in ['NORMAL', 'DETECTING', 'ISOLATING', 'ISOLATED', 'RECOVERING']:
    count = (df['state_name'] == s).sum()
    if count > 0:
        print(f"  {s:<11}: {count:>3} step ({100*count/len(df):.1f}%)")

print(f"\n=== Breaker Status ===")
n_open = int(df['breaker'].sum())
print(f"Breaker OPEN : {n_open} step ({100*n_open/len(df):.1f}%)")

print(f"\n=== Baris di Mana Breaker OPEN ===")
iso = df[df['breaker'] == 1]
if len(iso) > 0:
    print(iso[['t', 'fault_prob', 'state_name', 'isolated_string',
               'action']].head(10).to_string(index=False))

print(f"\n=== Total Isolasi ===")
n_iso = int((np.diff(np.concatenate([[0], df['breaker'].values])) == 1).sum())
print(f"Total isolasi : {n_iso} kali")
if n_iso > 0:
    first_iso_idx = np.where(np.diff(np.concatenate([[0], df['breaker'].values])) == 1)[0]
    print(f"Isolasi pertama pada time step: {int(first_iso_idx[0])+1}")

print(f"\n=== String yang Diisolasi ===")
iso_strings = df[df['isolated_string'] >= 0]['isolated_string'].unique()
if len(iso_strings) > 0:
    print(f"String terisolasi: {iso_strings.tolist()}")

if PRED_JSON.exists():
    import json
    with open(PRED_JSON) as f:
        pred = json.load(f)
    print(f"\n=== Verifikasi dengan predictions.json ===")
    print(f"Scenario ID  : {pred['scenario_id']}")
    print(f"N steps      : {pred['n_steps']}")
    print(f"N nodes      : {pred['n_nodes']}")
    print(f"Fault aktif  : {sum(pred['ground_truth_fault'])}")
    print(f"Rata-rata prob: {np.mean(pred['fault_prob']):.4f}")