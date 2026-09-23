"""
05_physics_consistency.py
Cek konsistensi fisis dataset fault.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = ROOT / "data" / "processed" / "fault_dataset.csv"

print("=" * 60)
print("  05. CEK KONSISTENSI FISIS DATASET")
print("=" * 60)

df = pd.read_csv(DATASET_CSV)

print(f"\n[1] Konsistensi P = V * I")
df['P_calc'] = df['V'] * df['I']
diff = (df['P'] - df['P_calc']).abs()
print(f"  Max |P - V*I|  : {diff.max():.6f}")
print(f"  Mean |P - V*I| : {diff.mean():.6f}")
print(f"  Status         : {'OK' if diff.max() < 1e-3 else 'CEK'}")

print(f"\n[2] Nilai Negatif")
n_vneg = (df['V'] < 0).sum()
n_ineg = (df['I'] < 0).sum()
n_pneg = (df['P'] < 0).sum()
print(f"  V < 0 : {n_vneg}")
print(f"  I < 0 : {n_ineg}")
print(f"  P < 0 : {n_pneg}")
ok = (n_vneg == 0 and n_ineg == 0 and n_pneg == 0)
print(f"  Status: {'OK' if ok else 'CEK'}")

print(f"\n[3] Missing Values (NaN)")
print(f"  NaN V : {df['V'].isna().sum()}")
print(f"  NaN I : {df['I'].isna().sum()}")
print(f"  NaN P : {df['P'].isna().sum()}")
ok = df[['V','I','P']].isna().sum().sum() == 0
print(f"  Status: {'OK' if ok else 'CEK'}")

print(f"\n[4] Konsistensi fault_active dengan onset + duration")
n_ok, n_bad = 0, 0
for sid in df[df['fault_type_name'] != 'normal']['scenario_id'].unique()[:20]:
    sub = df[df['scenario_id'] == sid]
    onset = sub['onset'].iloc[0]
    duration = sub['duration'].iloc[0]
    active_rows = sub[sub['fault_active'] == 1]['time_step'].unique()
    if len(active_rows) > 0:
        if len(active_rows) == duration:
            n_ok += 1
        else:
            n_bad += 1
            print(f"  Scenario {sid}: active={len(active_rows)}, "
                  f"duration={duration}")
print(f"  Diperiksa: {n_ok + n_bad}, cocok: {n_ok}, tidak: {n_bad}")

print(f"\n[5] Range Nilai Fisis")
print(f"  V : {df['V'].min():.3f} - {df['V'].max():.3f} V")
print(f"  I : {df['I'].min():.3f} - {df['I'].max():.3f} A")
print(f"  P : {df['P'].min():.3f} - {df['P'].max():.3f} W")

print(f"\n[6] Batas Arus LLF")
llf_max = df[df['fault_type_name'] == 'LLF']['I'].max()
isc = 10.23
limit = (3 - 1) * isc * 1.2
print(f"  I_max LLF   : {llf_max:.3f} A")
print(f"  Batas fisis : {limit:.3f} A")
print(f"  Status      : {'OK' if llf_max < limit else 'CEK'}")

print(f"\n[7] Distribusi Noise Sensor")
for ft in ['normal', 'LLF', 'LGF', 'PSC']:
    sub = df[df['fault_type_name'] == ft]
    if len(sub) > 0:
        cv_v = sub['V'].std() / sub['V'].mean()
        cv_i = sub['I'].std() / sub['I'].mean()
        print(f"  {ft:<8}: CV(V)={cv_v:.3f}, CV(I)={cv_i:.3f}")