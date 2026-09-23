"""
04_metadata_verification.py
Verifikasi metadata JSON dan konsistensinya dengan dataset.
"""
import sys, json
from pathlib import Path
import pandas as pd
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
METADATA_JSON = ROOT / "data" / "processed" / "fault_metadata.json"
DATASET_CSV = ROOT / "data" / "processed" / "fault_dataset.csv"

print("=" * 60)
print("  04. VERIFIKASI METADATA JSON")
print("=" * 60)

if not METADATA_JSON.exists():
    print(f"[ERROR] File tidak ditemukan: {METADATA_JSON}")
    sys.exit(1)

with open(METADATA_JSON) as f:
    meta = json.load(f)

print(f"\nTotal entri: {len(meta)}")

print(f"\n=== Contoh Metadata Tiap Jenis ===")
for ft in ['normal', 'LLF', 'LGF', 'PSC']:
    for m in meta:
        if m['fault_type'] == ft:
            print(f"\n--- {ft} ---")
            print(json.dumps(m, indent=2))
            break

print(f"\n=== Verifikasi Field Wajib ===")
required = ['scenario_id', 'fault_type', 'onset', 'duration',
            'affected_nodes', 'magnitude', 'gradual', 'n_steps']
missing = [(m['scenario_id'], f) for m in meta for f in required if f not in m]
if not missing:
    print("  Semua field lengkap: OK")
else:
    print(f"  Field hilang: {missing}")

print(f"\n=== Distribusi Jenis Fault ===")
types = [m['fault_type'] for m in meta]
print(Counter(types))

print(f"\n=== Range Onset dan Duration ===")
onsets = [m['onset'] for m in meta if m['onset'] >= 0]
durations = [m['duration'] for m in meta if m['duration'] > 0]
if onsets:
    print(f"Onset   : min={min(onsets)}, max={max(onsets)}, "
          f"mean={sum(onsets)/len(onsets):.1f}")
if durations:
    print(f"Duration: min={min(durations)}, max={max(durations)}, "
          f"mean={sum(durations)/len(durations):.1f}")

print(f"\n=== Verifikasi Konsistensi dengan Dataset ===")
if DATASET_CSV.exists():
    df = pd.read_csv(DATASET_CSV)
    df_first = df[df['time_step'] == 0].drop_duplicates('scenario_id')
    n_match = 0
    n_mismatch = 0
    for m in meta:
        sid = m['scenario_id']
        row = df_first[df_first['scenario_id'] == sid]
        if len(row) == 0:
            n_mismatch += 1
            continue
        row = row.iloc[0]
        if row['fault_type_name'] == m['fault_type'] and \
           int(row['onset']) == int(m['onset']):
            n_match += 1
        else:
            n_mismatch += 1
    print(f"Cocok      : {n_match}")
    print(f"Tidak cocok: {n_mismatch}")