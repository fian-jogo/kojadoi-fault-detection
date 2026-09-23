"""
11_latency_vs_severity.py
Analisis korelasi latency deteksi dengan severity fault.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LATENCY_CSV = ROOT / "logs" / "latency_per_scenario.csv"
DATASET_CSV = ROOT / "data" / "processed" / "fault_dataset.csv"

print("=" * 60)
print("  11. ANALISIS LATENCY vs SEVERITY")
print("=" * 60)

lat = pd.read_csv(LATENCY_CSV)
meta = pd.read_csv(DATASET_CSV)
meta = meta[meta['time_step'] == 0][
    ['scenario_id', 'magnitude', 'duration']].drop_duplicates()

det = lat[lat['detected']]
det_meta = det.merge(meta, on='scenario_id', how='left')

print(f"\n=== Korelasi ===")
corr = det_meta[['latency', 'magnitude', 'duration']].corr()
print(corr.round(3).to_string())

print(f"\n=== Latency per Jenis Fault dengan Severity ===")
for ft in ['LLF', 'LGF', 'PSC']:
    sub = det_meta[det_meta['fault_type'] == ft]
    if len(sub) > 0:
        print(f"\n{ft}:")
        print(f"  n              : {len(sub)}")
        print(f"  Latency median : {sub['latency'].median():.1f}")
        print(f"  Latency mean   : {sub['latency'].mean():.1f}")
        print(f"  Magnitude range: {sub['magnitude'].min():.2f} - "
              f"{sub['magnitude'].max():.2f}")
        print(f"  Duration range : {sub['duration'].min()}-"
              f"{sub['duration'].max()} step")

print(f"\n=== Kesimpulan ===")
if len(det_meta) > 1:
    corr_lat_mag = det_meta['latency'].corr(det_meta['magnitude'])
    corr_lat_dur = det_meta['latency'].corr(det_meta['duration'])
    print(f"  Korelasi latency vs magnitude : {corr_lat_mag:+.3f}")
    print(f"  Korelasi latency vs duration  : {corr_lat_dur:+.3f}")
    print(f"  Catatan: korelasi ini didorong oleh perbedaan antar-jenis fault")