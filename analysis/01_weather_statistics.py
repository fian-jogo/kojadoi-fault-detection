"""
01_weather_statistics.py
Statistik data iradiansi dan suhu NASA POWER.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "raw" / "koja_doi_irradiance_temperature_hourly.csv"

print("=" * 60)
print("  01. STATISTIK DATA NASA POWER")
print("=" * 60)

if not CSV.exists():
    print(f"[ERROR] File tidak ditemukan: {CSV}")
    sys.exit(1)

df = pd.read_csv(CSV, parse_dates=['datetime_local'])

print(f"\n=== Rentang Waktu ===")
print(f"Dari    : {df['datetime_local'].min()}")
print(f"Sampai  : {df['datetime_local'].max()}")
print(f"Baris   : {len(df)}")

print(f"\n=== Statistik Deskriptif ===")
print(df[['ALLSKY_SFC_SW_DWN', 'T2M']].describe().round(3).to_string())

print(f"\n=== Missing Values ===")
missing = df[['ALLSKY_SFC_SW_DWN', 'T2M']].isna().sum()
print(missing.to_string())
print(f"Total missing: {missing.sum()}")

print(f"\n=== Iradiansi Rata-rata per Jam (WITA) ===")
hourly = df.groupby(df['datetime_local'].dt.hour)['ALLSKY_SFC_SW_DWN'].mean().round(1)
print(hourly.to_string())
peak_hour = hourly.idxmax()
print(f"\nJam puncak: {peak_hour}:00 WITA ({hourly.max():.1f} W/m2)")

print(f"\n=== Distribusi Siang vs Malam ===")
n_night = (df['ALLSKY_SFC_SW_DWN'] < 1.0).sum()
print(f"Malam (G < 1)  : {n_night} ({100*n_night/len(df):.1f}%)")
print(f"Siang (G >= 1) : {len(df)-n_night} ({100*(len(df)-n_night)/len(df):.1f}%)")

print(f"\n=== Range Suhu ===")
print(f"Min  : {df['T2M'].min():.2f} °C")
print(f"Max  : {df['T2M'].max():.2f} °C")
print(f"Mean : {df['T2M'].mean():.2f} °C")
print(f"Std  : {df['T2M'].std():.2f} °C")