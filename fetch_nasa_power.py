"""
fetch_nasa_power.py
Ambil data iradiansi (ALLSKY_SFC_SW_DWN) dan suhu udara (T2M) per jam
untuk koordinat PLTS Koja Doi dari NASA POWER API.
Output: koja_doi_irradiance_temperature_hourly.csv
"""
import os
import requests
import pandas as pd

# ----------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------
OUTPUT_DIR = r"D:\Berkas Penting\JOBS\AEER\kojadoi-fault-detection\data\raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "koja_doi_irradiance_temperature_hourly.csv")

LATITUDE = -8.496
LONGITUDE = 122.399
START = "20250101"
END = "20250630"

# NASA POWER hourly selalu UTC. NTT = WITA = UTC+8.
LOCAL_TZ_OFFSET_HOURS = 8

# ----------------------------------------------------------------
# 1. Request ke NASA POWER
# ----------------------------------------------------------------
url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
params = {
    "parameters": "ALLSKY_SFC_SW_DWN,T2M",
    "community": "RE",
    "longitude": LONGITUDE,
    "latitude": LATITUDE,
    "start": START,
    "end": END,
    "format": "JSON",
}

print("Mengunduh data dari NASA POWER ...")
r = requests.get(url, params=params, timeout=60)
r.raise_for_status()
data = r.json()

# ----------------------------------------------------------------
# 2. Ambil parameter
# ----------------------------------------------------------------
parameters = data["properties"]["parameter"]
irradiance = parameters["ALLSKY_SFC_SW_DWN"]
temperature = parameters["T2M"]

df = pd.DataFrame({
    "ALLSKY_SFC_SW_DWN": pd.Series(irradiance),
    "T2M": pd.Series(temperature),
})

# ----------------------------------------------------------------
# 3. Parse timestamp (format YYYYMMDDHH, UTC)
# ----------------------------------------------------------------
df = df.sort_index()
df.index = pd.to_datetime(df.index, format="%Y%m%d%H", utc=True)
df.index.name = "datetime_utc"

# ----------------------------------------------------------------
# 4. Konversi UTC → WITA (UTC+8)
# ----------------------------------------------------------------
df.index = df.index + pd.Timedelta(hours=LOCAL_TZ_OFFSET_HOURS)
df.index.name = "datetime_local"

# ----------------------------------------------------------------
# 5. Ganti missing value NASA POWER (-999) menjadi NaN, lalu interpolasi
# ----------------------------------------------------------------
df = df.replace(-999.0, pd.NA).replace(-999, pd.NA)
df = df.interpolate(method="linear", limit_direction="both")

# Bulatkan agar rapi
df["ALLSKY_SFC_SW_DWN"] = df["ALLSKY_SFC_SW_DWN"].round(4)
df["T2M"] = df["T2M"].round(2)

# ----------------------------------------------------------------
# 6. Simpan
# ----------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_csv(OUTPUT_FILE, encoding="utf-8")

print(f"✅ Data tersimpan di: {OUTPUT_FILE}")
print(f"Total baris: {len(df)}")
print(f"Jumlah missing setelah interpolasi: {df.isna().sum().sum()}")
print(f"Rentang waktu (lokal WITA): {df.index.min()} s.d. {df.index.max()}")
print("\nCuplikan 5 baris pertama:")
print(df.head().to_string())