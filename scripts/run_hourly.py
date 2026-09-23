"""
run_hourly.py
Menjalankan model PV pada data iradiansi & suhu per jam untuk PLTS Koja Doi.

Konfigurasi sistem:
    - 3 string paralel
    - 4 modul seri per string
    - Modul monocrystalline 400 Wp
    - Kapasitas total: 4.800 Wp

Input:
    data/raw/koja_doi_irradiance_temperature_hourly.csv
    (dari fetch_nasa_power.py)

Output:
    data/processed/normal_condition_timeseries.csv
    figures/timeseries_7days.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# Deteksi ROOT otomatis: folder induk dari scripts/
# ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print(f"[debug] ROOT = {ROOT}")

# ----------------------------------------------------------------
# Muat parameter modul: prioritaskan hasil tuning
# ----------------------------------------------------------------
from src.pv_model import ModuleParams, analyze_string

try:
    from src.pv_model_tuned import TUNED_PARAMS as params
    print("[info] Memakai parameter hasil tuning dari src/pv_model_tuned.py")
    print(f"       R_s = {params.R_s:.4f} Ω | "
          f"R_sh = {params.R_sh:.2f} Ω | "
          f"I_o_ref = {params.I_o_ref:.3e} A")
except ImportError:
    params = ModuleParams()
    print("[warning] src/pv_model_tuned.py tidak ditemukan.")
    print("          Memakai parameter default dari ModuleParams().")
    print("          Jalankan validate_iv.py terlebih dahulu untuk tuning.")


# ----------------------------------------------------------------
# Konfigurasi sistem (proof-of-concept)
# ----------------------------------------------------------------
N_SERIES = 4       # 4 modul seri per string
N_STRINGS = 3      # 3 string paralel
CAPACITY_WP = N_SERIES * N_STRINGS * params.pdc0   # 4.800 Wp

INPUT_CSV = ROOT / "data" / "raw" / "koja_doi_irradiance_temperature_hourly.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "normal_condition_timeseries.csv"
FIGURE_PATH = ROOT / "figures" / "timeseries_7days.png"


# ----------------------------------------------------------------
# Cari file input di beberapa lokasi kandidat (fallback)
# ----------------------------------------------------------------
CANDIDATE_INPUTS = [
    INPUT_CSV,
    ROOT / "koja_doi_irradiance_temperature_hourly.csv",
    ROOT.parent / "koja_doi_irradiance_temperature_hourly.csv",
]
for candidate in CANDIDATE_INPUTS:
    if candidate.exists():
        INPUT_CSV = candidate
        break


def sanity_check(df: pd.DataFrame) -> None:
    """Cetak peringatan jika data tampak tidak wajar."""
    peak = df['P_mpp_total_W'].max()
    cf = 100.0 * df['P_mpp_total_W'].sum() / 1000.0 / (CAPACITY_WP * len(df) / 1000.0)

    print("\n--- Sanity check ---")
    if peak > CAPACITY_WP * 1.02:
        print(f"  [WARNING] Puncak daya {peak:.1f} W > kapasitas {CAPACITY_WP} Wp.")
        print("            Kemungkinan parameter modul belum dituning.")
    else:
        print(f"  Puncak daya {peak:.1f} W <= kapasitas {CAPACITY_WP} Wp. OK.")

    if cf < 12 or cf > 18:
        print(f"  [WARNING] Capacity factor {cf:.2f}% di luar rentang tipikal "
              f"PLTS NTT (12–18%).")
    else:
        print(f"  Capacity factor {cf:.2f}% dalam rentang wajar. OK.")

    # Cek pola harian: puncak iradiansi harus di sekitar jam 12–13 WITA
    if 'datetime_local' in df.columns:
        df_tmp = df.copy()
        df_tmp['hour'] = pd.to_datetime(df_tmp['datetime_local']).dt.hour
        peak_hour = df_tmp.groupby('hour')['G_Wm2'].mean().idxmax()
        if 10 <= peak_hour <= 14:
            print(f"  Jam puncak iradiansi rata-rata: {peak_hour}:00 WITA. OK.")
        else:
            print(f"  [WARNING] Jam puncak iradiansi rata-rata: {peak_hour}:00 WITA.")
            print("            Seharusnya di sekitar 12–13 WITA. Cek konversi timezone.")


def plot_first_week(df: pd.DataFrame, out_path: Path) -> None:
    """Plot 7 hari pertama untuk verifikasi visual."""
    df = df.copy()
    df['datetime_local'] = pd.to_datetime(df['datetime_local'])
    start = df['datetime_local'].min()
    end = start + pd.Timedelta(days=7)
    week = df[(df['datetime_local'] >= start) & (df['datetime_local'] < end)]

    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    ax1.plot(week['datetime_local'], week['P_mpp_total_W'],
             color='tab:blue', lw=1.5, label='Daya AC (W)')
    ax1.set_xlabel('Waktu (WITA)')
    ax1.set_ylabel('Daya (W)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(week['datetime_local'], week['G_Wm2'],
             color='tab:orange', lw=1.0, alpha=0.6, label='Iradiansi (W/m²)')
    ax2.set_ylabel('Iradiansi (W/m²)', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title(f'Daya PLTS {CAPACITY_WP/1000:.1f} kWp — 7 Hari Pertama')
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot 7 hari disimpan ke: {out_path}")


def main():
    # ------------------------------------------------------------
    # 1. Baca input
    # ------------------------------------------------------------
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            "File CSV tidak ditemukan di lokasi berikut:\n" +
            "\n".join(f"  - {p}" for p in CANDIDATE_INPUTS) +
            "\nJalankan fetch_nasa_power.py terlebih dahulu."
        )

    print(f"\n[info] Input CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, parse_dates=['datetime_local'])

    print(f"       Total baris  : {len(df)}")
    print(f"       Rentang waktu: {df['datetime_local'].min()} s.d. "
          f"{df['datetime_local'].max()}")
    print(f"       Kolom        : {list(df.columns)}")

    # ------------------------------------------------------------
    # 2. Iterasi per baris: hitung MPP string
    # ------------------------------------------------------------
    records = []
    n_night = 0
    n_day = 0

    for _, row in df.iterrows():
        G = float(row['ALLSKY_SFC_SW_DWN'])
        T = float(row['T2M'])

        # Sanity check nilai input
        if not np.isfinite(G) or G < 0:
            G = 0.0
        if not np.isfinite(T):
            T = 25.0

        # Malam: tidak ada produksi
        if G < 1.0:
            Vm = Im = Pm = 0.0
            n_night += 1
        else:
            Vm, Im, Pm = analyze_string(G, T, params, N_SERIES)
            n_day += 1

        records.append({
            'datetime_local':   row['datetime_local'],
            'G_Wm2':            round(G, 4),
            'T_cell_C':         round(T, 3),
            'V_mpp_string_V':   round(Vm, 4),
            'I_mpp_string_A':   round(Im, 4),
            'P_mpp_string_W':   round(Pm, 4),
            'P_mpp_total_W':    round(Pm * N_STRINGS, 4),
        })

    out = pd.DataFrame(records)

    # ------------------------------------------------------------
    # 3. Simpan CSV
    # ------------------------------------------------------------
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[info] CSV tersimpan: {OUTPUT_CSV}")
    print(f"       Baris siang : {n_day}")
    print(f"       Baris malam : {n_night}")

    # ------------------------------------------------------------
    # 4. Ringkasan
    # ------------------------------------------------------------
    total_energy_kwh = out['P_mpp_total_W'].sum() / 1000.0
    peak_power_w = out['P_mpp_total_W'].max()
    cf = 100.0 * total_energy_kwh / (CAPACITY_WP * len(out) / 1000.0)

    print("\n--- Ringkasan ---")
    print(f"  Kapasitas terpasang : {CAPACITY_WP:.1f} Wp")
    print(f"  Total energi        : {total_energy_kwh:.2f} kWh")
    print(f"  Puncak daya         : {peak_power_w:.1f} W")
    print(f"  Capacity factor     : {cf:.2f}%")

    sanity_check(out)

    # ------------------------------------------------------------
    # 5. Plot 7 hari pertama
    # ------------------------------------------------------------
    plot_first_week(out, FIGURE_PATH)

    # ------------------------------------------------------------
    # 6. Cuplikan baris siang hari
    # ------------------------------------------------------------
    print("\nCuplikan 8 baris siang hari (G > 100 W/m²):")
    day = out[out['G_Wm2'] > 100].head(8)
    if len(day) == 0:
        print("  (Tidak ada baris dengan G > 100 W/m² pada dataset.)")
    else:
        print(day.to_string(index=False))


if __name__ == '__main__':
    main()