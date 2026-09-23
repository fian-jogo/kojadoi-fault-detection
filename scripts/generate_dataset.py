"""
generate_dataset.py
Orkestrasi generasi dataset fault sintetik untuk PLTS Koja Doi.

Konfigurasi:
    - 3 string paralel, 4 modul seri per string
    - 12 node total (4 per string)
    - 3 jenis fault: LLF, LGF, PSC
    - 180 skenario: 30 normal + 50 LLF + 50 LGF + 50 PSC
    - Noise sensor 2% Gaussian

Output:
    data/processed/fault_dataset.csv
    data/processed/fault_metadata.json
    figures/fault_examples.png
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# Deteksi ROOT dan tambahkan ke sys.path
# ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print(f"[debug] ROOT = {ROOT}")
print(f"[debug] src exists = {(ROOT / 'src').exists()}")
print(f"[debug] fault_injection exists = "
      f"{(ROOT / 'src' / 'fault_injection.py').exists()}")

# ----------------------------------------------------------------
# Import dari src (WAJIB pakai prefix "src.")
# ----------------------------------------------------------------
from src.pv_model import ModuleParams
from src.fault_injection import FaultSpec, inject_fault

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


# ================================================================
# Konfigurasi
# ================================================================
N_STRINGS = 3
N_SERIES = 4
N_NODES = N_SERIES               # node per string
TOTAL_NODES = N_SERIES * N_STRINGS  # 12 node global
TIME_STEPS = 100
NOISE_STD = 0.02                 # 2% noise Gaussian

# Rentang parameter fault (SUDAH DIPERBAIKI)
LLF_R_RANGE = (0.1, 10.0)        # Ω
LGF_R_RANGE = (0.5, 20.0)        # Ω — rentang baru agar efek terlihat
PSC_SIGMA_RANGE = (0.3, 0.6)     # 30–60% penurunan iradiansi

# Jumlah skenario per jenis
N_NORMAL = 30
N_LLF = 50
N_LGF = 50
N_PSC = 50

# Output
DATASET_CSV = ROOT / "data" / "processed" / "fault_dataset.csv"
METADATA_JSON = ROOT / "data" / "processed" / "fault_metadata.json"
FIGURE_PATH = ROOT / "figures" / "fault_examples.png"

# Seed untuk reproducibility
RNG_SEED = 42


# ================================================================
# Load data cuaca
# ================================================================
def load_weather_data() -> pd.DataFrame:
    """Load data NASA POWER dan ambil baris siang hari (G > 50 W/m²)."""
    csv_path = ROOT / "data" / "raw" / "koja_doi_irradiance_temperature_hourly.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"File cuaca tidak ditemukan: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['datetime_local'])
    df_day = df[df['ALLSKY_SFC_SW_DWN'] > 50].reset_index(drop=True)
    if len(df_day) < TIME_STEPS:
        raise ValueError(
            f"Data siang hari hanya {len(df_day)} baris, "
            f"butuh minimal {TIME_STEPS}."
        )
    return df_day


# ================================================================
# Generasi satu skenario
# ================================================================
def generate_scenario(scenario_id: int,
                      fault_spec: FaultSpec or None, # type: ignore
                      weather_df: pd.DataFrame,
                      rng: np.random.Generator) -> dict:
    """
    Generate satu time series skenario.

    Return dict dengan:
        V, I, P       : array (n_steps, N_STRINGS, N_NODES)
        G, T          : array (n_steps, N_STRINGS)
        fault_active  : array bool (n_steps,)
        fault_type_arr: array int (n_steps,)
        metadata      : dict
    """
    n_steps = min(len(weather_df), TIME_STEPS)

    V_all = np.zeros((n_steps, N_STRINGS, N_NODES))
    I_all = np.zeros((n_steps, N_STRINGS, N_NODES))
    P_all = np.zeros((n_steps, N_STRINGS, N_NODES))
    G_all = np.zeros((n_steps, N_STRINGS))
    T_all = np.zeros((n_steps, N_STRINGS))

    fault_active = np.zeros(n_steps, dtype=bool)
    fault_type_arr = np.zeros(n_steps, dtype=int)

    # Mulai dari indeks acak dalam data cuaca
    start_idx = rng.integers(0, max(1, len(weather_df) - n_steps))

    for t in range(n_steps):
        row = weather_df.iloc[start_idx + t]
        G = float(row['ALLSKY_SFC_SW_DWN'])
        T = float(row['T2M'])
        if G < 0:
            G = 0.0

        if fault_spec is not None and \
           fault_spec.onset <= t < fault_spec.onset + fault_spec.duration:
            fault_active[t] = True
            fault_type_arr[t] = {'LLF': 1, 'LGF': 2, 'PSC': 3}[fault_spec.fault_type]

        for s in range(N_STRINGS):
            # Node terdampak di string ini
            if fault_spec is not None and fault_active[t]:
                string_nodes = [n for n in fault_spec.affected_nodes
                                if n // N_NODES == s]
                string_nodes = [n % N_NODES for n in string_nodes]
            else:
                string_nodes = []

            # Buat FaultSpec lokal per string
            if fault_spec is not None and string_nodes:
                local_spec = FaultSpec(
                    fault_type=fault_spec.fault_type,
                    onset=fault_spec.onset,
                    duration=fault_spec.duration,
                    affected_nodes=string_nodes,
                    magnitude=fault_spec.magnitude,
                    gradual=fault_spec.gradual,
                )
            else:
                local_spec = None

            # Hitung respons string
            status = inject_fault(
                G=G, T=T,
                params=params,
                n_series=N_SERIES,
                fault=local_spec,
                t_step=t,
                n_total_nodes=N_NODES,
                n_strings_parallel=N_STRINGS,
            )

            V_str = status['V_mpp']
            I_str = status['I_mpp']
            P_str = status['P_mpp']

            for n in range(N_NODES):
                V_all[t, s, n] = V_str / N_SERIES
                I_all[t, s, n] = I_str
                P_all[t, s, n] = P_str / N_SERIES

            G_all[t, s] = G
            T_all[t, s] = T

    # Tambahkan noise sensor
    V_all += rng.normal(0, NOISE_STD * np.abs(V_all) + 1e-6, V_all.shape)
    I_all += rng.normal(0, NOISE_STD * np.abs(I_all) + 1e-6, I_all.shape)
    P_all = V_all * I_all

    V_all = np.maximum(V_all, 0.0)
    I_all = np.maximum(I_all, 0.0)
    P_all = np.maximum(P_all, 0.0)

    metadata = {
        'scenario_id': int(scenario_id),
        'fault_type': fault_spec.fault_type if fault_spec else 'normal',
        'onset': int(fault_spec.onset) if fault_spec else -1,
        'duration': int(fault_spec.duration) if fault_spec else 0,
        'affected_nodes': [int(n) for n in fault_spec.affected_nodes]
                          if fault_spec else [],
        'magnitude': float(fault_spec.magnitude) if fault_spec else 0.0,
        'gradual': bool(fault_spec.gradual) if fault_spec else False,
        'n_steps': int(n_steps),
        'start_idx': int(start_idx),
    }

    return {
        'V': V_all,
        'I': I_all,
        'P': P_all,
        'G': G_all,
        'T': T_all,
        'fault_active': fault_active,
        'fault_type_arr': fault_type_arr,
        'metadata': metadata,
    }


# ================================================================
# Plot contoh fault (per-string, bukan rata-rata)
# ================================================================
def plot_fault_examples(scenarios: list) -> None:
    """
    Plot 4 contoh time series: normal, LLF, LGF, PSC.
    Setiap panel menampilkan daya per string sehingga fault lokal terlihat.
    """
    examples = {}
    for sc in scenarios:
        ft = sc['metadata']['fault_type']
        if ft == 'normal' and 'normal' not in examples:
            examples['normal'] = sc
        elif ft == 'LLF' and 'LLF' not in examples:
            examples['LLF'] = sc
        elif ft == 'LGF' and 'LGF' not in examples:
            examples['LGF'] = sc
        elif ft == 'PSC' and 'PSC' not in examples:
            examples['PSC'] = sc

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    order = ['normal', 'LLF', 'LGF', 'PSC']
    titles = ['Normal', 'Line-to-Line Fault',
              'Ground Fault', 'Partial Shading']
    string_colors = ['tab:blue', 'tab:green', 'tab:orange']

    for ax, label, title in zip(axes.flat, order, titles):
        sc = examples.get(label)
        if sc is None:
            ax.set_title(f'{title} (tidak ada)')
            continue

        meta = sc['metadata']
        n_steps = meta['n_steps']

        # Plot total daya per string
        for s in range(N_STRINGS):
            P_string = sc['P'][:, s, :].sum(axis=1)
            ax.plot(range(n_steps), P_string,
                    color=string_colors[s], lw=1.6,
                    label=f'String {s}')

        # Tandai periode fault
        if meta['onset'] >= 0:
            ax.axvspan(meta['onset'],
                       meta['onset'] + meta['duration'],
                       alpha=0.15, color='red',
                       label=f'Fault aktif (onset={meta["onset"]})')

        ax.set_xlabel('Time step')
        ax.set_ylabel('Daya string (W)')
        ax.set_title(
            f'{title}\n'
            f'scenario {meta["scenario_id"]}, '
            f'magnitude={meta["magnitude"]:.2f}, '
            f'gradual={meta["gradual"]}\n'
            f'affected nodes: {meta["affected_nodes"]}'
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)
    print(f"\n[info] Plot contoh fault tersimpan: {FIGURE_PATH}")


# ================================================================
# Main
# ================================================================
def main():
    rng = np.random.default_rng(RNG_SEED)
    weather_df = load_weather_data()
    print(f"[info] Data cuaca: {len(weather_df)} baris siang")

    all_scenarios = []

    # ------------------------------------------------------------
    # 1. Skenario normal
    # ------------------------------------------------------------
    print(f"\n[1/4] Generating {N_NORMAL} skenario normal ...")
    for i in range(N_NORMAL):
        scenario = generate_scenario(i, None, weather_df, rng)
        all_scenarios.append(scenario)
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{N_NORMAL}")

    # ------------------------------------------------------------
    # 2. Skenario LLF
    # ------------------------------------------------------------
    print(f"\n[2/4] Generating {N_LLF} skenario LLF ...")
    for i in range(N_LLF):
        R_f = float(rng.uniform(*LLF_R_RANGE))
        onset = int(rng.integers(20, 61))
        duration = int(rng.integers(10, 41))
        s1, s2 = rng.choice(N_STRINGS, size=2, replace=False)
        n1 = int(s1 * N_NODES + rng.integers(0, N_NODES))
        n2 = int(s2 * N_NODES + rng.integers(0, N_NODES))
        gradual = bool(rng.random() < 0.3)

        spec = FaultSpec(
            fault_type='LLF',
            onset=onset,
            duration=duration,
            affected_nodes=[n1, n2],
            magnitude=R_f,
            gradual=gradual,
        )
        scenario = generate_scenario(N_NORMAL + i, spec, weather_df, rng)
        all_scenarios.append(scenario)
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{N_LLF}")

    # ------------------------------------------------------------
    # 3. Skenario LGF
    # ------------------------------------------------------------
    print(f"\n[3/4] Generating {N_LGF} skenario LGF ...")
    for i in range(N_LGF):
        R_g = float(rng.uniform(*LGF_R_RANGE))
        onset = int(rng.integers(20, 61))
        duration = int(rng.integers(10, 41))
        s = int(rng.integers(0, N_STRINGS))
        n = int(s * N_NODES + rng.integers(0, N_NODES))

        spec = FaultSpec(
            fault_type='LGF',
            onset=onset,
            duration=duration,
            affected_nodes=[n],
            magnitude=R_g,
            gradual=True,
        )
        scenario = generate_scenario(N_NORMAL + N_LLF + i, spec, weather_df, rng)
        all_scenarios.append(scenario)
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{N_LGF}")

    # ------------------------------------------------------------
    # 4. Skenario PSC
    # ------------------------------------------------------------
    print(f"\n[4/4] Generating {N_PSC} skenario PSC ...")
    for i in range(N_PSC):
        sigma = float(rng.uniform(*PSC_SIGMA_RANGE))
        onset = int(rng.integers(20, 61))
        duration = int(rng.integers(10, 41))
        s = int(rng.integers(0, N_STRINGS))
        n_shaded = int(rng.integers(1, min(4, N_NODES) + 1))
        nodes_in_string = rng.choice(N_NODES, size=n_shaded, replace=False)
        nodes = [int(s * N_NODES + j) for j in nodes_in_string]

        spec = FaultSpec(
            fault_type='PSC',
            onset=onset,
            duration=duration,
            affected_nodes=nodes,
            magnitude=sigma,
            gradual=True,
        )
        scenario = generate_scenario(N_NORMAL + N_LLF + N_LGF + i,
                                     spec, weather_df, rng)
        all_scenarios.append(scenario)
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{N_PSC}")

    # ------------------------------------------------------------
    # Flatten ke DataFrame
    # ------------------------------------------------------------
    print(f"\n[info] Total skenario: {len(all_scenarios)}")
    print("[info] Flatten ke DataFrame ...")

    records = []
    for sc in all_scenarios:
        meta = sc['metadata']
        n_steps = meta['n_steps']
        for t in range(n_steps):
            for s in range(N_STRINGS):
                for n in range(N_NODES):
                    records.append({
                        'scenario_id': meta['scenario_id'],
                        'time_step': t,
                        'string': s,
                        'node': n,
                        'V': float(sc['V'][t, s, n]),
                        'I': float(sc['I'][t, s, n]),
                        'P': float(sc['P'][t, s, n]),
                        'G': float(sc['G'][t, s]),
                        'T': float(sc['T'][t, s]),
                        'fault_active': int(sc['fault_active'][t]),
                        'fault_type': int(sc['fault_type_arr'][t]),
                        'fault_type_name': meta['fault_type'],
                        'onset': meta['onset'],
                        'duration': meta['duration'],
                        'affected_nodes': str(meta['affected_nodes']),
                        'magnitude': meta['magnitude'],
                        'gradual': int(meta['gradual']),
                    })

    df = pd.DataFrame(records)
    DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_CSV, index=False)
    print(f"[info] Dataset tersimpan: {DATASET_CSV}")
    print(f"       Total baris: {len(df)}")

    # ------------------------------------------------------------
    # Simpan metadata ke JSON
    # ------------------------------------------------------------
    metadata_all = [sc['metadata'] for sc in all_scenarios]
    with open(METADATA_JSON, 'w', encoding='utf-8') as f:
        json.dump(metadata_all, f, indent=2)
    print(f"[info] Metadata tersimpan: {METADATA_JSON}")

    # ------------------------------------------------------------
    # Statistik ringkas
    # ------------------------------------------------------------
    print("\n--- Statistik Dataset ---")
    print(f"  Total skenario    : {len(all_scenarios)}")
    print(f"  Normal            : {N_NORMAL}")
    print(f"  LLF               : {N_LLF}")
    print(f"  LGF               : {N_LGF}")
    print(f"  PSC               : {N_PSC}")
    print(f"  Total baris       : {len(df)}")
    print(f"  Baris fault aktif : {int(df['fault_active'].sum())} "
          f"({100 * df['fault_active'].mean():.2f}%)")
    print("\n  Distribusi baris fault per jenis:")
    fault_rows = df[df['fault_active'] == 1]
    print(fault_rows['fault_type_name'].value_counts().to_string())

    print("\n  Range nilai per jenis fault:")
    for ft in ['normal', 'LLF', 'LGF', 'PSC']:
        sub = df[df['fault_type_name'] == ft]
        print(f"    {ft:8s}: V[{sub['V'].min():.2f}, {sub['V'].max():.2f}] "
              f"I[{sub['I'].min():.3f}, {sub['I'].max():.3f}] "
              f"P[{sub['P'].min():.2f}, {sub['P'].max():.2f}]")

    # ------------------------------------------------------------
    # Plot contoh fault
    # ------------------------------------------------------------
    plot_fault_examples(all_scenarios)

    print("\n[done] Generasi dataset selesai.")


if __name__ == '__main__':
    main()