"""
control_system.py
State machine kontrol untuk isolasi fault PLTS.
Pengganti MATLAB action layer dalam Python.

Alur:
    NORMAL -> DETECTING -> ISOLATING -> ISOLATED -> RECOVERING -> NORMAL

Input:  data/interface/predictions.json (dari inference_export.py)
Output:
    data/interface/control_log.csv
    figures/dashboard_operator.png
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PRED_JSON = ROOT / "data" / "interface" / "predictions.json"
LOG_CSV = ROOT / "data" / "interface" / "control_log.csv"
DASHBOARD_PNG = ROOT / "figures" / "dashboard_operator.png"


# ================================================================
# State machine
# ================================================================
def control_state_machine(fault_prob, top3_nodes,
                          conf_th=5, rec_th=10, prob_th=0.5):
    """
    State machine untuk isolasi fault.
    Return: list of dict per time step.
    """
    state_names = ['NORMAL', 'DETECTING', 'ISOLATING',
                   'ISOLATED', 'RECOVERING']
    alarm_map = {'NORMAL': 0, 'DETECTING': 1, 'ISOLATING': 2,
                 'ISOLATED': 3, 'RECOVERING': 1}

    n_steps = len(fault_prob)
    current_state = 0  # NORMAL
    confirm_counter = 0
    recover_counter = 0
    isolated_string = -1

    log = []
    for t in range(n_steps):
        fp = float(fault_prob[t])
        state_name = state_names[current_state]
        breaker = 0
        action = 'Monitoring'

        if current_state == 0:  # NORMAL
            if fp >= prob_th:
                current_state = 1  # DETECTING
                confirm_counter = 1
                action = f'Fault terdeteksi (p={fp:.2f}), konfirmasi...'

        elif current_state == 1:  # DETECTING
            if fp >= prob_th:
                confirm_counter += 1
                action = f'Konfirmasi {confirm_counter}/{conf_th} (p={fp:.2f})'
                if confirm_counter >= conf_th:
                    current_state = 2  # ISOLATING
            else:
                current_state = 0
                confirm_counter = 0
                action = 'False alarm, kembali normal'

        elif current_state == 2:  # ISOLATING
            top1_node = int(top3_nodes[t][0])
            isolated_string = top1_node // 4
            action = f'ISOLASI String {isolated_string} (node {top1_node})'
            current_state = 3  # ISOLATED
            breaker = 1

        elif current_state == 3:  # ISOLATED
            breaker = 1
            if fp < prob_th:
                recover_counter += 1
                action = f'Menunggu recovery {recover_counter}/{rec_th}'
                if recover_counter >= rec_th:
                    current_state = 4  # RECOVERING
            else:
                recover_counter = 0
                action = f'String {isolated_string} terisolasi, monitoring'

        elif current_state == 4:  # RECOVERING
            action = f'Recovery: breaker String {isolated_string} ditutup'
            current_state = 0
            isolated_string = -1
            recover_counter = 0

        state_name = state_names[current_state]
        log.append({
            't': t,
            'fault_prob': fp,
            'state_name': state_name,
            'state_code': current_state + 1,
            'breaker': breaker,
            'alarm_level': alarm_map[state_name],
            'isolated_string': isolated_string,
            'action': action,
        })

    return log


# ================================================================
# Dashboard
# ================================================================
def plot_dashboard(predictions, log_df, out_path):
    n_steps = predictions['n_steps']
    n_nodes = predictions['n_nodes']
    t = np.arange(n_steps)
    gt = np.array(predictions['ground_truth_fault'])
    fp = np.array(predictions['fault_prob'])
    top3 = np.array(predictions['top3_nodes'])

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Panel 1: probabilitas fault
    ax = axes[0]
    ax.fill_between(t, 0, 1.05, where=gt == 1, alpha=0.15, color='red',
                    label='Fault aktif (GT)')
    ax.plot(t, fp, 'b-', lw=1.8, label='Probabilitas fault')
    ax.axhline(0.5, color='k', ls='--', lw=1.2, label='Threshold 0.5')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Probabilitas')
    ax.set_title('Input: Prediksi Deteksi dari Model LSTM+GAT',
                 fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: state machine
    ax = axes[1]
    state_colors = ['#33cc33', '#ffcc33', '#ff6633', '#cc1a1a', '#6699ff']
    for k in range(n_steps):
        sc = log_df['state_code'].iloc[k] - 1
        ax.axvspan(k - 0.5, k + 0.5, color=state_colors[sc], alpha=0.55)
    ax.plot(t, log_df['state_code'], 'k-', lw=1.8)
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks(range(1, 6))
    ax.set_yticklabels(['NORMAL', 'DETECT', 'ISOLATE', 'ISOLATED', 'RECOVER'])
    ax.set_ylabel('State')
    ax.set_title('State Machine Kontrol', fontweight='bold')
    ax.grid(alpha=0.3)

    # Panel 3: breaker status
    ax = axes[2]
    breaker = log_df['breaker'].values
    ax.fill_between(t, -0.05, 1.1, where=breaker == 1,
                    alpha=0.4, color='red', label='Breaker OPEN')
    ax.plot(t, breaker, 'k-', lw=1.5)
    ax.set_ylim(-0.05, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['CLOSED', 'OPEN'])
    ax.set_ylabel('Breaker')
    ax.set_title('Status Breaker String', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 4: top-3 node
    ax = axes[3]
    ax.plot(t, top3[:, 0], 'r.-', lw=1.5, ms=6, label='Top-1')
    ax.plot(t, top3[:, 1], 'm.-', lw=1.2, ms=5, label='Top-2')
    ax.plot(t, top3[:, 2], 'b.-', lw=1.2, ms=5, label='Top-3')
    ax.set_ylim(-0.5, n_nodes - 0.5)
    ax.set_yticks(range(n_nodes))
    ax.set_ylabel('Node index')
    ax.set_xlabel('Time step')
    ax.set_title('Lokalisasi Fault (Top-3 Node)', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, ncol=3)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ================================================================
# Main
# ================================================================
def main():
    print("=" * 60)
    print("  KONTROL SISTEM PLTS KOJA DOI — PYTHON ACTION LAYER")
    print("=" * 60)

    # 1. Load predictions
    print("\n[1/4] Membaca predictions.json ...")
    if not PRED_JSON.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {PRED_JSON}\n"
            f"Jalankan dulu: python scripts/inference_export.py"
        )
    with open(PRED_JSON) as f:
        pred = json.load(f)

    print(f"       Skenario ID : {pred['scenario_id']}")
    print(f"       Jumlah step : {pred['n_steps']}")
    print(f"       Jumlah node : {pred['n_nodes']}")

    # 2. Run state machine
    print("\n[2/4] Menjalankan state machine ...")
    log = control_state_machine(
        fault_prob=pred['fault_prob'],
        top3_nodes=pred['top3_nodes'],
        conf_th=5, rec_th=10, prob_th=0.5,
    )
    log_df = pd.DataFrame(log)

    # 3. Save log
    print("\n[3/4] Menyimpan log ...")
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(LOG_CSV, index=False)
    print(f"       Tersimpan: {LOG_CSV}")

    # 4. Dashboard
    print("\n[4/4] Membuat dashboard ...")
    plot_dashboard(pred, log_df, DASHBOARD_PNG)
    print(f"       Tersimpan: {DASHBOARD_PNG}")

    # Ringkasan
    print("\n=== CONTROL STATE MACHINE SUMMARY ===")
    n_steps = len(log_df)
    print(f"Total time step       : {n_steps}")
    print(f"Fault aktif (GT)      : {sum(pred['ground_truth_fault'])}")
    for s in ['NORMAL', 'DETECTING', 'ISOLATING', 'ISOLATED', 'RECOVERING']:
        count = int((log_df['state_name'] == s).sum())
        if count > 0:
            print(f"Waktu dalam {s:11s}: {count:3d} ({100*count/n_steps:.1f}%)")

    n_iso = int(np.diff(np.concatenate([[0], log_df['breaker'].values])).sum())
    # Count rising edges
    n_iso = int((np.diff(np.concatenate([[0], log_df['breaker'].values])) == 1).sum())
    print(f"Total isolasi         : {n_iso} kali")
    if n_iso > 0:
        first_iso = int(np.where(np.diff(np.concatenate(
            [[0], log_df['breaker'].values])) == 1)[0][0]) + 1
        print(f"Isolasi pertama       : time step {first_iso}")

    print("\n" + "=" * 60)
    print("  Sistem kontrol selesai.")
    print("=" * 60)


if __name__ == '__main__':
    main()