"""
qualitative_analysis.py
Visualisasi contoh kasus:
    - Success: fault terdeteksi cepat dan lokalisasi tepat
    - Failure: fault terlewat (false negative) atau false alarm

Output:
    figures/qualitative_examples.png
    figures/qualitative_llf_success.png
    figures/qualitative_psc_failure.png
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import N_NODES_TOTAL, N_FEATURES, build_adjacency_matrix
from src.dataset import FaultWindowDataset
from src.model_hybrid import HybridDetector

SPLIT_NPZ = ROOT / "data" / "processed" / "train_val_test_split.npz"
DATASET_CSV = ROOT / "data" / "processed" / "fault_dataset.csv"
CKPT_PATH = ROOT / "logs" / "best_lstm_gat.pt"
OUT_PNG = ROOT / "figures" / "qualitative_examples.png"
T_WINDOW = 30
BATCH_SIZE = 128
DEVICE = torch.device('cpu')


def load_model():
    ckpt = torch.load(CKPT_PATH, weights_only=False)
    model = HybridDetector(
        n_nodes=N_NODES_TOTAL, n_features=N_FEATURES,
        hidden=64, n_layers=2, n_gat=2, n_heads=4,
        dropout=0.3, use_gat=True, use_pinn=False,
        norm_mean=ckpt['norm_mean'], norm_std=ckpt['norm_std'],
    )
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def inference_scenario(model, X_sel, adj):
    ds = FaultWindowDataset(X_sel, np.zeros(len(X_sel), dtype=int),
                             np.zeros((len(X_sel), N_NODES_TOTAL),
                                       dtype=np.float32))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    all_prob, all_loc = [], []
    with torch.no_grad():
        for X, _, _ in dl:
            det_logits, loc_logits, _ = model(X, adj)
            all_prob.append(torch.softmax(det_logits, dim=1)[:, 1].numpy())
            all_loc.append(torch.sigmoid(loc_logits).numpy())
    return np.concatenate(all_prob), np.concatenate(all_loc)


def find_examples(model, data, adj, type_map):
    sid_test = data['sid_test']
    X_test = data['X_test']
    y_det_test = data['y_det_test']

    fault_prob, loc_prob = [], []
    ds = FaultWindowDataset(X_test, y_det_test,
                             data['y_loc_test'])
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for Xb, y_det, y_loc in dl:
            det_logits, loc_logits, _ = model(Xb, adj)
            fault_prob.append(torch.softmax(det_logits, dim=1)[:, 1].numpy())
            loc_prob.append(torch.sigmoid(loc_logits).numpy())
    fault_prob = np.concatenate(fault_prob)
    loc_prob = np.concatenate(loc_prob)

    # Klasifikasi setiap window
    y_true = y_det_test
    y_pred = (fault_prob >= 0.5).astype(int)

    # Cari contoh
    examples = {}

    # Success: LLF dengan true positive dan latency kecil
    for sid in np.unique(sid_test):
        meta = type_map.get(int(sid))
        if not meta or meta['fault_type'] != 'LLF':
            continue
        mask = sid_test == sid
        idx = np.where(mask)[0]
        # Cek apakah ada true positive
        if (y_true[idx] * y_pred[idx]).sum() > 0:
            examples['success_LLF'] = {
                'sid': int(sid), 'meta': meta, 'idx': idx,
                'prob': fault_prob[idx], 'loc': loc_prob[idx],
                'y_true': y_true[idx], 'y_pred': y_pred[idx],
            }
            break

    # Failure: false negative (fault aktif tapi diprediksi normal)
    for sid in np.unique(sid_test):
        meta = type_map.get(int(sid))
        if not meta or meta['fault_type'] == 'normal':
            continue
        mask = sid_test == sid
        idx = np.where(mask)[0]
        # Cek apakah ada banyak false negative
        fn = ((y_true[idx] == 1) & (y_pred[idx] == 0)).sum()
        if fn > 5:
            examples['failure_FN'] = {
                'sid': int(sid), 'meta': meta, 'idx': idx,
                'prob': fault_prob[idx], 'loc': loc_prob[idx],
                'y_true': y_true[idx], 'y_pred': y_pred[idx],
            }
            break

    # Failure: false positive (normal tapi diprediksi fault)
    for sid in np.unique(sid_test):
        meta = type_map.get(int(sid))
        if not meta or meta['fault_type'] != 'normal':
            continue
        mask = sid_test == sid
        idx = np.where(mask)[0]
        fp = ((y_true[idx] == 0) & (y_pred[idx] == 1)).sum()
        if fp > 3:
            examples['failure_FP'] = {
                'sid': int(sid), 'meta': meta, 'idx': idx,
                'prob': fault_prob[idx], 'loc': loc_prob[idx],
                'y_true': y_true[idx], 'y_pred': y_pred[idx],
            }
            break

    return examples


def plot_example(ax, ex, title):
    idx_sorted = np.argsort(ex['idx'])
    prob = ex['prob'][idx_sorted]
    y_true = ex['y_true'][idx_sorted]
    y_pred = ex['y_pred'][idx_sorted]
    t = np.arange(len(prob))

    ax.fill_between(t, 0, 1.05, where=y_true == 1, alpha=0.15,
                     color='red', label='Fault aktif (GT)')
    ax.plot(t, prob, 'b-', lw=1.8, label='Probabilitas fault')
    ax.plot(t, y_pred, 'g--', lw=1.2, label='Prediksi (th=0.5)')
    ax.axhline(0.5, color='k', ls=':', lw=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Window index')
    ax.set_ylabel('Probabilitas')
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='upper left')


def main():
    print("[1/3] Load model dan data ...")
    model = load_model()
    data = np.load(SPLIT_NPZ)
    adj = torch.from_numpy(build_adjacency_matrix()).float()

    # Load metadata
    df_meta = pd.read_csv(DATASET_CSV)
    df_meta = df_meta[df_meta['time_step'] == 0][
        ['scenario_id', 'fault_type_name', 'onset']].drop_duplicates()
    type_map = {int(r['scenario_id']):
                {'fault_type': r['fault_type_name'],
                 'onset': int(r['onset'])}
                for _, r in df_meta.iterrows()}

    print("[2/3] Cari contoh kasus ...")
    examples = find_examples(model, data, adj, type_map)
    for k, v in examples.items():
        print(f"       {k}: scenario {v['sid']} ({v['meta']['fault_type']})")

    print("[3/3] Plot ...")
    n_ex = len(examples)
    if n_ex == 0:
        print("       Tidak ada contoh ditemukan.")
        return

    fig, axes = plt.subplots(1, n_ex, figsize=(6 * n_ex, 4))
    if n_ex == 1:
        axes = [axes]

    titles = {
        'success_LLF': 'Sukses: LLF terdeteksi',
        'failure_FN': 'Gagal: False Negative (fault terlewat)',
        'failure_FP': 'Gagal: False Positive (alarm palsu)',
    }
    for ax, (k, ex) in zip(axes, examples.items()):
        plot_example(ax, ex, f"{titles.get(k, k)}\n"
                              f"Scenario {ex['sid']} — "
                              f"{ex['meta']['fault_type']}")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"       Tersimpan: {OUT_PNG}")


if __name__ == '__main__':
    main()