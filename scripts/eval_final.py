"""
eval_final.py
Evaluasi final model LSTM+GAT pada test set:
    1. Metrik agregat (F1, precision, recall, confusion matrix)
    2. Latency analysis (deteksi dini)
    3. Per-class metrics (LLF vs LGF vs PSC)
    4. Threshold tuning untuk aplikasi keselamatan

Output:
    logs/best_lstm_gat.pt          — model checkpoint
    logs/final_evaluation.json      — semua metrik
    figures/confusion_matrix_final.png
    figures/latency_histogram.png
    figures/per_class_metrics.png
    figures/threshold_sweep.png
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import N_NODES_TOTAL, N_FEATURES, build_adjacency_matrix
from src.dataset import FaultWindowDataset
from src.model_hybrid import HybridDetector

# ----------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------
SPLIT_NPZ = ROOT / "data" / "processed" / "train_val_test_split.npz"
DATASET_CSV = ROOT / "data" / "processed" / "fault_dataset.csv"
CKPT_PATH = ROOT / "logs" / "best_lstm_gat.pt"
METRICS_JSON = ROOT / "logs" / "final_evaluation.json"
CM_PNG = ROOT / "figures" / "confusion_matrix_final.png"
LATENCY_PNG = ROOT / "figures" / "latency_histogram.png"
PERCLASS_PNG = ROOT / "figures" / "per_class_metrics.png"
THRESHOLD_PNG = ROOT / "figures" / "threshold_sweep.png"

T_WINDOW = 30
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_LOC = 1.0
SEED = 42
DEVICE = torch.device('cpu')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


# ================================================================
# Training LSTM+GAT (best config)
# ================================================================
def train_lstm_gat(data_split, adj, norm_mean, norm_std):
    set_seed(SEED)

    ds_train = FaultWindowDataset(
        data_split['train']['X'],
        data_split['train']['y_det'],
        data_split['train']['y_loc'])
    ds_val = FaultWindowDataset(
        data_split['val']['X'],
        data_split['val']['y_det'],
        data_split['val']['y_loc'])

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

    model = HybridDetector(
        n_nodes=N_NODES_TOTAL, n_features=N_FEATURES,
        hidden=64, n_layers=2, n_gat=2, n_heads=4,
        dropout=0.3, use_gat=True, use_pinn=False,
        norm_mean=norm_mean, norm_std=norm_std,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)
    crit_det = nn.CrossEntropyLoss()
    crit_loc = nn.BCEWithLogitsLoss()

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        for X, y_det, y_loc in dl_train:
            optimizer.zero_grad()
            det_logits, loc_logits, _ = model(X, adj)
            loss = crit_det(det_logits, y_det) + LAMBDA_LOC * crit_loc(
                loc_logits, y_loc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Val
        model.eval()
        y_true_v, y_pred_v = [], []
        with torch.no_grad():
            for X, y_det, y_loc in dl_val:
                det_logits, _, _ = model(X, adj)
                y_true_v.append(y_det.numpy())
                y_pred_v.append(det_logits.argmax(dim=1).numpy())
        y_true_v = np.concatenate(y_true_v)
        y_pred_v = np.concatenate(y_pred_v)
        m = compute_metrics(y_true_v, y_pred_v)

        if m['f1'] > best_f1:
            best_f1 = m['f1']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"val_F1={m['f1']:.4f} | val_acc={m['accuracy']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'norm_mean': norm_mean,
        'norm_std': norm_std,
        'best_val_f1': best_f1,
    }, CKPT_PATH)
    print(f"  Checkpoint disimpan: {CKPT_PATH}")

    return model, best_f1


# ================================================================
# Inference pada test set
# ================================================================
@torch.no_grad()
def inference_test(model, data_split, adj):
    model.eval()
    X = torch.from_numpy(data_split['test']['X']).float()
    ds = FaultWindowDataset(X.numpy(),
                             data_split['test']['y_det'],
                             data_split['test']['y_loc'])
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    all_prob = []
    all_loc = []
    for Xb, y_det, y_loc in dl:
        det_logits, loc_logits, _ = model(Xb, adj)
        all_prob.append(torch.softmax(det_logits, dim=1)[:, 1].numpy())
        all_loc.append(torch.sigmoid(loc_logits).numpy())

    fault_prob = np.concatenate(all_prob)
    loc_prob = np.concatenate(all_loc)
    return fault_prob, loc_prob


# ================================================================
# Latency analysis
# ================================================================
def compute_latency(fault_prob, y_true, sid_test, type_map, threshold=0.5):
    """
    Untuk setiap skenario dengan fault, hitung:
        - onset time step (dari metadata)
        - waktu deteksi pertama (t_last saat model prediksi fault)
        - latency = t_detect - onset
    """
    # Rekonstruksi t_last untuk setiap window
    # Window ke-k dalam skenario memiliki t_last = k + T - 1
    t_last_arr = np.zeros(len(sid_test), dtype=int)
    for sid in np.unique(sid_test):
        mask = sid_test == sid
        idx = np.where(mask)[0]
        for k, i in enumerate(idx):
            t_last_arr[i] = k + T_WINDOW - 1

    results = []
    for sid in np.unique(sid_test):
        meta = type_map.get(int(sid))
        if meta is None or meta['fault_type'] == 'normal':
            continue
        onset = meta['onset']
        if onset < 0:
            continue

        mask = sid_test == sid
        idx = np.where(mask)[0]
        # Urutkan berdasarkan t_last
        idx = idx[np.argsort(t_last_arr[idx])]
        t_last_sorted = t_last_arr[idx]
        prob_sorted = fault_prob[idx]

        # Cari window pertama dengan t_last >= onset dan prob >= threshold
        detected_at = None
        for i, t in enumerate(t_last_sorted):
            if t >= onset and prob_sorted[i] >= threshold:
                detected_at = t
                break

        if detected_at is None:
            latency = -1  # tidak terdeteksi
        else:
            latency = int(detected_at - onset)

        results.append({
            'scenario_id': int(sid),
            'fault_type': meta['fault_type'],
            'onset': int(onset),
            'detected': detected_at is not None,
            'detected_at': int(detected_at) if detected_at is not None else -1,
            'latency': latency,
        })

    return pd.DataFrame(results)


# ================================================================
# Per-class metrics
# ================================================================
def per_class_metrics(y_true, y_pred, sid_test, type_map):
    rows = []
    for ft in ['normal', 'LLF', 'LGF', 'PSC']:
        mask = np.array([type_map[int(s)].get('fault_type', 'normal') == ft
                         if int(s) in type_map else False
                         for s in sid_test])
        if mask.sum() == 0:
            continue
        m = compute_metrics(y_true[mask], y_pred[mask])
        m['fault_type'] = ft
        m['n_windows'] = int(mask.sum())
        m['n_fault_active'] = int(y_true[mask].sum())
        rows.append(m)
    return pd.DataFrame(rows)


# ================================================================
# Threshold sweep
# ================================================================
def threshold_sweep(fault_prob, y_true):
    thresholds = np.arange(0.10, 0.81, 0.05)
    rows = []
    for th in thresholds:
        y_pred = (fault_prob >= th).astype(int)
        m = compute_metrics(y_true, y_pred)
        m['threshold'] = float(th)
        rows.append(m)
    return pd.DataFrame(rows)


# ================================================================
# Main
# ================================================================
def main():
    print(f"[info] Device: {DEVICE}")
    set_seed(SEED)

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------
    print(f"\n[1/6] Load split ...")
    data = np.load(SPLIT_NPZ)
    norm_mean = data['mean']
    norm_std = data['std']

    data_split = {
        'train': {'X': data['X_train'], 'y_det': data['y_det_train'],
                  'y_loc': data['y_loc_train'], 'sid': data['sid_train']},
        'val':   {'X': data['X_val'], 'y_det': data['y_det_val'],
                  'y_loc': data['y_loc_val'], 'sid': data['sid_val']},
        'test':  {'X': data['X_test'], 'y_det': data['y_det_test'],
                  'y_loc': data['y_loc_test'], 'sid': data['sid_test']},
    }
    sid_test = data_split['test']['sid']
    y_true = data_split['test']['y_det']

    adj = torch.from_numpy(build_adjacency_matrix()).float()

    # ------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------
    print(f"[2/6] Load metadata fault ...")
    df_meta = pd.read_csv(DATASET_CSV)
    df_meta = df_meta[df_meta['time_step'] == 0][
        ['scenario_id', 'fault_type_name', 'onset', 'duration',
         'affected_nodes']].drop_duplicates()

    type_map = {}
    for _, row in df_meta.iterrows():
        type_map[int(row['scenario_id'])] = {
            'fault_type': str(row['fault_type_name']),
            'onset': int(row['onset']),
            'duration': int(row['duration']),
            'affected_nodes': row['affected_nodes'],
        }
    print(f"       Metadata: {len(type_map)} skenario")

    # ------------------------------------------------------------
    # Train / load model
    # ------------------------------------------------------------
    if CKPT_PATH.exists():
        print(f"[3/6] Load checkpoint ...")
        ckpt = torch.load(CKPT_PATH, weights_only=False)
        model = HybridDetector(
            n_nodes=N_NODES_TOTAL, n_features=N_FEATURES,
            hidden=64, n_layers=2, n_gat=2, n_heads=4,
            dropout=0.3, use_gat=True, use_pinn=False,
            norm_mean=ckpt['norm_mean'], norm_std=ckpt['norm_std'],
        )
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        print(f"       best_val_f1 = {ckpt['best_val_f1']:.4f}")
    else:
        print(f"[3/6] Train LSTM+GAT (30 epoch) ...")
        model, best_f1 = train_lstm_gat(data_split, adj, norm_mean, norm_std)
        model.eval()

    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------
    print(f"[4/6] Inference pada test set ...")
    fault_prob, loc_prob = inference_test(model, data_split, adj)

    # ------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------
    print(f"\n[5/6] Evaluasi ...")
    y_pred = (fault_prob >= 0.5).astype(int)
    metrics = compute_metrics(y_true, y_pred)
    print(f"       F1        = {metrics['f1']:.4f}")
    print(f"       Accuracy  = {metrics['accuracy']:.4f}")
    print(f"       Precision = {metrics['precision']:.4f}")
    print(f"       Recall    = {metrics['recall']:.4f}")
    print(f"       TP/FP/FN/TN = {metrics['tp']}/{metrics['fp']}/"
          f"{metrics['fn']}/{metrics['tn']}")

    # ------------------------------------------------------------
    # Latency
    # ------------------------------------------------------------
    latency_df = compute_latency(fault_prob, y_true, sid_test, type_map,
                                  threshold=0.5)
    detected = latency_df[latency_df['detected']]
    missed = latency_df[~latency_df['detected']]

    print(f"\n       Latency analysis:")
    print(f"         Total scenario fault : {len(latency_df)}")
    print(f"         Terdeteksi           : {len(detected)} "
          f"({100*len(detected)/max(len(latency_df),1):.1f}%)")
    print(f"         Tidak terdeteksi     : {len(missed)}")
    if len(detected) > 0:
        print(f"         Latency median       : {detected['latency'].median():.1f} step")
        print(f"         Latency mean         : {detected['latency'].mean():.1f} step")
        print(f"         Latency P90          : {detected['latency'].quantile(0.9):.1f} step")

    # ------------------------------------------------------------
    # Per-class
    # ------------------------------------------------------------
    pcm = per_class_metrics(y_true, y_pred, sid_test, type_map)
    print(f"\n       Per-class metrics:")
    print(pcm[['fault_type', 'n_windows', 'n_fault_active',
               'f1', 'precision', 'recall']].to_string(index=False))

    # ------------------------------------------------------------
    # Threshold sweep
    # ------------------------------------------------------------
    sweep = threshold_sweep(fault_prob, y_true)
    best_f1_row = sweep.loc[sweep['f1'].idxmax()]
    # Threshold untuk recall >= 0.75
    high_recall = sweep[sweep['recall'] >= 0.75]
    if len(high_recall) > 0:
        best_high_recall = high_recall.loc[high_recall['f1'].idxmax()]
    else:
        best_high_recall = None

    print(f"\n       Threshold sweep:")
    print(f"         Best F1: threshold = {best_f1_row['threshold']:.2f}, "
          f"F1 = {best_f1_row['f1']:.4f}, rec = {best_f1_row['recall']:.4f}")
    if best_high_recall is not None:
        print(f"         Recall >= 0.75: threshold = "
              f"{best_high_recall['threshold']:.2f}, "
              f"F1 = {best_high_recall['f1']:.4f}, "
              f"rec = {best_high_recall['recall']:.4f}")

    # ------------------------------------------------------------
    # Simpan metrik
    # ------------------------------------------------------------
    print(f"\n[6/6] Simpan output ...")
    metrics_all = {
        'aggregate': metrics,
        'latency': {
            'n_total_fault_scenarios': int(len(latency_df)),
            'n_detected': int(len(detected)),
            'n_missed': int(len(missed)),
            'detection_rate': float(len(detected) / max(len(latency_df), 1)),
            'latency_median': float(detected['latency'].median()) if len(detected) > 0 else None,
            'latency_mean': float(detected['latency'].mean()) if len(detected) > 0 else None,
            'latency_p90': float(detected['latency'].quantile(0.9)) if len(detected) > 0 else None,
        },
        'per_class': pcm.to_dict(orient='records'),
        'threshold_sweep': sweep.to_dict(orient='records'),
        'best_threshold': {
            'f1': {'threshold': float(best_f1_row['threshold']),
                   'f1': float(best_f1_row['f1']),
                   'recall': float(best_f1_row['recall']),
                   'precision': float(best_f1_row['precision'])},
            'high_recall': ({
                'threshold': float(best_high_recall['threshold']),
                'f1': float(best_high_recall['f1']),
                'recall': float(best_high_recall['recall']),
                'precision': float(best_high_recall['precision']),
            } if best_high_recall is not None else None),
        },
    }
    with open(METRICS_JSON, 'w') as f:
        json.dump(metrics_all, f, indent=2)
    print(f"       Metrik: {METRICS_JSON}")

    # ------------------------------------------------------------
    # Plot: confusion matrix final
    # ------------------------------------------------------------
    cm = np.array([[metrics['tn'], metrics['fp']],
                   [metrics['fn'], metrics['tp']]])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Fault'])
    ax.set_yticklabels(['Normal', 'Fault'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — Threshold 0.5 (F1 = {metrics["f1"]:.4f})')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=13)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=150); plt.close(fig)
    print(f"       CM: {CM_PNG}")

    # ------------------------------------------------------------
    # Plot: latency histogram
    # ------------------------------------------------------------
    if len(detected) > 0:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(detected['latency'], bins=15, color='tab:blue',
                edgecolor='k', alpha=0.7)
        ax.axvline(detected['latency'].median(), color='r', ls='--',
                    lw=2, label=f'Median = {detected["latency"].median():.1f}')
        ax.axvline(detected['latency'].mean(), color='orange', ls='--',
                    lw=2, label=f'Mean = {detected["latency"].mean():.1f}')
        ax.set_xlabel('Latency (time step setelah onset)')
        ax.set_ylabel('Jumlah skenario')
        ax.set_title('Distribusi Latency Deteksi')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(LATENCY_PNG, dpi=150); plt.close(fig)
        print(f"       Latency: {LATENCY_PNG}")

    # ------------------------------------------------------------
    # Plot: per-class metrics
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(pcm))
    w = 0.25
    ax.bar(x - w, pcm['f1'], w, label='F1', color='tab:blue')
    ax.bar(x, pcm['precision'], w, label='Precision', color='tab:orange')
    ax.bar(x + w, pcm['recall'], w, label='Recall', color='tab:green')
    ax.set_xticks(x)
    ax.set_xticklabels(pcm['fault_type'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance (Threshold 0.5)')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(PERCLASS_PNG, dpi=150); plt.close(fig)
    print(f"       Per-class: {PERCLASS_PNG}")

    # ------------------------------------------------------------
    # Plot: threshold sweep
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sweep['threshold'], sweep['f1'], 'o-', label='F1', lw=2)
    ax.plot(sweep['threshold'], sweep['precision'], 's-',
            label='Precision', lw=2, alpha=0.7)
    ax.plot(sweep['threshold'], sweep['recall'], '^-',
            label='Recall', lw=2, alpha=0.7)
    ax.axvline(0.5, color='k', ls=':', lw=1.5, label='Threshold 0.5')
    ax.axvline(best_f1_row['threshold'], color='r', ls='--', lw=1.5,
                label=f'Best F1 ({best_f1_row["threshold"]:.2f})')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Threshold Sweep')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(THRESHOLD_PNG, dpi=150); plt.close(fig)
    print(f"       Sweep: {THRESHOLD_PNG}")

    # Simpan latency per scenario
    latency_df.to_csv(ROOT / "logs" / "latency_per_scenario.csv", index=False)

    print("\n[done] Evaluasi final selesai.")


if __name__ == '__main__':
    main()