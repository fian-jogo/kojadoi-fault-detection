"""
inference_export.py
Mengekspor prediksi per time step dari model LSTM+GAT ke JSON untuk control layer.

Prioritas:
    1. Jika logs/best_lstm_gat.pt ada -> load checkpoint (cepat, <30 detik)
    2. Jika tidak ada -> latih dari nol (5 menit)

Output: data/interface/predictions.json
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ----------------------------------------------------------------
# Path setup
# ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print(f"[debug] ROOT = {ROOT}")

from src.preprocessing import N_NODES_TOTAL, N_FEATURES, build_adjacency_matrix
from src.dataset import FaultWindowDataset
from src.model_hybrid import HybridDetector


# ================================================================
# Konfigurasi
# ================================================================
SPLIT_NPZ = ROOT / "data" / "processed" / "train_val_test_split.npz"
CKPT_PATH = ROOT / "logs" / "best_lstm_gat.pt"
OUTPUT_JSON = ROOT / "data" / "interface" / "predictions.json"

T_WINDOW = 30
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_LOC = 1.0
SEED = 42
DEVICE = torch.device('cpu')


# ================================================================
# Utility
# ================================================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ================================================================
# Training fallback (jika checkpoint belum ada)
# ================================================================
def train_model(data_split, adj, norm_mean, norm_std):
    """Latih LSTM+GAT dari nol (fallback jika checkpoint tidak ada)."""
    set_seed(SEED)

    ds_train = FaultWindowDataset(
        data_split['train']['X'],
        data_split['train']['y_det'],
        data_split['train']['y_loc'])
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)

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

    for epoch in range(1, EPOCHS + 1):
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
        if epoch % 10 == 0 or epoch == 1:
            print(f"       Epoch {epoch}/{EPOCHS}")

    # Simpan checkpoint untuk dipakai lain kali
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'norm_mean': norm_mean,
        'norm_std': norm_std,
        'best_val_f1': -1.0,
    }, CKPT_PATH)
    print(f"       Checkpoint disimpan: {CKPT_PATH}")

    return model


# ================================================================
# Main
# ================================================================
def main():
    set_seed(SEED)
    print(f"[info] Device: {DEVICE}")

    # ------------------------------------------------------------
    # 1. Load split
    # ------------------------------------------------------------
    print(f"\n[1/4] Load split ...")
    if not SPLIT_NPZ.exists():
        raise FileNotFoundError(
            f"File split tidak ditemukan: {SPLIT_NPZ}\n"
            f"Jalankan dulu: python scripts/train_baseline.py"
        )

    data = np.load(SPLIT_NPZ)
    norm_mean = data['mean']
    norm_std = data['std']

    data_split = {
        'train': {
            'X': data['X_train'],
            'y_det': data['y_det_train'],
            'y_loc': data['y_loc_train'],
            'sid': data['sid_train'],
        },
        'test': {
            'X': data['X_test'],
            'y_det': data['y_det_test'],
            'y_loc': data['y_loc_test'],
            'sid': data['sid_test'],
        },
    }
    print(f"       Train window: {len(data_split['train']['X'])}")
    print(f"       Test window : {len(data_split['test']['X'])}")

    adj = torch.from_numpy(build_adjacency_matrix()).float()

    # ------------------------------------------------------------
    # 2. Load atau latih model
    # ------------------------------------------------------------
    if CKPT_PATH.exists():
        print(f"\n[2/4] Load checkpoint (cepat) ...")
        ckpt = torch.load(CKPT_PATH, weights_only=False)
        model = HybridDetector(
            n_nodes=N_NODES_TOTAL, n_features=N_FEATURES,
            hidden=64, n_layers=2, n_gat=2, n_heads=4,
            dropout=0.3, use_gat=True, use_pinn=False,
            norm_mean=ckpt['norm_mean'], norm_std=ckpt['norm_std'],
        )
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        print(f"       Loaded: {CKPT_PATH}")
        print(f"       best_val_f1: {ckpt.get('best_val_f1', 'N/A')}")
    else:
        print(f"\n[2/4] Checkpoint tidak ada, latih dari nol ...")
        model = train_model(data_split, adj, norm_mean, norm_std)
        model.eval()

    # ------------------------------------------------------------
    # 3. Inference pada skenario dengan fault aktif
    # ------------------------------------------------------------
    print(f"\n[3/4] Inference pada test set ...")

    sid_test = data_split['test']['sid']
    X_test = data_split['test']['X']
    y_det_test = data_split['test']['y_det']
    y_loc_test = data_split['test']['y_loc']

    # Pilih skenario dengan fault aktif terbanyak (demo yang bermakna)
    best_sid = None
    best_count = -1
    for sid in np.unique(sid_test):
        mask = sid_test == sid
        count = int(y_det_test[mask].sum())
        if count > best_count:
            best_count = count
            best_sid = int(sid)

    print(f"       Skenario dipilih: {best_sid}")
    print(f"       Fault aktif di skenario ini: {best_count} window")

    mask = sid_test == best_sid
    idx_sel = np.where(mask)[0]
    X_sel = X_test[idx_sel]
    y_det_sel = y_det_test[idx_sel]
    y_loc_sel = y_loc_test[idx_sel]

    n_export = min(100, len(X_sel))
    X_sel = X_sel[:n_export]
    y_det_sel = y_det_sel[:n_export]
    y_loc_sel = y_loc_sel[:n_export]

    ds_sel = FaultWindowDataset(X_sel, y_det_sel, y_loc_sel)
    dl_sel = DataLoader(ds_sel, batch_size=BATCH_SIZE, shuffle=False)

    all_det_prob = []
    all_loc_prob = []
    with torch.no_grad():
        for X, _, _ in dl_sel:
            det_logits, loc_logits, _ = model(X, adj)
            det_prob = torch.softmax(det_logits, dim=1)[:, 1]
            loc_prob = torch.sigmoid(loc_logits)
            all_det_prob.append(det_prob.numpy())
            all_loc_prob.append(loc_prob.numpy())

    det_prob_arr = np.concatenate(all_det_prob)
    loc_prob_arr = np.concatenate(all_loc_prob)
    top3_nodes = np.argsort(-loc_prob_arr, axis=1)[:, :3]

    # ------------------------------------------------------------
    # 4. Simpan ke JSON
    # ------------------------------------------------------------
    print(f"\n[4/4] Simpan ke JSON ...")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    predictions = {
        'scenario_id': int(best_sid),
        'n_steps': int(n_export),
        'n_nodes': int(N_NODES_TOTAL),
        'time_step': list(range(n_export)),
        'fault_prob': [float(x) for x in det_prob_arr.tolist()],
        'top3_nodes': [[int(n) for n in row] for row in top3_nodes.tolist()],
        'top3_scores': [
            [float(loc_prob_arr[i, n]) for n in top3_nodes[i]]
            for i in range(n_export)
        ],
        'ground_truth_fault': [int(x) for x in y_det_sel.tolist()],
        'ground_truth_nodes': [list(map(float, row)) for row in y_loc_sel.tolist()],
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"       Tersimpan: {OUTPUT_JSON}")
    print(f"       Total time step: {n_export}")
    print(f"       Fault aktif (GT): {int(y_det_sel.sum())}")
    print(f"       Rata-rata fault_prob: {det_prob_arr.mean():.4f}")
    print(f"       Max fault_prob: {det_prob_arr.max():.4f}")
    print(f"\n[done] Inference export selesai.")


if __name__ == '__main__':
    main()