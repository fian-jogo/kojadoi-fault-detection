"""
train_baseline.py
Training dan evaluasi baseline LSTM untuk fault detection & localization.

Pipeline:
    1. Load fault_dataset.csv
    2. Pivot ke tensor (n_scenarios, n_steps, n_nodes, n_features)
    3. Split by scenario (train/val/test)
    4. Normalisasi (fit pada train)
    5. Sliding window
    6. Train LSTM dengan dua head
    7. Evaluasi: akurasi, precision, recall, F1, latency, localization accuracy

Output:
    data/processed/train_val_test_split.npz
    figures/training_curves.png
    figures/confusion_matrix_lstm.png
    logs/lstm_baseline_metrics.json
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

# ----------------------------------------------------------------
# Path setup
# ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print(f"[debug] ROOT = {ROOT}")

from src.preprocessing import (
    load_fault_dataset, pivot_to_tensor,
    compute_normalization_stats, normalize,
    build_adjacency_matrix, make_sliding_windows,
    split_by_scenario, select_windows_by_scenario,
    N_NODES_TOTAL, N_FEATURES,
)
from src.dataset import FaultWindowDataset
from src.model_lstm import LSTMDetector


# ================================================================
# Konfigurasi
# ================================================================
DATASET_CSV = ROOT / "data" / "processed" / "fault_dataset.csv"
SPLIT_NPZ = ROOT / "data" / "processed" / "train_val_test_split.npz"
CURVES_PNG = ROOT / "figures" / "training_curves.png"
CM_PNG = ROOT / "figures" / "confusion_matrix_lstm.png"
METRICS_JSON = ROOT / "logs" / "lstm_baseline_metrics.json"

T_WINDOW = 30
STRIDE = 1
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Bobot untuk loss localization (relatif terhadap detection loss)
LAMBDA_LOC = 0.5


# ================================================================
# Utility
# ================================================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_pred):
    """Hitung akurasi, precision, recall, F1."""
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

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


# ================================================================
# Training loop
# ================================================================
def train_one_epoch(model, loader, optimizer, criterion_det, criterion_loc,
                    device):
    model.train()
    total_loss = 0.0
    n = 0

    for X, y_det, y_loc in loader:
        X = X.to(device)
        y_det = y_det.to(device)
        y_loc = y_loc.to(device)

        optimizer.zero_grad()
        det_logits, loc_logits = model(X)

        loss_det = criterion_det(det_logits, y_det)
        loss_loc = criterion_loc(loc_logits, y_loc)
        loss = loss_det + LAMBDA_LOC * loss_loc

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        n += X.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true_all, y_pred_all = [], []
    y_loc_true_all, y_loc_prob_all = [], []
    total_loss = 0.0
    n = 0

    criterion_det = nn.CrossEntropyLoss()
    criterion_loc = nn.BCEWithLogitsLoss()

    for X, y_det, y_loc in loader:
        X = X.to(device)
        y_det = y_det.to(device)
        y_loc = y_loc.to(device)

        det_logits, loc_logits = model(X)

        loss_det = criterion_det(det_logits, y_det)
        loss_loc = criterion_loc(loc_logits, y_loc)
        loss = loss_det + LAMBDA_LOC * loss_loc

        total_loss += loss.item() * X.size(0)
        n += X.size(0)

        y_pred = det_logits.argmax(dim=1)
        y_prob = torch.sigmoid(loc_logits)

        y_true_all.append(y_det.cpu().numpy())
        y_pred_all.append(y_pred.cpu().numpy())
        y_loc_true_all.append(y_loc.cpu().numpy())
        y_loc_prob_all.append(y_prob.cpu().numpy())

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    y_loc_true_all = np.concatenate(y_loc_true_all)
    y_loc_prob_all = np.concatenate(y_loc_prob_all)

    metrics = compute_metrics(y_true_all, y_pred_all)
    metrics['loss'] = total_loss / max(n, 1)

    return metrics, y_true_all, y_pred_all, y_loc_true_all, y_loc_prob_all


# ================================================================
# Main
# ================================================================
def main():
    set_seed(SEED)
    print(f"[info] Device: {DEVICE}")

    # ------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------
    print(f"\n[1/6] Load dataset ...")
    df, n_scenarios, n_steps = load_fault_dataset(DATASET_CSV)
    print(f"       n_scenarios = {n_scenarios}, n_steps = {n_steps}")

    # ------------------------------------------------------------
    # 2. Pivot ke tensor
    # ------------------------------------------------------------
    print(f"\n[2/6] Pivot ke tensor ...")
    X, y_detect, y_type, y_loc, metadata = pivot_to_tensor(
        df, n_scenarios, n_steps)
    print(f"       X shape = {X.shape}")

    # ------------------------------------------------------------
    # 3. Split by scenario
    # ------------------------------------------------------------
    print(f"\n[3/6] Split by scenario ...")
    splits = split_by_scenario(metadata, train_frac=0.7, val_frac=0.15,
                                seed=SEED)
    print(f"       train = {len(splits['train'])} skenario")
    print(f"       val   = {len(splits['val'])} skenario")
    print(f"       test  = {len(splits['test'])} skenario")

    for name in ['train', 'val', 'test']:
        count = {}
        for sid in splits[name]:
            ft = metadata[sid]['fault_type']
            count[ft] = count.get(ft, 0) + 1
        print(f"       {name}: {count}")

    # ------------------------------------------------------------
    # 4. Normalisasi (fit pada train)
    # ------------------------------------------------------------
    print(f"\n[4/6] Normalisasi ...")
    X_train_raw = X[splits['train']]
    mean, std = compute_normalization_stats(X_train_raw)
    print(f"       mean = {mean}")
    print(f"       std  = {std}")

    X_norm = normalize(X, mean, std)

    # ------------------------------------------------------------
    # 5. Sliding window
    # ------------------------------------------------------------
    print(f"\n[5/6] Sliding window (T={T_WINDOW}) ...")
    X_win, y_det_win, y_type_win, y_loc_win, t_last_win, sid_win = \
        make_sliding_windows(X_norm, y_detect, y_type, y_loc,
                              T=T_WINDOW, stride=STRIDE)
    print(f"       Total window = {len(X_win)}")

    # Filter per split
    data_split = {}
    for name, sids in splits.items():
        Xs, yds, yts, yls, sids_w = select_windows_by_scenario(
            X_win, y_det_win, y_type_win, y_loc_win, sid_win, sids)
        data_split[name] = {
            'X': Xs, 'y_det': yds, 'y_type': yts, 'y_loc': yls, 'sid': sids_w
        }
        pos = int(yds.sum())
        print(f"       {name}: {len(Xs)} window, "
              f"{pos} positif ({100*pos/max(len(Xs),1):.1f}%)")

    # Simpan split ke npz
    SPLIT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SPLIT_NPZ,
        X_train=data_split['train']['X'],
        y_det_train=data_split['train']['y_det'],
        y_loc_train=data_split['train']['y_loc'],
        sid_train=data_split['train']['sid'],
        X_val=data_split['val']['X'],
        y_det_val=data_split['val']['y_det'],
        y_loc_val=data_split['val']['y_loc'],
        sid_val=data_split['val']['sid'],
        X_test=data_split['test']['X'],
        y_det_test=data_split['test']['y_det'],
        y_loc_test=data_split['test']['y_loc'],
        sid_test=data_split['test']['sid'],
        mean=mean, std=std,
    )
    print(f"       Split disimpan: {SPLIT_NPZ}")

    # ------------------------------------------------------------
    # 6. Training
    # ------------------------------------------------------------
    print(f"\n[6/6] Training LSTM ...")

    ds_train = FaultWindowDataset(
        data_split['train']['X'],
        data_split['train']['y_det'],
        data_split['train']['y_loc'])
    ds_val = FaultWindowDataset(
        data_split['val']['X'],
        data_split['val']['y_det'],
        data_split['val']['y_loc'])
    ds_test = FaultWindowDataset(
        data_split['test']['X'],
        data_split['test']['y_det'],
        data_split['test']['y_loc'])

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0)

    model = LSTMDetector(
        n_nodes=N_NODES_TOTAL, n_features=N_FEATURES,
        hidden=64, n_layers=2, dropout=0.2,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"       Parameter: {n_params:,}")

    # Bobot untuk class imbalance detection
    n_pos = data_split['train']['y_det'].sum()
    n_neg = len(data_split['train']['y_det']) - n_pos
    w_pos = len(data_split['train']['y_det']) / max(2 * n_pos, 1)
    w_neg = len(data_split['train']['y_det']) / max(2 * n_neg, 1)
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32).to(DEVICE)
    print(f"       Class weights: neg={w_neg:.3f}, pos={w_pos:.3f}")

    criterion_det = nn.CrossEntropyLoss(weight=class_weights)
    criterion_loc = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)

    history = {'train_loss': [], 'val_loss': [],
               'val_acc': [], 'val_f1': [], 'val_prec': [], 'val_rec': []}

    best_val_f1 = -1.0
    best_state = None
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, dl_train, optimizer,
                                      criterion_det, criterion_loc, DEVICE)
        val_metrics, *_ = evaluate(model, dl_val, DEVICE)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_prec'].append(val_metrics['precision'])
        history['val_rec'].append(val_metrics['recall'])

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_loss={val_metrics['loss']:.4f} | "
                  f"acc={val_metrics['accuracy']:.4f} | "
                  f"F1={val_metrics['f1']:.4f}")

    elapsed = time.time() - t0
    print(f"\n       Training selesai dalam {elapsed:.1f} s")

    # ------------------------------------------------------------
    # Evaluasi test dengan model terbaik
    # ------------------------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, y_true, y_pred, y_loc_true, y_loc_prob = evaluate(
        model, dl_test, DEVICE)

    print(f"\n--- Metrik Test ---")
    print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"  Precision : {test_metrics['precision']:.4f}")
    print(f"  Recall    : {test_metrics['recall']:.4f}")
    print(f"  F1        : {test_metrics['f1']:.4f}")
    print(f"  TP/FP/FN/TN: {test_metrics['tp']}/{test_metrics['fp']}/"
          f"{test_metrics['fn']}/{test_metrics['tn']}")

    # ------------------------------------------------------------
    # Localization accuracy (top-1)
    # ------------------------------------------------------------
    # Ambil window yang label detection = 1 dan punya minimal 1 node fault
    fault_mask = y_true == 1
    loc_top1_acc = 0.0
    loc_top3_acc = 0.0
    loc_mean_iou = 0.0
    n_loc = int(fault_mask.sum())

    if n_loc > 0:
        y_loc_true_f = y_loc_true[fault_mask]
        y_loc_prob_f = y_loc_prob[fault_mask]

        # Top-1: argmax probabilitas harus termasuk node yang fault
        top1_idx = y_loc_prob_f.argmax(axis=1)
        loc_top1_acc = float(y_loc_true_f[np.arange(n_loc), top1_idx].mean())

        # Top-3
        top3_idx = np.argsort(-y_loc_prob_f, axis=1)[:, :3]
        hits = 0
        for i in range(n_loc):
            if y_loc_true_f[i, top3_idx[i]].max() > 0:
                hits += 1
        loc_top3_acc = hits / n_loc

        # Mean IoU
        pred_bin = (y_loc_prob_f > 0.5).astype(np.float32)
        inter = (pred_bin * y_loc_true_f).sum(axis=1)
        union = ((pred_bin + y_loc_true_f) > 0).sum(axis=1)
        iou = inter / np.maximum(union, 1)
        loc_mean_iou = float(iou.mean())

    print(f"\n--- Localization (pada {n_loc} window fault) ---")
    print(f"  Top-1 accuracy : {loc_top1_acc:.4f}")
    print(f"  Top-3 accuracy : {loc_top3_acc:.4f}")
    print(f"  Mean IoU       : {loc_mean_iou:.4f}")

    # ------------------------------------------------------------
    # Simpan metrik ke JSON
    # ------------------------------------------------------------
    metrics_all = {
        'config': {
            'T_window': T_WINDOW, 'stride': STRIDE, 'batch_size': BATCH_SIZE,
            'epochs': EPOCHS, 'lr': LR, 'weight_decay': WEIGHT_DECAY,
            'lambda_loc': LAMBDA_LOC, 'n_params': n_params,
            'device': str(DEVICE),
        },
        'split': {
            'n_train_scenarios': len(splits['train']),
            'n_val_scenarios': len(splits['val']),
            'n_test_scenarios': len(splits['test']),
            'n_train_windows': len(data_split['train']['X']),
            'n_val_windows': len(data_split['val']['X']),
            'n_test_windows': len(data_split['test']['X']),
        },
        'test': test_metrics,
        'localization': {
            'top1_accuracy': loc_top1_acc,
            'top3_accuracy': loc_top3_acc,
            'mean_iou': loc_mean_iou,
            'n_fault_windows': n_loc,
        },
        'training_time_sec': elapsed,
        'history': history,
    }
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_JSON, 'w') as f:
        json.dump(metrics_all, f, indent=2)
    print(f"\n[info] Metrik disimpan: {METRICS_JSON}")

    # ------------------------------------------------------------
    # Plot training curves
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history['val_acc'], label='Accuracy')
    axes[1].plot(history['val_f1'], label='F1')
    axes[1].plot(history['val_prec'], label='Precision', alpha=0.6)
    axes[1].plot(history['val_rec'], label='Recall', alpha=0.6)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Validation Metrics')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    CURVES_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CURVES_PNG, dpi=150)
    plt.close(fig)
    print(f"[info] Training curves: {CURVES_PNG}")

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------
    cm = np.array([[test_metrics['tn'], test_metrics['fp']],
                   [test_metrics['fn'], test_metrics['tp']]])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Fault'])
    ax.set_yticklabels(['Normal', 'Fault'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix — Test Set')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=12)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=150)
    plt.close(fig)
    print(f"[info] Confusion matrix: {CM_PNG}")

    print("\n[done] Baseline LSTM selesai.")


if __name__ == '__main__':
    main()