"""
train_hybrid.py
Training hybrid model (LSTM + GAT + PINN v3) dengan ablation study.

PINN v3 (fixed):
    - Physics head memprediksi (V, I) dalam satuan fisik (denormalisasi)
    - Physics loss = residual persamaan diode pada prediksi itu
    - Clamp V ∈ [0, 60] V, I ∈ [0, 15] A untuk stabilitas numerik
    - LAMBDA_PHYS = 0.01 (sangat kecil)
    - Curriculum: warmup 10 epoch tanpa physics
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print(f"[debug] ROOT = {ROOT}")

from src.preprocessing import N_NODES_TOTAL, N_FEATURES, build_adjacency_matrix
from src.dataset import FaultWindowDataset
from src.model_hybrid import HybridDetector, diode_residual_loss, MODULE_PARAMS


# ================================================================
# Konfigurasi
# ================================================================
SPLIT_NPZ = ROOT / "data" / "processed" / "train_val_test_split.npz"
ABLATION_JSON = ROOT / "logs" / "ablation_results.json"
ABLATION_PNG = ROOT / "figures" / "ablation_study.png"
CURVES_PNG = ROOT / "figures" / "training_curves_hybrid.png"

BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LAMBDA_LOC = 1.0
LAMBDA_PHYS = 0.01          # sangat kecil
WARMUP_EPOCHS = 10

ABLATION_CONFIGS = [
    {'name': 'LSTM_baseline',       'use_gat': False, 'use_pinn': False, 'frac': 1.00},
    {'name': 'LSTM+GAT',            'use_gat': True,  'use_pinn': False, 'frac': 1.00},
    {'name': 'LSTM+GAT+PINN',       'use_gat': True,  'use_pinn': True,  'frac': 1.00},
    {'name': 'LSTM+GAT_50pct',      'use_gat': True,  'use_pinn': False, 'frac': 0.50},
    {'name': 'LSTM+GAT_20pct',      'use_gat': True,  'use_pinn': False, 'frac': 0.20},
    {'name': 'LSTM+GAT+PINN_50pct', 'use_gat': True,  'use_pinn': True,  'frac': 0.50},
    {'name': 'LSTM+GAT+PINN_20pct', 'use_gat': True,  'use_pinn': True,  'frac': 0.20},
]


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


def subsample_train(X, y_det, y_loc, sid, frac, seed=42):
    if frac >= 1.0:
        return X, y_det, y_loc, sid
    rng = np.random.default_rng(seed)
    unique_sids = np.unique(sid)
    n_keep = max(2, int(len(unique_sids) * frac))
    keep_sids = rng.choice(unique_sids, size=n_keep, replace=False)
    mask = np.isin(sid, keep_sids)
    return X[mask], y_det[mask], y_loc[mask], sid[mask]


# ================================================================
# Training
# ================================================================
def train_one_epoch(model, loader, optimizer, adj, device,
                    lambda_loc, lambda_phys, use_pinn,
                    norm_mean_t, norm_std_t):
    model.train()
    total_loss = 0.0
    n = 0
    crit_det = nn.CrossEntropyLoss()
    crit_loc = nn.BCEWithLogitsLoss()

    for X, y_det, y_loc in loader:
        X = X.to(device)
        y_det = y_det.to(device)
        y_loc = y_loc.to(device)
        adj_d = adj.to(device)

        optimizer.zero_grad()
        det_logits, loc_logits, phys_pred = model(X, adj_d)

        loss_det = crit_det(det_logits, y_det)
        loss_loc = crit_loc(loc_logits, y_loc)
        loss = loss_det + lambda_loc * loss_loc

        # PINN v3: residual diode pada prediksi (V, I)
        if use_pinn and phys_pred is not None and lambda_phys > 0.0:
            V_pred = phys_pred[..., 0]   # (B, N) sudah di-clamp
            I_pred = phys_pred[..., 1]

            # Denormalisasi G dari input
            G_obs_norm = X[:, -1, :, 3]  # (B, N)
            G_obs = G_obs_norm * norm_std_t[3] + norm_mean_t[3]
            G_obs = torch.clamp(G_obs, min=0.0)

            loss_phys = diode_residual_loss(
                V_pred, I_pred, G_obs, MODULE_PARAMS
            )
            loss = loss + lambda_phys * loss_phys

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        n += X.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, adj, device):
    model.eval()
    y_true_all, y_pred_all = [], []
    y_loc_true_all, y_loc_prob_all = [], []
    crit_det = nn.CrossEntropyLoss()
    crit_loc = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n = 0

    for X, y_det, y_loc in loader:
        X = X.to(device)
        y_det = y_det.to(device)
        y_loc = y_loc.to(device)
        adj_d = adj.to(device)

        det_logits, loc_logits, _ = model(X, adj_d)
        loss = crit_det(det_logits, y_det) + LAMBDA_LOC * crit_loc(loc_logits, y_loc)
        total_loss += loss.item() * X.size(0)
        n += X.size(0)

        y_true_all.append(y_det.cpu().numpy())
        y_pred_all.append(det_logits.argmax(dim=1).cpu().numpy())
        y_loc_true_all.append(y_loc.cpu().numpy())
        y_loc_prob_all.append(torch.sigmoid(loc_logits).cpu().numpy())

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    y_loc_true_all = np.concatenate(y_loc_true_all)
    y_loc_prob_all = np.concatenate(y_loc_prob_all)

    metrics = compute_metrics(y_true_all, y_pred_all)
    metrics['loss'] = total_loss / max(n, 1)

    fault_mask = y_true_all == 1
    n_loc = int(fault_mask.sum())
    loc_metrics = {'n_fault_windows': n_loc}

    if n_loc > 0:
        y_loc_true_f = y_loc_true_all[fault_mask]
        y_loc_prob_f = y_loc_prob_all[fault_mask]

        top1_idx = y_loc_prob_f.argmax(axis=1)
        loc_metrics['top1_accuracy'] = float(
            y_loc_true_f[np.arange(n_loc), top1_idx].mean())

        top3_idx = np.argsort(-y_loc_prob_f, axis=1)[:, :3]
        hits = sum(1 for i in range(n_loc)
                   if y_loc_true_f[i, top3_idx[i]].max() > 0)
        loc_metrics['top3_accuracy'] = hits / n_loc

        pred_bin = (y_loc_prob_f > 0.5).astype(np.float32)
        inter = (pred_bin * y_loc_true_f).sum(axis=1)
        union = ((pred_bin + y_loc_true_f) > 0).sum(axis=1)
        loc_metrics['mean_iou'] = float((inter / np.maximum(union, 1)).mean())

    return metrics, loc_metrics


# ================================================================
# Run one config
# ================================================================
def run_config(cfg, data_split, adj, device, norm_mean, norm_std):
    name = cfg['name']
    print(f"\n{'='*60}")
    print(f"  Konfigurasi: {name}")
    print(f"  use_gat={cfg['use_gat']}, use_pinn={cfg['use_pinn']}, "
          f"frac={cfg['frac']}")
    print(f"{'='*60}")

    set_seed(SEED)

    X_train, y_det_train, y_loc_train, sid_train = subsample_train(
        data_split['train']['X'], data_split['train']['y_det'],
        data_split['train']['y_loc'], data_split['train']['sid'],
        frac=cfg['frac'], seed=SEED,
    )
    print(f"  Train window: {len(X_train)} "
          f"(dari {len(data_split['train']['X'])})")

    ds_train = FaultWindowDataset(X_train, y_det_train, y_loc_train)
    ds_val = FaultWindowDataset(data_split['val']['X'],
                                 data_split['val']['y_det'],
                                 data_split['val']['y_loc'])
    ds_test = FaultWindowDataset(data_split['test']['X'],
                                  data_split['test']['y_det'],
                                  data_split['test']['y_loc'])

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

    model = HybridDetector(
        n_nodes=N_NODES_TOTAL, n_features=N_FEATURES,
        hidden=64, n_layers=2, n_gat=2, n_heads=4,
        dropout=0.3,
        use_gat=cfg['use_gat'],
        use_pinn=cfg['use_pinn'],
        norm_mean=norm_mean,
        norm_std=norm_std,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameter: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)

    norm_mean_t = torch.tensor(norm_mean, dtype=torch.float32)
    norm_std_t = torch.tensor(norm_std, dtype=torch.float32)

    history = {'train_loss': [], 'val_f1': [], 'val_acc': [], 'val_rec': [],
               'lambda_phys_eff': []}
    best_val_f1 = -1.0
    best_state = None
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        if cfg['use_pinn'] and epoch > WARMUP_EPOCHS:
            progress = min((epoch - WARMUP_EPOCHS) / 10.0, 1.0)
            lambda_phys_eff = LAMBDA_PHYS * progress
        else:
            lambda_phys_eff = 0.0

        train_loss = train_one_epoch(
            model, dl_train, optimizer, adj, device,
            LAMBDA_LOC, lambda_phys_eff, cfg['use_pinn'],
            norm_mean_t, norm_std_t,
        )
        val_metrics, _ = evaluate(model, dl_val, adj, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_f1'].append(val_metrics['f1'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_rec'].append(val_metrics['recall'])
        history['lambda_phys_eff'].append(lambda_phys_eff)

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"train_loss={train_loss:.4f} | "
                  f"F1={val_metrics['f1']:.4f} | "
                  f"acc={val_metrics['accuracy']:.4f} | "
                  f"λ={lambda_phys_eff:.4f}")

    elapsed = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, loc_metrics = evaluate(model, dl_test, adj, device)

    print(f"\n  Test F1: {test_metrics['f1']:.4f} | "
          f"Acc: {test_metrics['accuracy']:.4f} | "
          f"Rec: {test_metrics['recall']:.4f} | "
          f"Prec: {test_metrics['precision']:.4f}")
    if 'top1_accuracy' in loc_metrics:
        print(f"  Top-1 Loc: {loc_metrics['top1_accuracy']:.4f} | "
              f"Top-3 Loc: {loc_metrics['top3_accuracy']:.4f} | "
              f"IoU: {loc_metrics['mean_iou']:.4f}")
    print(f"  Training time: {elapsed:.1f} s")

    return {
        'name': name, 'config': cfg, 'n_params': n_params,
        'n_train_windows': int(len(X_train)),
        'test': test_metrics, 'localization': loc_metrics,
        'training_time_sec': elapsed, 'history': history,
    }


# ================================================================
# Main
# ================================================================
def main():
    print(f"[info] Device: {DEVICE}")

    print(f"\n[1/3] Load split ...")
    data = np.load(SPLIT_NPZ)
    norm_mean = data['mean']
    norm_std = data['std']
    print(f"  norm_mean = {norm_mean}")
    print(f"  norm_std  = {norm_std}")

    data_split = {
        'train': {'X': data['X_train'], 'y_det': data['y_det_train'],
                  'y_loc': data['y_loc_train'], 'sid': data['sid_train']},
        'val':   {'X': data['X_val'], 'y_det': data['y_det_val'],
                  'y_loc': data['y_loc_val'], 'sid': data['sid_val']},
        'test':  {'X': data['X_test'], 'y_det': data['y_det_test'],
                  'y_loc': data['y_loc_test'], 'sid': data['sid_test']},
    }
    print(f"  train: {len(data_split['train']['X'])} window")

    adj = torch.from_numpy(build_adjacency_matrix()).float()
    print(f"  adj shape: {adj.shape}, nnz: {(adj > 0).sum().item()}")

    # Sanity check physics residual pada data normal
    print(f"\n[sanity] Physics residual pada data normal:")
    with torch.no_grad():
        X_sanity = torch.from_numpy(
            data_split['test']['X'][:64]).float().to(DEVICE)
        V_norm = X_sanity[:, -1, :, 0]
        I_norm = X_sanity[:, -1, :, 1]
        G_norm = X_sanity[:, -1, :, 3]
        V_phys = V_norm * float(norm_std[0]) + float(norm_mean[0])
        I_phys = I_norm * float(norm_std[1]) + float(norm_mean[1])
        G_phys = torch.clamp(
            G_norm * float(norm_std[3]) + float(norm_mean[3]), min=0.0)
        r = diode_residual_loss(V_phys, I_phys, G_phys)
        print(f"  residual mean = {r.item():.6e} (harusnya kecil, < 1)")

    print(f"\n[2/3] Menjalankan {len(ABLATION_CONFIGS)} konfigurasi ...")
    results = []
    for cfg in ABLATION_CONFIGS:
        result = run_config(cfg, data_split, adj, DEVICE,
                             norm_mean, norm_std)
        results.append(result)

    ABLATION_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(ABLATION_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[info] Hasil ablation disimpan: {ABLATION_JSON}")

    print(f"\n[3/3] Ringkasan")
    print(f"\n{'Config':<25}{'F1':>8}{'Acc':>8}{'Rec':>8}{'Top1Loc':>10}{'IoU':>8}")
    print("-" * 70)
    for r in results:
        t = r['test']
        loc = r['localization']
        print(f"{r['name']:<25}{t['f1']:>8.4f}{t['accuracy']:>8.4f}"
              f"{t['recall']:>8.4f}"
              f"{loc.get('top1_accuracy', 0):>10.4f}"
              f"{loc.get('mean_iou', 0):>8.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    for use_pinn, color, label in [(False, 'tab:blue', 'LSTM+GAT'),
                                    (True, 'tab:red', 'LSTM+GAT+PINN')]:
        xs, ys = [], []
        for r in results:
            if r['config']['use_pinn'] == use_pinn and r['config']['use_gat'] \
               and not r['config']['name'].startswith('LSTM_baseline'):
                xs.append(r['config']['frac'] * 100)
                ys.append(r['test']['f1'])
        if xs:
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], 'o-',
                    color=color, label=label, lw=2, ms=8)
    ax.set_xlabel('Fraksi data training (%)')
    ax.set_ylabel('Test F1')
    ax.set_title('Ketahanan terhadap Keterbatasan Data')
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    for use_pinn, color, label in [(False, 'tab:blue', 'LSTM+GAT'),
                                    (True, 'tab:red', 'LSTM+GAT+PINN')]:
        xs, ys = [], []
        for r in results:
            if r['config']['use_pinn'] == use_pinn and r['config']['use_gat'] \
               and not r['config']['name'].startswith('LSTM_baseline'):
                xs.append(r['config']['frac'] * 100)
                ys.append(r['localization'].get('top1_accuracy', 0))
        if xs:
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], 'o-',
                    color=color, label=label, lw=2, ms=8)
    ax.set_xlabel('Fraksi data training (%)')
    ax.set_ylabel('Top-1 Localization Accuracy')
    ax.set_title('Kualitas Lokalisasi vs Data')
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[2]
    names = [r['name'].replace('LSTM+', '').replace('LSTM_', '')
             for r in results]
    f1s = [r['test']['f1'] for r in results]
    colors = ['tab:gray' if r['config']['frac'] == 1.0 and not r['config']['use_gat']
              else ('tab:red' if r['config']['use_pinn'] else 'tab:blue')
              for r in results]
    ax.barh(range(len(names)), f1s, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Test F1')
    ax.set_title('Semua Konfigurasi')
    ax.grid(alpha=0.3, axis='x'); ax.set_xlim(0, 1)

    plt.tight_layout()
    ABLATION_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ABLATION_PNG, dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for r in results:
        if r['config']['frac'] == 1.0:
            axes[0].plot(r['history']['train_loss'], label=r['name'], lw=1.5)
            axes[1].plot(r['history']['val_f1'], label=r['name'], lw=1.5)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Train Loss')
    axes[0].set_title('Training Loss'); axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3); axes[0].set_yscale('log')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Val F1')
    axes[1].set_title('Validation F1'); axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURVES_PNG, dpi=150); plt.close(fig)

    print(f"\n[done] Ablation study selesai.")


if __name__ == '__main__':
    main()