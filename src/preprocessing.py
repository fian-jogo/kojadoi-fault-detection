"""
preprocessing.py
Loading, normalisasi, sliding window, dan adjacency matrix
untuk dataset fault PLTS Koja Doi.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
_ROOT = _SRC_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ----------------------------------------------------------------
# Konstanta
# ----------------------------------------------------------------
N_STRINGS = 3
N_NODES_PER_STRING = 4
N_NODES_TOTAL = N_STRINGS * N_NODES_PER_STRING  # 12
N_FEATURES = 5
FEATURE_COLS = ['V', 'I', 'P', 'G', 'T']


def load_fault_dataset(csv_path):
    """Load fault_dataset.csv (long format)."""
    df = pd.read_csv(csv_path)
    n_scenarios = df['scenario_id'].nunique()
    n_steps = df['time_step'].nunique()
    return df, n_scenarios, n_steps


def pivot_to_tensor(df, n_scenarios, n_steps):
    """
    Pivot DataFrame long ke tensor (n_scenarios, n_steps, N_NODES_TOTAL, N_FEATURES).
    Juga return label arrays.
    """
    df = df.copy()
    df['node_global'] = df['string'] * N_NODES_PER_STRING + df['node']

    # Index flat untuk scatter
    idx = (df['scenario_id'].values * n_steps * N_NODES_TOTAL
           + df['time_step'].values * N_NODES_TOTAL
           + df['node_global'].values)

    # Scatter features
    flat = np.zeros(n_scenarios * n_steps * N_NODES_TOTAL * N_FEATURES,
                    dtype=np.float32)
    for fi, col in enumerate(FEATURE_COLS):
        flat[idx * N_FEATURES + fi] = df[col].values.astype(np.float32)

    X = flat.reshape(n_scenarios, n_steps, N_NODES_TOTAL, N_FEATURES)

    # ------------------------------------------------------------
    # Labels per time step
    # ------------------------------------------------------------
    # Ambil hanya node_global=0 untuk label (karena sama semua node dalam 1 skenario)
    label_df = df[df['node_global'] == 0][
        ['scenario_id', 'time_step', 'fault_active', 'fault_type',
         'fault_type_name', 'onset', 'duration', 'affected_nodes']
    ].sort_values(['scenario_id', 'time_step']).reset_index(drop=True)

    y_detect = label_df['fault_active'].values.astype(np.int64).reshape(
        n_scenarios, n_steps)
    y_type = label_df['fault_type'].values.astype(np.int64).reshape(
        n_scenarios, n_steps)

    # Localization labels
    y_loc = np.zeros((n_scenarios, n_steps, N_NODES_TOTAL), dtype=np.float32)

    metadata = {}
    for sid in range(n_scenarios):
        sub = label_df[label_df['scenario_id'] == sid]
        meta = sub.iloc[0]
        affected = meta['affected_nodes']
        if isinstance(affected, str):
            try:
                affected = eval(affected)
            except Exception:
                affected = []
        affected = [int(x) for x in affected]

        metadata[sid] = {
            'fault_type': str(meta['fault_type_name']),
            'onset': int(meta['onset']),
            'duration': int(meta['duration']),
            'affected_nodes': affected,
        }

        # Isi y_loc pada time step fault aktif
        for t in range(n_steps):
            if sub.iloc[t]['fault_active'] == 1:
                for n in affected:
                    if 0 <= n < N_NODES_TOTAL:
                        y_loc[sid, t, n] = 1.0

    return X, y_detect, y_type, y_loc, metadata


def compute_normalization_stats(X_train):
    """Hitung mean dan std per fitur dari training set saja."""
    mean = X_train.mean(axis=(0, 1, 2))  # (N_FEATURES,)
    std = X_train.std(axis=(0, 1, 2))
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(X, mean, std):
    """Normalisasi fitur dengan broadcasting."""
    return ((X - mean) / std).astype(np.float32)


def build_adjacency_matrix():
    """
    Bangun adjacency matrix (12 node) dengan:
        - Series edge dalam string: (s, n)-(s, n+1)
        - Cross-string edge pada posisi sama: (s1, n)-(s2, n)
        - Self-loop
    Return: A_norm (12, 12) dengan symmetric normalization D^-1/2 A D^-1/2.
    """
    A = np.eye(N_NODES_TOTAL, dtype=np.float32)

    # Series
    for s in range(N_STRINGS):
        for n in range(N_NODES_PER_STRING - 1):
            i = s * N_NODES_PER_STRING + n
            j = s * N_NODES_PER_STRING + (n + 1)
            A[i, j] = 1.0
            A[j, i] = 1.0

    # Cross-string
    for n in range(N_NODES_PER_STRING):
        for s1 in range(N_STRINGS):
            for s2 in range(s1 + 1, N_STRINGS):
                i = s1 * N_NODES_PER_STRING + n
                j = s2 * N_NODES_PER_STRING + n
                A[i, j] = 1.0
                A[j, i] = 1.0

    # Symmetric normalization
    deg = A.sum(axis=1, keepdims=True)
    deg = np.where(deg < 1e-6, 1.0, deg)
    A_norm = A / np.sqrt(deg * deg.T)
    return A_norm.astype(np.float32)


def make_sliding_windows(X, y_detect, y_type, y_loc, T=30, stride=1):
    """
    Buat sliding windows dari tensor skenario.
    Return: X_win, y_detect_win, y_type_win, y_loc_win, t_last_list, sid_list.
    """
    n_scenarios, n_steps, n_nodes, n_features = X.shape
    n_win_per_scenario = (n_steps - T) // stride + 1

    X_win = np.zeros((n_scenarios * n_win_per_scenario, T, n_nodes, n_features),
                     dtype=np.float32)
    y_detect_win = np.zeros(n_scenarios * n_win_per_scenario, dtype=np.int64)
    y_type_win = np.zeros(n_scenarios * n_win_per_scenario, dtype=np.int64)
    y_loc_win = np.zeros((n_scenarios * n_win_per_scenario, n_nodes),
                          dtype=np.float32)
    t_last_list = np.zeros(n_scenarios * n_win_per_scenario, dtype=np.int64)
    sid_list = np.zeros(n_scenarios * n_win_per_scenario, dtype=np.int64)

    k = 0
    for sid in range(n_scenarios):
        for w in range(n_win_per_scenario):
            t_start = w * stride
            t_end = t_start + T
            t_last = t_end - 1

            X_win[k] = X[sid, t_start:t_end]
            y_detect_win[k] = y_detect[sid, t_last]
            y_type_win[k] = y_type[sid, t_last]
            y_loc_win[k] = y_loc[sid, t_last]
            t_last_list[k] = t_last
            sid_list[k] = sid
            k += 1

    return X_win, y_detect_win, y_type_win, y_loc_win, t_last_list, sid_list


def split_by_scenario(metadata, train_frac=0.7, val_frac=0.15, seed=42):
    """
    Split skenario (bukan window) menjadi train/val/test, stratifikasi
    berdasarkan fault_type. Semua window dari satu skenario masuk ke split
    yang sama untuk menghindari data leakage.
    """
    rng = np.random.default_rng(seed)

    groups = {}
    for sid, meta in metadata.items():
        ft = meta['fault_type']
        groups.setdefault(ft, []).append(sid)

    splits = {'train': [], 'val': [], 'test': []}
    for ft, sids in groups.items():
        sids = np.array(sorted(sids))
        rng.shuffle(sids)
        n = len(sids)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        splits['train'].extend(sids[:n_train].tolist())
        splits['val'].extend(sids[n_train:n_train + n_val].tolist())
        splits['test'].extend(sids[n_train + n_val:].tolist())

    return splits


def select_windows_by_scenario(X_win, y_detect, y_type, y_loc, sid_win, sids):
    """Filter window berdasarkan daftar scenario_id."""
    mask = np.isin(sid_win, sids)
    return (X_win[mask], y_detect[mask], y_type[mask], y_loc[mask],
            sid_win[mask])