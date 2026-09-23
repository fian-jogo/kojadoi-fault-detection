"""
06_split_verification.py
Verifikasi train/val/test split.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLIT_NPZ = ROOT / "data" / "processed" / "train_val_test_split.npz"

print("=" * 60)
print("  06. VERIFIKASI TRAIN/VAL/TEST SPLIT")
print("=" * 60)

if not SPLIT_NPZ.exists():
    print(f"[ERROR] File tidak ditemukan: {SPLIT_NPZ}")
    sys.exit(1)

data = np.load(SPLIT_NPZ)

print(f"\n=== Ukuran Split ===")
for name in ['train', 'val', 'test']:
    X = data[f'X_{name}']
    y = data[f'y_det_{name}']
    print(f"{name:>5}: X={X.shape}, y_det={y.shape}, "
          f"positif={int(y.sum())} ({100*y.mean():.1f}%)")

print(f"\n=== Normalisasi Statistik ===")
mean = data['mean']
std = data['std']
names = ['V', 'I', 'P', 'G', 'T']
for i, n in enumerate(names):
    print(f"  {n}: mean={mean[i]:.4f}, std={std[i]:.4f}")

sid_train = set(data['sid_train'].tolist())
sid_val = set(data['sid_val'].tolist())
sid_test = set(data['sid_test'].tolist())

print(f"\n=== Verifikasi Data Leakage ===")
print(f"Overlap train-val : {len(sid_train & sid_val)} "
      f"{'(OK)' if len(sid_train & sid_val) == 0 else '(CEK)'}")
print(f"Overlap train-test: {len(sid_train & sid_test)} "
      f"{'(OK)' if len(sid_train & sid_test) == 0 else '(CEK)'}")
print(f"Overlap val-test  : {len(sid_val & sid_test)} "
      f"{'(OK)' if len(sid_val & sid_test) == 0 else '(CEK)'}")

print(f"\n=== Distribusi Skenario ===")
print(f"Train : {len(sid_train)} skenario")
print(f"Val   : {len(sid_val)} skenario")
print(f"Test  : {len(sid_test)} skenario")
print(f"Total : {len(sid_train)+len(sid_val)+len(sid_test)} skenario")

print(f"\n=== Cek Range Nilai Setelah Normalisasi ===")
X_train = data['X_train']
print(f"X_train:")
print(f"  min  : {X_train.min():.3f}")
print(f"  max  : {X_train.max():.3f}")
print(f"  mean : {X_train.mean():.3f}")
print(f"  std  : {X_train.std():.3f}")

print(f"\n=== Cek Konsistensi Train-Val-Test ===")
n_total = len(sid_train) + len(sid_val) + len(sid_test)
print(f"Total skenario unik: {n_total}")
print(f"Harus 180: {'OK' if n_total == 180 else 'CEK'}")