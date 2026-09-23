"""
15_environment_verification.py
Verifikasi environment dan reproducibility.
"""
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import N_NODES_TOTAL, N_FEATURES
from src.model_hybrid import HybridDetector

print("=" * 60)
print("  15. VERIFIKASI ENVIRONMENT DAN REPRODUCIBILITY")
print("=" * 60)

print(f"\n=== Versi Library ===")
print(f"torch   : {torch.__version__}")
print(f"numpy   : {np.__version__}")
print(f"pandas  : {pd.__version__}")
try:
    import pvlib
    print(f"pvlib   : {pvlib.__version__}")
except ImportError:
    print(f"pvlib   : tidak terpasang")
try:
    import matplotlib
    print(f"matplotlib: {matplotlib.__version__}")
except ImportError:
    print(f"matplotlib: tidak terpasang")
try:
    import scipy
    print(f"scipy   : {scipy.__version__}")
except ImportError:
    print(f"scipy   : tidak terpasang")

print(f"\n=== Device ===")
print(f"CUDA tersedia  : {torch.cuda.is_available()}")
print(f"Device         : {'cuda' if torch.cuda.is_available() else 'cpu'}")

print(f"\n=== Checkpoint Model ===")
CKPT = ROOT / "logs" / "best_lstm_gat.pt"
if CKPT.exists():
    ckpt = torch.load(CKPT, weights_only=False)
    print(f"File         : {CKPT}")
    print(f"Ukuran       : {CKPT.stat().st_size / 1024:.1f} KB")
    print(f"best_val_f1  : {ckpt.get('best_val_f1', 'N/A')}")
    print(f"norm_mean    : {ckpt['norm_mean']}")
    print(f"norm_std     : {ckpt['norm_std']}")
else:
    print(f"Checkpoint tidak ditemukan")

print(f"\n=== Parameter Model per Konfigurasi ===")
for use_gat, use_pinn, name in [(False, False, 'LSTM_baseline'),
                                  (True, False, 'LSTM+GAT'),
                                  (True, True, 'LSTM+GAT+PINN')]:
    m = HybridDetector(N_NODES_TOTAL, N_FEATURES, 64, 2, 2, 4, 0.3,
                        use_gat, use_pinn)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"  {name:<20}: {n_params:,} parameter")

print(f"\n=== Hyperparameter LAPORAN AKHIR ===")
print(f"  T_WINDOW       : 30")
print(f"  BATCH_SIZE     : 128")
print(f"  EPOCHS         : 30")
print(f"  LR             : 1e-3")
print(f"  WEIGHT_DECAY   : 1e-4")
print(f"  LAMBDA_LOC     : 1.0")
print(f"  LAMBDA_PHYS    : 0.01")
print(f"  WARMUP_EPOCHS  : 10")
print(f"  SEED           : 42")

print(f"\n=== Struktur Folder ===")
for subdir in ['src', 'scripts', 'data/raw', 'data/processed',
                'data/interface', 'figures', 'logs', 'analysis']:
    p = ROOT / subdir
    if p.exists():
        n_files = len(list(p.glob('*')))
        print(f"  {subdir:<20}: {n_files} file")
    else:
        print(f"  {subdir:<20}: TIDAK ADA")