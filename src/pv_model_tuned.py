"""
pv_model_tuned.py — parameter modul hasil auto-tuning.
Di-generate otomatis oleh validate_iv.py.
"""
from dataclasses import replace
from src.pv_model import ModuleParams

TUNED_PARAMS = replace(
    ModuleParams(),
    R_s=0.10652582558386603,
    R_sh=528536054.7642789,
    I_o_ref=2.5021780533197524e-09,
)

# Metadata tuning:
# J_opt = 5.858754e-01
# Converged = True
# Iterations = 321
