"""
02_iv_validation_stats.py
Verifikasi hasil auto-tuning parameter PV.
"""
import sys, json
from pathlib import Path
from dataclasses import replace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pv_model import ModuleParams, analyze_module

print("=" * 60)
print("  02. VERIFIKASI AUTO-TUNING PARAMETER PV")
print("=" * 60)

# Parameter sebelum tuning
base = ModuleParams()
print(f"\n=== Parameter Sebelum Tuning ===")
print(f"R_s       : {base.R_s:.4f} Ohm")
print(f"R_sh      : {base.R_sh:.2f} Ohm")
print(f"I_o_ref   : {base.I_o_ref:.4e} A")

V_oc_b, I_sc_b, V_mpp_b, I_mpp_b, P_max_b = analyze_module(1000.0, 25.0, base)
print(f"\nHasil model (sebelum):")
print(f"  V_oc  : {V_oc_b:.3f} V (DS: {base.v_oc_ref})")
print(f"  I_sc  : {I_sc_b:.3f} A (DS: {base.i_sc_ref})")
print(f"  P_max : {P_max_b:.3f} W (DS: {base.pdc0})")
print(f"  Error V_oc  : {100*(V_oc_b-base.v_oc_ref)/base.v_oc_ref:+.2f}%")
print(f"  Error I_sc  : {100*(I_sc_b-base.i_sc_ref)/base.i_sc_ref:+.2f}%")
print(f"  Error P_max : {100*(P_max_b-base.pdc0)/base.pdc0:+.2f}%")

# Parameter setelah tuning (dari file)
try:
    from src.pv_model_tuned import TUNED_PARAMS as tuned
    print(f"\n=== Parameter Sesudah Tuning ===")
    print(f"R_s       : {tuned.R_s:.4f} Ohm")
    print(f"R_sh      : {tuned.R_sh:.2f} Ohm")
    print(f"I_o_ref   : {tuned.I_o_ref:.4e} A")

    V_oc_a, I_sc_a, V_mpp_a, I_mpp_a, P_max_a = analyze_module(1000.0, 25.0, tuned)
    print(f"\nHasil model (sesudah):")
    print(f"  V_oc  : {V_oc_a:.3f} V (DS: {tuned.v_oc_ref})")
    print(f"  I_sc  : {I_sc_a:.3f} A (DS: {tuned.i_sc_ref})")
    print(f"  V_mpp : {V_mpp_a:.3f} V (DS: {tuned.v_mp_ref})")
    print(f"  I_mpp : {I_mpp_a:.3f} A (DS: {tuned.i_mp_ref})")
    print(f"  P_max : {P_max_a:.3f} W (DS: {tuned.pdc0})")
    print(f"\nError (sesudah):")
    err_voc = 100*(V_oc_a-tuned.v_oc_ref)/tuned.v_oc_ref
    err_isc = 100*(I_sc_a-tuned.i_sc_ref)/tuned.i_sc_ref
    err_vmpp = 100*(V_mpp_a-tuned.v_mp_ref)/tuned.v_mp_ref
    err_impp = 100*(I_mpp_a-tuned.i_mp_ref)/tuned.i_mp_ref
    err_pmax = 100*(P_max_a-tuned.pdc0)/tuned.pdc0
    print(f"  V_oc  : {err_voc:+.2f}%  {'OK' if abs(err_voc)<5 else 'CEK'}")
    print(f"  I_sc  : {err_isc:+.2f}%  {'OK' if abs(err_isc)<5 else 'CEK'}")
    print(f"  V_mpp : {err_vmpp:+.2f}%  {'OK' if abs(err_vmpp)<5 else 'CEK'}")
    print(f"  I_mpp : {err_impp:+.2f}%  {'OK' if abs(err_impp)<5 else 'CEK'}")
    print(f"  P_max : {err_pmax:+.2f}%  {'OK' if abs(err_pmax)<5 else 'CEK'}")

    print(f"\n=== Perbaikan Error ===")
    print(f"  V_oc  : {100*(V_oc_b-base.v_oc_ref)/base.v_oc_ref:+.2f}% -> "
          f"{err_voc:+.2f}%  (perbaikan "
          f"{abs(100*(V_oc_b-base.v_oc_ref)/base.v_oc_ref) - abs(err_voc):.2f} pp)")
    print(f"  P_max : {100*(P_max_b-base.pdc0)/base.pdc0:+.2f}% -> "
          f"{err_pmax:+.2f}%  (perbaikan "
          f"{abs(100*(P_max_b-base.pdc0)/base.pdc0) - abs(err_pmax):.2f} pp)")
except ImportError:
    print("\n[WARNING] src/pv_model_tuned.py tidak ditemukan")

# Tuning report
TR = ROOT / "logs" / "tuning_report.json"
if TR.exists():
    with open(TR) as f:
        info = json.load(f)
    print(f"\n=== Tuning Report ===")
    for k, v in info.items():
        print(f"  {k}: {v}")