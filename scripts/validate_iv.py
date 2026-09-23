"""
validate_iv.py
Validasi kurva I-V dan P-V model single-diode terhadap datasheet 400 Wp.
Auto-tuning R_s, R_sh, dan I_o_ref menggunakan scipy.optimize.minimize.

Output:
    figures/iv_curve_validation.png     (sebelum & sesudah tuning)
    src/pv_model_tuned.py               (parameter optimal)
    logs/tuning_report.txt              (log iterasi)
"""
import sys
import json
from pathlib import Path
from dataclasses import replace, asdict

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# Deteksi ROOT otomatis
# ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pv_model import ModuleParams, analyze_module, iv_curve_module

from scipy.optimize import minimize


# ================================================================
# 1. Fungsi objektif untuk tuning
# ================================================================
def objective(x, base_params):
    """
    x = [R_s, R_sh, I_o_ref]
    Return weighted sum of squared normalized errors.
    """
    Rs, Rsh, I0 = x

    # Guard: nilai harus positif dan dalam rentang wajar
    if Rs <= 0 or Rsh <= 0 or I0 <= 0:
        return 1e9
    if not (np.isfinite(Rs) and np.isfinite(Rsh) and np.isfinite(I0)):
        return 1e9

    # Buat salinan params dengan nilai baru (tidak memutasi base_params)
    p = replace(base_params, R_s=float(Rs), R_sh=float(Rsh), I_o_ref=float(I0))

    try:
        V_oc, I_sc, V_mpp, I_mpp, P_max = analyze_module(1000.0, 25.0, p)
    except Exception:
        return 1e9

    # Cek nilai fisis
    if not all(np.isfinite(v) for v in [V_oc, I_sc, V_mpp, I_mpp, P_max]):
        return 1e9
    if V_oc <= 0 or P_max <= 0:
        return 1e9

    # Error ternormalisasi
    err_voc  = (V_oc  - base_params.v_oc_ref) / base_params.v_oc_ref
    err_vmpp = (V_mpp - base_params.v_mp_ref) / base_params.v_mp_ref
    err_pmax = (P_max - base_params.pdc0)    / base_params.pdc0

    # Bobot: prioritaskan Voc dan Pmax
    w_voc, w_vmpp, w_pmax = 2.0, 1.0, 2.0
    return w_voc*err_voc**2 + w_vmpp*err_vmpp**2 + w_pmax*err_pmax**2


def tune_params(base_params, verbose=True):
    """
    Tuning 2 tahap: Nelder-Mead lalu L-BFGS-B.
    Return (params_optimal, hasil_optimasi_dict).
    """
    x0 = np.array([base_params.R_s, base_params.R_sh, base_params.I_o_ref])

    # Rentang wajar
    bounds = [
        (0.05, 1.50),        # R_s: 0.05–1.5 Ω
        (50.0, 2000.0),      # R_sh: 50–2000 Ω
        (1e-12, 1e-7),       # I_o_ref: 1e-12 – 1e-7 A
    ]

    # ---- Tahap 1: Nelder-Mead (global-ish, tanpa gradien) ----
    if verbose:
        print("\n[Tuning Tahap 1] Nelder-Mead ...")
    res_nm = minimize(
        objective, x0, args=(base_params,),
        method='Nelder-Mead',
        options={'xatol': 1e-6, 'fatol': 1e-10, 'maxiter': 2000, 'disp': verbose},
    )

    # ---- Tahap 2: L-BFGS-B (lokal, dengan gradien numerik) ----
    if verbose:
        print("\n[Tuning Tahap 2] L-BFGS-B (refinement) ...")
    res_lbfgs = minimize(
        objective, res_nm.x, args=(base_params,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 500, 'disp': verbose},
    )

    # Pilih hasil terbaik
    best = res_lbfgs if res_lbfgs.fun < res_nm.fun else res_nm

    Rs_opt, Rsh_opt, I0_opt = best.x
    params_opt = replace(
        base_params,
        R_s=float(Rs_opt),
        R_sh=float(Rsh_opt),
        I_o_ref=float(I0_opt),
    )

    info = {
        'x0': x0.tolist(),
        'Rs_opt': float(Rs_opt),
        'Rsh_opt': float(Rsh_opt),
        'Io_opt': float(I0_opt),
        'J_opt': float(best.fun),
        'n_iter': int(best.nit),
        'converged': bool(best.success),
        'message': str(best.message),
    }
    return params_opt, info


# ================================================================
# 2. Helper: cetak tabel perbandingan
# ================================================================
def print_comparison(label, params, G_test=1000.0, T_test=25.0):
    V_oc, I_sc, V_mpp, I_mpp, P_max = analyze_module(G_test, T_test, params)

    ds = {
        'Voc':  params.v_oc_ref,
        'Isc':  params.i_sc_ref,
        'Vmpp': params.v_mp_ref,
        'Impp': params.i_mp_ref,
        'Pmax': params.pdc0,
    }
    sim = {
        'Voc': V_oc, 'Isc': I_sc,
        'Vmpp': V_mpp, 'Impp': I_mpp,
        'Pmax': P_max,
    }

    print(f"\n=== {label} ===")
    print(f"  R_s = {params.R_s:.4f} Ω | "
          f"R_sh = {params.R_sh:.2f} Ω | "
          f"I_o_ref = {params.I_o_ref:.3e} A")
    print(f"  {'Parameter':<8}{'Datasheet':>12}{'Model':>12}{'Error %':>10}")
    print("  " + "-" * 44)
    for k in ds:
        err = 100.0 * (sim[k] - ds[k]) / ds[k]
        flag = "OK" if abs(err) < 5 else "CEK"
        print(f"  {k:<8}{ds[k]:>12.3f}{sim[k]:>12.3f}{err:>10.2f}  {flag}")
    return sim


# ================================================================
# 3. Helper: simpan parameter hasil tuning ke file Python
# ================================================================
def save_tuned_params(params, info, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('"""\n')
        f.write('pv_model_tuned.py — parameter modul hasil auto-tuning.\n')
        f.write('Di-generate otomatis oleh validate_iv.py.\n')
        f.write('"""\n')
        f.write('from dataclasses import replace\n')
        f.write('from src.pv_model import ModuleParams\n\n')
        f.write('TUNED_PARAMS = replace(\n')
        f.write('    ModuleParams(),\n')
        for key, val in asdict(params).items():
            if key in ('R_s', 'R_sh', 'I_o_ref'):
                if isinstance(val, float):
                    f.write(f'    {key}={val!r},\n')
        f.write(')\n\n')
        f.write(f'# Metadata tuning:\n')
        f.write(f'# J_opt = {info["J_opt"]:.6e}\n')
        f.write(f'# Converged = {info["converged"]}\n')
        f.write(f'# Iterations = {info["n_iter"]}\n')
    print(f"\nParameter hasil tuning disimpan ke: {out_path}")


# ================================================================
# 4. Main
# ================================================================
def main():
    base_params = ModuleParams()
    G_test, T_test = 1000.0, 25.0

    # ------------------------------------------------------------
    # 4a. Evaluasi sebelum tuning
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUASI AWAL (sebelum tuning)")
    print("=" * 60)
    sim_before = print_comparison("Sebelum tuning", base_params, G_test, T_test)

    # ------------------------------------------------------------
    # 4b. Tuning
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("AUTO-TUNING R_s, R_sh, I_o_ref")
    print("=" * 60)
    params_opt, info = tune_params(base_params, verbose=True)

    # ------------------------------------------------------------
    # 4c. Evaluasi sesudah tuning
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUASI AKHIR (sesudah tuning)")
    print("=" * 60)
    sim_after = print_comparison("Sesudah tuning", params_opt, G_test, T_test)

    print(f"\nRingkasan tuning:")
    print(f"  x0            = {info['x0']}")
    print(f"  R_s optimal   = {info['Rs_opt']:.4f} Ω")
    print(f"  R_sh optimal  = {info['Rsh_opt']:.2f} Ω")
    print(f"  I_o optimal   = {info['Io_opt']:.4e} A")
    print(f"  Objective     = {info['J_opt']:.6e}")
    print(f"  Converged     = {info['converged']} ({info['n_iter']} iterasi)")
    print(f"  Message       = {info['message']}")

    # ------------------------------------------------------------
    # 4d. Plot perbandingan sebelum vs sesudah
    # ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Baris 1: kurva I-V
    for ax, params, label in [
        (axes[0, 0], base_params, 'Sebelum tuning'),
        (axes[0, 1], params_opt,  'Sesudah tuning'),
    ]:
        V, I = iv_curve_module(G_test, T_test, params, n_points=500)
        V_oc_, I_sc_, V_mpp_, I_mpp_, P_max_ = analyze_module(G_test, T_test, params)
        ax.plot(V, I, 'b-', lw=2, label='Model')
        ax.plot(params.v_oc_ref, 0, 'ro', label='Voc datasheet')
        ax.plot(0, params.i_sc_ref, 'go', label='Isc datasheet')
        ax.plot(V_mpp_, I_mpp_, 'ms', ms=8,
                label=f'MPP model = {P_max_:.1f} W')
        ax.plot(params.v_mp_ref, params.i_mp_ref, 'k*', ms=12,
                label='MPP datasheet')
        ax.set_xlabel('Tegangan (V)')
        ax.set_ylabel('Arus (A)')
        ax.set_title(f'Kurva I-V @ STC — {label}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Baris 2: kurva P-V
    for ax, params, label in [
        (axes[1, 0], base_params, 'Sebelum tuning'),
        (axes[1, 1], params_opt,  'Sesudah tuning'),
    ]:
        V, I = iv_curve_module(G_test, T_test, params, n_points=500)
        P = V * I
        _, _, V_mpp_, I_mpp_, P_max_ = analyze_module(G_test, T_test, params)
        ax.plot(V, P, 'r-', lw=2, label='Model')
        ax.plot(V_mpp_, P_max_, 'ms', ms=8,
                label=f'MPP model = {P_max_:.1f} W')
        ax.axhline(params.pdc0, color='k', ls='--', lw=1,
                   label=f'Pmax datasheet = {params.pdc0} W')
        ax.set_xlabel('Tegangan (V)')
        ax.set_ylabel('Daya (W)')
        ax.set_title(f'Kurva P-V @ STC — {label}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_dir = ROOT / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'iv_curve_validation.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot disimpan ke: {out_path}")

    # ------------------------------------------------------------
    # 4e. Simpan parameter hasil tuning
    # ------------------------------------------------------------
    save_tuned_params(
        params_opt, info,
        ROOT / 'src' / 'pv_model_tuned.py',
    )

    # ------------------------------------------------------------
    # 4f. Simpan log tuning
    # ------------------------------------------------------------
    log_path = ROOT / 'logs' / 'tuning_report.json'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'before': {
                'R_s': base_params.R_s,
                'R_sh': base_params.R_sh,
                'I_o_ref': base_params.I_o_ref,
                'sim': {k: float(v) for k, v in sim_before.items()},
            },
            'after': {
                'R_s': params_opt.R_s,
                'R_sh': params_opt.R_sh,
                'I_o_ref': params_opt.I_o_ref,
                'sim': {k: float(v) for k, v in sim_after.items()},
            },
            'optimization': info,
        }, f, indent=2)
    print(f"Log tuning disimpan ke: {log_path}")


if __name__ == '__main__':
    main()