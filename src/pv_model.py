"""
pv_model.py
Model single-diode untuk modul PV 400 Wp (monocrystalline).
Dipakai oleh validate_iv.py, run_hourly.py, dan fault_injection.py.
"""
from dataclasses import dataclass
import numpy as np
from pvlib import pvsystem
from pvlib.pvsystem import i_from_v, v_from_i
from pvlib.singlediode import bishop88_mpp


# ----------------------------------------------------------------
# Parameter modul (datasheet generik monocrystalline 400 Wp)
# ----------------------------------------------------------------
@dataclass
class ModuleParams:
    pdc0: float = 400.0            # Wp
    v_oc_ref: float = 49.4         # V
    i_sc_ref: float = 10.23        # A
    v_mp_ref: float = 41.4         # V
    i_mp_ref: float = 9.66         # A
    alpha_sc: float = 0.0051       # A/°C  (+0.05%/°C * Isc)
    beta_voc: float = -0.143       # V/°C  (-0.29%/°C * Voc)
    gamma_pdc: float = -0.0034     # /°C
    cells_in_series: int = 72      # 72 sel → Voc ≈ 49.4 V @ STC
    n_ideality: float = 1.2        # faktor idealitas dioda
    R_s: float = 0.35              # Ω
    R_sh: float = 300.0            # Ω
    I_L_ref: float = 10.25         # A (light current @ STC)
    I_o_ref: float = 1e-10         # A (saturation current @ STC)


# ----------------------------------------------------------------
# Helper: hitung a_ref = n * Ns * Vth pada STC
# ----------------------------------------------------------------
def _a_ref(params: ModuleParams) -> float:
    Vth_stc = 0.025693  # kT/q pada 25°C
    return params.cells_in_series * params.n_ideality * Vth_stc


# ----------------------------------------------------------------
# Hitung parameter single-diode (IL, I0, Rs, Rsh, nNsVth) untuk G, T
# ----------------------------------------------------------------
def get_sdm_params(G: float, T: float, params: ModuleParams):
    """De Soto model. Return (IL, I0, Rs, Rsh, nNsVth)."""
    if G < 0:
        G = 0.0
    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
        effective_irradiance=G,
        temp_cell=T,
        alpha_sc=params.alpha_sc,
        a_ref=_a_ref(params),
        I_L_ref=params.I_L_ref,
        I_o_ref=params.I_o_ref,
        R_sh_ref=params.R_sh,
        R_s=params.R_s,
        EgRef=1.121,
        dEgdT=-0.0002677,
    )
    return IL, I0, Rs, Rsh, nNsVth


# ----------------------------------------------------------------
# Analisis satu modul (Voc, Isc, Vmpp, Impp, Pmpp)
# ----------------------------------------------------------------
def analyze_module(G: float, T: float, params: ModuleParams):
    """Return (V_oc, I_sc, V_mpp, I_mpp, P_max) untuk satu modul."""
    IL, I0, Rs, Rsh, nNsVth = get_sdm_params(G, T, params)

    if IL <= 0 or not np.isfinite(IL):
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Voc: tegangan saat arus = 0
    V_oc = float(v_from_i(0.0, IL, I0, Rs, Rsh, nNsVth, method='lambertw'))

    # Isc: arus saat tegangan = 0
    I_sc = float(i_from_v(0.0, IL, I0, Rs, Rsh, nNsVth, method='lambertw'))

    # MPP via bishop88
    V_mpp, I_mpp, P_max = bishop88_mpp(IL, I0, Rs, Rsh, nNsVth, method='brentq')
    return V_oc, I_sc, float(V_mpp), float(I_mpp), float(P_max)


# ----------------------------------------------------------------
# Analisis string: n_series modul seri
# ----------------------------------------------------------------
def analyze_string(G: float, T: float, params: ModuleParams, n_series: int):
    """
    String dengan n_series modul seri.
    Return (V_mpp_string, I_mpp_string, P_mpp_string).
    Tegangan dikali n_series; arus sama dengan satu modul.
    """
    _, _, V_mpp, I_mpp, _ = analyze_module(G, T, params)
    return V_mpp * n_series, I_mpp, V_mpp * n_series * I_mpp


# ----------------------------------------------------------------
# Kurva I-V satu modul (untuk plotting validasi)
# ----------------------------------------------------------------
def iv_curve_module(G: float, T: float, params: ModuleParams, n_points: int = 300):
    """Return (V_array, I_array) untuk satu modul."""
    IL, I0, Rs, Rsh, nNsVth = get_sdm_params(G, T, params)
    if IL <= 0 or not np.isfinite(IL):
        return np.array([0.0]), np.array([0.0])

    V_oc = float(v_from_i(0.0, IL, I0, Rs, Rsh, nNsVth, method='lambertw'))
    if not np.isfinite(V_oc) or V_oc <= 0:
        return np.array([0.0]), np.array([0.0])

    V = np.linspace(0.0, V_oc * 1.02, n_points)
    I = i_from_v(V, IL, I0, Rs, Rsh, nNsVth, method='lambertw')
    return V, I

if __name__ == "__main__":
    p = ModuleParams()
    V_oc, I_sc, V_mpp, I_mpp, P_max = analyze_module(1000, 25, p)

    print(f"Voc  = {V_oc:7.3f} V   (datasheet: {p.v_oc_ref})")
    print(f"Isc  = {I_sc:7.3f} A   (datasheet: {p.i_sc_ref})")
    print(f"Vmpp = {V_mpp:7.3f} V   (datasheet: {p.v_mp_ref})")
    print(f"Impp = {I_mpp:7.3f} A   (datasheet: {p.i_mp_ref})")
    print(f"Pmax = {P_max:7.2f} W   (datasheet: {p.pdc0})")