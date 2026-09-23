"""
fault_injection.py
Modul untuk injeksi fault pada model PV single-diode.
Dipakai oleh scripts/generate_dataset.py (Hari 2).

Mendukung tiga jenis fault:
    - LLF : Line-to-Line Fault (dengan batas arus fisis)
    - LGF : Ground Fault (dengan efek proporsional)
    - PSC : Partial Shading

Catatan: file ini menambahkan ROOT ke sys.path sendiri sehingga dapat
di-import dari skrip mana pun tanpa konfigurasi tambahan.
"""
import sys
from pathlib import Path

# ----------------------------------------------------------------
# Pastikan ROOT (folder induk src/) ada di sys.path
# ----------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
_ROOT = _SRC_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ----------------------------------------------------------------
# Import setelah sys.path fix
# ----------------------------------------------------------------
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pvlib.pvsystem import i_from_v, v_from_i

from src.pv_model import ModuleParams, get_sdm_params


# ================================================================
# 1. Struktur data spesifikasi fault
# ================================================================
@dataclass
class FaultSpec:
    """Spesifikasi satu fault event."""
    fault_type: str              # 'LLF', 'LGF', 'PSC'
    onset: int                   # time step onset
    duration: int                # durasi fault (time step)
    affected_nodes: List[int]    # node global yang terdampak
    magnitude: float             # R_f (Ω), R_g (Ω), atau sigma (0–1)
    gradual: bool = False        # True untuk onset gradual


# ================================================================
# 2. Helper: kurva I-V satu modul
# ================================================================
def _iv_curve_module(G: float, T: float,
                     params: ModuleParams,
                     n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Kurva I-V satu modul pada iradiansi G dan suhu T."""
    IL, I0, Rs, Rsh, nNsVth = get_sdm_params(G, T, params)
    if IL <= 0 or not np.isfinite(IL):
        return np.array([0.0]), np.array([0.0])

    V_oc = float(v_from_i(0.0, IL, I0, Rs, Rsh, nNsVth, method='lambertw'))
    if not np.isfinite(V_oc) or V_oc <= 0:
        return np.array([0.0]), np.array([0.0])

    V = np.linspace(0.0, V_oc * 1.02, n_points)
    I = i_from_v(V, IL, I0, Rs, Rsh, nNsVth, method='lambertw')
    return V, I


def _find_mpp(V: np.ndarray, I: np.ndarray) -> Tuple[float, float, float]:
    """Cari MPP dari kurva I-V. Return (V_mpp, I_mpp, P_mpp)."""
    if len(V) == 0 or len(I) == 0:
        return 0.0, 0.0, 0.0
    P = V * I
    idx = int(np.argmax(P))
    return float(V[idx]), float(I[idx]), float(P[idx])


# ================================================================
# 3. Kurva I-V string normal
# ================================================================
def iv_curve_string_normal(G: float, T: float,
                            params: ModuleParams,
                            n_series: int) -> Tuple[np.ndarray, np.ndarray]:
    """Kurva I-V string dengan semua modul identik (kondisi normal)."""
    V_mod, I_mod = _iv_curve_module(G, T, params)
    return V_mod * n_series, I_mod


# ================================================================
# 4. Model partial shading
# ================================================================
def apply_partial_shading(G_base: float, T: float,
                           params: ModuleParams,
                           shaded_nodes: List[int],
                           sigma: float,
                           n_total_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kurva I-V string dengan partial shading pada subset modul.

    Setiap modul dihitung terpisah, kemudian digabung secara seri
    (arus sama, tegangan dijumlahkan).
    """
    G_eff = np.full(n_total_nodes, G_base, dtype=float)
    for node in shaded_nodes:
        if 0 <= node < n_total_nodes:
            G_eff[node] = G_base * (1.0 - sigma)

    curves = [_iv_curve_module(G_eff[n], T, params) for n in range(n_total_nodes)]

    I_max_each = [I_mod.max() for _, I_mod in curves]
    I_limit = min(I_max_each) * 0.99

    if I_limit <= 0:
        return np.array([0.0]), np.array([0.0])

    I_sampled = np.linspace(0.0, I_limit, 300)
    V_total = np.zeros_like(I_sampled)

    for V_mod, I_mod in curves:
        idx = np.argsort(I_mod)
        I_sorted = I_mod[idx]
        V_sorted = V_mod[idx]

        I_u, idx_u = np.unique(I_sorted, return_index=True)
        V_u = V_sorted[idx_u]

        if len(I_u) < 2:
            continue

        V_interp = np.interp(I_sampled, I_u, V_u,
                              left=V_u[0], right=V_u[-1])
        V_total += V_interp

    return V_total, I_sampled


# ================================================================
# 5. Model line-to-line fault (dengan batas arus fisis)
# ================================================================
def apply_line_to_line_fault(V_mpp_normal: float,
                              I_mpp_normal: float,
                              R_f: float,
                              I_sc_module: float = 10.23,
                              n_strings_parallel: int = 3) -> Tuple[float, float, float]:
    """
    Model line-to-line fault dengan batas arus fisis.

    Arus fault dibatasi oleh kontribusi string-string sehat:
        I_fault_max = (N_parallel - 1) × I_sc_module × 1.2

    Referensi:
        Alam et al. (2015), "A comprehensive review of catastrophic faults
        in PV arrays: types, detection, and mitigation techniques",
        IEEE Journal of Photovoltaics, 5(3), 982-997.

    Return: (V_mpp_faulted, I_mpp_faulted, P_mpp_faulted)
    """
    # Batas atas arus fault dari string sehat
    I_fault_max = (n_strings_parallel - 1) * I_sc_module * 1.2

    # Arus fault dari Hukum Ohm
    I_fault_ohm = V_mpp_normal / max(R_f, 0.01)

    # Batas fisis: ambil yang lebih kecil
    I_fault = min(I_fault_ohm, I_fault_max)

    # Penurunan tegangan pada string yang fault
    V_drop = I_fault * R_f * 0.5
    V_faulted = max(V_mpp_normal - V_drop, 0.1 * V_mpp_normal)

    # Arus string fault: hanya sebagian arus fault mengalir ke terminal string
    I_faulted = I_mpp_normal + 0.3 * I_fault

    # Batas atas kedua: total arus string tidak melebihi 2× I_mpp
    I_faulted = min(I_faulted, 2.0 * I_mpp_normal)

    return V_faulted, I_faulted, V_faulted * I_faulted


# ================================================================
# 6. Model ground fault (dengan efek proporsional)
# ================================================================
def apply_ground_fault(V_mpp_normal: float,
                        I_mpp_normal: float,
                        R_g: float,
                        V_node: float) -> Tuple[float, float, float, float]:
    """
    Model ground fault dengan efek proporsional terhadap V_node.

    Arus bocor: I_leak = V_node / R_g
    Arus string: I_faulted = I_mpp - I_leak (floor 5%)

    Return: (V_mpp_faulted, I_mpp_faulted, P_mpp_faulted, I_leak)
    """
    I_leak = V_node / max(R_g, 0.01)

    # Arus string berkurang (floor 5% untuk menghindari collapse total)
    I_faulted = max(I_mpp_normal - I_leak, 0.05 * I_mpp_normal)

    # Tegangan turun proporsional terhadap arus
    V_faulted = V_mpp_normal * (I_faulted / max(I_mpp_normal, 1e-6))

    return V_faulted, I_faulted, V_faulted * I_faulted, I_leak


# ================================================================
# 7. Orkestrasi injeksi fault
# ================================================================
def inject_fault(G: float, T: float,
                 params: ModuleParams,
                 n_series: int,
                 fault: Optional[FaultSpec],
                 t_step: int,
                 n_total_nodes: int = 4,
                 n_strings_parallel: int = 3) -> dict:
    """
    Hitung respons string pada satu time step dengan kemungkinan fault.

    Return dict dengan kunci:
        - fault_active : bool
        - V_mpp        : tegangan MPP string (V)
        - I_mpp        : arus MPP string (A)
        - P_mpp        : daya MPP string (W)
        - I_leak       : arus bocor (A), hanya untuk LGF
    """
    # Hitung kondisi normal terlebih dahulu
    V_str, I_str = iv_curve_string_normal(G, T, params, n_series)
    V_mpp, I_mpp, P_mpp = _find_mpp(V_str, I_str)

    status = {
        'fault_active': False,
        'V_mpp': V_mpp,
        'I_mpp': I_mpp,
        'P_mpp': P_mpp,
        'I_leak': 0.0,
    }

    if fault is None:
        return status

    if not (fault.onset <= t_step < fault.onset + fault.duration):
        return status

    status['fault_active'] = True

    # Progres fault (untuk gradual)
    if fault.gradual and fault.duration > 1:
        progress = (t_step - fault.onset) / (fault.duration - 1)
        progress = float(min(max(progress, 0.0), 1.0))
    else:
        progress = 1.0

    # ------------------------------------------------------------
    # Terapkan fault sesuai jenisnya
    # ------------------------------------------------------------
    if fault.fault_type == 'PSC':
        sigma_eff = fault.magnitude * progress
        V_psc, I_psc = apply_partial_shading(
            G, T, params, fault.affected_nodes, sigma_eff, n_total_nodes
        )
        V_mpp_f, I_mpp_f, P_mpp_f = _find_mpp(V_psc, I_psc)
        status['V_mpp'] = V_mpp_f
        status['I_mpp'] = I_mpp_f
        status['P_mpp'] = P_mpp_f

    elif fault.fault_type == 'LLF':
        # R_f turun gradual dari nilai besar (normal) ke nilai fault
        R_f_eff = fault.magnitude * (1.0 - progress) + 0.1 * progress
        V_f, I_f, P_f = apply_line_to_line_fault(
            V_mpp, I_mpp, R_f_eff,
            I_sc_module=params.i_sc_ref,
            n_strings_parallel=n_strings_parallel,
        )
        status['V_mpp'] = V_f
        status['I_mpp'] = I_f
        status['P_mpp'] = P_f

    elif fault.fault_type == 'LGF':
        # R_g turun gradual dari nilai sangat besar (tak hingga) ke nilai fault
        R_g_eff = fault.magnitude * progress + 1e6 * (1.0 - progress)
        node_idx = fault.affected_nodes[0] if fault.affected_nodes else 0
        V_node = (node_idx + 1) * params.v_mp_ref
        V_f, I_f, P_f, I_leak = apply_ground_fault(
            V_mpp, I_mpp, R_g_eff, V_node
        )
        status['V_mpp'] = V_f
        status['I_mpp'] = I_f
        status['P_mpp'] = P_f
        status['I_leak'] = I_leak

    return status