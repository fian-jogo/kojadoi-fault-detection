"""
model_hybrid.py
Model hybrid: LSTM + Graph Attention Network + Physics head.

PINN v3 (fixed):
    Physics head memprediksi (V, I) dalam ruang ternormalisasi,
    lalu di-denormalisasi ke volt/ampere sebelum dihitung residual
    persamaan diode. Physics loss adalah residual itu sendiri.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# Parameter modul (hasil tuning Hari 1)
# ================================================================
MODULE_PARAMS = {
    'I_L_ref': 10.25,
    'I_o_ref': 2.502e-9,
    'R_s': 0.1065,
    'R_sh': 528536054.76,
    'a_ref': 72 * 1.2 * 0.0257,   # ≈ 2.22 V
}


# ================================================================
# Dense GAT
# ================================================================
class DenseGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.2):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // n_heads

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(self.head_dim, 1, bias=False)
        self.a_dst = nn.Linear(self.head_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.n_heads, self.head_dim)
        e = self.leaky(self.a_src(Wh).unsqueeze(2) +
                        self.a_dst(Wh).unsqueeze(1)).squeeze(-1)
        adj_mask = (adj > 0).unsqueeze(0).unsqueeze(-1)
        e = e.masked_fill(~adj_mask, -1e9)
        alpha = self.dropout(F.softmax(e, dim=2))
        h_new = torch.einsum('bijh,bjhd->bihd', alpha, Wh).reshape(
            B, N, self.out_dim)
        return F.elu(h_new)


class GATBlock(nn.Module):
    def __init__(self, dim, n_heads=4, dropout=0.2):
        super().__init__()
        self.gat = DenseGATLayer(dim, dim, n_heads=n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        return self.norm(h + self.dropout(self.gat(h, adj)))


# ================================================================
# Physics Residual (diode equation)
# ================================================================
def diode_residual_loss(V_pred, I_pred, G_obs, params=None):
    """
    Hitung residual persamaan single-diode pada prediksi (V, I).

    Args:
        V_pred, I_pred : (B, N) tegangan (V) dan arus (A) — sudah di-denormalisasi
        G_obs          : (B, N) iradiansi (W/m²) — sudah di-denormalisasi
    Returns:
        loss : scalar (mean residual ternormalisasi)
    """
    if params is None:
        params = MODULE_PARAMS

    I_L_ref = params['I_L_ref']
    I_o_ref = params['I_o_ref']
    R_s = params['R_s']
    R_sh = params['R_sh']
    a_ref = params['a_ref']

    # Pastikan G >= 0, dan V, I sudah di-clamp oleh caller
    G_clamped = torch.clamp(G_obs, min=0.0)
    I_ph = G_clamped / 1000.0 * I_L_ref + 1e-3   # minimum 1 mA untuk stabilitas

    V_diode = V_pred + I_pred * R_s
    exp_arg = torch.clamp(V_diode / a_ref, min=-20.0, max=20.0)
    I_diode = I_o_ref * (torch.exp(exp_arg) - 1.0)
    I_shunt = V_diode / R_sh

    I_calc = I_ph - I_diode - I_shunt
    residual = (I_pred - I_calc) ** 2 / (I_ph ** 2 + 1.0)
    return residual.mean()


# ================================================================
# Model Hybrid
# ================================================================
class HybridDetector(nn.Module):
    def __init__(self, n_nodes=12, n_features=5,
                 hidden=64, n_layers=2, n_gat=2, n_heads=4,
                 dropout=0.2, use_gat=True, use_pinn=True,
                 norm_mean=None, norm_std=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden = hidden
        self.use_gat = use_gat
        self.use_pinn = use_pinn

        # Register normalisasi statistik sebagai buffer
        if norm_mean is not None and norm_std is not None:
            self.register_buffer('norm_mean',
                                  torch.tensor(norm_mean, dtype=torch.float32))
            self.register_buffer('norm_std',
                                  torch.tensor(norm_std, dtype=torch.float32))
        else:
            self.norm_mean = None
            self.norm_std = None

        # LSTM
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False,
        )

        # GAT
        if use_gat:
            self.gat_blocks = nn.ModuleList([
                GATBlock(hidden, n_heads=n_heads, dropout=dropout)
                for _ in range(n_gat)
            ])
        else:
            self.gat_blocks = None

        # Detection head
        self.detect_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

        # Localization head
        self.localize_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

        # Physics head: output 2 nilai (V_norm, I_norm)
        if use_pinn:
            self.physics_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 2),
            )
        else:
            self.physics_head = None

    def forward(self, x, adj=None):
        """
        Returns:
            det_logits : (B, 2)
            loc_logits : (B, N)
            phys_pred  : (B, N, 2) — (V, I) dalam satuan fisik, atau None
        """
        B, T, N, F = x.shape

        # LSTM per node
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        out, _ = self.lstm(x_flat)
        h = out[:, -1, :].reshape(B, N, self.hidden)

        # GAT
        if self.use_gat and self.gat_blocks is not None and adj is not None:
            for block in self.gat_blocks:
                h = block(h, adj)

        # Detection
        det_logits = self.detect_head(h.mean(dim=1))

        # Localization
        loc_logits = self.localize_head(h).squeeze(-1)

        # Physics head: prediksi (V, I) ternormalisasi, lalu denormalisasi
        if self.use_pinn and self.physics_head is not None:
            phys_norm = self.physics_head(h)  # (B, N, 2)
            if self.norm_mean is not None and self.norm_std is not None:
                # Denormalisasi: [V, I] pakai mean/std fitur 0 dan 1
                mean_VI = self.norm_mean[:2].view(1, 1, 2)
                std_VI = self.norm_std[:2].view(1, 1, 2)
                phys_pred = phys_norm * std_VI + mean_VI
            else:
                phys_pred = phys_norm

            # Clamp ke rentang fisis: V ∈ [0, 60] V, I ∈ [0, 15] A
            V_pred = torch.clamp(phys_pred[..., 0], min=0.0, max=60.0)
            I_pred = torch.clamp(phys_pred[..., 1], min=0.0, max=15.0)
            phys_pred = torch.stack([V_pred, I_pred], dim=-1)
        else:
            phys_pred = None

        return det_logits, loc_logits, phys_pred