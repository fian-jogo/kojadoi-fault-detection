"""
model_lstm.py
Baseline model: LSTM dengan dua head (deteksi + lokalisasi).

Arsitektur:
    1. LSTM shared across nodes
    2. Ambil hidden state terakhir
    3. Detection head: pool across nodes -> linear -> 2 kelas
    4. Localization head: per-node linear -> 1 logit
"""
import torch
import torch.nn as nn


class LSTMDetector(nn.Module):
    def __init__(self, n_nodes=12, n_features=5,
                 hidden=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden = hidden

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.detect_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

        self.localize_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        # x: (B, T, N, F)
        B, T, N, F = x.shape

        # Reshape ke (B*N, T, F)
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, F)

        # LSTM
        out, _ = self.lstm(x_flat)  # (B*N, T, H)

        # Ambil hidden state terakhir
        h_last = out[:, -1, :]                       # (B*N, H)
        h_last = h_last.reshape(B, N, self.hidden)   # (B, N, H)

        # Detection: pool across nodes
        h_pool = h_last.mean(dim=1)                  # (B, H)
        detect_logits = self.detect_head(h_pool)     # (B, 2)

        # Localization: per-node
        localize_logits = self.localize_head(h_last).squeeze(-1)  # (B, N)

        return detect_logits, localize_logits