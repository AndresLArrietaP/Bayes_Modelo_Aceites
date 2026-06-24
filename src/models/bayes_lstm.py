"""Bayesian LSTM (aproximación por MC Dropout).

NOTA METODOLÓGICA
-----------------
El paper de referencia (Chen et al., 2026) usa *Bayes by Backprop* (BBB):
pesos con distribución variacional q(W)=N(mu, sigma^2) y muestreo de pesos.
Aquí implementamos la aproximación más ligera y estándar — **MC Dropout**
(Gal & Ghahramani, 2016): se deja el dropout activo en inferencia y se hacen
N pasadas estocásticas; la media es la predicción y la varianza estima la
incertidumbre. Es suficiente para una primera versión y mucho más simple de
entrenar. En `models/bayes_lstm_bbb.py` (TODO) puede implementarse el BBB
completo si se requiere fidelidad al paper.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class BayesianLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)        # (B, T, H)
        h_last = self.drop(out[:, -1, :])
        return self.fc(h_last)       # (B, output_dim)

    @torch.no_grad()
    def predict_with_uncertainty(self, x, n_samples=30):
        """Devuelve (mean, var) sobre n_samples pasadas con dropout activo."""
        self.train()                 # mantiene dropout encendido
        preds = [self.forward(x).cpu().numpy() for _ in range(n_samples)]
        preds = np.stack(preds, axis=0)          # (n_samples, B, d)
        return preds.mean(axis=0), preds.var(axis=0)
