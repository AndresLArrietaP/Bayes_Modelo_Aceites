"""Clasificador de pronóstico: encoder LSTM + cabeza binaria con MC Dropout.

Predice P(condición adversa en (t, t+H]) desde la ventana de metales. Mantiene la
identidad bayesiana del proyecto: el dropout queda activo en inferencia y N pasadas
estocásticas dan media (probabilidad) y desviación (incertidumbre).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)              # (B, T, H)
        h = self.drop(out[:, -1, :])
        return self.fc(h).squeeze(-1)      # (B,) logit

    @torch.no_grad()
    def predict_proba(self, x, n_samples: int = 30):
        """Devuelve (prob_media, prob_std) con MC Dropout (dropout activo)."""
        self.train()
        probs = [torch.sigmoid(self.forward(x)).cpu().numpy() for _ in range(n_samples)]
        probs = np.stack(probs, axis=0)    # (n_samples, B)
        return probs.mean(axis=0), probs.std(axis=0)
