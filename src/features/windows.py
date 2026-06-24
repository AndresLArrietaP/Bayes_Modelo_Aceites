"""Escalado y construcción de ventanas temporales.

Convierte la tabla de muestras (una fila por muestreo de aceite) en ventanas
deslizantes por motor para alimentar el LSTM:

  entrada  W_t = [u_{t-T+1}, ..., u_t]   (T pasos, cada uno con d+ctx features)
  objetivo y       = x_{t+H}             (vector de metales a horizonte H)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FleetScaler:
    """Estandariza oil_vars + context_vars. Por simplicidad, un scaler global;
    puede extenderse a un scaler por familia_motor."""

    def __init__(self, cols: list[str]):
        self.cols = cols
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame) -> "FleetScaler":
        self.scaler.fit(df[self.cols].astype(float).values)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(df[self.cols].astype(float).values)

    def inverse_oil(self, x_scaled: np.ndarray, n_oil: int) -> np.ndarray:
        """Invierte el escalado solo para las primeras n_oil columnas (metales)."""
        mean = self.scaler.mean_[:n_oil]
        scale = self.scaler.scale_[:n_oil]
        return x_scaled * scale + mean


def make_windows(df: pd.DataFrame, cfg: dict, scaler: FleetScaler):
    """Devuelve X (N, T, F), Y (N, d), y meta (DataFrame con equipo/fecha/modo)."""
    T = cfg["model"]["window_size"]
    H = cfg["model"]["horizon"]
    oil_vars = cfg["oil_vars"]
    feat_cols = oil_vars + cfg["context_vars"]
    d = len(oil_vars)

    X_list, Y_list, meta = [], [], []
    for equipo, g in df.groupby("equipo"):
        g = g.sort_values("fecha_muestra").reset_index(drop=True)
        if len(g) < T + H:
            continue
        feats = scaler.transform(g)          # (len, F)
        for t in range(T - 1, len(g) - H):
            X_list.append(feats[t - T + 1 : t + 1, :])     # (T, F)
            Y_list.append(feats[t + H, :d])                # objetivo: metales escalados
            row = g.iloc[t + H]
            meta.append({
                "equipo": equipo,
                "fecha_muestra": row["fecha_muestra"],
                "modo_real": row.get("_modo_real", "NA"),
            })
    X = np.asarray(X_list, dtype=np.float32)
    Y = np.asarray(Y_list, dtype=np.float32)
    return X, Y, pd.DataFrame(meta)
