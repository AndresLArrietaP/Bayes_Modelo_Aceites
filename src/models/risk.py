"""Score de riesgo multinivel.

Para cada muestra:
  R_i = alpha * e + beta * |m_hat_i| + gamma * var(m_hat_i)
  R_motor = max_i R_i
  Nivel = clasificación 0/1/2/3 según tau1, tau2, tau3.

Donde:
  e        = error de reconstrucción del VAE (anomalía global)
  m_hat_i  = activación pronosticada del modo de falla i
  var      = incertidumbre del modo i (propagada desde el LSTM)
"""
from __future__ import annotations

import numpy as np


def propagate_var_to_modes(x_var: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Varianza de m = F x (asumiendo independencia): Var(m) = (F^2) Var(x)."""
    return x_var @ (F.T ** 2)        # (N, k)


def risk_scores(e, m_hat, m_var, cfg):
    """Devuelve (R_modes (N,k), R_motor (N,), niveles (N,))."""
    r = cfg["risk"]
    e = np.asarray(e).reshape(-1, 1)
    R_modes = r["alpha"] * e + r["beta"] * np.abs(m_hat) + r["gamma"] * m_var
    R_motor = R_modes.max(axis=1)

    niveles = np.zeros_like(R_motor, dtype=int)
    niveles[R_motor >= r["tau1"]] = 1
    niveles[R_motor >= r["tau2"]] = 2
    niveles[R_motor >= r["tau3"]] = 3
    return R_modes, R_motor, niveles


def dominant_mode(R_modes: np.ndarray, failure_modes: list[str]) -> list[str]:
    idx = R_modes.argmax(axis=1)
    return [failure_modes[i] for i in idx]
