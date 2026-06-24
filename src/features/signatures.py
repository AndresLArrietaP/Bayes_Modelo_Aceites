"""Matriz de firmas metálicas F (modos de falla).

m_t = F @ x_t   convierte el vector de metales en activaciones por modo
de falla (Cojinetes, Cilindro, Aire, Refrigerante, Combustion).
"""
from __future__ import annotations

import numpy as np
import torch


def build_F(cfg: dict) -> torch.Tensor:
    """Devuelve F (k x d) como tensor float32, en el orden de cfg['oil_vars']."""
    oil_vars = cfg["oil_vars"]
    rows = []
    for mode in cfg["failure_modes"]:
        weights = cfg["signature_matrix"][mode]
        if len(weights) != len(oil_vars):
            raise ValueError(
                f"La fila '{mode}' de signature_matrix tiene {len(weights)} "
                f"pesos pero oil_vars tiene {len(oil_vars)} variables."
            )
        rows.append(weights)
    return torch.tensor(np.asarray(rows, dtype=np.float32))


def apply_F(F: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """m = x @ F^T  -> soporta x de forma (..., d). Devuelve (..., k)."""
    return x @ F.T
