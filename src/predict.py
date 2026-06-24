"""Inferencia: tabla de estado de flota.

Uso:
    python -m src.predict

Carga los artefactos entrenados, recalcula el score de riesgo sobre la última
muestra de cada motor y produce una tabla con: nivel de riesgo (0-3), modo de
falla dominante y score, ordenada de mayor a menor riesgo.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import torch

from .config import ARTIFACTS, load_config
from .data.load import load_data
from .features.signatures import apply_F
from .features.windows import make_windows
from .models.bayes_lstm import BayesianLSTM
from .models.risk import dominant_mode, propagate_var_to_modes, risk_scores
from .models.vae import VAE


def _normalize(e, m_hat, m_var, calib):
    """Normaliza cada término por su referencia 'sana' -> ~0..1+ (calibrable)."""
    e_n = e / calib["e_ref"]
    m_n = np.abs(m_hat) / calib["m_ref"]
    v_n = m_var / calib["mvar_ref"]
    return e_n, m_n, v_n


def main():
    cfg = load_config()
    meta_art = joblib.load(ARTIFACTS / "meta.joblib")
    scaler = joblib.load(ARTIFACTS / "scaler.joblib")
    F = torch.tensor(meta_art["F"])
    calib = meta_art["calib"]
    d = len(cfg["oil_vars"])

    df = load_data(cfg)
    X, Y, meta = make_windows(df, cfg, scaler)

    method = meta_art.get("method", "mc_dropout")
    if method == "bbb":
        from .models.bayes_lstm_bbb import BayesianLSTM_BBB
        lstm = BayesianLSTM_BBB(meta_art["lstm_input"], cfg["model"]["lstm_hidden"], d,
                                cfg["model"]["bbb_prior_sigma"])
    else:
        lstm = BayesianLSTM(meta_art["lstm_input"], cfg["model"]["lstm_hidden"], d,
                            cfg["model"]["lstm_layers"], cfg["model"]["mc_dropout"])
    lstm.load_state_dict(torch.load(ARTIFACTS / "lstm.pt"))
    vae = VAE(meta_art["z_dim"], cfg["model"]["vae_latent"], cfg["model"]["vae_hidden"])
    vae.load_state_dict(torch.load(ARTIFACTS / "vae.pt"))

    x_hat, x_var = lstm.predict_with_uncertainty(
        torch.tensor(X), n_samples=cfg["model"]["mc_samples"])
    m_hat = apply_F(F, torch.tensor(x_hat)).numpy()
    m_var = propagate_var_to_modes(x_var, F.numpy())
    ctx = X[:, -1, d:]
    z = np.concatenate([x_hat, m_hat, ctx], axis=1).astype(np.float32)
    e = vae.reconstruction_error(torch.tensor(z)).numpy()

    # Usa umbrales auto-calibrados (guardados en train) si están disponibles.
    if "auto_tau" in calib:
        cfg["risk"].update(calib["auto_tau"])
        print(f"Umbrales (auto-calibrados): {calib['auto_tau']}")

    e_n, m_n, v_n = _normalize(e, m_hat, m_var, calib)
    R_modes, R_motor, niveles = risk_scores(e_n, m_n, v_n, cfg)
    modo_dom = dominant_mode(R_modes, cfg["failure_modes"])

    out = meta.copy()
    out["R_motor"] = R_motor
    out["nivel"] = niveles
    out["modo_dominante"] = modo_dom

    # Última muestra por motor = estado actual
    estado = (out.sort_values("fecha_muestra")
                 .groupby("equipo")
                 .tail(1)
                 .sort_values("R_motor", ascending=False)
                 .reset_index(drop=True))

    pd.set_option("display.width", 200)
    print("\n===== TABLA DE ESTADO DE FLOTA (última muestra por motor) =====")
    print(estado[["equipo", "fecha_muestra", "nivel", "modo_dominante",
                  "R_motor", "modo_real"]].to_string(index=False))

    estado.to_csv(ARTIFACTS / "estado_flota.csv", index=False)
    out.to_csv(ARTIFACTS / "riesgo_series.csv", index=False)
    print(f"\nGuardado: {ARTIFACTS/'estado_flota.csv'} y riesgo_series.csv")


if __name__ == "__main__":
    main()
