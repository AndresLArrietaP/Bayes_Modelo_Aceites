"""Entrenamiento end-to-end: LSTM (predicción + incertidumbre) y VAE (anomalías).

Uso:
    python -m src.train

Pasos:
  1. Carga datos (synthetic o SQL según DATA_SOURCE).
  2. Escala features y construye ventanas temporales por motor.
  3. Entrena el BayesianLSTM para predecir x_{t+H}.
  4. Construye z = [x_hat, m_hat, contexto] y entrena el VAE SOLO con motores sanos.
  5. Calibra escalas de referencia (sano) para el score de riesgo.
  6. Guarda todos los artefactos en artifacts/.
"""
from __future__ import annotations

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ARTIFACTS, load_config
from .data.load import load_data
from .features.signatures import apply_F, build_F
from .features.windows import FleetScaler, make_windows
from .models.bayes_lstm import BayesianLSTM
from .models.risk import propagate_var_to_modes
from .models.vae import VAE, vae_loss


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    cfg = load_config()
    _set_seed(cfg["train"]["seed"])
    ARTIFACTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    # ---- 1. Datos ----
    df = load_data(cfg)
    print(f"Datos: {len(df)} filas | {df['equipo'].nunique()} motores")

    feat_cols = cfg["oil_vars"] + cfg["context_vars"]
    scaler = FleetScaler(feat_cols).fit(df)
    X, Y, meta = make_windows(df, cfg, scaler)
    print(f"Ventanas: X={X.shape}  Y={Y.shape}")

    d = len(cfg["oil_vars"])
    F = build_F(cfg)

    # ---- 2. Entrenar LSTM ----
    Xt = torch.tensor(X)
    Yt = torch.tensor(Y)
    ds = TensorDataset(Xt, Yt)
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True)

    method = cfg["model"].get("method", "mc_dropout")
    print(f"\n== Entrenando LSTM bayesiano (método: {method}) ==")

    if method == "bbb":
        from .models.bayes_lstm_bbb import BayesianLSTM_BBB, elbo_loss
        lstm = BayesianLSTM_BBB(
            input_dim=X.shape[2], hidden_dim=cfg["model"]["lstm_hidden"],
            output_dim=d, prior_sigma=cfg["model"]["bbb_prior_sigma"],
        ).to(device)
        opt = torch.optim.Adam(lstm.parameters(), lr=cfg["train"]["lr"])
        kl_w = cfg["model"]["bbb_kl_weight"] / max(1, len(dl))
        for ep in range(cfg["train"]["epochs_lstm"]):
            lstm.train()
            tot = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                y = lstm(xb)
                loss, nll = elbo_loss(y, yb, lstm.kl_loss(), kl_w)
                loss.backward()
                opt.step()
                tot += nll.item() * len(xb)
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  epoch {ep+1:>3}  nll={tot/len(ds):.4f}")
    else:
        lstm = BayesianLSTM(
            input_dim=X.shape[2], hidden_dim=cfg["model"]["lstm_hidden"],
            output_dim=d, num_layers=cfg["model"]["lstm_layers"],
            dropout=cfg["model"]["mc_dropout"],
        ).to(device)
        opt = torch.optim.Adam(lstm.parameters(), lr=cfg["train"]["lr"])
        mse = nn.MSELoss()
        for ep in range(cfg["train"]["epochs_lstm"]):
            lstm.train()
            tot = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = mse(lstm(xb), yb)
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  epoch {ep+1:>3}  mse={tot/len(ds):.4f}")

    # ---- 3. Predicciones + incertidumbre sobre todas las ventanas ----
    x_hat, x_var = lstm.predict_with_uncertainty(
        Xt.to(device), n_samples=cfg["model"]["mc_samples"])
    m_hat = apply_F(F, torch.tensor(x_hat)).numpy()              # (N, k)
    m_var = propagate_var_to_modes(x_var, F.numpy())            # (N, k)

    # contexto (sin escalar-inverso; usamos las features de contexto del último paso)
    ctx = X[:, -1, d:]                                          # (N, ctx)
    z = np.concatenate([x_hat, m_hat, ctx], axis=1).astype(np.float32)

    # ---- 4. Entrenar VAE solo con motores SANOS ----
    healthy_mask = (meta["modo_real"] == "Sano").values
    if healthy_mask.sum() == 0:        # si no hay etiqueta (datos SQL reales), usa todo
        print("Aviso: sin etiqueta de salud; el VAE se entrena con toda la data.")
        healthy_mask = np.ones(len(z), dtype=bool)
    z_healthy = torch.tensor(z[healthy_mask])

    vae = VAE(input_dim=z.shape[1], latent_dim=cfg["model"]["vae_latent"],
              hidden=cfg["model"]["vae_hidden"]).to(device)
    opt_v = torch.optim.Adam(vae.parameters(), lr=cfg["train"]["lr"])
    dl_v = DataLoader(TensorDataset(z_healthy), batch_size=cfg["train"]["batch_size"],
                      shuffle=True)

    print("\n== Entrenando VAE (motores sanos) ==")
    for ep in range(cfg["train"]["epochs_vae"]):
        vae.train()
        tot = 0.0
        for (zb,) in dl_v:
            zb = zb.to(device)
            opt_v.zero_grad()
            zhat, mu, logvar = vae(zb)
            loss, _, _ = vae_loss(zb, zhat, mu, logvar)
            loss.backward()
            opt_v.step()
            tot += loss.item() * len(zb)
        if (ep + 1) % 15 == 0 or ep == 0:
            print(f"  epoch {ep+1:>3}  loss={tot/len(z_healthy):.4f}")

    # ---- 5. Calibración: escalas de referencia + umbrales desde la dist. sana ----
    from .models.risk import risk_scores
    e_all = vae.reconstruction_error(torch.tensor(z).to(device)).cpu().numpy()
    e_healthy = e_all[healthy_mask]
    calib = {
        "e_ref": float(np.percentile(e_healthy, 95) + 1e-6),
        "m_ref": (np.percentile(np.abs(m_hat[healthy_mask]), 95, axis=0) + 1e-6),
        "mvar_ref": (np.percentile(m_var[healthy_mask], 95, axis=0) + 1e-6),
    }
    # Normaliza términos y calcula R en sanos para fijar tau1/tau2/tau3.
    e_n = e_healthy / calib["e_ref"]
    m_n = np.abs(m_hat[healthy_mask]) / calib["m_ref"]
    v_n = m_var[healthy_mask] / calib["mvar_ref"]
    _, R_healthy, _ = risk_scores(e_n, m_n, v_n, cfg)
    # Umbrales más sensibles: tau1 en la mediana de sanos, tau2/tau3 más arriba.
    # Sube la detección a costa de algunos falsos positivos (aceptable aquí).
    auto_tau = {
        "tau1": float(np.percentile(R_healthy, 80)),
        "tau2": float(np.percentile(R_healthy, 92)),
        "tau3": float(np.percentile(R_healthy, 97)),
    }
    calib["auto_tau"] = auto_tau
    print(f"\nUmbrales auto-calibrados (desde motores sanos): {auto_tau}")

    # ---- 6. Guardar artefactos ----
    torch.save(lstm.state_dict(), ARTIFACTS / "lstm.pt")
    torch.save(vae.state_dict(), ARTIFACTS / "vae.pt")
    joblib.dump(scaler, ARTIFACTS / "scaler.joblib")
    joblib.dump({"F": F.numpy(), "calib": calib, "z_dim": z.shape[1],
                 "lstm_input": X.shape[2], "method": method},
                ARTIFACTS / "meta.joblib")
    print(f"\nArtefactos guardados en {ARTIFACTS}/")
    print("Listo. Ejecuta `python -m src.predict` para generar el tablero de flota.")


if __name__ == "__main__":
    main()
