"""Tablero de flota con el modelo supervisado.

Uso:
    python -m src.predict_supervised

Para cada motor toma su ÚLTIMA ventana de metales y estima P(condición adversa en el
horizonte). Usa el mejor modelo (LSTM o GBT) según PR-AUC. Reporta:
  - banda de riesgo (Bajo/Medio/Alto) con doble umbral por precisión,
  - antigüedad de la última muestra (motores inactivos no son accionables),
  - modo de falla dominante (matriz de firmas) como explicabilidad.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import torch

from .config import ARTIFACTS, load_config
from .data.load import load_data
from .features.signatures import apply_F, build_F
from .features.windows import FleetScaler  # noqa: F401  (tipo del scaler serializado)
from .models.classifier import LSTMClassifier
from .train_supervised import artifact_suffix, window_features


def _last_windows(df, cfg, scaler):
    """Última ventana por motor: X (M,T,F), metales escalados del ancla, meta."""
    T = cfg["model"]["window_size"]
    d = len(cfg["oil_vars"])
    X, x_last, meta = [], [], []
    for equipo, g in df.groupby("equipo"):
        g = g.sort_values("fecha_muestra").reset_index(drop=True)
        if len(g) < T:
            continue
        feats = scaler.transform(g)
        X.append(feats[-T:, :])
        x_last.append(feats[-1, :d])
        meta.append({"equipo": equipo, "fecha_ultima": g.iloc[-1]["fecha_muestra"]})
    return (np.asarray(X, dtype=np.float32), np.asarray(x_last, dtype=np.float32),
            pd.DataFrame(meta))


def main():
    os.environ.setdefault("DATA_SOURCE", "sql")
    cfg = load_config()
    sfx = artifact_suffix(cfg)
    meta_sup = joblib.load(ARTIFACTS / f"meta_sup{sfx}.joblib")
    scaler = joblib.load(ARTIFACTS / f"scaler_sup{sfx}.joblib")

    df = load_data(cfg)
    X, x_last, meta = _last_windows(df, cfg, scaler)
    if len(X) == 0:
        raise SystemExit("Sin motores con ventana completa.")

    mejor = meta_sup["mejor"]
    if mejor == "LSTM":
        clf = LSTMClassifier(meta_sup["input_dim"], cfg["model"]["lstm_hidden"],
                             cfg["model"]["lstm_layers"], cfg["model"]["mc_dropout"])
        clf.load_state_dict(torch.load(ARTIFACTS / f"clf_lstm{sfx}.pt"))
        prob, std = clf.predict_proba(torch.tensor(X), n_samples=cfg["model"]["mc_samples"])
        thr, thr_alta = meta_sup["thr_lstm"], meta_sup["thr_lstm_alta"]
    else:
        gbt = joblib.load(ARTIFACTS / f"clf_gbt{sfx}.joblib")
        prob = gbt.predict_proba(window_features(X))[:, 1]
        std = None  # GBT no entrega incertidumbre (no MC Dropout)
        thr, thr_alta = meta_sup["thr_gbt"], meta_sup["thr_gbt_alta"]

    # Banda de riesgo por doble umbral (calibrados por precisión en train)
    banda = np.where(prob >= thr_alta, "Alto",
                     np.where(prob >= thr, "Medio", "Bajo"))

    # Antigüedad: referencia = última fecha de muestreo disponible en los datos
    ref = pd.to_datetime(df["fecha_muestra"]).max()
    dias = (ref - pd.to_datetime(meta["fecha_ultima"])).dt.days
    max_inact = int(cfg.get("predict", {}).get("max_dias_inactivo", 180))
    inactivo = dias > max_inact

    # Modo de falla dominante (explicabilidad) desde los metales del ancla
    F = build_F(cfg)
    m = apply_F(F, torch.tensor(x_last)).numpy()
    modo_dom = [cfg["failure_modes"][i] for i in m.argmax(axis=1)]

    out = meta.copy()
    out["prob_adversa"] = prob.round(3)
    out["banda"] = banda
    out["dias_desde_ultima"] = dias
    out["estado_dato"] = np.where(inactivo, "INACTIVO", "vigente")
    out["modo_dominante"] = modo_dom
    if std is not None:
        out["incertidumbre"] = std.round(3)

    # Ranking accionable: solo motores vigentes, por probabilidad
    vig = out[~inactivo].sort_values("prob_adversa", ascending=False).reset_index(drop=True)
    n_alto = (vig["banda"] == "Alto").sum()
    n_medio = (vig["banda"] == "Medio").sum()

    pd.set_option("display.width", 200)
    print(f"\n===== TABLERO DE FLOTA (modelo {mejor}) =====")
    print(f"Umbrales: Medio>={thr:.3f}  Alto>={thr_alta:.3f}  | "
          f"vigentes={len(vig)} inactivos={int(inactivo.sum())}  | "
          f"ALTO={n_alto}  MEDIO={n_medio}")
    print("\n--- Motores VIGENTES en banda Alto/Medio (accionables) ---")
    foco = vig[vig["banda"] != "Bajo"]
    cols = ["equipo", "fecha_ultima", "prob_adversa", "banda", "modo_dominante"]
    print(foco[cols].to_string(index=False) if len(foco) else "  (ninguno)")

    out.sort_values("prob_adversa", ascending=False).to_csv(
        ARTIFACTS / "flota_supervisado.csv", index=False)
    print(f"\nGuardado completo (incl. inactivos): {ARTIFACTS/'flota_supervisado.csv'}")


if __name__ == "__main__":
    main()
