"""Evaluación del modelo de riesgo.

Ejecuta:  python -m src.evaluate

Tres referencias, de menor a mayor valor:
  1. vs_estado   — nivel predicho vs flag `Estado` del laboratorio (proxy débil).
  2. vs_limits   — nivel predicho vs etiqueta objetiva por límites lc (proxy; SQL).
  3. vs_faults   — nivel predicho vs FALLA REAL en el horizonte (Eqpcare.Fault).

(1) y (2) miden auto-consistencia: la etiqueta deriva de los mismos metales que
el modelo ya ve. (3) es la única evaluación de PRONÓSTICO real (verdad de campo).
Requiere artifacts/riesgo_series.csv generado por `python -m src.predict`.
"""
from __future__ import annotations

import os

import pandas as pd

from .config import ARTIFACTS, load_config


def _load_series() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "riesgo_series.csv", parse_dates=["fecha_muestra"])


def _report(df: pd.DataFrame, label_col: str, pos_value: str, neg_value: str) -> None:
    sub = df.dropna(subset=[label_col])
    print(f"\nMuestras con etiqueta '{label_col}': {len(sub)} de {len(df)}")
    if not len(sub):
        return
    print(pd.crosstab(sub["nivel"], sub[label_col],
                      rownames=["Nivel predicho"], colnames=[label_col]))
    print("\nR_motor por etiqueta:")
    print(sub.groupby(label_col)["R_motor"].agg(["mean", "median", "count"]))
    pos = sub[sub[label_col] == pos_value]
    neg = sub[sub[label_col] == neg_value]
    if len(pos) and len(neg):
        det = (pos["nivel"] >= 2).mean() * 100
        fp = (neg["nivel"] >= 2).mean() * 100
        print(f"\nDetección ({pos_value} en Nivel>=2): {det:.1f}%")
        print(f"Falsos positivos ({neg_value} en Nivel>=2): {fp:.1f}%")


def vs_estado() -> None:
    """(1) Nivel vs Estado/modo_real de la BD (Sano/Anomalo)."""
    print("\n" + "=" * 60 + "\n1) vs Estado del laboratorio (proxy)\n" + "=" * 60)
    df = _load_series()
    _report(df[df["modo_real"].isin(["Sano", "Anomalo"])].copy(),
            "modo_real", "Anomalo", "Sano")


def vs_limits(min_excedencias: int = 2) -> None:
    """(2) Nivel vs etiqueta objetiva por límites lc (Critico si >=N metales sobre LC)."""
    print("\n" + "=" * 60 + "\n2) vs límites de laboratorio lc (proxy)\n" + "=" * 60)
    os.environ["DATA_SOURCE"] = "sql"
    from .features.limits import make_label
    cfg = load_config()
    lab = make_label(cfg, min_excedencias=min_excedencias)
    df = _load_series().merge(lab, on=["equipo", "fecha_muestra"], how="left")
    _report(df, "label_obj", "Critico", "Normal")


def vs_faults() -> None:
    """(3) Nivel vs FALLA REAL en el horizonte (Eqpcare.Fault). Evaluación de pronóstico."""
    print("\n" + "=" * 60 + "\n3) vs falla real en el horizonte (Eqpcare.Fault)\n" + "=" * 60)
    os.environ["DATA_SOURCE"] = "sql"
    from .data.faults import filter_engine_faults, label_by_horizon, load_faults
    cfg = load_config()
    faults = filter_engine_faults(load_faults(cfg), cfg)
    df = _load_series()
    labeled = label_by_horizon(df[["equipo", "fecha_muestra"]], faults, cfg)
    df = df.merge(labeled[["equipo", "fecha_muestra", "y_fail", "lead_time_dias"]],
                  on=["equipo", "fecha_muestra"], how="left")
    df["etiqueta"] = df["y_fail"].map({1: "Falla", 0: "Sin falla"})
    _report(df, "etiqueta", "Falla", "Sin falla")
    fallas = df[df["y_fail"] == 1]
    if len(fallas):
        print(f"\nLead time (días) en positivas: "
              f"media={fallas['lead_time_dias'].mean():.0f}  "
              f"mediana={fallas['lead_time_dias'].median():.0f}")


def main() -> None:
    vs_estado()
    for fn in (vs_limits, vs_faults):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - dependen de conexión SQL
            print(f"  (omitido: {e})")


if __name__ == "__main__":
    main()
