"""Carga de datos unificada (SQL o sintético) con limpieza, imputación y
normalización por límites de laboratorio (Eqpcare.lc).
"""
from __future__ import annotations

import os

import pandas as pd
from sqlalchemy import text

from . import synthetic
from .db import get_engine


def _clean_impute(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # extra_vars solo existen en modo SQL; se ignoran si no están en el df.
    num_cols = [c for c in cfg["oil_vars"] + cfg["context_vars"] + cfg.get("extra_vars", [])
                if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["equipo", "fecha_muestra"])
    # imputación temporal por motor, luego mediana global, luego 0
    df[num_cols] = df.groupby("equipo")[num_cols].transform(lambda s: s.ffill().bfill())
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[num_cols] = df[num_cols].fillna(0.0)

    # --- Normalización por límite crítico (lc) con fallback, solo en modo SQL ---
    if os.environ.get("DATA_SOURCE", "synthetic") == "sql":
        from ..features.limits import attach_limits
        df = attach_limits(df, cfg)
        for v in cfg["oil_vars"]:
            lc_col = f"LC_{v}"
            if lc_col in df.columns:
                fb = df[v].quantile(0.95) or 1.0
                denom = df[lc_col].fillna(fb).replace(0, fb)
                df[v] = df[v] / denom
                df.drop(columns=[lc_col], inplace=True, errors="ignore")

    return df.reset_index(drop=True)


def _load_sql(cfg: dict) -> pd.DataFrame:
    db = cfg["db"]
    cmap = db["column_map"]

    select_parts = [
        f"[{db['equipment_col']}] AS equipo",
        f"[{db['date_col']}] AS fecha_muestra",
        f"[{db.get('family_col', db['equipment_col'])}] AS familia_motor",
    ]
    for name in cfg["oil_vars"] + cfg["context_vars"] + cfg.get("extra_vars", []):
        real = cmap.get(name)
        if real is None:
            raise KeyError(f"Falta mapeo de columna para '{name}' en config.db.column_map")
        select_parts.append(f"[{real}] AS {name}")

    status_col = db.get("status_col")
    if status_col:
        select_parts.append(f"[{status_col}] AS estado_bd")

    query = f"SELECT {', '.join(select_parts)} FROM {db['table']}"
    filt = db.get("compartment_filter") or []
    if filt and db.get("compartment_col"):
        vals = ", ".join("'" + v.replace("'", "''") + "'" for v in filt)
        query += f" WHERE [{db['compartment_col']}] IN ({vals})"

    with get_engine().connect() as conn:
        df = pd.read_sql(text(query), conn)

    df["fecha_muestra"] = pd.to_datetime(df["fecha_muestra"], errors="coerce")

    if "estado_bd" in df.columns:
        healthy = set(v.upper() for v in db.get("status_healthy_values", []))

        def lab(s):
            if pd.isna(s):
                return "NA"
            return "Sano" if str(s).upper() in healthy else "Anomalo"

        df["_modo_real"] = df["estado_bd"].apply(lab)
    else:
        df["_modo_real"] = "NA"

    df = df.dropna(subset=["equipo", "fecha_muestra"])
    internal = (["equipo", "fecha_muestra", "familia_motor"]
                + cfg["oil_vars"] + cfg["context_vars"] + cfg.get("extra_vars", [])
                + ["_modo_real"])
    return df[internal]


def load_data(cfg: dict, source: str | None = None) -> pd.DataFrame:
    source = source or os.environ.get("DATA_SOURCE", "synthetic")
    if source == "sql":
        df = _load_sql(cfg)
    elif source == "synthetic":
        df = synthetic.generate_fleet(seed=cfg["train"]["seed"])
    else:
        raise ValueError(f"DATA_SOURCE desconocido: {source!r} (usa 'sql' o 'synthetic')")
    return _clean_impute(df, cfg).sort_values(["equipo", "fecha_muestra"]).reset_index(drop=True)
