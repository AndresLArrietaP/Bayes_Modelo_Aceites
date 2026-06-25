"""Eventos de falla (Eqpcare.Fault) y etiquetado por horizonte.

Convierte el problema de detección NO supervisada (anomalía vs. proxy de límites)
en PRONÓSTICO SUPERVISADO con verdad de campo:

    Una muestra de aceite en t es POSITIVA (y_fail = 1) si el MISMO equipo sufre
    una falla de motor dentro de un horizonte hacia adelante (t, t+H], medido en
    días calendario o en horas de operación (Smr).

Esto habilita métricas reales —precision/recall/PR-AUC y tiempo de anticipación
(lead time)— en lugar de medir auto-consistencia contra los mismos metales.

Antes de confiar en los filtros de motor, corre docs/VALIDACION_SSMS.sql (Bloques
2 y 5) y ajusta `faults.engine_codes` / `faults.engine_keywords` en config.yaml.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text

from .db import get_engine


def load_faults(cfg: dict) -> pd.DataFrame:
    """Carga Eqpcare.Fault a columnas internas: equipo, fault_fecha, fault_smr, code, descripcion."""
    fc = cfg["faults"]
    q = f"""
        SELECT [{fc['equipment_col']}] AS equipo,
               [{fc['date_col']}]      AS fault_fecha,
               [{fc['smr_col']}]       AS fault_smr,
               [{fc['code_col']}]      AS code,
               [{fc['desc_col']}]      AS descripcion
        FROM {fc['table']}
        WHERE [{fc['equipment_col']}] IS NOT NULL
    """
    with get_engine().connect() as c:
        f = pd.read_sql(text(q), c)
    f["fault_fecha"] = pd.to_datetime(f["fault_fecha"], errors="coerce")
    f["equipo"] = f["equipo"].astype(str).str.upper()
    return f


def filter_engine_faults(faults: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Filtra a fallas relevantes a motor/aceite por prefijo de Code o palabra clave
    en Description. Si no hay filtros configurados, devuelve todas (con aviso)."""
    fc = cfg["faults"]
    codes = [c.upper() for c in (fc.get("engine_codes") or [])]
    kws = [k.upper() for k in (fc.get("engine_keywords") or [])]
    if not codes and not kws:
        print("[faults] Sin engine_codes/engine_keywords -> se usan TODAS las fallas.")
        return faults

    code_u = faults["code"].astype(str).str.upper()
    desc_u = faults["descripcion"].astype(str).str.upper()
    mask = pd.Series(False, index=faults.index)
    for c in codes:
        mask |= code_u.str.startswith(c)
    for k in kws:
        mask |= desc_u.str.contains(k, na=False, regex=False)
    out = faults[mask].copy()
    print(f"[faults] Fallas de motor: {len(out)} de {len(faults)} totales.")
    return out


def label_by_horizon(samples: pd.DataFrame, faults: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Etiqueta cada muestra con y_fail (0/1) y lead_time_dias.

    samples: DataFrame con al menos [equipo, fecha_muestra].
    Positiva si existe una falla del mismo equipo en (fecha_muestra, fecha_muestra + H].
    Usa merge_asof (siguiente falla hacia adelante por equipo).
    """
    hdays = int(cfg["faults"].get("horizon_days", 90))

    s = samples.copy()
    s["equipo"] = s["equipo"].astype(str).str.upper()
    s["fecha_muestra"] = pd.to_datetime(s["fecha_muestra"], errors="coerce")
    s = s.dropna(subset=["fecha_muestra"]).sort_values("fecha_muestra")

    f = (faults[["equipo", "fault_fecha"]]
         .dropna()
         .sort_values("fault_fecha")
         .rename(columns={"fault_fecha": "next_fault"}))

    if f.empty:
        s["y_fail"] = 0
        s["lead_time_dias"] = np.nan
        return s

    m = pd.merge_asof(
        s, f,
        left_on="fecha_muestra", right_on="next_fault",
        by="equipo", direction="forward", allow_exact_matches=False,
    )
    gap = (m["next_fault"] - m["fecha_muestra"]).dt.days
    within = gap.le(hdays).fillna(False)
    m["y_fail"] = within.astype(int)
    m["lead_time_dias"] = gap.where(within)
    pos = int(m["y_fail"].sum())
    print(f"[faults] Positivas (falla en {hdays}d): {pos} de {len(m)} "
          f"({100.0*pos/max(1,len(m)):.1f}%)")
    return m
