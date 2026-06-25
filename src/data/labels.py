"""Etiqueta supervisada de PRONÓSTICO por severidad de condición del laboratorio.

Verdad de campo = veredicto del laboratorio en Oil.LaboratoryData, unificando dos
columnas en una severidad ordinal (0=Normal, 1=Monitoreo, 2=Precaución, 3=Crítico):

    severidad = coalesce(Condicion_map, Estado_map)   # Condicion primaria, Estado fallback

Esto sube la cobertura etiquetada (~3.4k solo Condicion -> ~9k con Estado) frente a
Eqpcare.Fault, que es un log de telemetría (~1.9M eventos) sin valor clínico para v1.

Etiqueta binaria de pronóstico:
    Una muestra en t es POSITIVA (y_target = 1) si el MISMO motor alcanza severidad
    >= target.adverse_min_severity en una muestra futura dentro de (t, t+H] días.
    La condición de la propia muestra t NO es feature -> pronóstico real, sin fuga.

Diseño features densos / etiqueta dispersa: las ventanas se arman sobre metales
(densos); el entrenamiento conserva solo ventanas cuyo target tiene severidad
codificada. Ver config.yaml -> target. Mapeos a verificar con VALIDACION B13.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text

from .db import get_engine


def load_condition_series(cfg: dict) -> pd.DataFrame:
    """Carga [equipo, fecha_muestra, condicion, estado, horometro] de muestras MOTOR."""
    db = cfg["db"]
    q = f"""
        SELECT [{db['equipment_col']}] AS equipo,
               [{db['date_col']}]      AS fecha_muestra,
               [Condicion]             AS condicion,
               [{db.get('status_col', 'Estado')}] AS estado,
               [Horometro]             AS horometro
        FROM {db['table']}
        WHERE [{db['compartment_col']}] = 'MOTOR'
    """
    with get_engine().connect() as c:
        df = pd.read_sql(text(q), c)
    df["fecha_muestra"] = pd.to_datetime(df["fecha_muestra"], errors="coerce")
    df["equipo"] = df["equipo"].astype(str).str.upper()
    return df.dropna(subset=["equipo", "fecha_muestra"])


def unified_severity(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Severidad ordinal 0..3 desde Condicion (primaria) y Estado (fallback)."""
    t = cfg["target"]
    cmap = {str(k).upper().strip(): int(v) for k, v in (t.get("condicion_map") or {}).items()}
    emap = {str(k).upper().strip(): int(v) for k, v in (t.get("estado_map") or {}).items()}

    def _map(col, mp):
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return df[col].astype(str).str.upper().str.strip().map(mp)

    sev = _map("condicion", cmap)
    return sev.fillna(_map("estado", emap))


def label_future_adverse(samples: pd.DataFrame, cfg: dict,
                         condition_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Añade y_target (0/1), label_valido (bool) y lead_time_dias a cada muestra.

    samples: df con [equipo, fecha_muestra] (+ opcional condicion/estado).
    condition_df: si se pasa (load_condition_series), aporta las muestras codificadas
    futuras; si no, se usan las del propio `samples`.

    y_target = 1 si hay una muestra con severidad >= adverse_min en (t, t+H].
    label_valido = True si el desenlace es OBSERVABLE: o bien es positiva, o bien hay
    al menos una muestra codificada (no adversa) en (t, t+H]. Si no hay ninguna muestra
    codificada futura dentro del horizonte, el desenlace es desconocido (censura por
    derecha) y la muestra NO debe usarse como negativo. Filtra por label_valido.
    """
    t = cfg["target"]
    hdays = int(t.get("horizon_days", 120))
    adverse_min = int(t.get("adverse_min_severity", 2))

    s = samples.copy()
    s["equipo"] = s["equipo"].astype(str).str.upper()
    s["fecha_muestra"] = pd.to_datetime(s["fecha_muestra"], errors="coerce")

    cond = condition_df.copy() if condition_df is not None else s
    cond["equipo"] = cond["equipo"].astype(str).str.upper()
    cond["sev"] = unified_severity(cond, cfg)

    coded = (cond.loc[cond["sev"].notna(), ["equipo", "fecha_muestra", "sev"]]
                 .dropna(subset=["fecha_muestra"]).sort_values("fecha_muestra"))
    adv = (coded.loc[coded["sev"] >= adverse_min, ["equipo", "fecha_muestra"]]
               .rename(columns={"fecha_muestra": "adv_date"}))
    coded_any = coded[["equipo", "fecha_muestra"]].rename(columns={"fecha_muestra": "coded_date"})

    s = s.sort_values("fecha_muestra")

    def _next(right):
        if right.empty:
            return pd.Series(pd.NaT, index=s.index)
        col = right.columns[1]
        merged = pd.merge_asof(
            s[["equipo", "fecha_muestra"]], right,
            left_on="fecha_muestra", right_on=col,
            by="equipo", direction="forward", allow_exact_matches=False,
        )
        return merged[col].values

    s["adv_date"] = _next(adv)
    s["coded_date"] = _next(coded_any)
    gap_adv = (pd.to_datetime(s["adv_date"]) - s["fecha_muestra"]).dt.days
    gap_coded = (pd.to_datetime(s["coded_date"]) - s["fecha_muestra"]).dt.days

    y = gap_adv.le(hdays).fillna(False)
    s["y_target"] = y.astype(int)
    s["label_valido"] = (y | gap_coded.le(hdays).fillna(False)).astype(bool)
    s["lead_time_dias"] = gap_adv.where(y)

    val = s[s["label_valido"]]
    pos = int(val["y_target"].sum())
    print(f"[labels] Etiquetables: {len(val)} de {len(s)} | positivas (sev>={adverse_min} "
          f"en {hdays}d): {pos} ({100.0*pos/max(1,len(val)):.1f}%)")
    return s.drop(columns=["adv_date", "coded_date"])
