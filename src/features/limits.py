"""Integración de límites de laboratorio (Eqpcare.lc) para normalización física.

- Une cada muestra de aceite con su proyecto/modelo real (via MiningEquipment).
- Trae el límite CRÍTICO (LC) de cada parámetro desde lc para COMPONENTE='MOTOR'.
- Normaliza cada variable como  valor / límite_crítico  (comparable entre minas).
- Donde no hay límite (mina/modelo no está en lc), deja NaN -> el caller usa fallback.
"""
from __future__ import annotations
import pandas as pd
from sqlalchemy import text
from ..data.db import get_engine

# nombre interno -> nombre de columna LC en Eqpcare.lc (solo el límite crítico)
_LC_MAP = {
    "Fe": "FIERRO - LC", "Cu": "COBRE - LC", "Pb": "PLOMO - LC", "Sn": "ESTAÑO - LC",
    "Al": "ALUMINIO - LC", "Cr": "CROMO - LC", "Ni": "NIQUEL - LC", "Si": "SILICIO - LC",
    "Na": "SODIO - LC", "K": "POTASIO - LC", "Ox": "OXI - LC", "Nit": "NIT - LC",
    "Hollin": "HOLLIN - LC", "V100": "VISC - LCS", "TBN": "TBN - LC",
}

def _norm(s):
    return s.astype(str).str.strip().str.upper() if hasattr(s, "str") else str(s).strip().upper()

def load_limits(cfg) -> pd.DataFrame:
    """Devuelve límites críticos por (proyecto, modelo) del componente activo, en
    columnas internas. Solo trae los límites de los oil_vars del componente."""
    comp_lc = cfg.get("db", {}).get("lc_componente", "MOTOR")
    used = [v for v in cfg["oil_vars"] if v in _LC_MAP]
    cols = ", ".join(f'[{_LC_MAP[k]}] AS "{k}"' for k in used)
    safe = comp_lc.replace("'", "''")
    q = f"SELECT Proyecto, MODELO, {cols} FROM Eqpcare.lc WHERE COMPONENTE = '{safe}'"
    with get_engine().connect() as c:
        lc = pd.read_sql(text(q), c)
    lc["_key"] = _norm(lc["Proyecto"]) + "|" + _norm(lc["MODELO"])
    return lc

def equipment_project_model() -> pd.DataFrame:
    """Mapa MiningEquipmentId -> proyecto/modelo reales."""
    q = """
    SELECT me.Id AS equipo_id, mp.Name AS proyecto, ef.Model AS modelo
    FROM Mine.MiningEquipment me
    LEFT JOIN Mine.MiningProject mp ON me.MiningProjectId = mp.Id
    LEFT JOIN Mine.EquipmentFleet ef ON me.EquipmentFleetId = ef.Id
    """
    with get_engine().connect() as c:
        return pd.read_sql(text(q), c)

def attach_limits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Añade columnas LC_<var> a cada fila segun su proyecto/modelo. NaN si no hay límite."""
    emap = equipment_project_model()
    emap["_key"] = _norm(emap["proyecto"]) + "|" + _norm(emap["modelo"])
    lc = load_limits(cfg)

    # df.equipo es MiningEquipmentId (string). Unimos por ese id.
    df = df.copy()
    df["_equipo_str"] = df["equipo"].astype(str).str.upper()
    emap["_equipo_str"] = emap["equipo_id"].astype(str).str.upper()
    df = df.merge(emap[["_equipo_str", "_key"]], on="_equipo_str", how="left")

    avail = [k for k in _LC_MAP if k in lc.columns]   # solo límites del componente
    lc_cols = {k: f"LC_{k}" for k in avail}
    lc_small = lc[["_key"] + avail].rename(columns=lc_cols)
    df = df.merge(lc_small, on="_key", how="left")

    probe = f"LC_{avail[0]}" if avail else None
    cobertura = (df[probe].notna().mean() * 100) if probe else 0.0
    print(f"[limits] Cobertura de límites lc ({cfg.get('db', {}).get('lc_componente', '?')}): "
          f"{cobertura:.1f}% de las muestras")
    return df.drop(columns=["_equipo_str", "_key"])

def load_raw_for_label(cfg):
    """Carga valores CRUDOS (ppm, sin normalizar) directo de la BD, para etiquetar."""
    from sqlalchemy import text
    from ..data.db import get_engine
    db = cfg["db"]; cmap = db["column_map"]
    filt = db.get("compartment_filter") or []
    vals = ", ".join("'" + v.replace("'", "''") + "'" for v in filt)
    where = f"WHERE [{db['compartment_col']}] IN ({vals})" if filt else ""
    sel = [f"[{db['equipment_col']}] AS equipo", f"[{db['date_col']}] AS fecha_muestra"]
    sel += [f"[{cmap[v]}] AS {v}" for v in cfg["oil_vars"]]
    q = f"SELECT {', '.join(sel)} FROM {db['table']} {where}"
    import pandas as pd
    with get_engine().connect() as c:
        df = pd.read_sql(text(q), c)
    df["fecha_muestra"] = pd.to_datetime(df["fecha_muestra"], errors="coerce")
    return df.dropna(subset=["equipo","fecha_muestra"])


def make_label(cfg, min_excedencias=2):
    """Etiqueta objetiva sobre valores CRUDOS.
    'Critico' si >= min_excedencias metales superan su LC; 'Precaucion' si >=1 supera LC;
    si no, 'Normal'. Devuelve df con [equipo, fecha_muestra, label_obj]."""
    import pandas as pd
    d = load_raw_for_label(cfg)
    d = attach_limits(d, cfg)
    n_exc = pd.Series(0, index=d.index)
    for v in cfg["oil_vars"]:
        lc = pd.to_numeric(d.get(f"LC_{v}"), errors="coerce")
        val = pd.to_numeric(d[v], errors="coerce")
        if lc is None:
            continue
        exc = (val.notna() & lc.notna() & (val > lc))
        n_exc = n_exc + exc.astype(int)
    out = pd.Series("Normal", index=d.index)
    out[n_exc >= 1] = "Precaucion"
    out[n_exc >= min_excedencias] = "Critico"
    d["label_obj"] = out
    return d[["equipo","fecha_muestra","label_obj"]]