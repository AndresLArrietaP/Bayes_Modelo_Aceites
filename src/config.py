"""Carga y resolución de configuración.

El proyecto entrena un modelo por TIPO DE COMPONENTE (motor, transmisión, mando
final, …). El `config.yaml` trae una base (motor) y un registro `components:` con
las particularidades de cada componente (compartimiento en la BD, nombre en la
tabla de límites `Eqpcare.lc`, y opcionalmente `oil_vars` / matriz de firmas /
modos de falla propios de su tribología).

`load_config()` resuelve el componente activo (env `COMPONENT` > `active_component`
del yaml) superponiendo su bloque sobre la base. Así el código aguas abajo recibe
un cfg "plano" idéntico al de antes, pero específico del componente elegido. Por
defecto el componente es `engine` (MOTOR) → comportamiento idéntico al histórico.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"
ARTIFACTS = ROOT / "artifacts"

# Claves que un componente puede sobrescribir respecto de la base.
_OVERRIDABLE = ("oil_vars", "context_vars", "extra_vars", "failure_modes",
                "signature_matrix")


def resolve_component(cfg: dict, name: str | None = None) -> dict:
    """Devuelve un cfg específico del componente `name` (deepcopy, no muta el original).

    Prioridad del nombre: argumento > env COMPONENT > cfg['active_component'] > 'engine'.
    Superpone components[name] sobre la base y deja trazas en cfg['_component'] y
    cfg['db']['lc_componente'] (nombre del componente en Eqpcare.lc).
    """
    name = name or os.environ.get("COMPONENT") or cfg.get("active_component", "engine")
    registry = cfg.get("components") or {}
    if name not in registry:
        disponibles = ", ".join(sorted(registry)) or "(ninguno)"
        raise KeyError(f"Componente '{name}' no está en config.components. "
                       f"Disponibles: {disponibles}")
    comp = registry[name] or {}
    out = copy.deepcopy(cfg)
    out["_component"] = name
    for k in _OVERRIDABLE:
        if k in comp:
            out[k] = copy.deepcopy(comp[k])
    db = out.setdefault("db", {})
    db["compartment_filter"] = comp.get("compartment_filter",
                                        db.get("compartment_filter", []))
    # Nombre del componente en la tabla de límites Eqpcare.lc (para normalización).
    db["lc_componente"] = comp.get("lc_componente", "MOTOR")
    # T (window_size) puede ajustarse por componente (cadencia de muestreo distinta).
    if "window_size" in comp:
        out.setdefault("model", {})["window_size"] = int(comp["window_size"])
    # Override por componente del objetivo (severidad/horizonte), p.ej. crítico solo
    # tiene sentido en motor; el resto usa precaución+ (sev>=2).
    for k in ("adverse_min_severity", "horizon_days", "year_min"):
        if k in comp:
            out.setdefault("target", {})[k] = comp[k]
    _apply_env_overrides(out)
    return out


def _apply_env_overrides(cfg: dict) -> None:
    """Overrides rápidos por variable de entorno (para barridos sin editar el yaml).
       ADVERSE_MIN=2  HORIZON_DAYS=90  YEAR_MIN=2025  WINDOW_SIZE=6
    """
    env = os.environ
    if env.get("ADVERSE_MIN"):
        cfg.setdefault("target", {})["adverse_min_severity"] = int(env["ADVERSE_MIN"])
    if env.get("HORIZON_DAYS"):
        cfg.setdefault("target", {})["horizon_days"] = int(env["HORIZON_DAYS"])
    if env.get("YEAR_MIN"):
        cfg.setdefault("target", {})["year_min"] = int(env["YEAR_MIN"])
    if env.get("WINDOW_SIZE"):
        cfg.setdefault("model", {})["window_size"] = int(env["WINDOW_SIZE"])


def load_config(path: str | Path = DEFAULT_CONFIG, component: str | None = None) -> dict:
    """Carga el yaml y resuelve el componente activo (engine por defecto).

    Si el yaml no trae `components:` (formato antiguo), devuelve la base tal cual
    para no romper configuraciones previas.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("components"):
        return resolve_component(cfg, component)
    return cfg


def list_components(path: str | Path = DEFAULT_CONFIG) -> list[str]:
    """Lista los componentes definidos en el registro."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return sorted((cfg.get("components") or {}).keys())
