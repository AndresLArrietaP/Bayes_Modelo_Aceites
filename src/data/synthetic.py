"""Generador de datos sintéticos de análisis de aceite.

Permite desarrollar y probar TODO el pipeline sin acceso a Azure SQL.
Simula una flota con motores sanos y algunos con falla progresiva.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OIL_VARS = ["Fe", "Cu", "Pb", "Sn", "Al", "Cr", "Ni", "Si",
            "Na", "K", "Ox", "Nit", "Hollin", "V100", "TBN"]
CONTEXT_VARS = ["HorasAceite", "HorasComp", "Horometro", "Carga", "Altitud", "TempAmb"]

_BASE = {
    "Fe": 15, "Cu": 4, "Pb": 3, "Sn": 1.5, "Al": 3, "Cr": 1.5, "Ni": 0.8,
    "Si": 8, "Na": 3, "K": 2, "Ox": 10, "Nit": 8, "Hollin": 0.2,
    "V100": 14.5, "TBN": 9.0,
}
_MODE_DRIVERS = {
    "Cojinetes":    {"Cu": 6.0, "Pb": 5.0, "Sn": 3.0},
    "Cilindro":     {"Fe": 10.0, "Cr": 2.5, "Al": 4.0},
    "Aire":         {"Si": 12.0, "Al": 5.0, "Fe": 6.0},
    "Refrigerante": {"Na": 8.0, "K": 6.0, "TBN": -4.0},
    "Combustion":   {"Ox": 15.0, "Nit": 12.0, "Hollin": 1.5},
}


def _one_engine(engine_id, n_samples, rng, mode=None, onset=0.5):
    rows = []
    horas = 0.0
    horas_comp = rng.uniform(2000, 8000)
    altitud = rng.uniform(2000, 4500)
    for t in range(n_samples):
        horas += rng.uniform(180, 260)
        horas_comp += horas
        rec = {v: max(0.0, _BASE[v] + rng.normal(0, _BASE[v] * 0.08)) for v in OIL_VARS}

        if mode is not None:
            frac = t / (n_samples - 1)
            if frac >= onset:
                sev = (frac - onset) / (1 - onset)
                sev = sev ** 1.6
                for metal, gain in _MODE_DRIVERS[mode].items():
                    rec[metal] = max(0.0, rec[metal] + gain * sev * (3 + 2 * frac))

        rec.update({
            "equipo": engine_id,
            "fecha_muestra": pd.Timestamp("2024-01-01") + pd.Timedelta(days=14 * t),
            "familia_motor": "QSK95" if int(engine_id[-1]) % 2 == 0 else "QSK78",
            "HorasAceite": horas,
            "HorasComp": horas_comp,
            "Horometro": horas_comp + rng.uniform(0, 500),
            "Carga": rng.uniform(60, 95),
            "Altitud": altitud,
            "TempAmb": rng.uniform(5, 30),
            "_modo_real": mode or "Sano",
        })
        rows.append(rec)
    return rows


def generate_fleet(n_healthy=20, n_faulty=5, n_samples=40, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_healthy):
        rows += _one_engine(f"T3{100 + i:03d}", n_samples, rng, mode=None)
    modes = list(_MODE_DRIVERS)
    for i in range(n_faulty):
        mode = modes[i % len(modes)]
        rows += _one_engine(f"T3{170 + i:03d}", n_samples, rng,
                            mode=mode, onset=rng.uniform(0.4, 0.6))
    df = pd.DataFrame(rows)
    cols = (["equipo", "fecha_muestra", "familia_motor"]
            + OIL_VARS + CONTEXT_VARS + ["_modo_real"])
    return df[cols].sort_values(["equipo", "fecha_muestra"]).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_fleet()
    print(df.head(12))
    print(f"\nTotal filas: {len(df)} | motores: {df['equipo'].nunique()}")
