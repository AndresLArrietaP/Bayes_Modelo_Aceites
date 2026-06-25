"""Diagnóstico de la BD real (consolida diagnose/diagnose2/diagnose_join/diagnose_lc).

Ejecuta:  python -m src.data.diagnose

Recorre, en orden, el estado de las tablas que alimentan el modelo:
  - Oil.LaboratoryData (volumen, compartimientos, estado, nulos, muestras/motor)
  - Mine.* (join a proyecto/modelo) y Eqpcare.lc (límites)
  - Eqpcare.Fault (eventos de falla) y su alineación temporal con el aceite

La alineación temporal (sección FALLAS) es el equivalente en Python de los Bloques
5-7 de docs/VALIDACION_SSMS.sql: confirma si hay señal para pronóstico supervisado.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import inspect, text

from ..config import load_config
from .db import get_engine

cfg = load_config()
TABLA = cfg["db"]["table"]
eng = get_engine()


def show(title: str, query: str) -> None:
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64)
    try:
        with eng.connect() as c:
            print(pd.read_sql(text(query), c).to_string(index=False))
    except Exception as e:  # noqa: BLE001 - diagnóstico best-effort
        print("  ERROR:", e)


def main() -> None:
    # ---------------- Oil.LaboratoryData ----------------
    show("Total de filas", f"SELECT COUNT(*) AS filas FROM {TABLA}")

    show("Valores de Compartimiento (top 30)", f"""
        SELECT TOP 30 Compartimiento, COUNT(*) AS n
        FROM {TABLA} GROUP BY Compartimiento ORDER BY n DESC""")

    show("Valores de Estado", f"""
        SELECT Estado, COUNT(*) AS n
        FROM {TABLA} GROUP BY Estado ORDER BY n DESC""")

    show("Muestras de MOTOR: resumen", """
        SELECT COUNT(*) AS n_muestras,
               COUNT(DISTINCT MiningEquipmentId) AS motores,
               SUM(CASE WHEN MiningEquipmentId IS NULL THEN 1 ELSE 0 END) AS nulos_equipo
        FROM [Oil].[LaboratoryData] WHERE Compartimiento = 'MOTOR'""")

    show("Muestras por motor (top 15)", """
        SELECT TOP 15 MiningEquipmentId, COUNT(*) n
        FROM [Oil].[LaboratoryData] WHERE Compartimiento='MOTOR'
        GROUP BY MiningEquipmentId ORDER BY n DESC""")

    show("% de nulos en metales clave", f"""
        SELECT COUNT(*) AS total,
               COUNT(Fe_ppm) AS Fe_ok, COUNT(Cu_ppm) AS Cu_ok,
               COUNT(Oxidacion) AS Ox_ok, COUNT(TBN) AS TBN_ok
        FROM {TABLA}""")

    # ---------------- Join a proyecto/modelo + límites lc ----------------
    show("Proyecto/Modelo de motores con aceite (MOTOR)", """
        SELECT mp.Name AS proyecto, ef.Model AS modelo,
               COUNT(DISTINCT ld.MiningEquipmentId) AS motores, COUNT(*) AS muestras
        FROM [Oil].[LaboratoryData] ld
        JOIN Mine.MiningEquipment me ON ld.MiningEquipmentId = me.Id
        LEFT JOIN Mine.MiningProject mp ON me.MiningProjectId = mp.Id
        LEFT JOIN Mine.EquipmentFleet ef ON me.EquipmentFleetId = ef.Id
        WHERE ld.Compartimiento = 'MOTOR'
        GROUP BY mp.Name, ef.Model ORDER BY muestras DESC""")

    show("lc: Proyecto/MODELO/TIPO para COMPONENTE=MOTOR", """
        SELECT DISTINCT Proyecto, MODELO, TIPO,
               [FIERRO - LP] AS Fe_LP, [FIERRO - LC] AS Fe_LC, [TBN - LP] AS TBN_LP
        FROM Eqpcare.lc WHERE COMPONENTE = 'MOTOR' ORDER BY Proyecto""")

    # ---------------- Eqpcare.Fault (verdad de campo) ----------------
    show("FALLAS: panorama de Eqpcare.Fault", """
        SELECT COUNT(*) AS total_fallas,
               COUNT(DISTINCT MiningEquipmentId) AS equipos_con_falla,
               MIN(DateFrom) AS fecha_min, MAX(DateFrom) AS fecha_max,
               SUM(CASE WHEN SmrFrom IS NULL THEN 1 ELSE 0 END) AS sin_smr
        FROM Eqpcare.Fault""")

    show("FALLAS: top Code", """
        SELECT TOP 25 Code, COUNT(*) AS n, MIN(Description) AS ejemplo
        FROM Eqpcare.Fault GROUP BY Code ORDER BY n DESC""")

    show("FALLAS: solapamiento equipos falla <-> aceite motor", """
        SELECT
          (SELECT COUNT(DISTINCT MiningEquipmentId) FROM Eqpcare.Fault
             WHERE MiningEquipmentId IS NOT NULL) AS equipos_falla,
          (SELECT COUNT(DISTINCT MiningEquipmentId) FROM Oil.LaboratoryData
             WHERE Compartimiento='MOTOR') AS equipos_aceite,
          (SELECT COUNT(*) FROM (
              SELECT DISTINCT f.MiningEquipmentId FROM Eqpcare.Fault f
              JOIN Oil.LaboratoryData o ON o.MiningEquipmentId=f.MiningEquipmentId
                AND o.Compartimiento='MOTOR') t) AS equipos_en_ambos""")

    show("FALLAS: alineación temporal (aceite en 90/180d previos)", """
        ;WITH f AS (
            SELECT Id, MiningEquipmentId, DateFrom FROM Eqpcare.Fault
            WHERE MiningEquipmentId IS NOT NULL AND DateFrom IS NOT NULL)
        SELECT COUNT(*) AS fallas,
               SUM(CASE WHEN x.c90  > 0 THEN 1 ELSE 0 END) AS con_aceite_90d,
               SUM(CASE WHEN x.c180 > 0 THEN 1 ELSE 0 END) AS con_aceite_180d
        FROM f CROSS APPLY (
            SELECT
              SUM(CASE WHEN o.FechaMuestreo > DATEADD(day,-90,f.DateFrom)
                       AND o.FechaMuestreo <= f.DateFrom THEN 1 ELSE 0 END) AS c90,
              SUM(CASE WHEN o.FechaMuestreo > DATEADD(day,-180,f.DateFrom)
                       AND o.FechaMuestreo <= f.DateFrom THEN 1 ELSE 0 END) AS c180
            FROM Oil.LaboratoryData o
            WHERE o.MiningEquipmentId=f.MiningEquipmentId AND o.Compartimiento='MOTOR') x""")

    # ---------------- Metadatos de tablas de catálogo ----------------
    insp = inspect(eng)
    for sch, t in [("Mine", "MiningProject"), ("Mine", "EquipmentFleet")]:
        print(f"\n=== columnas {sch}.{t} ===")
        try:
            for col in insp.get_columns(t, schema=sch):
                print(" ", col["name"], col["type"])
        except Exception as e:  # noqa: BLE001
            print("  ERROR:", e)


if __name__ == "__main__":
    main()
