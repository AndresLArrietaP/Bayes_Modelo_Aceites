"""Diagnóstico de la BD real antes de entrenar.  Ejecuta: python -m src.data.diagnose"""
import pandas as pd
from sqlalchemy import text

from ..config import load_config
from .db import get_engine


def show(title, query):
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)
    try:
        with get_engine().connect() as c:
            print(pd.read_sql(text(query), c).to_string(index=False))
    except Exception as e:
        print("  ERROR:", e)


cfg = load_config()
tabla = cfg["db"]["table"]

show("Total de filas", f"SELECT COUNT(*) AS filas FROM {tabla}")

show("Valores de Compartimiento (top 30)", f"""
    SELECT Compartimiento, COUNT(*) AS n
    FROM {tabla} GROUP BY Compartimiento ORDER BY n DESC""")

show("Valores de Estado", f"""
    SELECT Estado, COUNT(*) AS n
    FROM {tabla} GROUP BY Estado ORDER BY n DESC""")

show("Muestras por motor (ComponentSerialNumber) - resumen", f"""
    SELECT MIN(n) AS minimo, MAX(n) AS maximo, AVG(n*1.0) AS promedio,
           SUM(CASE WHEN n>=11 THEN 1 ELSE 0 END) AS motores_con_11omas
    FROM (SELECT ComponentSerialNumber, COUNT(*) AS n
          FROM {tabla} GROUP BY ComponentSerialNumber) t""")

show("% de nulos en metales clave", f"""
    SELECT COUNT(*) AS total,
           COUNT(Fe_ppm) AS Fe_ok, COUNT(Cu_ppm) AS Cu_ok,
           COUNT(Oxidacion) AS Ox_ok, COUNT(TBN) AS TBN_ok
    FROM {tabla}""")
