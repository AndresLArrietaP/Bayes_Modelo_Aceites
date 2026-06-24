"""Explorador de esquema de la base de datos.

Ejecuta:
    python -m src.data.explore_schema
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import inspect, text

from .db import get_engine


def list_tables() -> list[str]:
    eng = get_engine()
    insp = inspect(eng)
    tables = []
    for schema in insp.get_schema_names():
        for t in insp.get_table_names(schema=schema):
            tables.append(f"{schema}.{t}")
        for v in insp.get_view_names(schema=schema):
            tables.append(f"{schema}.{v} (view)")
    return tables


def describe_table(qualified_name: str, sample: int = 5) -> None:
    eng = get_engine()
    insp = inspect(eng)
    schema, name = qualified_name.split(".", 1)
    name = name.replace(" (view)", "")
    print(f"\n=== Columnas de {schema}.{name} ===")
    for col in insp.get_columns(name, schema=schema):
        print(f"  {col['name']:<30} {col['type']}")
    with eng.connect() as conn:
        df = pd.read_sql(text(f"SELECT TOP {sample} * FROM [{schema}].[{name}]"), conn)
    print(f"\n--- Muestra ({len(df)} filas) ---")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df)


if __name__ == "__main__":
    print("Tablas y vistas disponibles:")
    for t in list_tables():
        print("  -", t)
