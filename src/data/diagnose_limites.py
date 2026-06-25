"""python -m src.data.diagnose_limites"""
import pandas as pd
from sqlalchemy import text, inspect
from ..config import load_config
from .db import get_engine

eng = get_engine()
# 1. nombre y columnas de la tabla de límites (ajusta el nombre si lo conoces)
insp = inspect(eng)
for sch in insp.get_schema_names():
    for t in insp.get_table_names(schema=sch):
        if "lim" in t.lower() or t.lower() in ("lb","lc","lc2","trend"):
            print(f"\n=== {sch}.{t} ===")
            for c in insp.get_columns(t, schema=sch):
                print(" ", c["name"], c["type"])