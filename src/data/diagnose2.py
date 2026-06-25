"""python -m src.data.diagnose2"""
import pandas as pd
from sqlalchemy import text
from ..config import load_config
from .db import get_engine

cfg = load_config()
q = """
SELECT n_muestras = COUNT(*) ,
       motores = COUNT(DISTINCT MiningEquipmentId),
       nulos_equipo = SUM(CASE WHEN MiningEquipmentId IS NULL THEN 1 ELSE 0 END)
FROM [Oil].[LaboratoryData] WHERE Compartimiento = 'MOTOR'
"""
q2 = """
SELECT TOP 15 MiningEquipmentId, COUNT(*) n
FROM [Oil].[LaboratoryData] WHERE Compartimiento='MOTOR'
GROUP BY MiningEquipmentId ORDER BY n DESC
"""
with get_engine().connect() as c:
    print(pd.read_sql(text(q), c).to_string(index=False))
    print("\nMuestras por motor (top 15):")
    print(pd.read_sql(text(q2), c).to_string(index=False))