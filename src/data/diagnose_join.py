"""python -m src.data.diagnose_join"""
import pandas as pd
from sqlalchemy import text

from .db import get_engine

eng = get_engine()


def show(t, q):
    print("\n" + "=" * 60 + f"\n{t}\n" + "=" * 60)
    with eng.connect() as c:
        print(pd.read_sql(text(q), c).to_string(index=False))


show("Proyecto/Modelo de los motores con aceite (MOTOR)", """
    SELECT mp.Name AS proyecto, ef.Model AS modelo,
           COUNT(DISTINCT ld.MiningEquipmentId) AS motores, COUNT(*) AS muestras
    FROM [Oil].[LaboratoryData] ld
    JOIN Mine.MiningEquipment me ON ld.MiningEquipmentId = me.Id
    LEFT JOIN Mine.MiningProject mp ON me.MiningProjectId = mp.Id
    LEFT JOIN Mine.EquipmentFleet ef ON me.EquipmentFleetId = ef.Id
    WHERE ld.Compartimiento = 'MOTOR'
    GROUP BY mp.Name, ef.Model ORDER BY muestras DESC""")

show("lc: Proyecto/MODELO para COMPONENTE=MOTOR", """
    SELECT DISTINCT Proyecto, MODELO, TIPO FROM Eqpcare.lc
    WHERE COMPONENTE = 'MOTOR' ORDER BY Proyecto""")
