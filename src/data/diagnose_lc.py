"""python -m src.data.diagnose_lc"""
import pandas as pd
from sqlalchemy import inspect, text

from .db import get_engine

eng = get_engine()


def show(t, q):
    print("\n" + "=" * 60 + f"\n{t}\n" + "=" * 60)
    try:
        with eng.connect() as c:
            print(pd.read_sql(text(q), c).to_string(index=False))
    except Exception as e:
        print("ERROR:", e)


show("lc: combinaciones (COMPONENTE/MODELO/TIPO) para Proyecto", """
    SELECT TOP 40 Proyecto, COMPONENTE, MODELO, TIPO,
           [FIERRO - LP] AS Fe_LP, [FIERRO - LC] AS Fe_LC, [TBN - LP] AS TBN_LP
    FROM Eqpcare.lc ORDER BY Proyecto, COMPONENTE""")

show("LaboratoryData: columnas de texto candidatas a cruce", """
    SELECT TOP 5 Compartimiento, CliCodigoHis, EquCodigoHis, ComCodigoHis,
           ComponentStatus, Ubicacion
    FROM [Oil].[LaboratoryData] WHERE Compartimiento='MOTOR'""")

show("MiningEquipment join a proyecto/fleet", """
    SELECT TOP 5 me.Id, me.Code, mp.Id AS ProjId, ef.Id AS FleetId
    FROM Mine.MiningEquipment me
    LEFT JOIN Mine.MiningProject mp ON me.MiningProjectId = mp.Id
    LEFT JOIN Mine.EquipmentFleet ef ON me.EquipmentFleetId = ef.Id""")

insp = inspect(eng)
for sch, t in [("Mine", "MiningProject"), ("Mine", "EquipmentFleet")]:
    print(f"\n=== {sch}.{t} ===")
    for c in insp.get_columns(t, schema=sch):
        print(" ", c["name"], c["type"])
