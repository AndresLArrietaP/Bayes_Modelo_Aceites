"""Valida el modelo: nivel predicho vs Estado real de la BD.
Ejecuta: python -m src.evaluate"""
import pandas as pd
from .config import ARTIFACTS

df = pd.read_csv(ARTIFACTS / "riesgo_series.csv")
# usa solo muestras que SÍ tienen etiqueta real
lab = df[df["modo_real"].isin(["Sano", "Anomalo"])].copy()
print(f"Muestras con etiqueta real: {len(lab)} de {len(df)}\n")

if len(lab):
    # tabla cruzada: nivel predicho (0-3) vs realidad
    tabla = pd.crosstab(lab["nivel"], lab["modo_real"],
                        rownames=["Nivel predicho"], colnames=["Realidad"])
    print("=== Cruce nivel predicho vs realidad ===")
    print(tabla, "\n")

    # ¿el riesgo medio es mayor en los Anomalo que en los Sano?
    print("=== R_motor promedio por realidad ===")
    print(lab.groupby("modo_real")["R_motor"].agg(["mean", "median", "count"]), "\n")

    # tasa de detección: % de Anomalo que cae en Nivel >=2
    anom = lab[lab["modo_real"] == "Anomalo"]
    sano = lab[lab["modo_real"] == "Sano"]
    if len(anom) and len(sano):
        det = (anom["nivel"] >= 2).mean() * 100
        fp = (sano["nivel"] >= 2).mean() * 100
        print(f"Detección (Anomalo en Nivel>=2): {det:.1f}%")
        print(f"Falsos positivos (Sano en Nivel>=2): {fp:.1f}%")