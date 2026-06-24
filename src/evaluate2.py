"""python -m src.evaluate2"""
import os
os.environ["DATA_SOURCE"] = "sql"
import pandas as pd
from .config import load_config, ARTIFACTS
from .features.limits import make_label

cfg = load_config()
lab = make_label(cfg, min_excedencias=2)   # crítico = 2+ metales sobre LC

ser = pd.read_csv(ARTIFACTS / "riesgo_series.csv", parse_dates=["fecha_muestra"])
m = ser.merge(lab, on=["equipo","fecha_muestra"], how="left").dropna(subset=["label_obj"])

print("Etiquetadas:", len(m), "de", len(ser))
print("\nDistribucion de etiqueta objetiva:")
print(m["label_obj"].value_counts())
print("\n=== R_motor por etiqueta objetiva ===")
print(m.groupby("label_obj")["R_motor"].agg(["mean","median","count"]))
print("\n=== Nivel predicho vs etiqueta objetiva ===")
print(pd.crosstab(m["nivel"], m["label_obj"]))

crit = m[m["label_obj"]=="Critico"]; norm = m[m["label_obj"]=="Normal"]
if len(crit) and len(norm):
    print(f"\nDeteccion (Critico en Nivel>=2): {(crit['nivel']>=2).mean()*100:.1f}%")
    print(f"Falsos positivos (Normal en Nivel>=2): {(norm['nivel']>=2).mean()*100:.1f}%")