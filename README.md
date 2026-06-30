# Pronóstico de fallas por análisis de aceite — BayesLSTM-VAE

Modelo predictivo híbrido para flota diésel minera (QSK78 / QSK95) que combina
predicción de tendencias con incertidumbre (**Bayesian LSTM**), detección de
anomalías (**VAE**) y una **matriz de firmas metálicas** que vincula los metales
del análisis de aceite con modos de falla concretos, produciendo un **score de
riesgo multinivel (0–3)** por motor.

Basado en: Chen, Y. et al. (2026). *Diesel engine lubricating oil fault prognosis:
A hybrid Bayesian LSTM and deep generative model architecture for multilayer
anomaly detection.* **Tribology International, 215**, 111434.

---

## ⚠️ Antes de subir a GitHub — léelo

- El archivo **`.env` contiene la cadena de conexión real** (usuario/contraseña de
  lectura a Azure SQL). Está incluido en **`.gitignore`** para que **no se suba**.
- **Verifica siempre** antes del primer push:
  ```bash
  git status        # .env NO debe aparecer entre los archivos a commitear
  git check-ignore .env   # debe imprimir ".env"
  ```
- Si por error apareciera, **no hagas el commit** hasta corregir el `.gitignore`.
  Nunca pongas credenciales dentro de archivos `.py` o `config.yaml`.

> Recomendación adicional: la contraseña que compartiste viajó en texto. Conviene
> **rotarla** en Azure cuando puedas y, si es posible, usar autenticación por
> *Azure AD / Managed Identity* en lugar de usuario-contraseña.

---

## Arquitectura

```
Datos de aceite (Azure SQL o sintético)
        │
        ▼
 Escalado + ventanas temporales (T muestras por motor)
        │
        ▼
 BayesianLSTM  ──►  x̂_{t+H}  +  incertidumbre (Var)      [MC Dropout]
        │
        ├─► m̂ = F · x̂     (activación por modo de falla, matriz de firmas)
        │
        ▼
 z = [x̂, m̂, contexto]  ──►  VAE  ──►  error de reconstrucción e
        │
        ▼
 Score de riesgo:  R_i = α·e + β·|m̂_i| + γ·Var(m̂_i)
                   R_motor = máx_i R_i  →  Nivel 0/1/2/3
```

Los **5 modos de falla** y su firma metálica (editable en `config/config.yaml`):

| Modo | Metales clave |
|------|---------------|
| Cojinetes | Cu, Pb, Sn |
| Cilindro–pistón | Fe, Cr, Al |
| Aire / polvo | Si, Al, Fe |
| Refrigerante | Na, K, TBN↓ |
| Combustión / inyección | Ox, Nit, Hollín |

---

## Estructura del proyecto

```
prognosis-aceite-bayeslstm-vae/
├── config/config.yaml        # variables, matriz F, mapeo de columnas, hiperparámetros
├── src/
│   ├── config.py             # carga de config
│   ├── data/
│   │   ├── db.py             # engine SQLAlchemy desde .env
│   │   ├── explore_schema.py # descubre tablas/columnas de Azure SQL
│   │   ├── synthetic.py      # datos sintéticos para desarrollo
│   │   └── load.py           # carga unificada (sql | synthetic)
│   ├── features/
│   │   ├── signatures.py     # matriz de firmas F
│   │   └── windows.py        # escalado + ventanas temporales
│   ├── models/
│   │   ├── bayes_lstm.py     # LSTM + MC Dropout (incertidumbre)
│   │   ├── bayes_lstm_bbb.py # LSTM Bayes by Backprop (fiel al paper)
│   │   ├── vae.py            # VAE de anomalías
│   │   └── risk.py           # score de riesgo multinivel
│   ├── train.py              # entrena LSTM y VAE, autocalibra umbrales
│   └── predict.py            # tabla de estado de flota
├── notebooks/
│   └── prognosis_aceite_BayesLSTM_VAE.ipynb   # notebook explicativo (Colab/Jupyter)
├── scripts/setup.ps1|.sh     # crea venv e instala dependencias
├── requirements.txt
├── .env.example              # plantilla (sin secretos)
└── .gitignore
```

---

## Puesta en marcha

### 1. Crear el entorno
**Windows (PowerShell):**
```powershell
.\scripts\setup.ps1
```
**Linux / Mac:**
```bash
bash scripts/setup.sh
```

### 2. Probar de inmediato con datos sintéticos
No necesita base de datos. En `.env`, `DATA_SOURCE=synthetic` (por defecto):
```bash
python -m src.train      # entrena y autocalibra umbrales
python -m src.predict    # genera artifacts/estado_flota.csv
```

### 3. Conectar a tu Azure SQL real
1. Instala **ODBC Driver 18 for SQL Server** en el sistema operativo.
2. Descubre el esquema real:
   ```bash
   python -m src.data.explore_schema
   ```
3. Rellena en `config/config.yaml` el bloque `db:` (tabla, columnas de fecha y
   equipo, y `column_map` de cada metal).
4. Cambia `DATA_SOURCE=sql` en `.env` y vuelve a entrenar.

---

## Notebook explicativo (Colab / Jupyter)

`notebooks/prognosis_aceite_BayesLSTM_VAE.ipynb` reproduce todo el modelo en un solo
archivo **autocontenido**, con celdas de **markdown que explican la teoría** (BayesLSTM,
matriz de firmas, VAE, score de riesgo y Bayes by Backprop) intercaladas con el código.
Corre tal cual en Google Colab o Jupyter: por defecto usa datos sintéticos, e incluye una
celda de **conexión segura a Azure SQL** (la contraseña se pide con `getpass`, nunca se
escribe en el notebook).

## Notas técnicas y decisiones de diseño

- **Dos variantes de "Bayesian LSTM"**, elegibles con `model.method` en `config.yaml`:
  - `mc_dropout`: aproximación por *Monte Carlo Dropout* (rápida).
  - `bbb`: **Bayes by Backprop** (Blundell et al., 2015) — pesos variacionales
    `q(W)=N(μ,σ²)` y energía libre variacional, fiel al paper. Está en
    `src/models/bayes_lstm_bbb.py`.
- **Esquema real ya mapeado**: el `config.yaml` apunta a `[Oil].[LaboratoryData]`
  (`Fe_ppm`, `Oxidacion`, `HorasDeAceite`, …), agrupa por `ComponentSerialNumber` y filtra
  `Compartimiento` a motor. Carga/Altitud/TempAmb no existen en esa tabla, así que el
  contexto usa horas/horómetro; pueden unirse luego desde tablas operativas.
- **Umbrales auto-calibrados**: `train.py` fija τ1/τ2/τ3 desde percentiles de la
  distribución de riesgo de los **motores sanos** (funciona aun sin etiquetas).
- **Matriz F** simplificada (pesos 0/1/−1). Conviene refinarla con criterio
  tribológico y validación contra fallas históricas.
- El generador sintético crea 20 motores sanos + 5 con falla progresiva (uno por
  modo) para validar todo el pipeline. **No reemplaza datos reales**: sirve para
  desarrollo, pruebas y para que Claude Code itere sin acceso a la BD.

## Hacia pronóstico supervisado (precisión real) — v2 con `Eqpcare.Fault`

La evaluación actual (`src/evaluate.py`, refs 1 y 2) mide **auto-consistencia**: la
etiqueta deriva de los mismos metales que el modelo ya ve, así que no demuestra
capacidad de **predecir fallas**. La verdad de campo está en **`Eqpcare.Fault`**
(eventos reales con `DateFrom`/`SmrFrom` por equipo).

Reformulación: una muestra de aceite en t es **POSITIVA** si el mismo motor sufre
una falla dentro del horizonte (t, t+H] (días o, mejor, horas de operación `Smr`).
Esto habilita precision/recall/PR-AUC y **lead time** (anticipación) reales.

**Hallazgo de la validación (SSMS):** `Eqpcare.Fault` es un LOG DE TELEMETRÍA
(~1.9M eventos, alarmas de sensor), no fallas de mantenimiento → inservible como
etiqueta v1 (tasa base 57%). La verdad de campo útil es la **condición del
laboratorio** (`Condicion`/`Estado`), unificada en severidad ordinal
**0=Normal, 1=Monitoreo, 2=Precaución, 3=Crítico** = `coalesce(Condicion, Estado)`
(~11.9k muestras etiquetables, ~20% positivas en 120 días). `Fault` queda para v2.

### Pipeline supervisado (v1, implementado)

```bash
# 1) Validar la BD (en SSMS): docs/VALIDACION_SSMS.sql  (Fases 1-3)
# 2) Entrenar el clasificador de pronóstico (usa Azure SQL):
python -m src.train_supervised
# 3) Tablero de flota: P(condición adversa) por motor + modo dominante
python -m src.predict_supervised
```

- **Objetivo:** `P(severidad >= 2 en (t, t+H])` desde la ventana de metales.
- **Etiqueta** ([src/data/labels.py](src/data/labels.py)): severidad unificada con
  **censura por derecha** (un negativo solo cuenta si su desenlace es observable).
- **Features densos / etiqueta dispersa:** ventanas sobre metales (23.8k), se
  conservan solo las de desenlace observable ([features/windows.py](src/features/windows.py)).
- **Modelos:** `LSTMClassifier` (encoder LSTM + cabeza binaria, MC Dropout para
  incertidumbre) y baseline `HistGradientBoosting` sobre features de tendencia
  (explicable). Se queda el mejor por PR-AUC.
- **Split sin fuga:** la condición tiene **dos eras** (`Estado` 2019-2022, `Condicion`
  2025-2026; ver `VALIDACION_SSMS.sql` B16). El split temporal sobre toda la data
  mezcla eras y NO generaliza (ROC≈0.5); por eso se usa **split por equipo**
  (`train.split_mode: group`), o restringir a una era con `target.year_min`. Pérdida
  ponderada (`pos_weight`) por desbalance.
- **Dos modos** (`target.adverse_min_severity`): **2** = screen general (precaución+,
  ~48% base); **3** = alerta crítica accionable (~20% base). Artefactos con sufijo
  (`_sev2`/`_sev3`) → ambos coexisten.
- **Tablero** ([src/predict_supervised.py](src/predict_supervised.py)): bandas
  **Bajo/Medio/Alto** (doble umbral por precisión), filtro de **motores inactivos**
  (`predict.max_dias_inactivo`) y modo de falla dominante (firmas) por explicabilidad.

**Resultados (jun 2026, split por equipo):**

| Modelo | Objetivo | PR-AUC | ROC-AUC | recall@prec≥50% | lead time |
|--------|----------|--------|---------|-----------------|-----------|
| GBT    | sev≥2 (screen)   | **0.87** | 0.86 | — | 30 d |
| LSTM   | sev=3 (crítico)  | 0.49 | **0.85** | 0.73 | 43 d |

Top features (GBT): `Oxidacion`, `Sn`, `Cr`, `Fe`, `Nit`, `TBN`, `HorasComp`, `Hollin` —
coherentes con tribología (degradación de aceite, cojinetes, desgaste cilindro).

**Naturaleza del modelo crítico — RANKER, no alarma.** La precisión tiene un techo
estructural ~50% porque la condición crítica es **reversible** (transiciones 3→1
frecuentes; ver `VALIDACION_SSMS.sql` B11): un motor puede entrar en crítico y
recuperarse, así que "crítico en 120 d" tiene incertidumbre irreducible. El valor
está en el **ranking** (ROC 0.85): la banda **Alto** es una lista priorizada de
inspección con ~50% de aciertos (2.5× sobre base rate 20%) y 43 d de anticipación.
Las tasas de desgaste (`extra_vars`: `Tasa_*_100h`, `Indice_PQ`) se probaron y dieron
mejora marginal (no superan al `slope` de la ventana); están dispersas en la era antigua.

Config en `config/config.yaml → target:` y `→ train:`. `Fault` (v2):
`config → faults:` + [src/data/faults.py](src/data/faults.py).

> Diagnóstico de BD: `python -m src.data.diagnose` (incluye sección FALLAS).

## Multi-componente (v3) — un modelo por tipo de componente

El objetivo del proyecto es predecir fallas **por componente, según la naturaleza
de cada uno**. La BD (`Oil.LaboratoryData`) tiene muchos compartimientos además del
motor: transmisión, mandos finales, ruedas, diferenciales, hidráulico, motores de
tracción, etc. Cada uno tiene su **tribología** (qué metales importan), sus
**límites LC** (en `Eqpcare.lc`, columna `COMPONENTE`) y su **base rate** propia.

El pipeline es **parametrizable por componente** mediante un registro en
`config.yaml → components:`. Cada entrada declara su `compartment_filter` (en la BD),
su `lc_componente` (en `Eqpcare.lc`) y, según su física, `oil_vars` / `failure_modes`
/ `signature_matrix` propios. La base = motor; los demás solo declaran lo que cambia.
Los componentes de **engranaje** (transmisión, mando final, ruedas, tracción)
comparten config vía anclas YAML (`*gear_oil_vars`, `*gear_F`).

```bash
# Selección por variable de entorno COMPONENT (engine por defecto):
COMPONENT=transmision python -m src.train_supervised
COMPONENT=transmision python -m src.predict_supervised
# Overrides rápidos para barridos (sin editar el yaml):
COMPONENT=mando_final ADVERSE_MIN=2 HORIZON_DAYS=90 python -m src.train_supervised
```

Los artefactos llevan sufijo por componente y severidad (`clf_gbt_transmision_sev2.joblib`,
etc.) → **todos los modelos coexisten** sin pisarse.

**Resultados (jun 2026, split por equipo, sev≥2 salvo motor):**

| Componente | Objetivo | Base rate | PR-AUC | ROC-AUC | recall@prec≥50% |
|---|---|---|---|---|---|
| engine (motor)    | sev≥3 crítico | 0.20 | 0.53 | 0.86 | 0.85 |
| transmision       | sev≥2 | 0.42 | 0.59* | 0.73* | 0.73 |
| mando_final       | sev≥2 | 0.31 | 0.75 | 0.84 | 0.90 |
| rueda_delantera   | sev≥2 | 0.36 | 0.85 | 0.85 | 0.85 |
| hidraulico        | sev≥2 | 0.41 | 0.85 | 0.89 | 0.98 |
| motor_traccion    | sev≥2 | 0.69 | 0.97† | 0.93 | 1.00 |

\* Transmisión tiene pocas muestras (val ~32 positivos) → métrica de **alta varianza**
con split único; necesita k-fold por grupo (pendiente). † `motor_traccion` inflado por
base rate 69% (ver abajo).

### Endurecimiento (v3.1)

Tres correcciones que aplican a todos los componentes:
- **Guarda temporal de ventana** (`model.max_window_days`, 540 d): descarta ventanas
  cuyo span es absurdo por muestreo esporádico (no forman trayectoria coherente).
- **Escalado sin fuga**: el `StandardScaler` se ajusta SOLO en train tras el split
  (antes veía todo el dataset). Subió el LSTM del motor a ROC 0.86 / PR-AUC 0.53.
- **Calibración isotónica** (ajustada en val): la `prob_adversa` del tablero es una
  probabilidad interpretable y las bandas viven en ese espacio (antes el GBT pegaba
  todo a ~0.98). `meta_sup` guarda los calibradores (`iso_lstm`, `iso_gbt`).

**Lecciones clave:**
- **sev≥3 (crítico) solo aplica al motor.** En componentes de engranaje el estado
  crítico es rarísimo (transmisión ~3% base) → inaprendible. El objetivo accionable
  es **sev≥2 (precaución+)**: screen de desgaste con anticipación. Por eso cada
  componente no-motor fija `adverse_min_severity: 2` en su config.
- **Los componentes de engranaje son MÁS predecibles que el motor.** Su degradación
  es monótona (desgaste acumulativo), no reversible como la condición crítica del
  motor. PR-AUC 0.74–0.85 con features tribológicas correctas (Cu/Fe/Ni/PQ).
- **⚠️ `motor_traccion` (PR-AUC 0.97) está bajo sospecha:** base rate 69% infla la
  métrica (flota crónicamente degradada o sesgo en la condición de ese
  compartimiento). No tomar como logro hasta auditar la etiqueta.
- **Cobertura de límites LC variable:** MOTOR 55%, TRANSMISION 10%. Donde no hay LC
  se usa fallback (percentil 95). Ampliar `Eqpcare.lc` mejoraría la normalización.
