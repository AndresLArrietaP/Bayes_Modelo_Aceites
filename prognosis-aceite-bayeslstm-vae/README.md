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

## Próximos pasos sugeridos
1. Mapear el esquema real de `bd_kmmp_osconfiabilidad` y completar `column_map`.
2. Definir qué columna marca la fecha/horas y cómo se identifican los componentes.
3. Calibrar umbrales con al menos un caso de falla documentado.
4. (Opcional) dashboard de flota y propagación temporal a horizonte 200–500 h.
