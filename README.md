# Examen Final — Fundamentos de Machine Learning
### S&P 500 Market Direction Forecasting

> **Universidad de las Espérides · Máster en Finanzas Cuantitativas y Métodos Computacionales**

---

## Índice

1. [El problema](#1-el-problema)
2. [Cómo ejecutar el proyecto](#2-cómo-ejecutar-el-proyecto)
3. [Arquitectura del pipeline](#3-arquitectura-del-pipeline)
4. [Estructura de archivos](#4-estructura-de-archivos)
5. [Cómo se evalúa](#5-cómo-se-evalúa)
6. [El modelo baseline](#6-el-modelo-baseline)
7. [Cómo mejorar el modelo](#7-cómo-mejorar-el-modelo)
8. [Guía de contribución de componentes](#8-guía-de-contribución-de-componentes)
9. [Bases de la modelización por componentes](#9-bases-de-la-modelización-por-componentes)
10. [Qué está permitido y qué no](#10-qué-está-permitido-y-qué-no)

---

## 1. El problema

El objetivo del examen es construir un **modelo de predicción del S&P 500** que maximice el **Sharpe Ratio** de la estrategia de trading resultante.

El modelo recibe datos históricos OHLCV (Open, High, Low, Close, Volume) del índice S&P 500 y debe producir una señal de posición diaria: **long (+1)** o **short (−1)**. La calificación se determina por el Sharpe Ratio anualizado de esa estrategia sobre un **test set fijo (2022–2023)**.

```
Datos S&P 500 → Modelo → Predicción → Señal de trading → Sharpe Ratio → Nota
```

### Conjuntos temporales (fijos e intocables)

| Conjunto | Período | Días de trading |
|---|---|---|
| **Train** | 2010-01-01 → 2019-12-31 | ~2.496 días |
| **Val** | 2020-01-01 → 2021-12-31 | ~505 días (COVID crash + recovery) |
| **Test OOS** | 2022-01-03 → 2023-12-29 | ~501 días ← **tu nota** |

> ⚠️ Los cortes de fecha son inmutables. El test set incluye el mercado bajista de 2022 (−19.4%) y la recuperación alcista de 2023 (+24.2%) — dos regímenes opuestos que distinguen modelos robustos de modelos que solo capturan tendencia.

---

## 2. Cómo ejecutar el proyecto

### Instalación

```bash
# Requisitos: Python ≥ 3.11 y uv instalado
# https://docs.astral.sh/uv/getting-started/installation/

cd hesperides_final_exam
uv sync
```

### Comandos principales

```bash
# Smoke test completo (ETL + train + imprime Sharpe)
uv run python -m code.apps.time_series_model.main

# Solo descarga datos del S&P 500
uv run python -m code.apps.time_series_model.main etl

# Entrena el modelo y muestra el Sharpe Ratio
uv run python -m code.apps.time_series_model.main train

# Carga el modelo guardado y genera predicciones
uv run python -m code.apps.time_series_model.main predict
```

### Salida esperada de `train`

```
  Split: train=2496, val=505, test=501
  Model saved to models/sp500/LogisticBaselineModel_v1/pipeline.pkl
   sharpe_ratio  annualized_return  max_drawdown  win_rate
0         0.083             0.016        -0.254     0.488
```

> El Sharpe baseline es ~0.08. Tu objetivo es superarlo. Cuanto mayor, mejor nota.

---

## 3. Arquitectura del pipeline

El proyecto utiliza un **pipeline ML orientado a slots** con configuración YAML. Cada slot es una etapa del flujo de datos con un protocolo de interfaz fijo. El pipeline es inmutable; los módulos que lo rellenan son intercambiables.

### Flujo de datos

```
YahooFinanceETL          →  datos OHLCV descargados de Yahoo Finance
    ↓
DatasetLoader            →  DataFrame (Date, Open, High, Low, Close, Volume)
    ↓
TimeSeriesXYSplit        →  X = [SMA_window(pct_return_1d)]
                            y = binary_direction_1d (1=long, 0=short)
                            metadata["close_prices"]
    ↓
TemporalValidationSplit  →  (X_train, y_train, X_val, y_val, X_test, y_test)
                            metadata["close_prices_test"]
    ↓
DPP [EXTENSIBLE]         →  fit en train, transform en train/val/test
                            (Identity por defecto → sin preprocessing)
    ↓
Model [EXTENSIBLE]       →  fit(X_train, y_train) → predict(X_test) → y_pred
    ↓
TradingSignalPOS         →  señal = 2·y_pred − 1  (0→−1 short, 1→+1 long)
    [INMUTABLE]             strategy_returns = señal × retornos_reales_test
    ↓
MetricsSharpe            →  Sharpe anualizado sobre strategy_returns
    [INMUTABLE]          →  score_df [sharpe_ratio, annualized_return, max_drawdown, win_rate]
```

### Slots y su estado

| Slot | Clase | Estado | Descripción |
|---|---|---|---|
| `etl` | `YahooFinanceETL` | Fijo | Descarga OHLCV de Yahoo Finance |
| `dataset_loader` | `DatasetLoader` | Fijo | Lee CSV desde disco |
| `xy_split` | `TimeSeriesXYSplit` | Fijo (params ajustables) | Calcula target y feature SMA |
| `validation_split` | `TemporalValidationSplit` | Fijo | Split temporal por fechas fijas |
| `dpp` | `Identity` (baseline) | **Extensible** | Preprocessing / scaling |
| `model` | `LogisticBaselineModel` | **Extensible** | Núcleo del examen |
| `pos` | `TradingSignalPOS` | **INMUTABLE** | Convierte predicción → señal |
| `metrics` | `MetricsSharpe` | **INMUTABLE** | Calcula el Sharpe (tu nota) |
| `model_loader` | `BasicModelLoader` | Fijo | Serializa el pipeline a disco |

---

## 4. Estructura de archivos

```
hesperides_final_exam/
│
├── code/
│   ├── apps/
│   │   └── time_series_model/
│   │       ├── main.py                          ← INMUTABLE — punto de entrada
│   │       ├── utils.py                         ← ConfigLoader + instantiate
│   │       ├── configs/
│   │       │   └── time_series_model.yaml       ← TU ARCHIVO DE TRABAJO
│   │       └── orchestrators/
│   │           └── ml_pipeline.py               ← orquestador del pipeline
│   │
│   └── modules/
│       └── stages/
│           ├── data/
│           │   └── yahoo_finance_etl.py         ← fijo
│           ├── transforms/
│           │   ├── dataset_loader.py            ← fijo
│           │   ├── time_series_xy_split.py      ← fijo (params en YAML)
│           │   ├── temporal_validation_split.py ← fijo
│           │   ├── identity.py                  ← placeholder DPP
│           │   └── trading_signal_pos.py        ← INMUTABLE
│           ├── metrics/
│           │   └── metrics_sharpe.py            ← INMUTABLE
│           ├── loaders/
│           │   └── basic_model_loader.py        ← fijo
│           └── models/
│               └── logistic_baseline.py         ← baseline — reemplazable
│
├── data/
│   └── sp500/processed/sp500_dataset/table/    ← CSV generado por ETL
│
├── models/
│   └── sp500/                                  ← pipeline.pkl generado por train
│
└── pyproject.toml
```

---

## 5. Cómo se evalúa

La corrección es **automática y objetiva**. El profesor ejecuta:

```bash
uv run python -m code.apps.time_series_model.main train
```

El `sharpe_ratio` impreso sobre el **test set fijo 2022-01-03 → 2023-12-29** determina la calificación. Mayor Sharpe = mejor nota.

### Fórmula del Sharpe Ratio

```
Sharpe = mean(strategy_returns) / std(strategy_returns) × √252
```

Donde `strategy_returns[t] = señal[t] × retorno_real[t+1]`.

### Mecánica de evaluación paso a paso

1. El modelo predice `y_pred` sobre el test set
2. `TradingSignalPOS` convierte `y_pred` en posición: `+1` (long) si `y_pred == 1`, `−1` (short) si `y_pred == 0`
3. La posición se multiplica por los retornos reales diarios del S&P 500 → `strategy_returns`
4. `MetricsSharpe` calcula el Sharpe anualizado sobre `strategy_returns`

> 💡 **Clave**: el modelo no se evalúa por precisión de clasificación sino por rentabilidad ajustada al riesgo. Un modelo con 55% de accuracy puede tener un Sharpe mucho mayor que uno con 65% accuracy si acierta en los días de mayor movimiento.

### Referencia de calificación orientativa

| Sharpe Ratio | Calificación orientativa |
|---|---|
| < 0.08 | Por debajo del baseline |
| 0.08 – 0.30 | Aprobado |
| 0.30 – 0.60 | Notable |
| 0.60 – 1.00 | Sobresaliente |
| > 1.00 | Matrícula de Honor |

---

## 6. El modelo baseline

El baseline entregado es una **Regresión Logística** que clasifica la dirección del mercado usando como única feature la **media móvil simple (SMA-20) del retorno diario**.

```
Feature: SMA_20(pct_return_1d)    ← media de los últimos 20 retornos diarios
Target:  binary_direction_1d      ← 1 si Close[t+1] > Close[t], 0 en caso contrario
Modelo:  LogisticRegression(C=1.0)
Sharpe:  ~0.08
```

Este baseline es deliberadamente simple. Existe un margen real y significativo de mejora.

---

## 7. Cómo mejorar el modelo

Tienes **tres palancas de mejora** principales, no excluyentes:

### Palanca 1 — Cambiar el modelo (`model`)

Es el cambio con mayor impacto potencial. Reemplaza `LogisticBaselineModel` por cualquier modelo más expresivo en el YAML:

```yaml
model:
  class: "code.modules.stages.models.mi_modelo.MiModelo"
  params:
    n_estimators: 200
    max_depth: 5
```

Opciones recomendadas:

| Modelo | Tipo | Cuándo usar |
|---|---|---|
| `RandomForestClassifier` | Clasificación | Captura no linealidades sin sobreajuste |
| `GradientBoostingClassifier` / XGBoost | Clasificación | Alta capacidad, requiere regularización |
| `SVM` (kernel RBF) | Clasificación | Bueno con features normalizadas |
| `Ridge` / `Lasso` | Regresión | Si cambias a target continuo |
| `LightGBM` / CatBoost | Clasificación o Regresión | Rápido y robusto |
| LSTM / Transformer | Secuencial | Mayor expresividad, requiere más datos |

### Palanca 2 — Cambiar el preprocessing (`dpp`)

Añade normalización o transformaciones antes del modelo. El slot `dpp` acepta **una lista de transformers** que se encadenan:

```yaml
dpp:
  - class: "code.modules.stages.transforms.standard_scaler.StandardScalerTransform"
    params: {}
  - class: "code.modules.stages.transforms.mi_transform.MiTransform"
    params:
      param1: valor
```

> ⚠️ Importante: el DPP hace `fit` solo en train y `transform` en train/val/test. Nunca ajustes sobre datos de validación o test.

### Palanca 3 — Cambiar el target y/o features

**Cambio de target** en el YAML:
```yaml
xy_split:
  class: "code.modules.stages.transforms.time_series_xy_split.TimeSeriesXYSplit"
  params:
    target: "pct_return_1d"    # regresión de retorno continuo
    sma_window: 20
```

Si usas un target continuo (regresión), actualiza también el modo del POS:
```yaml
pos:
  - class: "code.modules.stages.transforms.trading_signal_pos.TradingSignalPOS"
    params:
      signal_mode: "regression"   # sign(y_pred) → señal
```

Targets disponibles:
- `binary_direction_1d` — clasificación binaria (baseline)
- `pct_return_1d` — retorno porcentual continuo (regresión)
- `log_return_1d` — log-return continuo (regresión)

**Ampliar el histórico de entrenamiento**: el split es por fecha, no por proporción, así que puedes ampliar el dataset sin afectar el test set:
```yaml
etl:
  - class: "code.modules.stages.data.yahoo_finance_etl.YahooFinanceETL"
    params:
      ticker: "^GSPC"
      start_date: "2000-01-01"    # más datos de entrenamiento
      end_date: "2023-12-31"
```

---

## 8. Guía de contribución de componentes

Para añadir un nuevo modelo, transformer o cualquier componente al pipeline, sigue estos pasos:

### Paso 1 — Crea el archivo en el lugar correcto

```
code/modules/stages/
├── models/          ← nuevos modelos aquí
├── transforms/      ← nuevos transformers DPP aquí
└── data/            ← nuevas fuentes de datos aquí
```

### Paso 2 — Implementa el protocolo correcto

Cada tipo de componente tiene un **protocolo de interfaz** que debe respetar:

#### Protocolo Model
```python
class MiModelo:
    def __init__(self, param1=..., param2=..., **kwargs):
        # Recibe parámetros desde el YAML via **kwargs
        ...

    def initialize(self):
        # Instancia el modelo interno (llamado por el pipeline antes de fit)
        ...

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            X_test=None, y_test=None, metadata=None, **kwargs):
        # Entrena el modelo. Puede usar X_val/y_val para early stopping.
        # Retorna: (None, metadata)
        ...

    def predict(self, X, y=None, metadata=None, **kwargs):
        # Genera predicciones sobre X.
        # Retorna: (X, y_pred_dataframe, metadata)
        # y_pred debe ser pd.DataFrame con columna "target"
        # Valores: 0/1 para clasificación, float para regresión
        ...
```

#### Protocolo Transformer (para DPP)
```python
class MiTransformer:
    def __init__(self, **kwargs):
        ...

    def fit(self, X, y=None, metadata=None, **kwargs):
        # Aprende los parámetros de la transformación solo sobre train
        # Retorna: (X, y, metadata)  ← sin modificar X todavía
        ...

    def transform(self, X, y=None, metadata=None, **kwargs):
        # Aplica la transformación aprendida
        # Retorna: (X_transformed, y, metadata)
        ...
```

### Paso 3 — Registra el componente en el YAML

No hay registro manual. Basta con indicar la ruta completa de importación en el YAML:

```yaml
model:
  class: "code.modules.stages.models.mi_modelo.MiModelo"   # módulo.Clase
  params:
    param1: valor1
    param2: valor2
```

El pipeline instancia la clase dinámicamente: `MiModelo(**params)`.

### Paso 4 — Verifica el formato de `y_pred`

El componente `TradingSignalPOS` (inmutable) espera que `y_pred` sea un `pd.DataFrame` con columna `"target"`:
- **Clasificación** (`signal_mode: "classification"`): valores `0` o `1`
- **Regresión** (`signal_mode: "regression"`): valores float (el signo determina la posición)

```python
# Ejemplo correcto de retorno en predict()
y_pred = pd.DataFrame({"target": self.model.predict(X.values)})
return X, y_pred, metadata
```

### Paso 5 — Ejecuta y compara

```bash
uv run python -m code.apps.time_series_model.main train
```

Anota el `sharpe_ratio` y compara con la iteración anterior.

---

## 9. Bases de la modelización por componentes

El pipeline sigue el **patrón Slots & Protocols**: el orquestador define una secuencia fija de slots; cada slot acepta cualquier objeto que respete su protocolo. Esto permite:

- **Intercambiabilidad**: cualquier modelo, transformer o métrica puede sustituirse sin tocar el orquestador
- **Reproducibilidad**: el flujo de datos es siempre el mismo; solo cambian los módulos
- **Comparabilidad**: todos los alumnos son evaluados con el mismo orquestador, mismo POS y mismo MetricsSharpe

### El contrato de datos entre stages

Cada stage recibe y devuelve la misma tupla: `(X, y, metadata)`.

- **`X`**: `pd.DataFrame` con las features
- **`y`**: `pd.DataFrame` con el target
- **`metadata`**: diccionario Python para pasar información auxiliar entre stages (e.g., precios Close del test set para calcular retornos reales)

El metadata es la herramienta de comunicación entre stages desacoplados. `TradingSignalPOS` necesita `metadata["close_prices_test"]` para calcular los retornos estratégicos; `MetricsSharpe` necesita `metadata["strategy_returns"]`.

### El rol del YAML

El YAML es la **única fuente de configuración**. Cambiar de modelo, de target, de preprocessing o de hiperparámetros no requiere tocar código Python — solo el YAML. El archivo relevante es:

```
code/apps/time_series_model/configs/time_series_model.yaml
```

### Separación inmutable / extensible

Esta separación es intencional y didáctica:

| Capa | Qué contiene | Razón |
|---|---|---|
| **Inmutable** | `main.py`, `TradingSignalPOS`, `MetricsSharpe` | Garantiza que todos se evalúan igual |
| **Extensible** | `dpp`, `model` | Aquí reside la libertad del alumno |
| **Parametrizable** | `xy_split` (target, sma_window), ETL (start_date) | Ajustable desde el YAML sin escribir código |

---

## 10. Qué está permitido y qué no

### ✅ Permitido

- Reemplazar `model` en el YAML por cualquier modelo con el protocolo correcto
- Añadir steps al slot `dpp` (normalización, feature engineering adicional)
- Cambiar el `target` y la `sma_window` en `xy_split`
- Cambiar el `signal_mode` en `TradingSignalPOS` (si cambias a target de regresión)
- Ampliar el histórico de entrenamiento (`start_date` en el ETL)
- Añadir nuevos archivos en `code/modules/stages/models/` o `code/modules/stages/transforms/`
- Añadir dependencias en `pyproject.toml`

### ❌ No permitido

- Modificar `main.py` (invalida el examen)
- Modificar `trading_signal_pos.py` o `metrics_sharpe.py` (el juez del examen)
- Modificar los cortes de fecha en `temporal_validation_split.py` o en el YAML (`val_start_date`, `test_start_date`)
- Entrenar con datos posteriores a 2023-12-31
- Guardar manualmente predicciones sobre el test set y cargarlas como modelo

---

## Entrega

El repositorio debe incluir:
1. El código modificado (nuevos módulos, YAML actualizado)
2. El archivo `models/sp500/<NombreModelo>_<version>/pipeline.pkl` generado por `train`

La corrección se realiza ejecutando `uv run python -m code.apps.time_series_model.main train` sobre el repositorio entregado. El `sharpe_ratio` impreso determina la calificación.

---

*Máster en Finanzas Cuantitativas y Métodos Computacionales — Universidad de las Espérides*

