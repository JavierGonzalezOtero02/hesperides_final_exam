# Examen Final — Fundamentos de Machine Learning
### S&P 500 Market Direction Forecasting

> **Universidad de las Espérides · Máster en Finanzas Cuantitativas y Métodos Computacionales**

---

## Índice

1. [El problema](#1-el-problema)
2. [Instalación y ejecución](#2-instalación-y-ejecución)
3. [Estructura del proyecto](#3-estructura-del-proyecto)
4. [Conjuntos de datos y periodo](#4-conjuntos-de-datos-y-periodo)
5. [Cómo se evalúa: el Sharpe Ratio](#5-cómo-se-evalúa-el-sharpe-ratio)
6. [Arquitectura del pipeline](#6-arquitectura-del-pipeline)
7. [Qué está permitido y qué no](#7-qué-está-permitido-y-qué-no)
8. [Cómo contribuir con nueva metodología](#8-cómo-contribuir-con-nueva-metodología)

---

## 1. El problema

El objetivo del examen es construir un **modelo de predicción de la dirección semanal del S&P 500** que maximice el **Sharpe Ratio** de la estrategia de trading resultante sobre un test set fijo y fuera de muestra.

```
Datos S&P 500 (OHLCV semanal) → Modelo → Señal long/short → Backtest → Sharpe Ratio
```

La evaluación es **automática y objetiva**: se ejecuta `predict` sobre el test set fijo y se reporta el Sharpe Ratio por consola.

---

## 2. Instalación y ejecución

### ¿Qué es `uv`?

[`uv`](https://docs.astral.sh/uv/) es una herramienta moderna de gestión de entornos y dependencias para Python. Puedes pensar en ella como un reemplazo rápido y más cómodo de la combinación clásica `pip` + `venv`: en lugar de crear el entorno virtual manualmente, instalar dependencias una a una y activar el entorno, `uv` hace todo eso con un único comando.

Ventajas clave para este proyecto:
- **Un solo comando** (`uv sync`) instala todas las dependencias exactas del proyecto.
- **Sin conflictos**: crea un entorno aislado específico para este repositorio, sin afectar al resto de tu sistema.
- **Reproducibilidad**: el archivo `uv.lock` fija las versiones exactas de cada paquete, garantizando que el entorno es idéntico en cualquier máquina.

### Requisitos

- Python ≥ 3.11 (puedes comprobarlo con `python --version` o `python3 --version` en tu terminal)
- `uv` instalado (ver instrucciones a continuación)

### Instalar `uv`

Ejecuta **uno** de los siguientes comandos según tu sistema operativo:

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Una vez instalado, cierra y vuelve a abrir la terminal para que el comando `uv` quede disponible. Puedes verificarlo con:

```bash
uv --version
```

> Si ya tienes `pip` instalado, también puedes instalar `uv` con `pip install uv`, aunque el método anterior es el recomendado.

### Instalación del proyecto

```bash
cd hesperides_final_exam
uv sync
```

`uv sync` lee el archivo `uv.lock`, crea automáticamente un entorno virtual en la carpeta `.venv/` del proyecto e instala todas las dependencias en sus versiones exactas. No se necesita ningún paso adicional ni activar el entorno manualmente: los comandos `uv run ...` lo hacen de forma transparente.

### Comandos

```bash
# 1. Descargar datos y guardar el test set en disco
uv run python -m code.apps.time_series_model.main etl

# 2. Entrenar el modelo → reporta Sharpe Ratio en test
uv run python -m code.apps.time_series_model.main train

# 3. Inferencia sobre el test set guardado → reporta Sharpe Ratio
uv run python -m code.apps.time_series_model.main predict

# Pipeline completo en secuencia (etl → train → predict)
uv run python -m code.apps.time_series_model.main
```

### Salida esperada (baseline)

```
╔════════════════════════════════════════════════════╗
║          BACKTEST RESULTS — OOS TEST SET           ║
╠════════════════════════════════════════════════════╣
║  Sharpe Ratio (ann.)            +0.1469            ║
║  Annualised Return               +2.79%            ║
║  Max Drawdown                   -23.16%            ║
║  Win Rate                        50.49%            ║
║  Periods evaluated                  103            ║
╚════════════════════════════════════════════════════╝
```

---

## 3. Estructura del proyecto

```
hesperides_final_exam/
│
├── code/
│   ├── apps/
│   │   └── time_series_model/
│   │       ├── main.py                    ← 🔒 INMUTABLE — entry point
│   │       ├── utils.py                   ← instanciación dinámica desde YAML
│   │       ├── configs/
│   │       │   └── time_series_model.yaml ← ✏️  TU ARCHIVO DE TRABAJO
│   │       └── orchestrators/
│   │           └── ml_pipeline.py         ← orquestador del pipeline
│   │
│   └── modules/
│       └── stages/
│           ├── data/
│           │   └── yahoo_finance_etl.py         ← descarga OHLCV de Yahoo Finance
│           ├── transforms/
│           │   ├── dataset_loader.py            ← lee el CSV desde disco
│           │   ├── time_series_xy_split.py      ← construye features y target
│           │   ├── temporal_validation_split.py ← split temporal por fechas fijas
│           │   └── identity.py                  ← DPP placeholder (no-op)
│           ├── metrics/
│           │   └── metrics_sharpe.py            ← 🔒 INMUTABLE — backtest + Sharpe
│           ├── loaders/
│           │   └── basic_model_loader.py        ← serialización del modelo
│           └── models/
│               └── logistic_baseline.py         ← baseline — reemplazable
│
├── data/
│   └── sp500/processed/
│       ├── sp500_dataset/table/table.csv  ← serie histórica completa (ETL)
│       └── sp500_dataset/test/test.csv    ← 🔒 TEST SET FIJO — no modificar
│
├── models/
│   └── sp500/<NombreModelo>_<version>/
│       └── pipeline.pkl                   ← modelo entrenado (artefacto de entrega)
│
├── pyproject.toml                         ← dependencias
└── uv.lock                                ← versiones fijadas
```

---

## 4. Conjuntos de datos y periodo

Los datos son **barras semanales OHLCV del S&P 500** descargadas de Yahoo Finance (ticker `^GSPC`).

La resolución semanal es deliberada: la señal de momentum de medias móviles tiene evidencia empírica robusta a escala de semanas/meses, no de días. Permite además contar con suficientes observaciones históricas manteniendo un ratio señal/ruido razonable.

### Splits temporales (fijos e inmutables)

| Conjunto | Periodo | Barras semanales | Uso |
|---|---|---|---|
| **Train** | 1990-01-01 → 2019-12-31 | ~1.556 | Ajuste del modelo |
| **Val** | 2020-01-01 → 2021-12-31 | ~104 | Selección de hiperparámetros |
| **Test OOS** | 2022-01-03 → 2023-12-29 | ~104 | Evaluación final |

> ⚠️ Los cortes de fecha son **inmutables**. El test set incluye el mercado bajista de 2022 (−19.4%) y la recuperación alcista de 2023 (+24.2%) — dos regímenes opuestos que distinguen modelos robustos de modelos que solo capturan tendencia.

### El test set commiteado

El archivo `data/sp500/processed/sp500_dataset/test/test.csv` está versionado en el repositorio. Contiene las features, el target y los precios de cierre del periodo 2022-2023. El modo `predict` lo carga directamente para evaluar el modelo guardado sin necesidad de reconstruir el pipeline completo.

**Este archivo no debe ser modificado manualmente bajo ningún concepto.**

---

## 5. Cómo se evalúa: el Sharpe Ratio

### Qué es el Sharpe Ratio

El Sharpe Ratio mide la **rentabilidad ajustada al riesgo** de una estrategia. Responde a la pregunta: ¿cuánto retorno obtengo por cada unidad de volatilidad que asumo?

```
Sharpe = media(retornos_estrategia) / desv_típica(retornos_estrategia) × √52
```

- Un Sharpe de **0** significa que la estrategia no genera valor ajustado por riesgo.
- Un Sharpe de **1** equivale, intuitivamente, a ganar 1 unidad de retorno por cada unidad de riesgo. Es considerado excelente en gestión de activos real.
- El factor **√52** anualiza el ratio (52 semanas por año).

### Por qué Sharpe y no accuracy

### Referencia de valores

A modo orientativo, la industria de gestión cuantitativa de activos utiliza los siguientes rangos:

| Sharpe anualizado | Interpretación |
|---|---|
| < 0 | La estrategia destruye valor ajustado al riesgo |
| 0 – 0.3 | Señal débil; rentabilidad apenas justifica el riesgo asumido |
| 0.3 – 0.6 | Señal moderada; estrategia competitiva en mercados líquidos |
| 0.6 – 1.0 | Señal sólida; difícil de sostener fuera de muestra |
| > 1.0 | Excepcional; muy raro en estrategias sobre índices líquidos |

### Mecánica del backtest

```
1. El modelo predice y_pred sobre el test set
2. MetricsSharpe convierte y_pred en señal de posición:
      clasificación: 1 → +1 (long), 0 → -1 (short)
      regresión:     sign(y_pred) → +1 / -1
3. señal[t] × retorno_real[t→t+1] = retorno_estrategia[t]
4. Sharpe = media / desv_típica × √52
```

La estrategia es siempre **long/short**: el modelo está 100% invertido en todo momento, largo o corto, sin posición neutral. Esto hace que sea exigente — un modelo con win rate < 50% puede destruir valor activamente.

---

## 6. Arquitectura del pipeline

El pipeline sigue el patrón **Slots & Protocols**: el orquestador define una secuencia fija de etapas; cada etapa acepta cualquier objeto que respete su protocolo de interfaz.

### Flujo de datos

```
YahooFinanceETL          →  OHLCV semanal desde Yahoo Finance
    ↓
DatasetLoader            →  DataFrame (Date, Open, High, Low, Close, Volume)
    ↓
TimeSeriesXYSplit        →  X = [SMA_10(retorno_semanal)]
                            y = binary_direction (1=sube, 0=baja)
                            metadata["close_prices"]
    ↓
TemporalValidationSplit  →  (X_train, y_train, X_val, y_val, X_test, y_test)
                            metadata["close_prices_test"]
    ↓
DPP [EXTENSIBLE]         →  fit en train, transform en train/val/test
    ↓
Model [EXTENSIBLE]       →  fit(X_train, y_train) → predict(X_test) → y_pred
    ↓
MetricsSharpe [INMUTABLE]→  y_pred → señal long/short → backtest → Sharpe
```

### Estado de cada slot

| Slot | Clase baseline | Estado |
|---|---|---|
| `etl` | `YahooFinanceETL` | Fijo (parámetros ajustables) |
| `dataset_loader` | `DatasetLoader` | Fijo |
| `xy_split` | `TimeSeriesXYSplit` | Fijo (parámetros ajustables) |
| `validation_split` | `TemporalValidationSplit` | **Fechas inmutables** |
| `dpp` | `Identity` | **Extensible** |
| `model` | `LogisticBaselineModel` | **Extensible** |
| `metrics` | `MetricsSharpe` | **INMUTABLE** |
| `model_loader` | `BasicModelLoader` | Fijo |

El contrato entre etapas es siempre la tupla `(X, y, metadata)` donde `metadata` es un diccionario Python para pasar información auxiliar (precios de cierre, strategy returns, etc.).

El único archivo de configuración relevante es `code/apps/time_series_model/configs/time_series_model.yaml`. Cambiar de modelo, features o hiperparámetros no requiere tocar código Python.

---

## 7. Qué está permitido y qué no

### ✅ Permitido

- Reemplazar `model` en el YAML por cualquier modelo que respete el protocolo
- Añadir steps al slot `dpp` (normalización, feature engineering)
- Cambiar `target` y `sma_window` en `xy_split`
- Cambiar `signal_mode` en `metrics` si usas target de regresión (`"regression"`)
- Ampliar el histórico de entrenamiento (`start_date` en el ETL)
- Añadir nuevos archivos en `code/modules/stages/models/` o `code/modules/stages/transforms/`
- Añadir dependencias en `pyproject.toml`

### ❌ No permitido

| Archivo | Razón |
|---|---|
| `code/apps/time_series_model/main.py` | Entry point del examen — inmutable |
| `code/modules/stages/metrics/metrics_sharpe.py` | El juez — modificarlo invalida la evaluación |
| `data/sp500/processed/sp500_dataset/test/test.csv` | Test set fijo — modificarlo es trampa |
| Fechas en `temporal_validation_split.py` o en el YAML (`val_start_date`, `test_start_date`) | Garantizan evaluación homogénea |
| Guardar predicciones manuales sobre el test set y cargarlas como modelo | Invalida el examen |

---

## 8. Cómo contribuir con nueva metodología

### Palanca 1 — Cambiar el modelo

Es el cambio con mayor impacto. Crea un nuevo archivo en `code/modules/stages/models/` e implementa el protocolo:

```python
class MiModelo:
    def __init__(self, param1=..., **kwargs):
        ...

    def initialize(self):
        # Instancia el modelo interno (llamado por el pipeline antes de fit)
        ...

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            X_test=None, y_test=None, metadata=None, **kwargs):
        # Entrena. Puede usar X_val/y_val para early stopping.
        # Retorna: (None, metadata)
        ...

    def predict(self, X, y=None, metadata=None, **kwargs):
        # Retorna: (X, y_pred_dataframe, metadata)
        # y_pred debe ser pd.DataFrame con columna "target"
        # Clasificación: valores 0/1 | Regresión: float
        ...
```

Luego regístralo en el YAML:

```yaml
model:
  class: "code.modules.stages.models.mi_modelo.MiModelo"
  params:
    n_estimators: 200
    max_depth: 5
```

### Palanca 2 — Añadir preprocessing (DPP)

Crea un transformer en `code/modules/stages/transforms/` e impleméntalo con el protocolo:

```python
class MiTransformer:
    def __init__(self, **kwargs): ...

    def fit(self, X, y=None, metadata=None, **kwargs):
        # Aprende parámetros solo sobre train
        return X, y, metadata

    def transform(self, X, y=None, metadata=None, **kwargs):
        # Aplica la transformación aprendida
        return X_transformed, y, metadata
```

Encadena varios transformers en el YAML:

```yaml
dpp:
  - class: "code.modules.stages.transforms.standard_scaler.StandardScalerTransform"
    params: {}
  - class: "code.modules.stages.transforms.mi_transform.MiTransform"
    params: {}
```

### Palanca 3 — Cambiar el target o las features

**Target continuo** (regresión en lugar de clasificación):

```yaml
xy_split:
  params:
    target: "pct_return_1d"   # retorno continuo en lugar de dirección binaria
    sma_window: 10
    feature_groups: ["sma", "momentum", "rsi", "volatility", "bollinger", "volume"]

metrics:
  params:
    trading_days_per_year: 52
    signal_mode: "regression"  # sign(y_pred) determina la posición
```

**Targets disponibles:**

| Target | Tipo | Descripción |
|---|---|---|
| `binary_direction_1d` | Clasificación | 1 si la semana siguiente sube, 0 si baja (baseline) |
| `pct_return_1d` | Regresión | Retorno porcentual de la semana siguiente |
| `log_return_1d` | Regresión | Log-return de la semana siguiente |

**Feature groups disponibles** (parámetro `feature_groups` en `xy_split`):

| Grupo | Features generadas | Descripción |
|---|---|---|
| `"sma"` | `sma_N` | Media móvil del retorno semanal (baseline) |
| `"momentum"` | `mom_1d/3d/5d/10d/20d/60d` | Retornos a múltiples horizontes |
| `"rsi"` | `rsi_14` | RSI clásico (señal de mean-reversion) |
| `"volatility"` | `vol_5d`, `vol_20d`, `atr` | Volatilidad realizada + rango intradiario |
| `"bollinger"` | `bb_pct`, `bb_width` | Posición y amplitud de las Bandas de Bollinger |
| `"volume"` | `vol_ratio`, `vol_change` | Volumen relativo a su media |
| `"calendar"` | `dow` | Día de la semana (efecto lunes documentado) |

Ejemplo para un modelo de árboles que puede aprovechar múltiples features:

```yaml
xy_split:
  params:
    target: "binary_direction_1d"
    sma_window: 10
    feature_groups: ["sma", "momentum", "rsi", "volatility", "bollinger", "volume"]
```

### Flujo de trabajo recomendado

```bash
# 1. Modifica el YAML o añade nuevos módulos
# 2. Entrena y observa el Sharpe
uv run python -m code.apps.time_series_model.main train

# 3. Verifica que predict reproduce el mismo resultado
uv run python -m code.apps.time_series_model.main predict
```

---

## Entrega

El repositorio entregado debe incluir:

1. El código nuevo (`code/modules/stages/models/mi_modelo.py`, etc.)
2. El YAML actualizado (`configs/time_series_model.yaml`)
3. El modelo serializado (`models/sp500/<NombreModelo>_<version>/pipeline.pkl`)
4. El test set commiteado (`data/sp500/processed/sp500_dataset/test/test.csv`)

### Criterio de evaluación

La evaluación consiste en ejecutar el modo `predict` sobre el test set fijo y observar el Sharpe Ratio reportado por consola:

```bash
uv run python -m code.apps.time_series_model.main predict
```

Para que la entrega sea válida, este comando debe ejecutarse sin errores, cargar el modelo guardado, correr la inferencia sobre `test.csv` e imprimir el Sharpe Ratio. El valor obtenido es el resultado del examen.

---

*Máster en Finanzas Cuantitativas y Métodos Computacionales — Universidad de las Espérides*
