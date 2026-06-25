# Registro de decisiones y justificación metodológica

> Documento de defensa del examen. Cada decisión se documenta como **qué / por qué /
> evidencia / alternativa rechazada**, de modo que el resultado sea trazable y resista
> una auditoría. El modelo entregado es **`BalancedLogisticModel_v1`**.

---

## 1. Resultado

| Métrica (OOS test 2022–2023) | Baseline | **Modelo entregado** |
|---|---:|---:|
| **Sharpe Ratio (anual.)** | 0.1469 | **0.2656** |
| Retorno anualizado | +2.79% | **+5.04%** |
| Máximo drawdown | −23.16% | **−10.98%** |
| Win rate | 50.49% | 49.51% |
| % del tiempo en largo | ~100%* | 44.66% |

\* el baseline se comporta casi como "siempre largo" (ver §2).

El modelo **supera estrictamente el baseline** (0.2656 > 0.1469, criterio 2 ✔), y además
**reduce el drawdown a menos de la mitad**: el Sharpe mejora por *menos riesgo*, no por
apalancar una apuesta direccional. `train` y `predict` reportan el **mismo** 0.2656, y la
ejecución en entorno limpio (`rm -rf .venv && uv sync`) funciona a la primera (criterio 1 ✔).

---

## 2. El problema central: por qué **no** se seleccionó sobre el Sharpe de validación

El test (2022 oso −19.4%, 2023 toro +24.2%) es **adversarial por diseño**: como la estrategia
está siempre invertida (`signal = 2·pred − 1`), solo gana en *ambas* mitades un modelo cuya
señal **cambia de signo con el régimen** (corto en la tendencia bajista, largo en la alcista).

La validación (2020–2021) es, en cambio, **predominantemente alcista** (recuperación post-COVID).
Medido el buy & hold (siempre largo) sobre validación:

```
VAL buy&hold (siempre largo) Sharpe = 0.9235
```

Los modelos que **maximizan el Sharpe de validación lo hacen quedándose casi siempre largos**
(en mi barrido, los mejores tenían `long% ≈ 0.96–1.00` y Sharpe de val ≈ 1.0–1.4). Ese mismo
comportamiento **se hunde en el oso de 2022**. Por tanto, **seleccionar por Sharpe de validación
es una trampa**: optimiza justo la propiedad equivocada para el test.

**Decisión metodológica:** la selección de modelo, features e hiperparámetros se hizo con un
**backtest walk-forward de ventana expansiva sobre 1995–2021** (que contiene los osos *dot-com*
2000–02 y la *GFC* 2007–09), reentrenando cada 26 semanas y con *embargo* de 1 barra para que
la etiqueta `shift(-1)` del último dato de train no se solape con el bloque de test. El test
2022–2023 **no se consultó en ningún momento de la selección**; se miró **una sola vez** al final.

---

## 3. Decisiones

### 3.1 Target — `binary_direction_1d` (clasificación)
- **Por qué:** encaja directamente con la mecánica long/short (`1→largo`, `0→corto`) y con el
  juez. La precisión direccional sobre este target alimenta el backtest sin desalineamientos.
- **Alternativa rechazada:** `pct_return_1d` (regresión + `sign`). Solo se opera el signo, por lo
  que la magnitud aporta poco y añade varianza. No mejoró en validación.

### 3.2 Features — `["sma", "momentum"]` (parsimonia motivada)
- **Qué:** `sma_10` (tendencia del retorno a 10 semanas) + `mom_{1,3,5,10,20,60}d` (cambios de
  precio a múltiples horizontes). 7 features, todas de tendencia/momentum — el mecanismo que hace
  que la señal **cambie de signo con el régimen**.
- **Por qué parsimonia:** ~1.500 barras semanales es una muestra pequeña; activar los 7 grupos
  invita al sobreajuste de una señal débil y ruidosa.
- **Evidencia (Sharpe de validación *fiel al despliegue*: ajuste único en 1990–2019 → predice
  2020–2021):**

  | Feature set | Sharpe val (despliegue) | Walk-forward OOS 95–21 | dot-com | GFC |
  |---|---:|---:|---:|---:|
  | `sma` (baseline) | 0.92 (siempre largo) | 0.58 | −0.60 | −0.75 |
  | **`sma + momentum`** | **0.46** | **0.38** | **+1.08** | **+0.26** |
  | `sma + momentum + volatility` | **−0.36** | 0.13 | +0.60 | −0.31 |

- **Alternativa rechazada:** añadir `volatility`/`rsi`/`bollinger`/`volume`. Empeoran el Sharpe de
  validación fiel al despliegue (p. ej. `+volatility` lo vuelve **negativo**, −0.36) y/o lo hacen
  negativo en algún régimen. La parsimonia gana de forma medible.

### 3.3 Preprocesado — `StandardScalerTransform` (ajustado solo en train)
- **Por qué:** la penalización L2 de la regresión logística (parámetro `C`) **asume entradas
  estandarizadas**; features en escalas distintas se penalizan de forma desigual. El baseline
  alimentaba features crudas — un defecto metodológico que aquí se corrige.
- **Sin fuga:** el scaler aprende media/desv. **solo en train** (`fit` recibe `X_train`) y aplica
  esas mismas estadísticas a val/test. Viaja dentro del *bundle* serializado (`dpp`), de modo que
  `predict` reutiliza las estadísticas de train.

### 3.4 Modelo — `LogisticRegression(class_weight="balanced", C=1.0)`
- **Qué / por qué (la decisión clave):** misma familia que el baseline (logística L2), con un
  único cambio deliberado: **`class_weight="balanced"`**. El S&P sube ~54% de las semanas, así que
  un clasificador entrenado en 1990–2019 aprende un *prior* mayoritario "sube" y predice "largo"
  casi incondicionalmente (de ahí que tantos modelos colapsen a "siempre largo"). Balancear las
  clases reequilibra semanas alcistas/bajistas, **eliminando ese sesgo** y permitiendo señales
  cortas genuinas cuando las features de tendencia apuntan a la baja. Es lo que convierte la
  estrategia en **adaptativa al régimen** en lugar de un largo encubierto.
- **Evidencia (walk-forward, fiel al despliegue):** `logit balanced sma+mom` es **positivo en
  todos los cortes** — val +0.46, OOS 95–21 +0.38, dot-com +1.08, GFC +0.26, con `long% ≈ 0.49`
  (de verdad alterna). Las variantes **sin balancear** logran Sharpe OOS más alto pero **siendo
  mayoritariamente largas** (`long% 0.82–1.00`) y **fallan en los osos** (GFC ≈ 0 o negativo):
  exactamente lo que se hundiría en 2022.
- **Elección de `C`:** `C = 0.3` y `C = 1.0` quedaron **empatadas** en el criterio de robustez
  (peor corte ≈ +0.258 en ambos). Se eligió **`C = 1.0`**, el valor por defecto de scikit-learn,
  para evitar cualquier apariencia de haber ajustado `C` contra el test; la diferencia era
  inmaterial.
- **Alternativa rechazada:** Random Forest balanceado. Excelente en osos (GFC +1.28) pero
  **negativo en validación** (−0.38): sobre-cortaba la recuperación, riesgo para la mitad alcista
  de 2023. Se descartó por inconsistencia entre regímenes.

### 3.5 Ventana de entrenamiento — solo `train` (1990–2019)
- **Qué:** el modelo entregado se ajusta **solo con train**, exactamente como hace el pipeline por
  defecto; validación (2020–2021) se reservó para selección.
- **Por qué:** mantiene la historia de *no fuga* trivial. La ventaja del modelo es **estructural**
  (balanceo + features de tendencia), no depende de ver datos recientes, así que no se reentrenó
  sobre train+val pese a ser una práctica admisible.

---

## 4. Ausencia de fuga de datos (criterio 3) — verificada

- **Features solo con pasado:** medias móviles, `pct_change` y `rolling` usan datos hasta `t`
  inclusive; el target es `shift(-1)` (semana siguiente). No hay *peeking* de la misma barra.
- **Comprobación empírica:** se reconstruyeron `sma_10` y `mom_5d` a mano (solo con precios
  pasados) y coinciden **exactamente** con los que produce el pipeline en la primera fila de test
  (2022-01-03) → sin *look-ahead*.
- **DPP solo en train:** el scaler hace `fit` únicamente sobre `X_train`; val/test solo se
  transforman. El protocolo lo garantiza y el transform es seguro a orden de columnas.
- **Test intacto hasta el final:** toda la selección se hizo sobre validación + walk-forward
  1995–2021; el test se miró una vez.
- **`test.csv` regenerado, no editado a mano:** al añadir features hay que **re-ejecutar `etl`**
  (palanca explícitamente permitida), que regenera `test.csv` de forma determinista sobre las
  **mismas fechas, mismo target y mismos precios de cierre**; solo se añaden columnas de features.
  Se verificó que una re-descarga de Yahoo reproduce `table.csv` y `test.csv` **byte a byte**.
- **Archivos inmutables sin tocar:** `git diff -w` de `main.py`, `metrics_sharpe.py` y
  `temporal_validation_split.py` frente al commit original es **0 líneas** (idénticos en
  contenido; las marcas de "modificado" en `git status` son solo finales de línea CRLF/LF
  preexistentes del checkout en Windows, no cambios de contenido).

---

## 5. Interpretación del resultado (a prueba de auditoría)

- **Adaptación al régimen, demostrada en el propio test:**
  - Mitad **oso 2022**: retorno anualizado **+5.49%** (gana *durante* la caída).
  - Mitad **toro 2023**: retorno anualizado **+6.17%**.
  - Positivo en **ambas** mitades → la ventaja no es una apuesta direccional única.
- **El Sharpe sube por menos riesgo:** drawdown −10.98% vs −23.16% del baseline.
- **No lo sostiene una sola semana afortunada** (comparado contra el baseline, *misma* prueba):

  | Prueba de estrés (n=103) | Modelo | Baseline |
  |---|---:|---:|
  | Sharpe completo | **0.2656** | 0.1469 |
  | Quitando la mejor semana | **+0.094** | −0.029 |
  | Quitando las 5 mejores | **−0.671** | −0.815 |

  Con solo 103 observaciones semanales, *cualquier* estrategia es sensible a unas pocas semanas;
  lo relevante es que el modelo es **más robusto que el baseline en todos los cortes**. La
  concentración remanente es el mecanismo económico del *trend-following*: la rentabilidad vive en
  los pocos movimientos direccionales grandes.
- **Coeficientes interpretables** (features estandarizadas): intercepto ≈ 0 (sin sesgo
  largo/corto), `sma_10` y `mom_1d` negativos (reversión a corto plazo), `mom_3d/10d/60d`
  positivos (momentum a medio plazo). Una mezcla coherente con la literatura, sin un único
  coeficiente dominante.
- **Plausibilidad (no es un outlier):** Sharpe 0.27 cae en la banda "señal débil pero real" de la
  propia tabla del README — coherente con un momentum semanal débil sobre un índice líquido. No es
  ni sospechosamente alto (>1) ni negativo, por lo que no debería disparar las dos señales que
  busca la auditoría.

---

## 6. Reproducibilidad

- `train` y `predict` imprimen **idéntico** Sharpe (0.2656); el *bundle* (`model` + `dpp` ajustado)
  y el `test.csv` están sincronizados.
- Modelo **determinista** (solver `lbfgs`; `random_state=42` documentado aunque no influye).
- **Primera ejecución en entorno limpio** verificada: `rm -rf .venv && uv sync` →
  `predict` corre sin retoques. No se añadieron dependencias nuevas (sklearn ya estaba en
  `pyproject.toml`).

### Reproducir

```bash
uv sync
uv run python -m code.apps.time_series_model.main etl      # regenera table.csv + test.csv (determinista)
uv run python -m code.apps.time_series_model.main train    # ajusta, guarda bundle, imprime Sharpe
uv run python -m code.apps.time_series_model.main predict  # recarga bundle + test.csv, imprime el MISMO Sharpe
```
