# Propuesta de cambios — coherencia examen teórico ↔ entrega práctica

> **Estado: BORRADOR para visto bueno.** Nada de esto está codeado todavía.
> Cada propuesta cita la(s) pregunta(s) del examen teórico
> (`examen_teorico_respuestas_revisado.tex`) que la justifica, para que la entrega
> práctica sea *coherente* con la metodología que el propio alumno defiende por escrito.

---

## 0. Por qué partir del examen teórico

La auditoría mira teoría y práctica **a la vez** y declara que su objetivo es
*"trazar la causa raíz de cualquier problema y descartar el uso irresponsable/acrítico
de la IA"*. La defensa más robusta ante eso no es un Sharpe alto, es la **coherencia**:
que el modelo entregado sea exactamente lo que la teoría predica para este problema
(muestra pequeña, señal débil, test adversarial por régimen).

Hay dos tipos de hueco:

- **Huecos de coherencia** — la teoría promete una metodología (walk-forward con embargo,
  ensamblajes de modelos complementarios, métricas estadísticas + económicas) que el
  código entregado **no demuestra**. Un auditor que lea el `.tex` y luego el repo lo nota.
- **Coherencias ya logradas** — decisiones del pipeline que ya encajan con la teoría y que
  conviene **explicitar** en `DECISIONS.md` citando la pregunta, en vez de dejarlas implícitas.

Cuidado con la tensión interna del propio examen: **Q14** elige LightGBM como apuesta de
por vida y **Q4/Q18** defienden ensamblajes; pero **Q2, Q15 y Q16** advierten que en
muestras pequeñas, ruidosas y de señal débil ganan los modelos simples y regularizados.
La resolución correcta para *este* dataset es la segunda rama — y eso hay que **argumentarlo
explícitamente**, no dejar que parezca una contradicción.

---

## 1. Coherencias ya logradas — solo documentar (coste casi nulo, sin tocar código)

Estas decisiones del pipeline ya están alineadas con la teoría. Propongo añadir a
`DECISIONS.md` una columna/nota "respaldo teórico" que las enganche a la pregunta concreta.

| Decisión del pipeline | Pregunta del examen | Conexión a explicitar |
|---|---|---|
| `class_weight="balanced"` (no SMOTE) | **Q19** | Q19 recomienda *ponderar la pérdida por clase* en vez de generar sintéticos. La entrega hace justo eso. Coherencia directa y fuerte. |
| 7 features de tendencia, parsimonia | **Q2, Q15, Q17** | Q2 (reducir complejidad / selección de variables como antídoto al sobreajuste), Q15·1 y Q15·4 (modelos simples con señal débil y pocas obs.), Q17 (medias móviles / pct-changes suavizan ruido con pocos datos). |
| Logística L2 interpretable + lectura de coeficientes | **Q1, Q15·5** | Q1 (interpretabilidad para clientes institucionales), Q15·5 (relación ~lineal conocida → logística preferible a red). La lectura de coeficientes de §5 de DECISIONS es exactamente lo que Q1 pide. |
| `StandardScaler` ajustado solo en train | **Q4, Q10·1** | Q4 (estandarización pre-modelo), Q10·1 (mismo pipeline de transformación en train e inferencia evita *training-serving skew* → el scaler viaja en el bundle). |
| Test intacto + features con solo pasado + target `shift(-1)` | **Q4, Q5, Q10·2, Q12** | Q4 y Q10·2 (auditar fuga antes de entrenar), Q5 (definición del target como decisión de diseño), Q12 (definir régimen solo con datos hasta `t`). |
| Señal momentum adaptativa al régimen | **Q12, Q16·4** | Q12 (regímenes alcista/bajista), Q16·4 (ventana temporal = gestión sesgo-varianza). |

**Acción propuesta:** edición de `DECISIONS.md` para citar estas preguntas. Sin riesgo,
sube mucho la nota de coherencia en una auditoría conjunta.

---

## 2. Huecos de coherencia — propuestas de cambio (ordenadas por valor/riesgo)

### P1 — Script de selección walk-forward con embargo (committed y reproducible) ⭐ recomendado

- **Base teórica:** Q4 ("validación walk-forward… separación temporal entre entrenamiento
  y validación para evitar que se solapen observaciones"), Q12 (evitar fuga de régimen),
  Q18 (pesos/selección con walk-forward).
- **Hueco actual:** `DECISIONS.md` §2 y el `CHANGELOG` afirman que la selección se hizo con
  *"backtest walk-forward de ventana expansiva 1995–2021, reentrenando cada 26 semanas, con
  embargo de 1 barra"* y citan números concretos (val 0.46, OOS 95–21 0.38, dot-com +1.08,
  GFC +0.26). **No existe ningún script committeado que reproduzca esos números.** Es el
  hueco más peligroso: toda la defensa metodológica se apoya en evidencia no reproducible.
- **Qué:** añadir `experiments/walk_forward_selection.py` (carpeta nueva, **fuera** de la ruta
  inmutable) que reconstruya el backtest expansivo con embargo, reentrene cada 26 semanas y
  emita la tabla de ablación de features y el bake-off de modelos exactamente como en
  DECISIONS. No toca el pipeline de entrega; es una herramienta de respaldo.
- **Riesgo:** ninguno para los 3 criterios (no toca modelo, features ni test). Único riesgo:
  si al reproducir, los números no coinciden con los citados, hay que **corregir DECISIONS**
  para que coincidan con la realidad (mejor descubrirlo nosotros que el auditor).
- **Validación:** que el script corra con `uv run` y reproduzca (±) las cifras de DECISIONS.
- **Recomendación:** **Hacerlo.** Convierte una afirmación en evidencia. Riesgo nulo, defensa máxima.

### P2 — Métricas estadísticas + económicas en el análisis (no toca el juez) ⭐ recomendado

- **Base teórica:** Q4 ("métricas estadísticas y económicas: correlación de Spearman señal↔retorno,
  calibración; Sharpe, **máximo drawdown y turnover con su coste de transacción**"), Q18 (Spearman
  + consistencia), Q11 (una métrica agregada no basta).
- **Hueco actual:** `metrics_sharpe.py` es inmutable y **ignora costes de transacción y turnover**.
  La teoría insiste en evaluar utilidad económica con costes. Un Sharpe 0.27 sin haircut por
  costes es atacable.
- **Qué:** en el mismo `experiments/` (NO en el juez), reportar para el modelo entregado:
  (a) Spearman(señal, retorno realizado), (b) **turnover** y Sharpe **neto de un coste razonable**
  (p. ej. 1–5 pb por cambio de posición), (c) desglose por mitad oso/toro (ya en DECISIONS §5),
  (d) hit rate. Documentar en DECISIONS que el edge **sobrevive** a un haircut de costes.
- **Riesgo:** ninguno (análisis aparte, el juez no cambia). Si el Sharpe neto cae por debajo del
  baseline, es información valiosa que hay que conocer y discutir, no ocultar.
- **Recomendación:** **Hacerlo.** Demuestra rigor económico y blinda la plausibilidad del resultado.

### P3 — Ensamblaje mínimo de modelos complementarios — condicional

- **Base teórica:** **Q18** y **Q4** (el alumno escribe *dos ensayos largos* defendiendo
  ensamblar familias complementarias —lineal + boosting + factores— por diversidad de errores,
  con promedio simple "difícil de batir fuera de muestra"). Q1 (árbol + SHAP como complemento).
- **Hueco de coherencia:** la teoría predica ensamblajes como núcleo de la metodología cuant,
  pero la entrega es **un único modelo**. Es la incoherencia teoría↔práctica más visible.
- **Qué:** nuevo módulo modelo que combine **logística balanceada + árbol somero regularizado**
  (RandomForest `max_depth` 2–3 balanceado, o `HistGradientBoostingClassifier` con
  regularización), promediando **probabilidades** y umbralizando. Familias complementarias =
  diversidad de errores (Q18).
- **Tensión a resolver explícitamente:** Q2/Q15 advierten contra complejidad en muestra
  pequeña y ruidosa. Por eso el ensamblaje debe ser **mínimo** (2 modelos, promedio simple,
  árbol muy somero) y **solo se adopta si mejora o iguala con menor varianza en el
  walk-forward de P1**. Si no mejora, se documenta como *"considerado y rechazado por nuestro
  propio principio de parsimonia (Q15·1, Q2)"* — lo cual **también es una victoria de coherencia**.
- **Riesgo:** medio. Puede no superar el 0.2656 actual, puede sobreajustar, y un árbol cambia
  el perfil de régimen (DECISIONS ya nota que el RF balanceado era negativo en validación).
  Hay que re-validar en walk-forward **antes** de tocar el YAML de entrega.
- **Recomendación:** **Probar en P1, decidir con datos.** No adoptar a ciegas. El resultado
  (adoptar *o* rechazar con argumento) es defendible en ambos casos.

### P4 — Umbral de decisión seleccionado en validación (no 0.5 por defecto) — opcional

- **Base teórica:** Q19·3 ("ajustar el umbral de clasificación… según el coste de negocio"),
  Q11 (rendimiento en el *umbral operativo*, no en media).
- **Qué:** en vez de `predict` con corte 0.5, elegir el umbral que maximiza el Sharpe en el
  walk-forward (P1) y congelarlo. Como la señal es `2·pred − 1`, el umbral controla el balance
  largo/corto directamente.
- **Riesgo:** medio. Es un grado de libertad extra → **aparenta** tuning. Solo admisible si se
  selecciona en val/walk-forward (nunca en test) y se documenta. Beneficio probablemente
  marginal sobre `class_weight="balanced"`, que ya reequilibra.
- **Recomendación:** **Opcional / baja prioridad.** Mencionarlo como "considerado"; implementar
  solo si P1 muestra una mejora clara y estable.

### P5 — Filtro de colinealidad/selección de variables como DPP — opcional

- **Base teórica:** Q4 ("eliminar variables de baja varianza o muy correlacionadas |ρ|>0.95…
  filtrado univariante"), Q2·5 (selección de variables contra el sobreajuste).
- **Hueco menor:** `mom_1/3/5/10/20/60d` son momentum solapados → **colineales**. La teoría
  predica depurar esa redundancia; la entrega mete las 6.
- **Qué:** o bien una DPP que descarte features con |ρ|>0.95 (ajustada en train), o bien
  reducir a mano el set de momentum a horizontes no redundantes (p. ej. `mom_1d, mom_10d, mom_60d`).
  *Requiere re-`etl`* si se cambia `feature_groups`.
- **Riesgo:** bajo, pero con 7 features el impacto es pequeño y puede no compensar el re-`etl`
  y la re-validación.
- **Recomendación:** **Opcional.** Más valor como *argumento documentado* (mostrar en val que
  quitar momentum redundante no daña) que como cambio en sí.

### P6 — Documentar el "considerado y rechazado": LightGBM / deep learning — recomendado (solo doc)

- **Base teórica:** Q14 (LightGBM como apuesta única de por vida), Q15·1 y Q15·4 (red/complejo
  inadecuado con `n` pequeño y señal débil ruidosa), Q2.
- **Hueco de coherencia:** Q14 nombra LightGBM como su algoritmo predilecto; la entrega usa
  logística. Sin explicación, parece contradicción.
- **Qué:** añadir a DECISIONS un párrafo "por qué *no* LightGBM/boosting/red aquí pese a Q14":
  ~1.500 barras semanales, señal débil, test adversarial por régimen → exactamente los casos en
  que Q15 dice que un boosting/red sobreajusta y un lineal regularizado generaliza mejor.
  Cierra la tensión Q14 ↔ entrega de forma explícita y honesta.
- **Riesgo:** nulo (solo documentación).
- **Recomendación:** **Hacerlo.** Cheap coherence: convierte una aparente contradicción en una
  decisión razonada.

---

## 3. Lo que NO se debe hacer

- No tocar los archivos inmutables (`main.py`, `metrics_sharpe.py`, fechas de split, edición
  manual de `test.csv`). Regenerar `test.csv` vía `etl` solo si P5 cambia `feature_groups`.
- No subir a un ensamblaje grande / boosting profundo / red: la propia teoría del alumno
  (Q2, Q15) lo desaconseja para este dataset. Añadiría riesgo de sobreajuste y de Sharpe
  "sospechosamente alto" que dispara la auditoría.
- No introducir umbrales/pesos elegidos contra el test. Todo se selecciona en
  validación/walk-forward.

---

## 4. Plan sugerido (si das el visto bueno)

1. **P1** (walk-forward committeado) + **P2** (Spearman/turnover/coste) → primero, porque
   convierten la defensa de DECISIONS en evidencia reproducible y no tocan la entrega.
2. Con el walk-forward en marcha, **P3** (ensamblaje mínimo) y, si procede, **P4** (umbral):
   adoptar **solo** si mejoran de forma estable; si no, documentar el rechazo.
3. **P6** + **§1** (documentación de coherencia y "considerado/rechazado") → cierre.
4. **P5** opcional al final.

> Dime cuáles apruebas (todos, un subconjunto, o ajustes) y los codeo en ese orden.
> Recomendación mínima de alto valor y riesgo nulo: **P1 + P2 + P6 + §1**.
