# Exam Playbook — S&P 500 Weekly Direction / Sharpe

Working guide for tackling the take-home. Read top to bottom once, then use it as a checklist.

---

## 0. The grading criteria — read these first

The practical part is judged by **correct execution** and the **value of the metric**. There are **3 strict pass/fail criteria — all three are mandatory**:

1. **Runs first-try from the provided `main.py`.** It must do inference on the test set and report the target metrics on the *first* attempt, in the environment installed from `pyproject.toml`. No manual patching, no missing dependencies, no "works on my machine."
2. **Beats the baseline.** The metric must be **strictly superior to the baseline (Sharpe ≈ 0.1469).** This is **required** — a Sharpe at or below baseline fails, no matter how well you explain it.
3. **No data leakage with the test set.** **Critical.** Any test information leaking into training fails the submission.

**On top of pass/fail:** submissions are **semi-randomly audited**, focusing on **extraordinarily high or negative** results, to find the root cause and to rule out irresponsible use of AI. Practical consequence:

> Aim for a Sharpe **comfortably above baseline but plausible** for weekly index direction — *not* a suspiciously huge number. An extreme result (very high *or* very negative) flags you for audit, and you must be able to **justify how it arose and prove it isn't leakage or a bug**. The **decision log (§9)** is what carries that justification. So: beat the baseline, keep it defensible, document every choice.

---

## 1. What the task actually is

- Predict the **weekly direction** of the S&P 500 (`^GSPC`, 1-week bars).
- Convert the prediction into an **always-invested long/short signal** (no neutral position).
- Maximize the **annualized Sharpe** (`mean/std × √52`) over a **fixed OOS test set: 2022-01-03 → 2023-12-29**.

### The adversarial test set (your central narrative)

| Half | Period | Market | Implication |
|---|---|---|---|
| Bear | 2022 | **−19.4%** | A permanently-long model bleeds here. |
| Bull | 2023 | **+24.2%** | A permanently-short model bleeds here. |

Because the strategy is **100% invested at all times** (`metrics_sharpe.py:29` → `signal = 2·pred − 1`), the only way to do well across *both* halves is a signal that **flips** with the regime. This is the thesis to build and defend:

> **A weekly momentum / trend-following signal is justified because it adapts to regime** — it goes long the 2023 uptrend and short the 2022 downtrend, instead of betting on a single direction. The README itself pre-justifies weekly-scale momentum (README §4, lines on weekly resolution).

---

## 2. Hard constraints — do NOT touch

| File / setting | Why |
|---|---|
| `code/apps/time_series_model/main.py` | Immutable entry point. |
| `code/modules/stages/metrics/metrics_sharpe.py` | The judge. Changing it invalidates the exam. |
| `data/.../test/test.csv` — **manual edits** | Hand-editing (pasting predictions/targets) = cheating. |
| Split dates `val_start_date: 2020-01-01`, `test_start_date: 2022-01-03` | Guarantee a homogeneous evaluation. |

### The subtle, important nuance about `test.csv`

`predict` **reads `test.csv` directly and does not rebuild features** (`ml_pipeline.py:138-143`). The file currently holds only `sma_10, target, Close`.

So if you change `feature_groups`, you **must re-run `etl`**, which **deterministically regenerates** `test.csv` with the new feature columns over the *same frozen dates and target* (`ml_pipeline.py:65-71`). That is the **legitimate, intended workflow** — adding features is an explicitly allowed lever (README §7). The "no modificar" rule forbids *hand-editing the file to cheat*, not regenerating it from a config change.

> **State this explicitly in your write-up** so a grader doesn't misread a legitimately regenerated `test.csv` as tampering. The dates, the target column, and the close prices are unchanged by regeneration — only feature columns are added.

---

## 3. Allowed levers (your whole toolbox)

1. **Swap the model** (`model:` block) — biggest impact.
2. **Add DPP preprocessing stages** (`dpp:` list) — scaling, feature transforms.
3. **Change `target`** — `binary_direction_1d` (default) / `pct_return_1d` / `log_return_1d`.
4. **Change `feature_groups` and `sma_window`** in `xy_split`.
5. **Set `signal_mode: "regression"`** in `metrics` *if and only if* you use a regression target.
6. **Extend ETL `start_date`** for more training history.
7. **Add new module files** under `models/` or `transforms/`.
8. **Add dependencies** to `pyproject.toml`.

---

## 4. Golden workflow (run this loop every time)

```bash
uv sync                                                        # once, after dep changes

# After ANY change to feature_groups / sma_window / target / dates:
uv run python -m code.apps.time_series_model.main etl          # regenerates table.csv + test.csv

# After ANY change to model / dpp / hyperparameters:
uv run python -m code.apps.time_series_model.main train        # fit, save bundle, print Sharpe

# Deliverable validity gate — MUST print the same Sharpe as train:
uv run python -m code.apps.time_series_model.main predict
```

**Rules of the loop:**
- Changed features/target/dates → re-run `etl` **before** `train` (otherwise `test.csv` is stale and `predict` will mismatch or error on missing columns).
- Changed only model/dpp/hyperparams → `train` then `predict` is enough.
- **`train` and `predict` must print identical Sharpe.** If they don't, the bundle and `test.csv` are out of sync — re-run `etl` then `train`.

---

## 5. Methodology discipline (where most of the points live)

### 5.1 Never touch the test set during development
2022–2023 is **final exam only**. Do all model selection and hyperparameter tuning on the **validation window 2020–2021** (`temporal_validation_split.py`). Look at the test Sharpe **once**, at the end. Document that you did this.

The val window is itself a regime stress test (COVID crash + recovery), which makes it a fair proxy for "does this generalize across regimes."

### 5.2 No leakage — verify and state it
- Features use only **past** data (rolling windows, lagged returns). ✔ confirm in `time_series_xy_split.py`.
- Target is `shift(-1)` (next week). ✔ no same-bar peeking.
- DPP stages **fit on train only**, transform val/test (`ml_pipeline.py:95-99`). ✔ the protocol enforces it — don't break it in custom transforms.
- Say all of this in the write-up. "We checked for look-ahead bias and found none because X" is a gradeable sentence.

### 5.3 Parsimony over kitchen-sink
~1,556 weekly training rows is **small**. Enabling all 7 feature groups invites overfitting on a weak, noisy signal. Prefer a **few well-motivated features** and *show* (on the val set) that adding more does not help OOS. The disciplined choice is itself the thing being graded.

### 5.4 Reproducibility
- Set `random_state` on any stochastic model (RF/GBM) so results are deterministic.
- Confirm `train` == `predict` Sharpe every iteration.

---

## 6. Concrete improvement ideas, ranked by effort/justifiability

### Tier 1 — Cheap, correct, easy to defend

**(a) Add a `StandardScaler` DPP stage.**
The baseline feeds **raw, unscaled** `sma_10` into `LogisticRegression`, whose `C` penalty assumes standardized inputs. This is a genuine methodological flaw in the baseline. Fitting a scaler on train only is textbook-correct and trivially defensible.

New file `code/modules/stages/transforms/standard_scaler.py`:
```python
"""StandardScalerTransform — z-scores features; fit on train only."""
import pandas as pd
from sklearn.preprocessing import StandardScaler


class StandardScalerTransform:
    def __init__(self, **kwargs):
        self.scaler = StandardScaler()
        self._cols = None

    def fit(self, X, y=None, metadata=None, **kwargs):
        self._cols = [c for c in X.columns if c != "Date"]
        self.scaler.fit(X[self._cols].values)
        return X, y, metadata or {}

    def transform(self, X, y=None, metadata=None, **kwargs):
        X = X.copy()
        X[self._cols] = self.scaler.transform(X[self._cols].values)
        return X, y, metadata or {}
```
> Note: in `predict`, `Date` has already been dropped by the temporal split, so guard on column presence. The `[c for c in X.columns if c != "Date"]` filter handles both paths. The fitted scaler travels inside the saved bundle (`dpp` list), so `predict` reuses train statistics — no leakage.

YAML:
```yaml
dpp:
  - class: "code.modules.stages.transforms.standard_scaler.StandardScalerTransform"
    params: {}
```

**(b) Pick features from the regime thesis, not by grabbing everything.**
Start with `["sma", "momentum"]`. Justify: momentum is the mechanism that flips the signal with the trend.
```yaml
xy_split:
  params:
    target: "binary_direction_1d"
    sma_window: 10
    feature_groups: ["sma", "momentum"]
```
Remember: **re-run `etl`** after this change.

### Tier 2 — Model bake-off (the core experiment)

Compare 2–3 models **on the val set** and record the numbers. Likely (and defensible) conclusion: regularized linear models beat trees here because trees overfit financial noise on a small sample.

Candidates:
- **Regularized logistic** — tune `C` ∈ {0.01, 0.1, 1, 10} on val.
- **Shallow RandomForest** — `max_depth` 2–4, `n_estimators` ~200, `random_state=42`.
- **Gradient boosting** (sklearn `HistGradientBoostingClassifier` or add `lightgbm`).

Model skeleton — `code/modules/stages/models/<name>.py`:
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    def __init__(self, n_estimators=200, max_depth=3, random_state=42, **kwargs):
        self.kw = dict(n_estimators=n_estimators, max_depth=max_depth,
                       random_state=random_state)
        self.model = None

    def initialize(self):
        self.model = RandomForestClassifier(**self.kw)

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            X_test=None, y_test=None, metadata=None, **kwargs):
        self.model.fit(X_train.values, y_train.values.ravel())
        return None, metadata or {}

    def predict(self, X, y=None, metadata=None, **kwargs):
        preds = self.model.predict(X.values)
        return X, pd.DataFrame({"target": preds}), metadata or {}
```
> The `predict` `y_pred` must be a `pd.DataFrame` with a single `"target"` column (0/1 for classification, float for regression). The metrics module flattens it.

**How to read the val set** during the bake-off: temporarily inspect val performance by adding a throwaway print in a *scratch script* (not in the immutable files) — or simply reason from `train`'s behavior and keep the test untouched until the end. Do **not** tune against the test Sharpe.

### Tier 3 — Optional, only if time and clearly argued

- **Regression target** (`pct_return_1d` + `signal_mode: "regression"`): lets magnitude express confidence, but only `sign(y_pred)` is traded, so upside is marginal. Mention you considered it and why you kept/dropped it.
- **More history** (`start_date` earlier than 1990): more data, but older regimes may be less relevant. Argue the tradeoff.
- **Feature ablation**: show val Sharpe with `["sma"]` vs `["sma","momentum"]` vs all groups → empirical parsimony argument.

---

## 7. Things that will silently break you — checklist

- [ ] Changed `feature_groups`/`sma_window`/`target` but **forgot to re-run `etl`** → `predict` uses stale `test.csv`. **Always `etl` first.**
- [ ] `train` and `predict` Sharpe differ → bundle/test out of sync, or a DPP stage behaves differently on the two paths (e.g. assumes `Date` exists). Re-run `etl`+`train`; make transforms column-presence-safe.
- [ ] Regression target but left `signal_mode: "classification"` → signal math is wrong. Flip it.
- [ ] Stochastic model without `random_state` → non-reproducible Sharpe.
- [ ] Custom DPP that fits on val/test → **leakage**. Fit only in `fit()`, which only sees train.
- [ ] Enabled many features → check you're not overfitting (val Sharpe ≫ test Sharpe is a red flag to *report honestly*, not hide).

---

## 8. Definition of done

The three hard criteria first (§0):

- [ ] **Criterion 1 — first-try execution.** In a *clean* environment (`uv sync` from `pyproject.toml`), `uv run ... main predict` runs without errors and prints the metrics on the first attempt. Test it as if from scratch: `rm -rf .venv && uv sync`, then run. Every dependency you used must be declared in `pyproject.toml`.
- [ ] **Criterion 2 — beats baseline.** Final OOS Sharpe is **strictly > 0.1469**. (Above baseline but plausible — not a suspicious outlier.)
- [ ] **Criterion 3 — no leakage.** No test information touches training. DPP fits on train only; the test set is never used for fitting/selection (§5.2).

Pipeline integrity:

1. `uv run ... main etl` → regenerates `table.csv` and `test.csv` without error.
2. `uv run ... main train` → fits, saves `models/sp500/<model>_<version>/pipeline.pkl`, prints Sharpe.
3. `uv run ... main predict` → loads bundle + `test.csv`, prints **the same Sharpe** (and > baseline).
4. Decision log (§9) complete — strong enough to survive an audit of an extreme result.
5. `git status` clean except intended changes: YAML, new module files, `pipeline.pkl`, regenerated `test.csv`, the decision log.

---

## 9. Decision log — the real deliverable (fill this in)

Keep this as a section in the README or a separate `DECISIONS.md`. For **each** decision, record: **what / why / evidence / alternative rejected**.

```
### Target
- Chose: binary_direction_1d (classification).
- Why: matches the long/short mechanics (1→long, 0→short); cleaner to justify than
  trading the sign of a noisy regression.
- Evidence: <val numbers if you compared>.
- Rejected: pct_return_1d — only the sign is traded, marginal upside, more variance.

### Features
- Chose: ["sma", "momentum"].
- Why: weekly momentum is the mechanism that flips the signal with the regime,
  enabling profit in both the 2022 bear and 2023 bull halves.
- Evidence: feature-ablation val Sharpe — [sma]=.., [sma,momentum]=.., [all]=.. .
- Rejected: all 7 groups — overfits a weak signal on ~1,556 rows (val≫test gap).

### Preprocessing
- Chose: StandardScaler (fit on train only).
- Why: LogisticRegression's C penalty assumes standardized inputs; the baseline
  fed raw sma_10 — a methodological flaw we fixed.

### Model + hyperparameters
- Chose: <model>, <params>, selected on the 2020–2021 val window.
- Why: <linear regularized beats trees because trees overfit financial noise on a
  small sample / etc.>
- Evidence: bake-off table — logistic(C=..)=.., RF(depth=..)=.., GBM=.. (val Sharpe).
- Rejected: <the others>, with the reason.

### Methodology
- Test set (2022–2023) untouched until final predict. Tuning done on val only.
- No look-ahead bias: features use past data, target is shift(-1), DPP fits on train.
- test.csv was regenerated via `etl` (allowed lever: new features), NOT hand-edited.
  Dates/target/close unchanged; only feature columns added.

### Result interpretation (audit-ready)
- Final OOS Sharpe = <x>, vs baseline 0.1469 → **beats baseline by <Δ>** (criterion 2 ✔).
- Where the edge comes from: <the regime-flipping momentum signal / the scaler fix / etc.>,
  shown to hold on the 2020–2021 val window before ever touching the test set.
- Leakage check (criterion 3 ✔): DPP fits on train only; test untouched until final predict.
- Why the number is plausible (not an outlier): consistent with weak-but-real weekly
  momentum on a liquid index; no single lucky period drives it (see per-half breakdown).
```

> Two failure modes the audit hunts for: (1) **below baseline** → fails criterion 2 regardless of write-up; (2) **suspiciously extreme** Sharpe → flagged for review, and if you can't trace it to a legitimate mechanism (i.e. it's leakage or a bug), it fails criterion 3. Target the middle: **clearly above 0.1469, plausible, and fully explained.**

---

## 10. Quick reference — file map

| Slot | File | Editable? |
|---|---|---|
| entry point | `apps/.../main.py` | 🔒 no |
| config | `apps/.../configs/time_series_model.yaml` | ✏️ **your main workspace** |
| orchestrator | `apps/.../orchestrators/ml_pipeline.py` | read-only in practice |
| ETL | `modules/stages/data/yahoo_finance_etl.py` | params via YAML |
| features/target | `modules/stages/transforms/time_series_xy_split.py` | params via YAML |
| temporal split | `modules/stages/transforms/temporal_validation_split.py` | 🔒 dates |
| DPP | `modules/stages/transforms/*.py` | ✏️ add stages |
| model | `modules/stages/models/*.py` | ✏️ add/swap |
| metrics | `modules/stages/metrics/metrics_sharpe.py` | 🔒 the judge |
| loader | `modules/stages/loaders/basic_model_loader.py` | fixed |
