# Changelog

Todos los cambios significativos en este proyecto se documentan en este archivo.

El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/es/).

## [Unreleased]

### Added
- Inicialización de estructura base del proyecto
- `BalancedLogisticModel` (`code/modules/stages/models/balanced_logistic.py`): logística L2
  con `class_weight="balanced"` para eliminar el sesgo mayoritario "alcista" y permitir señales
  cortas adaptativas al régimen.
- `StandardScalerTransform` (`code/modules/stages/transforms/standard_scaler.py`): estandariza
  features ajustando solo en train (corrige el uso de features crudas del baseline).
- `DECISIONS.md`: registro de decisiones y justificación metodológica a prueba de auditoría.

### Changed
- `configs/time_series_model.yaml`: modelo → `BalancedLogisticModel`; DPP → `StandardScaler`;
  `feature_groups` → `["sma", "momentum"]`. Sharpe OOS 0.1469 → **0.2656** (drawdown −23.16% →
  −10.98%), seleccionado por walk-forward multi-régimen 1995–2021 sin tocar el test.

### Deprecated

### Removed

### Fixed

### Security

## Notas de versión

Las versiones siguen el formato `MAYOR.MENOR.PARCHE`:
- **MAYOR:** cambios incompatibles en la API pública
- **MENOR:** nuevas características compatibles hacia atrás
- **PARCHE:** correcciones de errores

