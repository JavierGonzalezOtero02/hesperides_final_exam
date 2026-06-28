# Changelog

Todos los cambios significativos en este proyecto se documentan en este archivo.

El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/es/).

## [Unreleased]

### Added
- Inicialización de estructura base del proyecto
- `RobustScalerTransform` (DPP): median/IQR feature scaling, fit on train only
  and applied to train/val/test. Robust to the fat-tailed outliers in the
  history (2008, March 2020) compared to a mean/std-based scaler.

### Changed
- `time_series_model.yaml`: enabled the `momentum` and `rsi` feature groups in
  addition to `sma` (from 1 to 8 input features), and replaced the `Identity`
  DPP step with `RobustScalerTransform` to homogenize feature scales
  (RSI 0–100 vs. returns ~0.01) before the logistic regression.

### Deprecated

### Removed

### Fixed

### Security

## Notas de versión

Las versiones siguen el formato `MAYOR.MENOR.PARCHE`:
- **MAYOR:** cambios incompatibles en la API pública
- **MENOR:** nuevas características compatibles hacia atrás
- **PARCHE:** correcciones de errores

