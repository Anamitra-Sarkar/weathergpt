# Verification

This file records checks actually run locally on 2026-09-01.

- Python: 3.10.12
- Test command: `pytest -q`
- Result: 20 passed
- Import/compile command: `python3 -m compileall -q app`
- Result: passed
- Live source check: Open-Meteo forecast request for Nagpur returned hourly data.
- Kaggle CLI: `arkosarkarhehe/weathergpt-official` reported `KernelWorkerStatus.COMPLETE` (2026-09-01). Published output files were not available through `kaggle kernels output`.

The Kaggle notebook source was inspected. Its current M2 printed precipitation RMSE uses the temperature validation target, and M1/M3 use random splits over generated examples. Consequently, no notebook model metric is reported here as scientifically verified.

Known limitations: CAP/IMD/GRIB2 depend on configuration or optional native dependencies; in-memory weather cache/evidence index do not survive a process restart; no live radar or satellite adapter is configured.
