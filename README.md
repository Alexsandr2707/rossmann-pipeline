# MLOps streaming ML MVP

MVP проекта для задания 1: построение ML-системы обработки потоковых табличных данных с заделом под CI/CD workflow задания 2.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI

```bash
python run.py -mode update
python run.py -mode inference -file ./path_to_file.csv
python run.py -mode summary
```

Текущая реализация поддерживает первые шаги pipeline: проверку исходного датасета, эмуляцию потока через батчи, сохранение raw batch-файлов, состояния сборщика, метаданных батчей, processed batch-файлов и отчетов качества данных.

## Project Layout

- `run.py` - единая CLI-точка входа.
- `config/config.yaml` - параметры pipeline.
- `app/` - код этапов MLOps pipeline.
- `data/` - исходные, потоковые и обработанные данные.
- `artifacts/`, `models/`, `logs/`, `reports/` - артефакты, пригодные для сохранения в GitHub Actions.

## Data

Ожидаемые исходные файлы Ethiopian Insurance:

- `data/external/motor_data11-14lats.csv`
- `data/external/motor_data14-2018.csv`

Временная колонка: `INSR_BEGIN`. Целевая переменная: вещественная `CLAIM_PAID`.

Пропуски в целевой переменной обрабатываются по политике из `config/config.yaml`.
По умолчанию используется `fill_zero`: пропущенные значения `CLAIM_PAID` заменяются на `0.0`, а рядом добавляется индикатор `CLAIM_PAID_WAS_MISSING`.

Команда `python run.py -mode update` берет следующий батч по времени и обновляет:

- `data/raw/batch_XXXX.csv`
- `data/processed/batch_XXXX_processed.csv`
- `artifacts/collector_state.json`
- `artifacts/batch_metadata_history.csv`
- `artifacts/data_quality_history.csv`
- `reports/eda_batch_XXXX.md`

Data quality/EDA считает generic summary только по analysis-признакам.
Из него исключаются служебные, временные, ID и target-колонки:
`INSR_BEGIN`, `INSR_END`, `OBJECT_ID`, `CLAIM_PAID`, `_source_file`, `CLAIM_PAID_WAS_MISSING`.

Фиксированная схема типов задается в секции `data_schema` файла `config/config.yaml`.
Она используется при подготовке данных и при расчете quality-отчетов, чтобы типы колонок не менялись от батча к батчу.

`duplicate_part` - доля полностью повторяющихся строк в батче: `duplicate_rows / rows_before`.
Параметр `max_duplicate_part` - это допустимый верхний порог этой доли; приставка `max_` означает, что batch проходит quality-check только если фактический `duplicate_part` не превышает заданный максимум.
