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
python run.py -mode reset
```

Текущая реализация поддерживает первые шаги pipeline: проверку исходного датасета, эмуляцию потока через батчи, сохранение raw batch-файлов, состояния сборщика, метаданных батчей, processed batch-файлов, отчетов качества данных, обучение регрессионных моделей и сохранение лучшей модели.

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
- `artifacts/model_metrics_history.csv`
- `models/model_vXXXX_<model_name>.pkl`
- `models/best_model.pkl`
- `reports/eda_batch_XXXX.md`

Data quality/EDA считает generic summary только по analysis-признакам.
Из него исключаются служебные, временные, ID и target-колонки:
`INSR_BEGIN`, `INSR_END`, `OBJECT_ID`, `CLAIM_PAID`, `_source_file`, `CLAIM_PAID_WAS_MISSING`.

Фиксированная схема типов задается в секции `data_schema` файла `config/config.yaml`.
Она используется при подготовке данных и при расчете quality-отчетов, чтобы типы колонок не менялись от батча к батчу.

`duplicate_part` - доля полностью повторяющихся строк в батче: `duplicate_rows / rows_before`.
Параметр `max_duplicate_part` - это допустимый верхний порог этой доли; приставка `max_` означает, что batch проходит quality-check только если фактический `duplicate_part` не превышает заданный максимум.

## Training

После data quality этапа `update` обучает или дообучает регрессионную модель.
По умолчанию используется режим `training_mode: single` и `update_strategy: incremental`: выбранная модель `selected_model: sgd_regression` загружается из `models/current_model.pkl` и дообучается через `partial_fit` только на новом processed-батче.
Если `models/current_model.pkl` еще нет, первый `update` может собрать несколько стартовых батчей подряд. Это управляется параметрами `model.initial_training_batches` и `model.initial_training_max_rows`.
Для сравнения моделей можно переключить `training_mode: all` и `update_strategy: refit`; тогда модели из `model.candidate_models` будут переобучаться на накопленных processed-батчах.
Основные параметры моделей задаются в секции `model_parameters` файла `config/config.yaml`.
Для признаков используется sklearn preprocessing pipeline:

- числовые признаки из `data_schema.numeric_columns`: median imputation и scaling;
- категориальные признаки из `data_schema.categorical_columns`: most frequent imputation и one-hot encoding;
- дополнительные признаки из `INSR_BEGIN`: год, месяц, квартал;
- ID, service и исходные datetime-колонки не используются как прямые признаки.

Метрики регрессии: `rmse`, `mae`, `r2`, `smape`, `pearson_corr`, `pearson_p_value`. Основная метрика выбирается через `model.primary_metric`, сейчас это `rmse`.
В `artifacts/model_metrics_history.csv` колонка `model_name` стоит первой, чтобы проще сравнивать модели между собой.
В историю также записываются `training_mode` и `update_strategy`, чтобы было видно, был ли запуск настоящим incremental update или полным refit.
Для стартового обучения дополнительно пишутся `initial_training` и `training_rows_total`.

Команда `python run.py -mode reset` сбрасывает pipeline к начальному runtime-состоянию:
удаляет сгенерированные raw/processed batch-файлы, EDA-отчеты, prediction/model-файлы и history/state-файлы, включая историю метрик моделей.
Исходные CSV в `data/external/`, конфиги, код и `.gitkeep` не удаляются.
