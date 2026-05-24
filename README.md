# Rossmann Store Sales regression pipeline

MVP MLOps pipeline for tabular regression on the Kaggle Rossmann Store Sales
dataset.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Docker CI/CD

The Docker setup keeps source data, models and generated runtime outputs outside
the image. They are mounted from the local working tree into the container:

- `data/`
- `models/`
- `artifacts/`
- `reports/`
- `logs/`

Build the local image:

```bash
docker compose build
```

Run the CI test command inside Docker:

```bash
docker compose run --rm ci
```

Run pipeline commands through the Docker image:

```bash
docker compose run --rm pipeline -mode pretrain
docker compose run --rm pipeline -mode update
docker compose run --rm pipeline -mode evaluate
docker compose run --rm pipeline -mode inference -file data/external/test.csv
docker compose run --rm summary
```

This Compose workflow is the first CD stage: after tests pass, run the desired
pipeline mode in the same image and persist outputs through mounted directories.
A future GitHub Actions workflow can reuse the same `docker compose build` and
`docker compose run --rm ci` commands.

## Data

Expected files:

- `data/external/train.csv`
- `data/external/store.csv`
- `data/external/test.csv` for inference
- `data/external/sample_submission.csv`

The training source is `train.csv`. Before preprocessing, it is joined with
`store.csv` by `Store` using a many-to-one left join. Offline evaluation does
not read `test.csv`.

Target: `Sales`.

Time column: `Date`.

Leakage rule: `Customers` is never used as a feature.

## CLI

```bash
python run.py -mode update
python run.py -mode update 3
python run.py -mode pretrain
python run.py -mode evaluate
python run.py -mode inference -file data/external/test.csv
python run.py -mode summary
python run.py -mode summary -open
python run.py -mode reset
```

`inference` loads an external CSV such as `data/external/test.csv`, merges
`store.csv` by `Store` when store metadata is available, applies the same
feature-building logic as training, loads `models/best_model.pkl`, and writes
the original input columns plus `predict` to:

- `artifacts/predictions/inference_<timestamp>.csv`

`evaluate` is the main sanity-check command: it loads full `train.csv`, merges
`store.csv`, applies the same date-period split as `pretrain`, trains configured
regression models on the initial period and compares validation-period
predictions with simple baselines.

Evaluation outputs:

- `reports/offline_model_evaluation.md`
- `reports/figures/offline_evaluation/actual_vs_prediction_timeline.svg`
- `artifacts/offline_model_evaluation.csv`

`summary` writes both the Markdown summary and a browser-friendly dashboard at
`reports/index.html`. Use `python run.py -mode summary -open` to generate it
and open it with the default browser.
Summary outputs include recent performance records (`inference`/`update`) and
the active model hyperparameters from config. After stream updates are available,
the report also includes an update-period prediction timeline that aggregates
daily actual sales and model predictions from all update batches.

CLI commands print their primary result to stdout: `update` prints `True` or
`False` for each executed update and then prints the refreshed Markdown report
path, `inference` prints the prediction CSV path, and `summary` prints the
Markdown report path. Use `python run.py -mode update 3` to process up to three
stream batches in sequence; execution stops early if no new batch is available.
After update mode finishes, the Markdown report and HTML dashboard are
regenerated automatically.

## Training

Configured models live in `config/config.yaml`.

`pretrain` splits the unique source dates into initial, validation and stream
periods. It trains the selected model on the initial period, evaluates it on the
validation period, then refits the starting stream model on initial plus
validation data. After successful pretraining, `current_model.pkl` is saved and
the stream state is initialized so the next `update` starts from the first stream
date.

Active regression models:

- `decision_tree_regression`
- `knn_regression`
- `ridge_regression`
- `sgd_regression`

Offline evaluation trains all configured `candidate_models`, so it compares
the active model set and baselines even when runtime training uses
`training_mode: single`.

The pipeline uses sklearn preprocessing:

- numeric columns: median imputation and standard scaling;
- categorical columns: missing bucket and frequency encoding;
- date-derived columns from `Date`: year, month, quarter, day and ISO week.

Runtime `update` mode emulates streaming by writing the next
`model.stream_batch_days` unique dates from the stream period:

- `data/raw/batch_XXXX.csv`
- `data/processed/batch_XXXX_processed.csv`
- `artifacts/collector_state.json`
- `artifacts/batch_metadata_history.csv`
- `artifacts/data_quality_history.csv`
- `artifacts/performance_history.csv`
- `artifacts/model_metrics_history.csv`
- `reports/eda_batch_XXXX.md`
- `models/model_vXXXX_<model_name>.pkl`
- `models/best_model.pkl`
- `reports/model_diagnostics_latest.md`
- `reports/figures/model/prediction_timeline.svg`
- `models/current_model.pkl` after `pretrain` or update

Regression metrics include `rmse`, `mae`, `r2`, `smape`, `pearson_corr` and
`pearson_p_value`. The primary model-selection metric is `rmse`.

Data Quality metrics include missingness, duplicate rows, constant columns,
schema drift, IQR outlier share and categorical cardinality. Summary reports
read these metrics from `artifacts/data_quality_history.csv`.
Performance monitoring writes operation duration/status metadata to
`artifacts/performance_history.csv`, and summary reports display recent rows.

`reset` removes generated runtime artifacts and keeps source CSV files, config,
code and `.gitkeep` files.
