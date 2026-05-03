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

На первом этапе реализована базовая структура проекта и маршрутизация CLI. Следующие этапы будут добавлять сбор батчей, quality checks, обучение, inference и summary.

## Project Layout

- `run.py` - единая CLI-точка входа.
- `config/config.yaml` - параметры pipeline.
- `app/` - код этапов MLOps pipeline.
- `data/` - исходные, потоковые и обработанные данные.
- `artifacts/`, `models/`, `logs/`, `reports/` - артефакты, пригодные для сохранения в GitHub Actions.

