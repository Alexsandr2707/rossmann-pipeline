# Rossmann Store Sales regression pipeline

MVP MLOps-пайплайн для табличной регрессии на датасете Kaggle Rossmann Store
Sales.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Docker CI/CD

В текущей версии проект использует локальный Docker-only CI/CD-процесс.
`docker compose` описывает воспроизводимые сервисы для тестов, обучения,
потоковых обновлений, offline evaluation, inference, генерации summary и
короткой локальной CD-последовательности. GitHub Actions намеренно не добавлен
в этот вариант, поэтому требования задания, которые явно требуют GitHub
Actions, здесь не закрываются и могут быть перенесены в отдельный workflow
позже.

Docker-образ содержит код, конфигурацию и Python-зависимости. Исходные данные,
обученные модели и runtime-результаты остаются вне образа: они зависят от
окружения, могут быть большими и должны сохраняться между запусками
контейнеров. Compose подключает их из локальной рабочей директории:

- `data/`
- `models/`
- `artifacts/`
- `reports/`
- `logs/`

Собрать локальный образ, когда скачивание Docker-образов разрешено:

```bash
docker compose build
```

Запустить Docker CI-сервис с unit-тестами:

```bash
docker compose run --rm ci
```

Запустить этапы пайплайна через отдельные Compose-сервисы:

```bash
docker compose run --rm pretrain
docker compose run --rm train
docker compose run --rm update
docker compose run --rm evaluate
docker compose run --rm inference
docker compose run --rm summary
```

`pipeline` остается универсальной точкой входа для произвольных CLI-аргументов:

```bash
docker compose run --rm pipeline -mode inference -file data/external/test.csv
docker compose run --rm pipeline -mode update
docker compose run --rm pipeline -mode update 3
```

Несколько потоковых update-шагов можно выполнить через
`pipeline -mode update <count>` или повторным запуском сервиса `update`.

Локальный CD-сервис последовательно запускает тесты, предварительное обучение и
генерацию summary в одном Compose-окружении:

```bash
docker compose run --rm local-cd
```

Docker-only процесс подтверждается следующими runtime-артефактами:

- `logs/pipeline.log`
- `models/*.pkl`
- `models/archive/*.pkl`
- `artifacts/collector_state.json`
- `artifacts/*history*.csv`
- `reports/index.html`
- `reports/summary_latest.md`

## Данные

Используется Kaggle dataset Rossmann Store Sales: исторические ежедневные
продажи 1 115 магазинов Rossmann. Задача проекта - прогнозировать колонку
`Sales` для новых строк, учитывая календарные признаки, открытие/закрытие
магазина, промо-акции, праздники, тип магазина, ассортимент и расстояние до
ближайшего конкурента. В данных также встречаются временные закрытия магазинов,
например из-за ремонта.

Ожидаемые исходные файлы:

- `data/external/train.csv`
- `data/external/store.csv`
- `data/external/test.csv` для inference

Опциональный справочный файл Kaggle:

- `data/external/sample_submission.csv`

Источник обучения - `train.csv`. Перед preprocessing он объединяется с
`store.csv` по колонке `Store` через many-to-one left join. Offline evaluation
не читает `test.csv`.

Target: `Sales`.

Временная колонка: `Date`.

Правило против leakage: `Customers` никогда не используется как feature.

Кратко по файлам:

- `train.csv` - исторические данные с `Sales`;
- `test.csv` - данные без `Sales`, используемые для inference;
- `store.csv` - дополнительная информация о магазинах;
- `sample_submission.csv` - пример Kaggle submission в нужном формате.

Ключевые поля:

- `Store` - идентификатор магазина;
- `Sales` - дневная выручка, которую прогнозирует модель;
- `Customers` - число покупателей за день, не используется как feature из-за
  leakage;
- `Open`, `Promo`, `StateHoliday`, `SchoolHoliday` - признаки режима работы,
  промо и календарных событий;
- `StoreType`, `Assortment`, `CompetitionDistance` - характеристики магазина и
  конкурентного окружения;
- `Promo2`, `Promo2SinceYear`, `Promo2SinceWeek`, `PromoInterval` - признаки
  длительной повторяющейся промо-акции.

## Архитектура проекта

Пакет `app/` разделен по ответственности этапов пайплайна. Новая структура
держит модули рядом с тем этапом, где они используются:

```text
app/
  core/           # CLI-оркестрация, чтение config и настройка logging
  data/           # загрузка данных, сбор stream batch-ей, validation и features
  models/         # model interfaces, sklearn-реализации и preprocessing blocks
  training/       # training/update workflow, diagnostics и interpretation
  serving/        # генерация inference-прогнозов
  evaluation/     # offline evaluation workflow
  monitoring/     # runtime-метрики операций
  reporting/      # Markdown/HTML-отчеты и prediction history charts
  visualization/  # переиспользуемые SVG chart writers
```

Логика группировки:

- `app/core/` содержит связующий код проекта: `config.py`, `logging_utils.py`
  и `pipeline.py`. Эти модули соединяют CLI с остальной системой, но не
  содержат ML-логику.
- `app/data/` содержит подготовку табличных данных до modeling: загрузку CSV,
  join с store metadata, разбиение периодов по датам, сбор stream batch-ей,
  проверку качества данных, preprocessing raw columns и построение признаков.
- `app/models/` содержит переиспользуемые блоки для построения моделей:
  wrappers, factories и sklearn-compatible preprocessing transformers.
- `app/training/` отвечает за жизненный цикл модели: initial pretraining,
  stream updates, diagnostics и артефакты интерпретации модели.
- `app/serving/` отвечает за prediction serving. Он загружает обученную модель,
  применяет тот же путь построения признаков, что и training, и пишет
  inference outputs.
- `app/evaluation/` отделен от runtime training, потому что offline comparison
  является аналитическим workflow, а не частью цикла stream update.
- `app/monitoring/` записывает operation-level performance metadata, которые
  затем используются в отчетах.
- `app/reporting/` и `app/visualization/` отвечают за представление
  результатов: reporting собирает outputs, visualization содержит
  переиспользуемый рендеринг графиков.

Внешняя точка входа остается небольшой. `run.py` разбирает CLI-аргументы,
загружает конфигурацию из `app.core.config`, настраивает logging и передает
выполнение в `app.core.pipeline.Pipeline`.

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

`inference` читает внешний CSV, например `data/external/test.csv`, объединяет
его со `store.csv` по `Store`, если доступны store metadata, применяет тот же
путь построения признаков, что и training, загружает `models/best_model.pkl` и
записывает исходные входные колонки плюс `predict` в:

- `artifacts/predictions/inference_<timestamp>.csv`

`evaluate` - основной sanity-check режим. Он читает полный `train.csv`,
объединяет его со `store.csv`, применяет то же разбиение периодов по датам, что
и `pretrain`, обучает настроенные regression models на initial period и
сравнивает predictions на validation period с простыми baseline-моделями.

Результаты evaluation:

- `reports/offline_model_evaluation.md`
- `reports/figures/offline_evaluation/actual_vs_prediction_timeline.svg`
- `artifacts/offline_model_evaluation.csv`

`summary` пишет Markdown summary и dashboard для браузера в
`reports/index.html`. Команда `python run.py -mode summary -open` генерирует
summary и открывает dashboard в браузере по умолчанию. Summary включает
последние performance records (`inference`/`update`), history качества данных,
history метрик модели, общий prediction timeline по всем update-запускам и
активные hyperparameters модели из config.

CLI-команды печатают основной результат в stdout:

- `update` печатает `True` или `False` для каждого выполненного update, затем
  печатает путь к обновленному Markdown report;
- `inference` печатает путь к CSV с прогнозами;
- `summary` печатает путь к Markdown report.

Команда `python run.py -mode update 3` обрабатывает до трех stream batch-ей
подряд. Выполнение останавливается раньше, если новых batch-ей нет. После
завершения каждого действия pipeline Markdown summary report и HTML dashboard
регенерируются автоматически. Актуальная summary лежит в
`reports/summary_latest.md`, а каждая генерация дополнительно сохраняется в
`reports/archive/summary/`.

## Тесты

Запустить focused unit-тесты локально:

```bash
python -m unittest discover -s tests -v
```

Проверить синтаксис модулей проекта:

```bash
python -m compileall -q run.py app
```

## Обучение

Настройки моделей находятся в `config/config.yaml`.

`pretrain` делит уникальные даты source dataset на initial, validation и stream
periods. Он обучает выбранную модель на initial period, оценивает ее на
validation period, затем заново обучает стартовую stream-модель на initial +
validation данных. После успешного предварительного обучения сохраняется
`current_model.pkl`, а stream state инициализируется так, чтобы следующий
`update` начался с первой stream-даты.

`pretrain` рассчитан на запуск после `reset`. Если в `models/` уже есть `.pkl`
файл модели, команда завершится ошибкой и предложит выполнить `reset`, чтобы
случайно не перезаписать существующий model lifecycle.

Активные regression-модели:

- `decision_tree_regression`
- `knn_regression`
- `ridge_regression`
- `sgd_regression`

Offline evaluation обучает все настроенные `candidate_models`, поэтому
сравнивает активный набор моделей и baseline-модели даже тогда, когда runtime
training использует `training_mode: single`.

Pipeline использует sklearn preprocessing:

- numeric columns: median imputation и standard scaling;
- categorical columns: missing bucket и frequency encoding;
- date-derived columns из `Date`: year, month, quarter, day и ISO week.

Runtime `update` mode эмулирует поток данных, записывая следующие
`model.stream_batch_days` уникальных дат из stream period:

- `data/raw/batch_XXXX.csv`
- `data/processed/batch_XXXX_processed.csv`
- `artifacts/collector_state.json`
- `artifacts/batch_metadata_history.csv`
- `artifacts/data_quality_history.csv`
- `artifacts/performance_history.csv`
- `artifacts/model_metrics_history.csv`
- `reports/archive/eda/eda_batch_XXXX.md`
- `reports/eda_latest.md`
- `models/archive/model_vXXXX_<model_name>.pkl`
- `models/best_model.pkl`
- `reports/summary_latest.md`
- `reports/archive/summary/summary_<timestamp>_<operation>.md`
- `reports/model_diagnostics_latest.md`
- `reports/figures/model/prediction_timeline.svg`
- `models/current_model.pkl` после `pretrain` или `update`

Regression-метрики:

- `rmse`
- `mae`
- `r2`
- `smape`
- `pearson_corr`
- `pearson_p_value`

Основная model-selection metric: `rmse`.

Data Quality metrics включают missingness, duplicate rows, constant columns,
schema drift, IQR outlier share и categorical cardinality. Summary reports
читают эти метрики из `artifacts/data_quality_history.csv`.

Performance monitoring пишет metadata о длительности и статусе операций в
`artifacts/performance_history.csv`, а summary reports показывают последние
строки из этого файла.

`reset` удаляет сгенерированные runtime artifacts и сохраняет исходные CSV,
конфигурацию, код и `.gitkeep` files.
