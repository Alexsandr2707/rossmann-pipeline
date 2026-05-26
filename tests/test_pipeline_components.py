from __future__ import annotations

import csv
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor

from app.core.config import Config, load_config
from app.core.pipeline import Pipeline
from app.data.data_collection import DataCollector
from app.data.feature_engineering import build_features
from app.models import FEATURE_PREPROCESSING_VERSION, model_signature
from app.serving.prediction_serving import PredictionServing


class PipelineComponentTests(unittest.TestCase):
    def test_build_features_excludes_ids_leakage_and_services(self) -> None:
        config = load_config("config/config.yaml")
        dataset = self._train_dataset()
        dataset["Id"] = [10, 20]

        features = build_features(dataset, config)

        self.assertNotIn("Id", features.columns)
        self.assertNotIn("Customers", features.columns)
        self.assertNotIn("_source_file", features.columns)
        self.assertNotIn(config.data.target_column, features.columns)

        self.assertIn("Date_year", features.columns)
        self.assertIn("Date_month", features.columns)
        self.assertIn("Date_quarter", features.columns)
        self.assertIn("Date_day", features.columns)
        self.assertIn("Date_weekofyear", features.columns)

    def test_prediction_serving_writes_predictions_for_temp_input(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._temp_config(root)

            self._write_store_metadata(config.data.store_path)
            self._write_dummy_model(config.paths.best_model_path)

            input_path = root / "inference.csv"
            input_rows = pd.DataFrame(
                {
                    "Id": [1, 2],
                    "Store": [1, 2],
                    "DayOfWeek": [1, 2],
                    "Date": ["2024-01-01", "2024-01-02"],
                    "Open": [1, 1],
                    "Promo": [0, 1],
                    "SchoolHoliday": [0, 0],
                }
            )
            input_rows.to_csv(input_path, index=False)

            output_path = PredictionServing(config).predict_file(input_path)

            output = pd.read_csv(output_path)

            self.assertEqual(len(output), len(input_rows))
            self.assertIn("predict", output.columns)
            self.assertEqual(set(output.columns) - set(input_rows.columns), {"predict"})
            self.assertEqual(output["predict"].tolist(), [42.0, 42.0])

            self.assertTrue(config.paths.performance_history_path.exists())

            with config.paths.performance_history_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(rows[-1]["operation"], "inference")
            self.assertEqual(rows[-1]["status"], "success")
            self.assertEqual(rows[-1]["input_rows"], "2")
            self.assertEqual(rows[-1]["output_rows"], "2")

    def test_stream_state_is_committed_once_after_successful_update(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config_with_source_data(root)

            current_model_path = config.paths.models_dir / "current_model.pkl"
            current_model_path.parent.mkdir(parents=True, exist_ok=True)
            current_model_path.write_bytes(b"existing")

            updated_model_path = config.paths.models_dir / "updated.pkl"

            pipeline = Pipeline(config)
            pipeline.model_trainer = _SuccessfulUpdateTrainer(  # type: ignore
                current_model_path,
                updated_model_path,
            )
            pipeline._refresh_reports = lambda _context: config.paths.reports_dir  # type: ignore

            self.assertEqual(pipeline.update(), 1)

            self.assertEqual(DataCollector(config).current_stream_batch_index(), 1)

            history = pd.read_csv(config.paths.batch_metadata_path)
            self.assertEqual(history["stream_batch_index"].tolist(), [0])

    def test_stream_state_is_not_committed_when_update_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config_with_source_data(root)

            current_model_path = config.paths.models_dir / "current_model.pkl"
            current_model_path.parent.mkdir(parents=True, exist_ok=True)
            current_model_path.write_bytes(b"existing")

            pipeline = Pipeline(config)
            pipeline.model_trainer = _FailingUpdateTrainer(current_model_path)  # type: ignore
            pipeline._refresh_reports = lambda _context: config.paths.reports_dir  # type: ignore

            with self.assertRaisesRegex(RuntimeError, "simulated update failure"):
                pipeline.update()

            self.assertEqual(DataCollector(config).current_stream_batch_index(), 0)
            self.assertFalse(config.paths.batch_metadata_path.exists())
            self.assertTrue((config.paths.raw_data_dir / "batch_0000.csv").exists())

    def _temp_config(self, root: Path) -> Config:
        config = load_config("config/config.yaml")

        paths = replace(
            config.paths,
            external_data_dir=root / "data" / "external",
            raw_data_dir=root / "data" / "raw",
            processed_data_dir=root / "data" / "processed",
            artifacts_dir=root / "artifacts",
            models_dir=root / "models",
            logs_dir=root / "logs",
            reports_dir=root / "reports",
            predictions_dir=root / "artifacts" / "predictions",
            collector_state_path=root / "artifacts" / "collector_state.json",
            batch_metadata_path=root / "artifacts" / "batch_metadata_history.csv",
            data_quality_history_path=root / "artifacts" / "data_quality_history.csv",
            performance_history_path=root / "artifacts" / "performance_history.csv",
            model_metrics_history_path=root / "artifacts" / "model_metrics_history.csv",
            best_model_path=root / "models" / "best_model.pkl",
            pipeline_log_path=root / "logs" / "pipeline.log",
        )

        data = replace(config.data, store_path=root / "store.csv")

        return replace(config, data=data, paths=paths)

    def _train_dataset(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Store": [1, 2],
                "DayOfWeek": [1, 2],
                "Date": ["2024-01-01", "2024-01-02"],
                "Sales": [100.0, 150.0],
                "Customers": [10, 20],
                "Open": [1, 1],
                "Promo": [0, 1],
                "StateHoliday": ["0", "a"],
                "SchoolHoliday": [0, 0],
                "StoreType": ["a", "b"],
                "Assortment": ["a", "c"],
                "CompetitionDistance": [100.0, 200.0],
                "CompetitionOpenSinceMonth": [1, 2],
                "CompetitionOpenSinceYear": [2020, 2021],
                "Promo2": [0, 1],
                "Promo2SinceWeek": [1, 5],
                "Promo2SinceYear": [2020, 2022],
                "PromoInterval": ["Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov"],
                "_source_file": ["train.csv", "train.csv"],
            }
        )

    def _stream_dataset(self, date_count: int) -> pd.DataFrame:
        rows = []

        for index in range(date_count):
            rows.append(
                {
                    "Store": 1 if index % 2 == 0 else 2,
                    "DayOfWeek": (index % 7) + 1,
                    "Date": f"2024-01-{index + 1:02d}",
                    "Sales": float(100 + index),
                    "Customers": 10 + index,
                    "Open": 1,
                    "Promo": index % 2,
                    "StateHoliday": "0",
                    "SchoolHoliday": 0,
                    "StoreType": "a" if index % 2 == 0 else "b",
                    "Assortment": "a" if index % 2 == 0 else "c",
                    "CompetitionDistance": 100.0 + index,
                    "CompetitionOpenSinceMonth": 1,
                    "CompetitionOpenSinceYear": 2020,
                    "Promo2": 0,
                    "Promo2SinceWeek": 1,
                    "Promo2SinceYear": 2020,
                    "PromoInterval": "Jan,Apr,Jul,Oct",
                    "_source_file": "train.csv",
                }
            )

        return pd.DataFrame(rows)

    def _write_store_metadata(self, store_path: Path | None) -> None:
        self.assertIsNotNone(store_path)
        assert store_path is not None

        store_path.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            {
                "Store": [1, 2],
                "StoreType": ["a", "b"],
                "Assortment": ["a", "c"],
                "CompetitionDistance": [100.0, 200.0],
                "CompetitionOpenSinceMonth": [1, 2],
                "CompetitionOpenSinceYear": [2020, 2021],
                "Promo2": [0, 1],
                "Promo2SinceWeek": [1, 5],
                "Promo2SinceYear": [2020, 2022],
                "PromoInterval": ["Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov"],
            }
        ).to_csv(store_path, index=False)

    def _write_dummy_model(self, model_path: Path, constant: float = 42.0) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)

        config = load_config("config/config.yaml")

        model = DummyRegressor(strategy="constant", constant=constant)
        model.fit(pd.DataFrame({"feature": [0.0, 1.0]}), [constant, constant])

        joblib.dump(
            {
                "pipeline": model,
                "model_name": config.model.selected_model,
                "feature_preprocessing_version": FEATURE_PREPROCESSING_VERSION,
                "model_signature": model_signature(
                    config.model.selected_model,
                    config.model.model_parameters[config.model.selected_model],
                ),
            },
            model_path,
        )

    def _config_with_source_data(self, root: Path) -> Config:
        config = self._temp_config(root)

        dataset = self._stream_dataset(date_count=20)
        source_path = root / "train.csv"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(source_path, index=False)

        self._write_store_metadata(config.data.store_path)

        data = replace(config.data, source_paths=(source_path,))

        return replace(config, data=data)


class ConfigLoadingTests(unittest.TestCase):
    def test_load_config_parses_nested_yaml_and_defaults(self) -> None:
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"

            config_path.write_text(
                dedent("""
                    project:
                      name: Test Project
                      random_seed: 123
                    data:
                      source_paths:
                        - data/raw/a.csv
                        - data/raw/b.csv
                      store_path: null
                      time_column: Date
                      target_column: Sales
                      min_rows: 10
                      min_features: 3
                      min_categorical_features: 1
                    target_preprocessing:
                      missing_strategy: fill_value
                      missing_fill_value: 0.5
                      add_missing_indicator: true
                      missing_indicator_suffix: _missing
                    data_schema:
                      numeric_columns:
                        - Sales
                        - Customers
                      categorical_columns:
                        - StoreType
                      datetime_columns:
                        - Date
                      id_columns:
                        - Id
                      service_columns:
                        - _source_file
                    paths:
                      external_data_dir: data/external
                      raw_data_dir: data/raw
                      processed_data_dir: data/processed
                      artifacts_dir: artifacts
                      models_dir: models
                      logs_dir: logs
                      reports_dir: reports
                      predictions_dir: artifacts/predictions
                      collector_state_path: artifacts/collector_state.json
                      batch_metadata_path: artifacts/batch_metadata_history.csv
                      model_metrics_history_path: artifacts/model_metrics_history.csv
                      best_model_path: models/best_model.pkl
                      pipeline_log_path: logs/pipeline.log
                    model:
                      primary_metric: rmse
                      candidate_models:
                        - decision_tree_regression
                        - random_forest_regression
                    model_parameters:
                      random_forest_regression:
                        n_estimators: 120
                        max_depth: 6
                    """).strip() + "\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.project.name, "Test Project")
            self.assertEqual(config.project.random_seed, 123)
            self.assertEqual(
                config.data.source_paths,
                (Path("data/raw/a.csv"), Path("data/raw/b.csv")),
            )
            self.assertIsNone(config.data.store_path)
            self.assertEqual(
                config.model.candidate_models,
                ("decision_tree_regression", "random_forest_regression"),
            )
            self.assertEqual(config.model.selected_model, "decision_tree_regression")
            self.assertEqual(
                config.model.model_parameters["random_forest_regression"]["max_depth"],
                6,
            )
            self.assertEqual(
                config.paths.data_quality_history_path,
                Path("artifacts/data_quality_history.csv"),
            )
            self.assertEqual(
                config.paths.performance_history_path,
                Path("artifacts/performance_history.csv"),
            )

    def test_load_config_rejects_invalid_model_settings(self) -> None:
        cases = {
            "training_mode": (
                "training_mode: single",
                "training_mode: invalid",
            ),
            "update_strategy": (
                "update_strategy: full_refit",
                "update_strategy: invalid",
            ),
            "primary_metric": (
                "primary_metric: rmse",
                "primary_metric: r2",
            ),
            "candidate_models": (
                "candidate_models:\n    - decision_tree_regression\n    - ridge_regression",
                "candidate_models: []",
            ),
            "selected_model": (
                "selected_model: decision_tree_regression",
                "selected_model: sgd_regression",
            ),
            "stream_batch_days": (
                "stream_batch_days: 7",
                "stream_batch_days: 0",
            ),
            "rolling_train_period_days": (
                "rolling_train_period_days: 365",
                "rolling_train_period_days: 0",
            ),
            "split ratios": (
                "initial_train_ratio: 0.50\n  validation_ratio: 0.20",
                "initial_train_ratio: 0.90\n  validation_ratio: 0.10",
            ),
        }

        for expected_message, (old, new) in cases.items():
            with self.subTest(expected_message=expected_message):
                with self.assertRaisesRegex(ValueError, expected_message):
                    self._load_config_text(self._valid_config_text().replace(old, new))

    def _load_config_text(self, text: str) -> Config:
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(text, encoding="utf-8")
            return load_config(config_path)

    def _valid_config_text(self) -> str:
        return dedent("""
            project:
              name: Test Project
              random_seed: 123
            data:
              source_paths:
                - data/raw/a.csv
              store_path: null
              time_column: Date
              target_column: Sales
              min_rows: 10
              min_features: 3
              min_categorical_features: 1
            target_preprocessing:
              missing_strategy: drop
              missing_fill_value: 0.0
              add_missing_indicator: false
              missing_indicator_suffix: _missing
            data_schema:
              numeric_columns:
                - Sales
              categorical_columns:
                - StoreType
              datetime_columns:
                - Date
              id_columns:
                - Id
              service_columns:
                - _source_file
            paths:
              external_data_dir: data/external
              raw_data_dir: data/raw
              processed_data_dir: data/processed
              artifacts_dir: artifacts
              models_dir: models
              logs_dir: logs
              reports_dir: reports
              predictions_dir: artifacts/predictions
              collector_state_path: artifacts/collector_state.json
              batch_metadata_path: artifacts/batch_metadata_history.csv
              model_metrics_history_path: artifacts/model_metrics_history.csv
              best_model_path: models/best_model.pkl
              pipeline_log_path: logs/pipeline.log
            model:
              primary_metric: rmse
              training_mode: single
              selected_model: decision_tree_regression
              update_strategy: full_refit
              stream_batch_days: 7
              initial_train_ratio: 0.50
              validation_ratio: 0.20
              stream_ratio: 0.30
              rolling_train_period_days: 365
              pretrain_mark_collector_state: true
              candidate_models:
                - decision_tree_regression
                - ridge_regression
            model_parameters:
              decision_tree_regression:
                min_samples_leaf: 10
              ridge_regression:
                alpha: 1.0
            """).lstrip()


class _SuccessfulUpdateTrainer:
    def __init__(self, current_model_path: Path, updated_model_path: Path) -> None:
        self.current_model_path = current_model_path
        self.updated_model_path = updated_model_path

    def has_compatible_current_model(self) -> bool:
        return True

    def update_on_stream_batch(self, **_kwargs):
        self.updated_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_model_path.write_bytes(b"updated")
        return self.updated_model_path, {"rmse": 0.0}


class _FailingUpdateTrainer:
    def __init__(self, current_model_path: Path) -> None:
        self.current_model_path = current_model_path

    def has_compatible_current_model(self) -> bool:
        return True

    def update_on_stream_batch(self, **_kwargs):
        raise RuntimeError("simulated update failure")


if __name__ == "__main__":
    unittest.main()
