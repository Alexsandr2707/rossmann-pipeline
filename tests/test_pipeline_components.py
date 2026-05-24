from __future__ import annotations

import csv
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor

from app.config import Config, load_config
from app.data_quality import DataQualityAnalyzer
from app.feature_engineering import build_features
from app.model_interpretation import ModelInterpretationWriter
from app.models.preprocessing import FrequencyEncoder, make_feature_preprocessor
from app.performance_monitoring import PerformanceMonitor, PerformanceRecord
from app.prediction_serving import PredictionServing
from app.preprocessing import DataPreprocessor
from app.reporting.html_report import generate_html_report
from app.reporting.summary_report import generate_summary_report


class PipelineComponentTests(unittest.TestCase):
    def test_data_quality_metrics_accept_train_schema_without_id(self) -> None:
        with TemporaryDirectory() as tmp:
            config = self._temp_config(Path(tmp))
            dataset = self._train_dataset()
            dataset = pd.concat([dataset, dataset.iloc[[0]]], ignore_index=True)

            metrics = DataQualityAnalyzer(config).analyze_batch(
                dataset,
                {
                    "batch_index": 1,
                    "stream_batch_index": 1,
                    "period_type": "stream",
                    "batch_path": "data/raw/batch_0001.csv",
                },
            )

            self.assertEqual(metrics["schema_missing_columns"], [])
            self.assertNotIn("Id", metrics["schema_missing_columns"])
            self.assertEqual(metrics["duplicate_rows"], 1)
            self.assertAlmostEqual(metrics["duplicate_part"], 1 / 3)
            self.assertIn("missing_part", metrics)
            self.assertIn("constant_columns", metrics)
            self.assertIn("numeric_outlier_part", metrics)
            self.assertEqual(metrics["category_cardinality"]["StoreType"], 2)

            history_path = config.paths.data_quality_history_path
            self.assertTrue(history_path.exists())
            with history_path.open("r", encoding="utf-8", newline="") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(json.loads(row["schema_missing_columns"]), [])
            self.assertTrue((config.paths.reports_dir / "eda_batch_0001.md").exists())

    def test_transform_features_does_not_require_target_or_keep_customers(self) -> None:
        config = load_config("config/config.yaml")
        dataset = pd.DataFrame(
            {
                "Id": [1],
                "Store": [1],
                "Date": ["2024-01-01"],
                "Customers": [100],
                "Open": [1],
            }
        )

        transformed = DataPreprocessor(config).transform_features(dataset)

        self.assertNotIn(config.data.target_column, transformed.columns)
        self.assertNotIn("Customers", transformed.columns)
        self.assertIn("Id", transformed.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(transformed["Date"]))

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

    def test_performance_monitor_records_failure_row(self) -> None:
        with TemporaryDirectory() as tmp:
            config = self._temp_config(Path(tmp))
            monitor = PerformanceMonitor(config)
            start_time = monitor.start()
            monitor.record(
                PerformanceRecord(
                    operation="update",
                    status="failure",
                    duration_seconds=0.0,
                    model_name=config.model.selected_model,
                    error_message="simulated error",
                ),
                start_time=start_time,
            )

            with config.paths.performance_history_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["operation"], "update")
            self.assertEqual(row["status"], "failure")
            self.assertEqual(row["error_message"], "simulated error")
            self.assertTrue(float(row["duration_seconds"]) >= 0.0)

    def test_summary_report_contains_hyperparameters_and_performance(self) -> None:
        with TemporaryDirectory() as tmp:
            config = self._temp_config(Path(tmp))
            config.paths.performance_history_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "timestamp": "2026-05-25T10:00:00+00:00",
                        "operation": "inference",
                        "status": "success",
                        "duration_seconds": "0.123",
                        "input_rows": "10",
                        "output_rows": "10",
                        "model_name": config.model.selected_model,
                    }
                ]
            ).to_csv(config.paths.performance_history_path, index=False)

            report_path = generate_summary_report(config)
            text = report_path.read_text(encoding="utf-8")
            self.assertIn("## Performance history", text)
            self.assertIn("operation", text)
            self.assertIn("## Model hyperparameters", text)
            self.assertIn(f"- selected_model: {config.model.selected_model}", text)

    def test_html_report_keeps_base_hyperparameters_without_explicit_params(self) -> None:
        with TemporaryDirectory() as tmp:
            config = self._temp_config(Path(tmp))
            model = replace(
                config.model,
                selected_model="model_without_params",
                model_parameters={
                    "decision_tree_regression": {"min_samples_leaf": 20},
                },
            )
            config = replace(config, model=model)

            report_path = generate_html_report(config)
            text = report_path.read_text(encoding="utf-8")
            self.assertIn("selected_model", text)
            self.assertIn("training_mode", text)
            self.assertIn("update_strategy", text)
            self.assertIn("primary_metric", text)
            self.assertIn("model_without_params", text)
            self.assertIn("No explicit parameters for selected model.", text)

    def test_model_interpretation_writer_uses_feature_importances(self) -> None:
        with TemporaryDirectory() as tmp:
            config = self._temp_config(Path(tmp))
            writer = ModelInterpretationWriter(config)
            features = pd.DataFrame(
                {
                    "num": [1.0, 2.0, 3.0, 4.0],
                    "cat": ["a", "b", "a", "b"],
                }
            )
            target = pd.Series([10.0, 20.0, 11.0, 19.0])
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", "passthrough", ["num"]),
                    ("cat", FrequencyEncoder(), ["cat"]),
                ],
                remainder="drop",
            )
            transformed = preprocessor.fit_transform(features)
            estimator = DecisionTreeRegressor(random_state=0)
            estimator.fit(transformed, target)

            output = writer.write_model_interpretation(
                {"model_name": "decision_tree_regression", "batch_index": 3},
                preprocessor=preprocessor,
                estimator=estimator,
            )

            latest_path = Path(output["latest_interpretation_report_path"])
            self.assertTrue(latest_path.exists())
            text = latest_path.read_text(encoding="utf-8")
            self.assertIn("interpretation_type: feature_importances", text)
            self.assertIn("num__num", text)
            self.assertIn("cat__cat_frequency", text)
            self.assertIn("| feature", text)
            self.assertIn("abs_value |", text)
            self.assertIn("interpretation_top_features_path", output)
            self.assertTrue(Path(output["interpretation_top_features_path"]).exists())

    def test_model_interpretation_writer_handles_unsupported_estimator(self) -> None:
        with TemporaryDirectory() as tmp:
            config = self._temp_config(Path(tmp))
            writer = ModelInterpretationWriter(config)
            estimator = DummyRegressor(strategy="mean")
            estimator.fit(pd.DataFrame({"x": [0.0, 1.0]}), [1.0, 1.0])

            output = writer.write_model_interpretation(
                {"model_name": "dummy", "batch_index": 1},
                preprocessor=None,
                estimator=estimator,
            )

            latest_path = Path(output["latest_interpretation_report_path"])
            self.assertTrue(latest_path.exists())
            text = latest_path.read_text(encoding="utf-8")
            self.assertIn("Interpretation unavailable", text)
            self.assertNotIn("interpretation_top_features_path", output)

    def test_model_interpretation_writer_extracts_feature_names_from_preprocessor(self) -> None:
        with TemporaryDirectory() as tmp:
            config = self._temp_config(Path(tmp))
            writer = ModelInterpretationWriter(config)
            features = pd.DataFrame(
                {
                    "num": [1.0, 2.0, 3.0, 4.0],
                    "cat": ["a", "b", "a", "b"],
                }
            )
            target = pd.Series([1.0, 2.0, 3.0, 4.0])
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", "passthrough", ["num"]),
                    ("cat", FrequencyEncoder(), ["cat"]),
                ],
                remainder="drop",
            )
            transformed = preprocessor.fit_transform(features)
            estimator = Ridge(alpha=1.0)
            estimator.fit(transformed, target)

            output = writer.write_model_interpretation(
                {"model_name": "ridge_regression", "batch_index": 2},
                preprocessor=preprocessor,
                estimator=estimator,
            )

            text = Path(output["latest_interpretation_report_path"]).read_text(
                encoding="utf-8"
            )
            self.assertIn("interpretation_type: coefficients", text)
            self.assertIn("num__num", text)
            self.assertIn("cat__cat_frequency", text)

    def test_make_feature_preprocessor_feature_names_keep_categorical_column_names(
        self,
    ) -> None:
        features = pd.DataFrame(
            {
                "CompetitionDistance": [100.0, 200.0, 300.0],
                "StoreType": ["a", "b", "a"],
                "PromoInterval": ["Jan", "Feb", "Mar"],
            }
        )
        preprocessor = make_feature_preprocessor(features)
        preprocessor.fit(features)
        names = preprocessor.get_feature_names_out().tolist()

        self.assertIn("categorical__StoreType_frequency", names)
        self.assertIn("categorical__PromoInterval_frequency", names)
        self.assertNotIn("categorical__0_frequency", names)
        self.assertNotIn("categorical__1_frequency", names)

    def _temp_config(self, root: Path) -> Config:
        config = load_config("config/config.yaml")
        paths = replace(
            config.paths,
            artifacts_dir=root / "artifacts",
            reports_dir=root / "reports",
            predictions_dir=root / "artifacts" / "predictions",
            data_quality_history_path=root / "artifacts" / "data_quality_history.csv",
            performance_history_path=root / "artifacts" / "performance_history.csv",
            best_model_path=root / "models" / "best_model.pkl",
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

    def _write_dummy_model(self, model_path: Path) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model = DummyRegressor(strategy="constant", constant=42.0)
        model.fit(pd.DataFrame({"feature": [0.0, 1.0]}), [42.0, 42.0])
        joblib.dump({"pipeline": model}, model_path)


class ConfigLoadingTests(unittest.TestCase):
    def test_load_config_parses_nested_yaml_and_defaults(self) -> None:
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                dedent(
                    """
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
                      missing_strategy: median
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
                        - linear
                        - random_forest
                    model_parameters:
                      random_forest:
                        n_estimators: 120
                        max_depth: 6
                    """
                ).strip()
                + "\n",
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
            self.assertEqual(config.model.candidate_models, ("linear", "random_forest"))
            self.assertEqual(config.model.selected_model, "linear")
            self.assertEqual(config.model.model_parameters["random_forest"]["max_depth"], 6)
            self.assertEqual(
                config.paths.data_quality_history_path,
                Path("artifacts/data_quality_history.csv"),
            )
            self.assertEqual(
                config.paths.performance_history_path,
                Path("artifacts/performance_history.csv"),
            )


if __name__ == "__main__":
    unittest.main()
