from __future__ import annotations

import csv
import json
import sys
import unittest
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor

from app.core.config import Config, load_config
from app.core.pipeline import Pipeline
from app.data.data_quality import DataQualityAnalyzer
from app.data.feature_engineering import build_features
from app.training.model_interpretation import ModelInterpretationWriter
from app.models.preprocessing import FrequencyEncoder, make_feature_preprocessor
from app.monitoring.performance_monitoring import PerformanceMonitor, PerformanceRecord
from app.serving.prediction_serving import PredictionServing
from app.data.preprocessing import DataPreprocessor
from app.reporting.summary_report import generate_summary_report
from run import open_report


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
            self.assertTrue(
                (
                    config.paths.reports_dir
                    / "archive"
                    / "eda"
                    / "eda_batch_0001.md"
                ).exists()
            )
            self.assertTrue((config.paths.reports_dir / "eda_latest.md").exists())

    def test_data_quality_markdown_limits_large_tables_to_five_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            config = self._temp_config(Path(tmp))
            dataset = self._train_dataset()

            DataQualityAnalyzer(config).analyze_batch(
                dataset,
                {
                    "batch_index": 1,
                    "stream_batch_index": 1,
                    "period_type": "stream",
                    "batch_path": "data/raw/batch_0001.csv",
                },
            )

            text = (config.paths.reports_dir / "eda_latest.md").read_text(
                encoding="utf-8"
            )
            profile = self._markdown_section(text, "## Column profile")
            profile_rows = [
                line
                for line in profile.splitlines()
                if line.startswith("| ") and not line.startswith("| ---")
            ]
            self.assertEqual(len(profile_rows), 6)
            self.assertNotIn("| Store ", profile)
            self.assertIn("| PromoInterval ", profile)

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
            config.paths.performance_history_path.parent.mkdir(
                parents=True, exist_ok=True
            )
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
            self.assertEqual(
                report_path,
                config.paths.reports_dir / "summary_latest.md",
            )
            self.assertTrue(
                any(
                    (config.paths.reports_dir / "archive" / "summary").glob(
                        "summary_*_manual.md"
                    )
                )
            )
            self.assertIn("## Performance history", text)
            self.assertIn("operation", text)
            self.assertIn("## Model hyperparameters", text)
            self.assertIn(f"- selected_model: {config.model.selected_model}", text)

    def test_summary_report_keeps_base_hyperparameters_without_explicit_params(
        self,
    ) -> None:
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

            report_path = generate_summary_report(config)
            text = report_path.read_text(encoding="utf-8")
            self.assertIn("selected_model", text)
            self.assertIn("training_mode", text)
            self.assertIn("update_strategy", text)
            self.assertIn("primary_metric", text)
            self.assertIn("model_without_params", text)
            self.assertIn("- model_parameters[model_without_params]:", text)
            self.assertIn("  - (empty)", text)
            self.assertFalse((config.paths.reports_dir / "index.html").exists())

    def test_pipeline_summary_and_refresh_do_not_create_html_index(self) -> None:
        with TemporaryDirectory() as tmp:
            config = self._temp_config(Path(tmp))
            pipeline = Pipeline(config)

            report_path = pipeline.summary()
            refresh_path = pipeline._refresh_reports("test")

            self.assertEqual(
                report_path,
                config.paths.reports_dir / "summary_latest.md",
            )
            self.assertEqual(
                refresh_path,
                config.paths.reports_dir / "summary_latest.md",
            )
            self.assertTrue(report_path.exists())
            self.assertFalse((config.paths.reports_dir / "index.html").exists())

    def test_open_report_skips_missing_wsl_tools(self) -> None:
        with TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "summary_latest.md"
            report_path.write_text("# Summary\n", encoding="utf-8")
            stderr = StringIO()

            with (
                patch("run.is_wsl", return_value=True),
                patch("run.shutil.which", return_value=None),
                patch("run.subprocess.check_output") as check_output,
                patch("run.subprocess.Popen") as popen,
                patch.object(sys, "stderr", stderr),
            ):
                open_report(report_path)

            check_output.assert_not_called()
            popen.assert_not_called()
            self.assertIn(
                "Automatic report opening is unavailable", stderr.getvalue()
            )
            self.assertIn(str(report_path.resolve()), stderr.getvalue())

    def test_summary_report_limits_history_tables_to_latest_five_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            config = self._temp_config(Path(tmp))
            config.paths.performance_history_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            pd.DataFrame(
                [
                    {
                        "timestamp": f"2026-05-25T10:0{index}:00+00:00",
                        "operation": f"operation_{index}",
                        "status": "success",
                        "duration_seconds": str(index),
                    }
                    for index in range(7)
                ]
            ).to_csv(config.paths.performance_history_path, index=False)
            config.paths.model_metrics_history_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            pd.DataFrame(
                [
                    {
                        "batch_index": index,
                        "model_name": f"model_{index}",
                        "rmse": index,
                        "mae": index,
                        "r2": index,
                    }
                    for index in range(7)
                ]
            ).to_csv(config.paths.model_metrics_history_path, index=False)

            text = generate_summary_report(config).read_text(encoding="utf-8")

            self.assertNotIn("operation_0", text)
            self.assertNotIn("operation_1", text)
            self.assertIn("operation_2", text)
            self.assertIn("operation_6", text)
            self.assertNotIn("model_0", text)
            self.assertNotIn("model_1", text)
            self.assertIn("model_2", text)
            self.assertIn("model_6", text)

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

    def test_model_interpretation_report_includes_all_features(self) -> None:
        with TemporaryDirectory() as tmp:
            config = self._temp_config(Path(tmp))
            writer = ModelInterpretationWriter(config)
            preprocessor = SimpleNamespace(
                get_feature_names_out=lambda: [f"feature_{index}" for index in range(7)]
            )
            estimator = SimpleNamespace(feature_importances_=pd.Series(range(7)))

            output = writer.write_model_interpretation(
                {"model_name": "decision_tree_regression", "batch_index": 4},
                preprocessor=preprocessor,
                estimator=estimator,
            )

            text = Path(output["latest_interpretation_report_path"]).read_text(
                encoding="utf-8"
            )
            self.assertIn("feature_6", text)
            self.assertIn("feature_0", text)

    def test_summary_report_keeps_single_top_features_table(self) -> None:
        with TemporaryDirectory() as tmp:
            config = self._temp_config(Path(tmp))
            writer = ModelInterpretationWriter(config)
            preprocessor = SimpleNamespace(
                get_feature_names_out=lambda: [f"feature_{index}" for index in range(7)]
            )
            estimator = SimpleNamespace(feature_importances_=pd.Series(range(7)))

            output = writer.write_model_interpretation(
                {"model_name": "decision_tree_regression", "batch_index": 4},
                preprocessor=preprocessor,
                estimator=estimator,
            )
            config.paths.model_metrics_history_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            pd.DataFrame(
                [
                    {
                        "batch_index": 4,
                        "model_name": "decision_tree_regression",
                        "interpretation_top_features_path": output[
                            "interpretation_top_features_path"
                        ],
                    }
                ]
            ).to_csv(config.paths.model_metrics_history_path, index=False)

            text = generate_summary_report(config).read_text(encoding="utf-8")

            self.assertEqual(
                len(
                    [
                        line
                        for line in text.splitlines()
                        if line.startswith("| feature ") and "abs_value" in line
                    ]
                ),
                1,
            )
            self.assertIn("### Top features preview", text)
            self.assertNotIn("### Top features (by absolute value)", text)
            self.assertIn("feature_6", text)
            self.assertNotIn("feature_0", text)

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

    def test_model_interpretation_writer_extracts_feature_names_from_preprocessor(
        self,
    ) -> None:
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

        self.assertIn("categorical__StoreType", names)
        self.assertIn("categorical__PromoInterval", names)
        self.assertIn("categorical__StoreType_frequency", names)
        self.assertIn("categorical__PromoInterval_frequency", names)
        self.assertNotIn("categorical__0", names)
        self.assertNotIn("categorical__1", names)
        self.assertNotIn("categorical__0_frequency", names)
        self.assertNotIn("categorical__1_frequency", names)

    def test_frequency_encoder_adds_frequency_without_replacing_category_code(
        self,
    ) -> None:
        features = pd.DataFrame({"StoreType": ["a", "b", "a", "c"]})
        encoder = FrequencyEncoder()

        transformed = encoder.fit_transform(features)

        self.assertEqual(transformed.shape, (4, 2))
        self.assertEqual(transformed[:, 0].tolist(), [0.0, 1.0, 0.0, 2.0])
        self.assertEqual(transformed[:, 1].tolist(), [0.5, 0.25, 0.5, 0.25])
        self.assertEqual(
            encoder.get_feature_names_out().tolist(),
            ["StoreType", "StoreType_frequency"],
        )

    def test_pretrain_rejects_existing_model_files(self) -> None:
        with TemporaryDirectory() as tmp:
            config = self._temp_config(Path(tmp))
            existing_model_path = config.paths.models_dir / "current_model.pkl"
            existing_model_path.parent.mkdir(parents=True, exist_ok=True)
            existing_model_path.write_bytes(b"existing model")

            with self.assertRaisesRegex(
                FileExistsError,
                "Run reset before pretrain",
            ):
                Pipeline(config).pretrain()

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

    def _markdown_section(self, text: str, heading: str) -> str:
        start = text.index(heading)
        next_heading = text.find("\n## ", start + len(heading))
        if next_heading == -1:
            return text[start:]
        return text[start:next_heading]


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
            self.assertEqual(config.model.candidate_models, ("linear", "random_forest"))
            self.assertEqual(config.model.selected_model, "linear")
            self.assertEqual(
                config.model.model_parameters["random_forest"]["max_depth"], 6
            )
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
