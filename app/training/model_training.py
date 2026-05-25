from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from scipy.stats import pearsonr

from app.core.config import Config
from app.data.feature_engineering import build_features_and_target
from app.training.model_diagnostics import ModelDiagnosticsWriter
from app.training.model_interpretation import ModelInterpretationWriter
from app.models import (
    FEATURE_PREPROCESSING_VERSION,
    SGD_REGRESSION_MODEL_NAME,
    canonical_model_name,
    make_feature_preprocessor,
    make_model,
    model_signature,
    supports_incremental_update,
)
from app.data.dataset_loading import load_source_dataset
from app.data.period_splitting import (
    period_boundaries,
    rows_for_dates,
    split_date_periods,
)
from app.data.preprocessing import DataPreprocessor


class ModelTrainer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.diagnostics = ModelDiagnosticsWriter(config)
        self.interpretation = ModelInterpretationWriter(config)

    @property
    def metrics_history_path(self) -> Path:
        return self.config.paths.model_metrics_history_path

    @property
    def best_model_path(self) -> Path:
        return self.config.paths.best_model_path

    @property
    def current_model_path(self) -> Path:
        return self.config.paths.models_dir / "current_model.pkl"

    def has_compatible_current_model(self) -> bool:
        if not self.current_model_path.exists():
            return False
        try:
            payload = joblib.load(self.current_model_path)
        except Exception as error:
            self.logger.warning("Cannot load current model: %s", error)
            return False
        return (
            payload.get("model_name")
            == canonical_model_name(self.config.model.selected_model)
            and payload.get("feature_preprocessing_version")
            == FEATURE_PREPROCESSING_VERSION
            and payload.get("model_signature")
            == self._model_signature(self.config.model.selected_model)
        )

    def pretrain_on_dataset(
        self,
        dataset: pd.DataFrame,
        processed_path: Path,
        batch_metadata: dict[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        split = split_date_periods(dataset, self.config)
        initial_dataset = rows_for_dates(
            dataset,
            self.config.data.time_column,
            split.initial_dates,
        )
        validation_dataset = rows_for_dates(
            dataset,
            self.config.data.time_column,
            split.validation_dates,
        )
        start_dataset = rows_for_dates(
            dataset,
            self.config.data.time_column,
            split.initial_dates.append(split.validation_dates),
        )
        x_train, y_train = self._build_features_and_target(initial_dataset)
        x_valid, y_valid = self._build_features_and_target(validation_dataset)
        valid_dates = self.target_dates(validation_dataset)

        if len(x_train) < 10 or len(x_valid) < 1:
            raise ValueError("Not enough rows for model pretraining.")

        results: list[dict[str, Any]] = []
        refitted_models: dict[str, SklearnPipeline] = {}

        for model_name in self._models_to_train():
            pipeline = self.fit_pipeline(x_train, y_train, model_name)
            predictions = pipeline.predict(x_valid)
            x_valid_transformed = pipeline.named_steps["preprocessor"].transform(
                x_valid
            )
            metrics = self._calculate_metrics(
                y_valid,
                predictions,
                estimator=pipeline.named_steps["model"],
                transformed_features=x_valid_transformed,
            )
            metrics.update(
                {
                    "model_name": model_name,
                    "batch_index": int(batch_metadata.get("batch_index", -1)),
                    "latest_processed_path": str(processed_path),
                    "train_rows": int(len(x_train)),
                    "valid_rows": int(len(x_valid)),
                    "training_mode": self.config.model.training_mode,
                    "update_strategy": "pretrain",
                    "period_type": "validation",
                    "stream_batch_index": "",
                    "date_min": split.validation_dates.min().strftime("%Y-%m-%d"),
                    "date_max": split.validation_dates.max().strftime("%Y-%m-%d"),
                    "date_count": int(len(split.validation_dates)),
                    "train_window_date_min": split.initial_dates.min().strftime(
                        "%Y-%m-%d"
                    ),
                    "train_window_date_max": split.initial_dates.max().strftime(
                        "%Y-%m-%d"
                    ),
                    "initial_training": True,
                    "training_rows_total": int(len(x_train)),
                    "pretrain_rows": int(
                        batch_metadata.get("pretrain_rows", len(dataset))
                    ),
                    "pretrain_time_min": str(batch_metadata.get("time_min", "")),
                    "pretrain_time_max": str(batch_metadata.get("time_max", "")),
                    **period_boundaries(split),
                }
            )
            metrics.update(
                self.diagnostics.write_model_diagnostics(
                    y_valid,
                    predictions,
                    metrics,
                    estimator=pipeline.named_steps["model"],
                    transformed_features=x_valid_transformed,
                    timeline_dates=valid_dates,
                )
            )
            metrics.update(
                self.interpretation.write_model_interpretation(
                    metrics,
                    preprocessor=pipeline.named_steps["preprocessor"],
                    estimator=pipeline.named_steps["model"],
                )
            )
            results.append(metrics)
            start_features, start_target = self._build_features_and_target(
                start_dataset
            )
            refitted_models[model_name] = self.fit_pipeline(
                start_features,
                start_target,
                model_name,
            )

        best_metrics = min(
            results,
            key=lambda item: item[self.config.model.primary_metric],
        )
        best_model_name = str(best_metrics["model_name"])
        best_pipeline = refitted_models[best_model_name]

        model_path = self._save_model_version(best_pipeline, best_metrics)
        self._save_current_model(
            best_pipeline, best_metrics, model_path, best_model_name
        )
        self._save_best_model(best_pipeline, best_metrics, model_path)
        self._append_metrics(results, model_path)
        self.diagnostics.write_metrics_history_plot(self.metrics_history_path)

        self.logger.info("Pretrained model: %s", best_metrics)
        return model_path, best_metrics

    def update_on_stream_batch(
        self,
        latest_processed_path: Path,
        raw_batch_path: Path,
        batch_metadata: dict[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        if not self.current_model_path.exists():
            raise FileNotFoundError(
                f"Current model not found: {self.current_model_path}. Run pretrain first."
            )

        payload = joblib.load(self.current_model_path)
        pipeline = payload["pipeline"]
        model_name = canonical_model_name(self.config.model.selected_model)
        payload_model_name = payload.get("model_name")
        if payload_model_name is not None and payload_model_name != model_name:
            raise ValueError(
                f"Current model is {payload_model_name}, expected {model_name}."
            )

        batch = self._load_processed_batch(latest_processed_path)
        features, target = self._build_features_and_target(batch)
        dates = self.target_dates(batch)
        predictions = pipeline.predict(features)
        transformed_features = pipeline.named_steps["preprocessor"].transform(features)
        metrics = self._calculate_metrics(
            target,
            predictions,
            estimator=pipeline.named_steps["model"],
            transformed_features=transformed_features,
        )

        stream_batch_index = int(batch_metadata["stream_batch_index"])
        metrics.update(
            {
                "model_name": model_name,
                "batch_index": stream_batch_index,
                "stream_batch_index": stream_batch_index,
                "latest_processed_path": str(latest_processed_path),
                "period_type": "stream",
                "date_min": str(batch_metadata["date_min"]),
                "date_max": str(batch_metadata["date_max"]),
                "date_count": int(batch_metadata["date_count"]),
                "prediction_model_path": str(
                    payload.get("model_path", self.current_model_path)
                ),
                "update_strategy": self.config.model.update_strategy,
                "training_mode": self.config.model.training_mode,
                "initial_training": False,
                "valid_rows": int(len(features)),
                "raw_batch_path": str(raw_batch_path),
                "processed_batch_path": str(latest_processed_path),
                "predictions_file_stem": (
                    f"model_predictions_batch_{stream_batch_index:04d}_{model_name}"
                ),
            }
        )
        metrics.update(
            self.diagnostics.write_model_diagnostics(
                target,
                predictions,
                metrics,
                estimator=pipeline.named_steps["model"],
                transformed_features=transformed_features,
                timeline_dates=dates,
            )
        )
        metrics.update(
            self.interpretation.write_model_interpretation(
                metrics,
                preprocessor=pipeline.named_steps["preprocessor"],
                estimator=pipeline.named_steps["model"],
            )
        )

        updated_pipeline, train_window = self._updated_pipeline(
            pipeline,
            batch,
            pd.to_datetime(str(batch_metadata["date_max"])),
            model_name,
        )
        metrics.update(
            {
                "train_rows": int(train_window["rows"]),
                "training_rows_total": int(train_window["rows"]),
                "train_window_date_min": train_window["date_min"],
                "train_window_date_max": train_window["date_max"],
            }
        )
        model_path = self._save_model_version(updated_pipeline, metrics)
        metrics["updated_model_path"] = str(model_path)
        self._save_current_model(updated_pipeline, metrics, model_path, model_name)
        self._save_best_model(updated_pipeline, metrics, model_path)
        self._append_metrics([metrics], model_path)
        self.diagnostics.write_metrics_history_plot(self.metrics_history_path)
        return model_path, metrics

    def _updated_pipeline(
        self,
        current_pipeline: SklearnPipeline,
        batch: pd.DataFrame,
        batch_end_date: pd.Timestamp,
        model_name: str,
    ) -> tuple[SklearnPipeline, dict[str, Any]]:
        strategy = self.config.model.update_strategy
        if strategy == "full_refit":
            training_dataset = self._known_period_dataset(batch_end_date)
            return (
                self._fit_pipeline_on_dataset(training_dataset, model_name),
                self._training_window_metadata(training_dataset),
            )
        if strategy == "rolling_refit":
            training_dataset = self._known_period_dataset(batch_end_date)
            window_start = batch_end_date.floor("D") - pd.Timedelta(
                days=self.config.model.rolling_train_period_days - 1
            )
            parsed_dates = pd.to_datetime(
                training_dataset[self.config.data.time_column],
                errors="coerce",
            ).dt.floor("D")
            training_dataset = training_dataset.loc[parsed_dates >= window_start].copy()
            return (
                self._fit_pipeline_on_dataset(training_dataset, model_name),
                self._training_window_metadata(training_dataset),
            )
        if strategy == "incremental":
            if model_name != SGD_REGRESSION_MODEL_NAME:
                raise ValueError(
                    "update_strategy=incremental is supported only for "
                    f"{SGD_REGRESSION_MODEL_NAME}."
                )
            estimator = current_pipeline.named_steps["model"]
            if not supports_incremental_update(estimator):
                raise ValueError(
                    f"Model {model_name} does not support incremental updates."
                )
            features, target = self._build_features_and_target(batch)
            transformed = current_pipeline.named_steps["preprocessor"].transform(
                features
            )
            estimator.update(transformed, target)
            return current_pipeline, self._training_window_metadata(batch)
        raise ValueError(f"Unsupported update strategy: {strategy}")

    def _known_period_dataset(self, max_date: pd.Timestamp) -> pd.DataFrame:
        raw_dataset = load_source_dataset(self.config)
        processed = DataPreprocessor(self.config).transform(raw_dataset)
        parsed_dates = pd.to_datetime(
            processed[self.config.data.time_column],
            errors="coerce",
        ).dt.floor("D")
        known = processed.loc[parsed_dates <= max_date.floor("D")].copy()
        return self._sort_by_time(known)

    def _fit_pipeline_on_dataset(
        self,
        dataset: pd.DataFrame,
        model_name: str,
    ) -> SklearnPipeline:
        features, target = self._build_features_and_target(dataset)
        if len(features) < 10:
            raise ValueError("Not enough rows for model refit.")
        return self.fit_pipeline(features, target, model_name)

    def fit_pipeline(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        model_name: str,
    ) -> SklearnPipeline:
        estimator = self._make_estimator(model_name)
        pipeline = SklearnPipeline(
            steps=[
                ("preprocessor", self._make_feature_preprocessor(features)),
                ("model", estimator),
            ]
        )
        pipeline.fit(features, target)
        return pipeline

    def _training_window_metadata(self, dataset: pd.DataFrame) -> dict[str, Any]:
        if dataset.empty:
            return {"rows": 0, "date_min": "", "date_max": ""}
        dates = pd.to_datetime(
            dataset[self.config.data.time_column],
            errors="coerce",
        ).dropna()
        return {
            "rows": int(len(dataset)),
            "date_min": "" if dates.empty else dates.min().strftime("%Y-%m-%d"),
            "date_max": "" if dates.empty else dates.max().strftime("%Y-%m-%d"),
        }

    def _load_processed_batch(self, processed_path: Path) -> pd.DataFrame:
        dataset = pd.read_csv(processed_path)
        return self._sort_by_time(dataset)

    def _sort_by_time(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset[self.config.data.time_column] = pd.to_datetime(
            dataset[self.config.data.time_column],
            errors="coerce",
        )
        return dataset.sort_values(self.config.data.time_column).reset_index(drop=True)

    def _build_features_and_target(
        self,
        dataset: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        return build_features_and_target(dataset, self.config)

    def target_dates(self, dataset: pd.DataFrame) -> pd.Series:
        target_column = self.config.data.target_column
        time_column = self.config.data.time_column
        target = pd.to_numeric(dataset[target_column], errors="coerce")
        dates = pd.to_datetime(
            dataset.loc[target.notna(), time_column],
            errors="coerce",
        )
        return dates.reset_index(drop=True)

    def _make_feature_preprocessor(self, features: pd.DataFrame) -> Any:
        return make_feature_preprocessor(features)

    def _make_estimator(self, model_name: str) -> Any:
        return make_model(
            model_name,
            self._model_parameters(model_name),
            self.config.project.random_seed,
        )

    def _models_to_train(self) -> tuple[str, ...]:
        if self.config.model.training_mode == "all":
            return tuple(
                dict.fromkeys(
                    canonical_model_name(model_name)
                    for model_name in self.config.model.candidate_models
                )
            )
        if self.config.model.training_mode == "single":
            selected_model = canonical_model_name(self.config.model.selected_model)
            candidate_models = {
                canonical_model_name(model_name)
                for model_name in self.config.model.candidate_models
            }
            if selected_model not in candidate_models:
                raise ValueError(
                    f"Selected model is not in candidate_models: {selected_model}"
                )
            return (selected_model,)
        raise ValueError(
            f"Unsupported training mode: {self.config.model.training_mode}"
        )

    def _model_parameters(self, model_name: str) -> dict[str, Any]:
        canonical_name = canonical_model_name(model_name)
        if canonical_name in self.config.model.model_parameters:
            return dict(self.config.model.model_parameters[canonical_name])
        if model_name in self.config.model.model_parameters:
            return dict(self.config.model.model_parameters[model_name])

        for configured_name, parameters in self.config.model.model_parameters.items():
            if canonical_model_name(configured_name) == canonical_name:
                return dict(parameters)
        return {}

    def _model_signature(self, model_name: str) -> dict[str, Any]:
        return model_signature(model_name, self._model_parameters(model_name))

    def _calculate_metrics(
        self,
        y_true: pd.Series,
        predictions: np.ndarray,
        estimator: Any | None = None,
        transformed_features: Any | None = None,
    ) -> dict[str, Any]:
        predictions = np.asarray(predictions, dtype=float)
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)
        y_true_array = y_true.to_numpy()
        pearson_corr, pearson_p_value = self._pearson_metrics(
            y_true_array,
            predictions,
        )
        metrics: dict[str, Any] = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "smape": self._smape(y_true_array, predictions),
            "pearson_corr": pearson_corr,
            "pearson_p_value": pearson_p_value,
            "prediction_mean": float(predictions.mean()),
            "target_mean": float(y_true_array.mean()),
        }

        return metrics

    def _smape(self, y_true: np.ndarray, predictions: np.ndarray) -> float:
        denominator = np.abs(y_true) + np.abs(predictions)
        smape_values = np.divide(
            2.0 * np.abs(predictions - y_true),
            denominator,
            out=np.zeros_like(predictions, dtype=float),
            where=denominator != 0,
        )
        return float(smape_values.mean())

    def _pearson_metrics(
        self,
        y_true: np.ndarray,
        predictions: np.ndarray,
    ) -> tuple[float | None, float | None]:
        if len(y_true) < 2:
            return None, None
        if np.isclose(np.std(y_true), 0.0) or np.isclose(np.std(predictions), 0.0):
            return None, None

        result = pearsonr(y_true, predictions)
        return float(result.statistic), float(result.pvalue)

    def _save_model_version(
        self,
        pipeline: SklearnPipeline,
        metrics: dict[str, Any],
    ) -> Path:
        self.config.paths.models_dir.mkdir(parents=True, exist_ok=True)
        batch_index = int(metrics["batch_index"])
        model_name = str(metrics["model_name"])
        if batch_index < 0:
            model_path = (
                self.config.paths.models_dir / f"model_pretrain_{model_name}.pkl"
            )
        else:
            model_path = self.config.paths.models_dir / (
                f"model_v{batch_index:04d}_{model_name}.pkl"
            )
        joblib.dump(
            {
                "pipeline": pipeline,
                "metrics": metrics,
                "feature_preprocessing_version": FEATURE_PREPROCESSING_VERSION,
                "model_signature": self._model_signature(str(metrics["model_name"])),
            },
            model_path,
        )
        return model_path

    def _save_best_model(
        self,
        pipeline: SklearnPipeline,
        metrics: dict[str, Any],
        model_path: Path,
    ) -> None:
        payload = {
            "pipeline": pipeline,
            "metrics": metrics,
            "model_path": str(model_path),
            "feature_preprocessing_version": FEATURE_PREPROCESSING_VERSION,
            "model_signature": self._model_signature(str(metrics["model_name"])),
        }
        if self.best_model_path.exists():
            current = joblib.load(self.best_model_path)
            if current.get(
                "feature_preprocessing_version"
            ) == FEATURE_PREPROCESSING_VERSION and current.get(
                "model_signature"
            ) == self._model_signature(
                str(metrics["model_name"])
            ):
                current_metrics = current.get("metrics", {})
                current_score = current_metrics.get(self.config.model.primary_metric)
                candidate_score = metrics[self.config.model.primary_metric]
                if current_score is not None and current_score <= candidate_score:
                    return

        joblib.dump(payload, self.best_model_path)

    def _save_current_model(
        self,
        pipeline: SklearnPipeline,
        metrics: dict[str, Any],
        model_path: Path,
        model_name: str,
    ) -> None:
        canonical_name = canonical_model_name(model_name)
        self.current_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": pipeline,
                "metrics": metrics,
                "model_path": str(model_path),
                "model_name": canonical_name,
                "feature_preprocessing_version": FEATURE_PREPROCESSING_VERSION,
                "model_signature": self._model_signature(canonical_name),
            },
            self.current_model_path,
        )

    def _append_metrics(
        self,
        metrics_rows: list[dict[str, Any]],
        model_path: Path,
    ) -> None:
        self.metrics_history_path.parent.mkdir(parents=True, exist_ok=True)
        rows = pd.DataFrame(metrics_rows)
        rows["model_path"] = str(model_path)
        rows["is_best_in_update"] = (
            rows[self.config.model.primary_metric]
            == rows[self.config.model.primary_metric].min()
        )
        if self.metrics_history_path.exists():
            history = pd.read_csv(self.metrics_history_path)
            rows = pd.concat([history, rows], ignore_index=True, sort=False)
        rows = rows[self._metrics_columns(rows)]
        rows.to_csv(self.metrics_history_path, index=False)

    def _metrics_columns(self, rows: pd.DataFrame) -> list[str]:
        preferred_columns = [
            "model_name",
            "batch_index",
            "period_type",
            "stream_batch_index",
            "date_min",
            "date_max",
            "date_count",
            "rmse",
            "mae",
            "r2",
            "smape",
            "pearson_corr",
            "pearson_p_value",
            "prediction_mean",
            "target_mean",
            "train_rows",
            "valid_rows",
            "is_best_in_update",
            "training_mode",
            "update_strategy",
            "model_update_method",
            "initial_training",
            "training_rows_total",
            "train_window_date_min",
            "train_window_date_max",
            "pretrain_rows",
            "pretrain_time_min",
            "pretrain_time_max",
            "latest_processed_path",
            "raw_batch_path",
            "processed_batch_path",
            "prediction_model_path",
            "updated_model_path",
            "diagnostics_report_path",
            "archive_diagnostics_report_path",
            "latest_diagnostics_report_path",
            "interpretation_report_path",
            "archive_interpretation_report_path",
            "latest_interpretation_report_path",
            "interpretation_top_features_path",
            "predictions_path",
            "actual_vs_prediction_plot_path",
            "latest_actual_vs_prediction_plot_path",
            "prediction_timeline_plot_path",
            "latest_prediction_timeline_plot_path",
            "archive_prediction_timeline_plot_path",
            "residuals_plot_path",
            "latest_residuals_plot_path",
            "model_path",
        ]
        return [
            *[column for column in preferred_columns if column in rows.columns],
            *[column for column in rows.columns if column not in preferred_columns],
        ]
