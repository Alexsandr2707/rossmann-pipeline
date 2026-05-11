from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import pearsonr

from app.config import Config


class ModelTrainer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def metrics_history_path(self) -> Path:
        return self.config.paths.model_metrics_history_path

    @property
    def best_model_path(self) -> Path:
        return self.config.paths.best_model_path

    @property
    def current_model_path(self) -> Path:
        return self.config.paths.models_dir / "current_model.pkl"

    def train_on_processed_data(
        self,
        latest_processed_path: Path,
        batch_metadata: dict[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        if self.config.model.update_strategy == "incremental":
            return self._incremental_update(latest_processed_path, batch_metadata)
        if self.config.model.update_strategy == "refit":
            return self._refit_on_processed_data(latest_processed_path, batch_metadata)
        raise ValueError(
            f"Unsupported update strategy: {self.config.model.update_strategy}"
        )

    def _refit_on_processed_data(
        self,
        latest_processed_path: Path,
        batch_metadata: dict[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        dataset = self._load_training_dataset()
        features, target = self._build_features_and_target(dataset)

        if len(features) < 10:
            raise ValueError("Not enough rows for model training.")

        x_train, x_valid, y_train, y_valid = self._time_holdout_split(
            features,
            target,
        )

        results: list[dict[str, Any]] = []
        fitted_models: dict[str, SklearnPipeline] = {}

        for model_name in self._models_to_train():
            estimator = self._make_estimator(model_name)
            pipeline = SklearnPipeline(
                steps=[
                    ("preprocessor", self._make_feature_preprocessor(features)),
                    ("model", estimator),
                ]
            )
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_valid)
            metrics = self._calculate_metrics(y_valid, predictions)
            metrics.update(
                {
                    "model_name": model_name,
                    "batch_index": int(batch_metadata.get("batch_index", -1)),
                    "latest_processed_path": str(latest_processed_path),
                    "train_rows": int(len(x_train)),
                    "valid_rows": int(len(x_valid)),
                    "training_mode": self.config.model.training_mode,
                    "update_strategy": self.config.model.update_strategy,
                    "initial_training": False,
                    "training_rows_total": int(len(features)),
                }
            )
            results.append(metrics)
            fitted_models[model_name] = pipeline

        best_metrics = min(
            results,
            key=lambda item: item[self.config.model.primary_metric],
        )
        best_model_name = str(best_metrics["model_name"])
        best_pipeline = fitted_models[best_model_name]

        model_path = self._save_model_version(best_pipeline, best_metrics)
        self._save_best_model(best_pipeline, best_metrics, model_path)
        self._append_metrics(results, model_path)

        self.logger.info("Best model for update: %s", best_metrics)
        return model_path, best_metrics

    def _incremental_update(
        self,
        latest_processed_path: Path,
        batch_metadata: dict[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        model_name = self.config.model.selected_model
        if model_name != self.config.model.incremental_model:
            raise ValueError(
                "Incremental update requires selected_model to match "
                "incremental_model."
            )
        if model_name != "sgd_regression":
            raise ValueError("Only sgd_regression supports incremental update now.")

        is_initial_training = not self.current_model_path.exists()
        training_dataset = self._incremental_training_dataset(
            latest_processed_path,
            is_initial_training,
        )
        features, target = self._build_features_and_target(training_dataset)
        if len(features) < 10:
            raise ValueError("Not enough rows for incremental model update.")

        x_train, x_valid, y_train, y_valid = self._time_holdout_split(
            features,
            target,
        )

        if not is_initial_training:
            payload = joblib.load(self.current_model_path)
            if payload.get("model_name") != model_name:
                raise ValueError(
                    f"Current model is {payload.get('model_name')}, expected {model_name}."
                )
            pipeline = payload["pipeline"]
            preprocessor = pipeline.named_steps["preprocessor"]
            estimator = pipeline.named_steps["model"]
            x_train_transformed = preprocessor.transform(x_train)
            x_valid_transformed = preprocessor.transform(x_valid)
        else:
            preprocessor = self._make_feature_preprocessor(features)
            estimator = self._make_estimator(model_name)
            x_train_transformed = preprocessor.fit_transform(x_train)
            x_valid_transformed = preprocessor.transform(x_valid)
            pipeline = SklearnPipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", estimator),
                ]
            )

        estimator.partial_fit(x_train_transformed, y_train)
        predictions = estimator.predict(x_valid_transformed)
        metrics = self._calculate_metrics(y_valid, predictions)
        metrics.update(
            {
                "model_name": model_name,
                "batch_index": int(batch_metadata.get("batch_index", -1)),
                "latest_processed_path": str(latest_processed_path),
                "train_rows": int(len(x_train)),
                "valid_rows": int(len(x_valid)),
                "training_mode": self.config.model.training_mode,
                "update_strategy": self.config.model.update_strategy,
                "initial_training": bool(is_initial_training),
                "training_rows_total": int(len(features)),
            }
        )

        model_path = self._save_model_version(pipeline, metrics)
        self._save_current_model(pipeline, metrics, model_path, model_name)
        self._save_best_model(pipeline, metrics, model_path)
        self._append_metrics([metrics], model_path)

        self.logger.info("Incremental model updated: %s", metrics)
        return model_path, metrics

    def _incremental_training_dataset(
        self,
        latest_processed_path: Path,
        is_initial_training: bool,
    ) -> pd.DataFrame:
        if not is_initial_training:
            return self._load_processed_batch(latest_processed_path)

        dataset = self._load_training_dataset()
        max_rows = self.config.model.initial_training_max_rows
        if max_rows > 0 and len(dataset) > max_rows:
            dataset = dataset.tail(max_rows).copy()
        return dataset.reset_index(drop=True)

    def _load_training_dataset(self) -> pd.DataFrame:
        paths = sorted(self.config.paths.processed_data_dir.glob("batch_*_processed.csv"))
        if not paths:
            raise FileNotFoundError("No processed batches found for training.")

        frames = [pd.read_csv(path) for path in paths]
        dataset = pd.concat(frames, ignore_index=True)
        return self._sort_by_time(dataset)

    def _load_processed_batch(self, processed_path: Path) -> pd.DataFrame:
        dataset = pd.read_csv(processed_path)
        return self._sort_by_time(dataset)

    def _sort_by_time(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset[self.config.data.time_column] = pd.to_datetime(
            dataset[self.config.data.time_column],
            errors="coerce",
        )
        return dataset.sort_values(self.config.data.time_column).reset_index(drop=True)

    def _time_holdout_split(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        train_size = max(int(len(features) * 0.8), 1)
        if train_size >= len(features):
            train_size = len(features) - 1
        return (
            features.iloc[:train_size],
            features.iloc[train_size:],
            target.iloc[:train_size],
            target.iloc[train_size:],
        )

    def _build_features_and_target(
        self,
        dataset: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        target_column = self.config.data.target_column
        if target_column not in dataset.columns:
            raise ValueError(f"Target column is missing: {target_column}")

        target = pd.to_numeric(dataset[target_column], errors="coerce")
        training = dataset[target.notna()].copy()
        target = target[target.notna()]

        features = pd.DataFrame(index=training.index)

        for column in self.config.data_schema.numeric_columns:
            if column in training.columns:
                features[column] = pd.to_numeric(training[column], errors="coerce")

        indicator_column = (
            f"{self.config.data.target_column}"
            f"{self.config.target_preprocessing.missing_indicator_suffix}"
        )
        if indicator_column in training.columns:
            features[indicator_column] = pd.to_numeric(
                training[indicator_column],
                errors="coerce",
            )

        for column in self.config.data_schema.categorical_columns:
            if column in training.columns:
                categorical = training[column]
                features[column] = (
                    categorical.astype("string")
                    .astype("object")
                    .where(categorical.notna(), np.nan)
                )

        time_column = self.config.data.time_column
        if time_column in training.columns:
            parsed_time = pd.to_datetime(training[time_column], errors="coerce")
            features[f"{time_column}_year"] = parsed_time.dt.year
            features[f"{time_column}_month"] = parsed_time.dt.month
            features[f"{time_column}_quarter"] = parsed_time.dt.quarter

        return features.reset_index(drop=True), target.reset_index(drop=True)

    def _make_feature_preprocessor(self, features: pd.DataFrame) -> ColumnTransformer:
        numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
        categorical_features = [
            column for column in features.columns if column not in numeric_features
        ]

        numeric_pipeline = SklearnPipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = SklearnPipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_features),
                ("categorical", categorical_pipeline, categorical_features),
            ],
            remainder="drop",
        )

    def _make_estimator(self, model_name: str) -> Any:
        random_seed = self.config.project.random_seed
        model_parameters = self._model_parameters(model_name)
        if model_name == "linear_regression":
            return LinearRegression(**model_parameters)
        if model_name == "elastic_net_regression":
            return ElasticNet(
                **self._with_random_state(model_parameters, random_seed)
            )
        if model_name == "knn_regression":
            return KNeighborsRegressor(**model_parameters)
        if model_name == "decision_tree_regression":
            return DecisionTreeRegressor(
                **self._with_random_state(model_parameters, random_seed)
            )
        if model_name == "random_forest_regression":
            return RandomForestRegressor(
                **self._with_random_state(model_parameters, random_seed)
            )
        if model_name == "sgd_regression":
            return SGDRegressor(
                **self._with_random_state(model_parameters, random_seed)
            )
        raise ValueError(f"Unsupported model name: {model_name}")

    def _models_to_train(self) -> tuple[str, ...]:
        if self.config.model.training_mode == "all":
            return self.config.model.candidate_models
        if self.config.model.training_mode == "single":
            selected_model = self.config.model.selected_model
            if selected_model not in self.config.model.candidate_models:
                raise ValueError(
                    f"Selected model is not in candidate_models: {selected_model}"
                )
            return (selected_model,)
        raise ValueError(f"Unsupported training mode: {self.config.model.training_mode}")

    def _model_parameters(self, model_name: str) -> dict[str, Any]:
        return dict(self.config.model.model_parameters.get(model_name, {}))

    def _with_random_state(
        self,
        model_parameters: dict[str, Any],
        random_seed: int,
    ) -> dict[str, Any]:
        parameters = dict(model_parameters)
        parameters.setdefault("random_state", random_seed)
        return parameters

    def _calculate_metrics(
        self,
        y_true: pd.Series,
        predictions: np.ndarray,
    ) -> dict[str, float]:
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)
        y_true_array = y_true.to_numpy()
        pearson_corr, pearson_p_value = self._pearson_metrics(
            y_true_array,
            predictions,
        )
        denominator = np.abs(y_true_array) + np.abs(predictions)
        smape_values = np.divide(
            2.0 * np.abs(predictions - y_true_array),
            denominator,
            out=np.zeros_like(predictions, dtype=float),
            where=denominator != 0,
        )
        smape = smape_values.mean()
        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "smape": float(smape),
            "pearson_corr": pearson_corr,
            "pearson_p_value": pearson_p_value,
        }

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
        model_path = self.config.paths.models_dir / (
            f"model_v{batch_index:04d}_{model_name}.pkl"
        )
        joblib.dump({"pipeline": pipeline, "metrics": metrics}, model_path)
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
        }
        if self.best_model_path.exists():
            current = joblib.load(self.best_model_path)
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
        self.current_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": pipeline,
                "metrics": metrics,
                "model_path": str(model_path),
                "model_name": model_name,
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
        rows["is_best_in_update"] = rows[self.config.model.primary_metric] == rows[
            self.config.model.primary_metric
        ].min()
        rows = rows[self._metrics_columns(rows)]
        rows.to_csv(
            self.metrics_history_path,
            mode="a",
            header=not self.metrics_history_path.exists(),
            index=False,
        )

    def _metrics_columns(self, rows: pd.DataFrame) -> list[str]:
        preferred_columns = [
            "model_name",
            "batch_index",
            "rmse",
            "mae",
            "r2",
            "smape",
            "pearson_corr",
            "pearson_p_value",
            "train_rows",
            "valid_rows",
            "is_best_in_update",
            "training_mode",
            "update_strategy",
            "initial_training",
            "training_rows_total",
            "latest_processed_path",
            "model_path",
        ]
        return [
            *[column for column in preferred_columns if column in rows.columns],
            *[column for column in rows.columns if column not in preferred_columns],
        ]
