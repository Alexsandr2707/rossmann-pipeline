from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.config import Config
from app.dataset_loading import load_source_dataset
from app.feature_engineering import build_features_and_target
from app.model_training import ModelTrainer
from app.models import canonical_model_name
from app.period_splitting import DatePeriodSplit, rows_for_dates, split_date_periods
from app.preprocessing import DataPreprocessor
from app.visualization import write_time_series_svg


class OfflineModelEvaluator:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.trainer = ModelTrainer(config)

    def evaluate(self) -> Path:
        dataset = self._load_full_processed_dataset()
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
        x_train, y_train = build_features_and_target(initial_dataset, self.config)
        x_valid, y_valid = build_features_and_target(validation_dataset, self.config)
        validation_dates = self.trainer.target_dates(validation_dataset)

        rows: list[dict[str, Any]] = [
            self._regression_metrics(
                "zero_baseline",
                y_valid,
                np.zeros(len(y_valid), dtype=float),
            ),
            self._regression_metrics(
                "mean_baseline",
                y_valid,
                np.full(len(y_valid), float(y_train.mean()), dtype=float),
            ),
        ]
        predictions_by_model: dict[str, np.ndarray] = {}
        for model_name in self._model_names():
            row, predictions = self._evaluate_project_model(
                model_name,
                x_train,
                x_valid,
                y_train,
                y_valid,
            )
            rows.append(row)
            predictions_by_model[model_name] = predictions

        results = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
        self.config.paths.reports_dir.mkdir(parents=True, exist_ok=True)
        self.config.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        results_path = self.config.paths.artifacts_dir / "offline_model_evaluation.csv"
        report_path = self.config.paths.reports_dir / "offline_model_evaluation.md"
        chart_path = self._write_best_model_timeline_chart(
            results,
            validation_dates,
            y_valid,
            predictions_by_model,
        )
        results.to_csv(results_path, index=False)
        self._write_report(
            report_path=report_path,
            results_path=results_path,
            chart_path=chart_path,
            results=results,
            y_train=y_train,
            y_valid=y_valid,
            feature_columns=list(x_train.columns),
            split=split,
        )
        return report_path

    def _load_full_processed_dataset(self) -> pd.DataFrame:
        raw_dataset = load_source_dataset(self.config)
        return self.preprocessor.transform(raw_dataset)

    def _model_names(self) -> list[str]:
        configured = (
            *self.config.model.candidate_models,
            self.config.model.selected_model,
        )
        seen: set[str] = set()
        model_names: list[str] = []
        for model_name in configured:
            canonical_name = canonical_model_name(model_name)
            if canonical_name in seen:
                continue
            seen.add(canonical_name)
            model_names.append(canonical_name)
        return model_names

    def _evaluate_project_model(
        self,
        model_name: str,
        x_train: pd.DataFrame,
        x_valid: pd.DataFrame,
        y_train: pd.Series,
        y_valid: pd.Series,
    ) -> tuple[dict[str, Any], np.ndarray]:
        pipeline = self.trainer.fit_pipeline(x_train, y_train, model_name)
        predictions = np.clip(
            np.asarray(pipeline.predict(x_valid), dtype=float),
            0.0,
            None,
        )
        row = self._regression_metrics(model_name, y_valid, predictions)
        row["notes"] = (
            "Project regression model; Customers is excluded from features."
        )
        return row, predictions

    def _write_best_model_timeline_chart(
        self,
        results: pd.DataFrame,
        validation_dates: pd.Series,
        y_valid: pd.Series,
        predictions_by_model: dict[str, np.ndarray],
    ) -> Path | None:
        best_model_name = self._best_project_model_name(results, predictions_by_model)
        if best_model_name is None:
            return None

        chart_path = (
            self.config.paths.reports_dir
            / "figures"
            / "offline_evaluation"
            / "actual_vs_prediction_timeline.svg"
        )
        daily = pd.DataFrame(
            {
                "date": pd.to_datetime(validation_dates, errors="coerce"),
                "actual": y_valid.to_numpy(dtype=float),
                "prediction": predictions_by_model[best_model_name],
            }
        ).dropna(subset=["date", "actual", "prediction"])
        if daily.empty:
            return None

        daily["date"] = daily["date"].dt.floor("D")
        daily = (
            daily.groupby("date", as_index=False)[["actual", "prediction"]]
            .sum()
            .sort_values("date")
        )
        first_date = daily["date"].min()
        day_offsets = (daily["date"] - first_date).dt.days.to_numpy(dtype=float)

        def format_day_offset(value: float) -> str:
            return (first_date + pd.to_timedelta(round(value), unit="D")).strftime(
                "%Y-%m-%d"
            )

        write_time_series_svg(
            chart_path,
            day_offsets,
            {
                "actual sales": (day_offsets, daily["actual"].to_numpy(dtype=float)),
                f"{best_model_name} prediction": (
                    day_offsets,
                    daily["prediction"].to_numpy(dtype=float),
                ),
            },
            "Validation timeline: actual vs model prediction",
            "Date",
            "Daily sales",
            format_day_offset,
        )
        return chart_path

    def _best_project_model_name(
        self,
        results: pd.DataFrame,
        predictions_by_model: dict[str, np.ndarray],
    ) -> str | None:
        for model_name in results["model_name"]:
            if model_name in predictions_by_model:
                return str(model_name)
        return None

    def _regression_metrics(
        self,
        model_name: str,
        y_true: pd.Series,
        predictions: np.ndarray,
    ) -> dict[str, Any]:
        y_true_array = y_true.to_numpy(dtype=float)
        predictions = np.asarray(predictions, dtype=float)
        return {
            "model_name": model_name,
            "rmse": float(np.sqrt(mean_squared_error(y_true_array, predictions))),
            "mae": float(mean_absolute_error(y_true_array, predictions)),
            "r2": float(r2_score(y_true_array, predictions)),
            "smape": self._smape(y_true_array, predictions),
            "target_mean": float(y_true_array.mean()),
            "prediction_mean": float(predictions.mean()),
            "prediction_min": float(predictions.min()),
            "prediction_max": float(predictions.max()),
            "actual_zero_rate": float((y_true_array == 0).mean()),
            "prediction_zero_rate": float((predictions <= 0).mean()),
            **self._top_actual_mse_shares(y_true_array, predictions),
        }

    def _top_actual_mse_shares(
        self,
        y_true: np.ndarray,
        predictions: np.ndarray,
    ) -> dict[str, float]:
        squared_error = np.square(predictions - y_true)
        total_squared_error = float(squared_error.sum())
        if np.isclose(total_squared_error, 0.0):
            return {
                "top_10_actual_mse_share": 0.0,
                "top_100_actual_mse_share": 0.0,
                "top_500_actual_mse_share": 0.0,
            }

        descending_actual_indices = np.argsort(-y_true)
        shares = {}
        for top_n in (10, 100, 500):
            top_indices = descending_actual_indices[: min(top_n, len(y_true))]
            shares[f"top_{top_n}_actual_mse_share"] = float(
                squared_error[top_indices].sum() / total_squared_error
            )
        return shares

    def _smape(self, y_true: np.ndarray, predictions: np.ndarray) -> float:
        denominator = np.abs(y_true) + np.abs(predictions)
        smape_values = np.divide(
            2.0 * np.abs(predictions - y_true),
            denominator,
            out=np.zeros_like(predictions, dtype=float),
            where=denominator != 0,
        )
        return float(smape_values.mean())

    def _write_report(
        self,
        report_path: Path,
        results_path: Path,
        chart_path: Path | None,
        results: pd.DataFrame,
        y_train: pd.Series,
        y_valid: pd.Series,
        feature_columns: list[str],
        split: DatePeriodSplit,
    ) -> None:
        best = results.iloc[0]
        zero_baseline = results[results["model_name"] == "zero_baseline"].iloc[0]
        mean_baseline = results[results["model_name"] == "mean_baseline"].iloc[0]
        zero_improvement = self._relative_improvement(
            zero_baseline["rmse"],
            best["rmse"],
        )
        mean_improvement = self._relative_improvement(
            mean_baseline["rmse"],
            best["rmse"],
        )
        lines = [
            "# Rossmann offline model evaluation",
            "",
            f"- Results CSV: `{results_path}`",
            f"- Train rows: {len(y_train)}",
            f"- Validation rows: {len(y_valid)}",
            (
                "- Time split: `split_date_periods` `initial_train_period` "
                "for training, `validation_period` for evaluation"
            ),
            (
                "- Initial train dates: "
                f"{split.initial_dates.min().strftime('%Y-%m-%d')} to "
                f"{split.initial_dates.max().strftime('%Y-%m-%d')} "
                f"({len(split.initial_dates)} dates)"
            ),
            (
                "- Validation dates: "
                f"{split.validation_dates.min().strftime('%Y-%m-%d')} to "
                f"{split.validation_dates.max().strftime('%Y-%m-%d')} "
                f"({len(split.validation_dates)} dates)"
            ),
            f"- Target: `{self.config.data.target_column}`",
            f"- Time column: `{self.config.data.time_column}`",
            "- Leakage control: `Customers` is excluded from the feature set.",
            f"- Best RMSE model: `{best['model_name']}`",
            f"- Best RMSE improvement over zero baseline: {zero_improvement:.4%}",
            f"- Best RMSE improvement over mean baseline: {mean_improvement:.4%}",
            "",
            "## Feature Columns",
            "",
            ", ".join(f"`{column}`" for column in feature_columns),
            "",
            "## Results",
            "",
            self._markdown_table(results),
            "",
            *self._timeline_report_section(report_path, chart_path),
            "## Baselines",
            "",
            (
                "- `zero_baseline` predicts `0` sales for every validation row. "
                "It is a sanity check for the closed-store zero-sales case, not a "
                "competitive forecast."
            ),
            (
                "- `mean_baseline` predicts the average train-period sales for "
                "every validation row. The project model should beat it clearly."
            ),
            (
                "- The split is time-based: older rows train the model and newer "
                "rows emulate future validation."
            ),
        ]
        report_path.write_text("\n".join(lines), encoding="utf-8")

    def _timeline_report_section(
        self,
        report_path: Path,
        chart_path: Path | None,
    ) -> list[str]:
        if chart_path is None:
            return []
        relative_chart_path = chart_path.relative_to(report_path.parent)
        return [
            "## Validation Timeline",
            "",
            f"![Actual vs model prediction]({relative_chart_path})",
            "",
        ]

    def _relative_improvement(self, baseline: float, candidate: float) -> float:
        if np.isclose(float(baseline), 0.0):
            return 0.0
        return float((baseline - candidate) / baseline)

    def _markdown_table(self, dataset: pd.DataFrame) -> str:
        return dataset.fillna("").to_markdown(index=False, floatfmt=".6g")
