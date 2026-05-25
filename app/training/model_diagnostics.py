from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import Config
from app.visualization import (
    write_histogram_svg,
    write_line_chart_svg,
    write_scatter_svg,
    write_time_series_svg,
)


class ModelDiagnosticsWriter:
    _METRIC_HISTORY_SPECS: tuple[tuple[str, str], ...] = (
        ("rmse", "RMSE"),
        ("mae", "MAE"),
        ("smape", "SMAPE"),
        ("r2", "R2"),
    )
    _LATEST_REPORT_NAME = "model_diagnostics_latest.md"
    _ARCHIVE_REPORT_DIR = "archive/model_diagnostics"
    _LATEST_PLOT_NAMES = {
        "actual_vs_prediction": "actual_vs_prediction.svg",
        "prediction_timeline": "prediction_timeline.svg",
        "residuals": "residuals.svg",
    }

    def __init__(self, config: Config) -> None:
        self.config = config

    def _model_figures_dir(self) -> Path:
        return self.config.paths.reports_dir / "figures" / "model"

    def _archive_model_figures_dir(self) -> Path:
        return self.config.paths.reports_dir / "figures" / "archive" / "model_diagnostics"

    def _history_figures_dir(self) -> Path:
        return self.config.paths.reports_dir / "figures" / "history"

    def _archive_legacy_model_figures(self) -> None:
        source_dir = self._model_figures_dir()
        if not source_dir.exists():
            return
        target_dir = self._archive_model_figures_dir() / "legacy"
        legacy_paths = list(source_dir.glob("model_diagnostics_batch_*.svg"))
        if not legacy_paths:
            return
        target_dir.mkdir(parents=True, exist_ok=True)
        for figure_path in legacy_paths:
            target_path = target_dir / figure_path.name
            if target_path.exists():
                target_path.unlink()
            shutil.move(str(figure_path), str(target_path))

    def write_model_diagnostics(
        self,
        y_true: pd.Series,
        predictions: np.ndarray,
        metrics: dict[str, Any],
        estimator: Any,
        transformed_features: Any,
        timeline_dates: pd.Series | None = None,
    ) -> dict[str, str]:
        self.config.paths.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.config.paths.reports_dir.mkdir(parents=True, exist_ok=True)
        self._model_figures_dir().mkdir(parents=True, exist_ok=True)
        self._archive_model_figures_dir().mkdir(parents=True, exist_ok=True)
        self._history_figures_dir().mkdir(parents=True, exist_ok=True)
        self._archive_legacy_model_figures()

        model_name = str(metrics["model_name"])
        batch_index = int(metrics["batch_index"])
        base_name = str(
            metrics.get(
                "predictions_file_stem",
                f"model_diagnostics_batch_{batch_index:04d}_{model_name}",
            )
        )
        predictions_path = self.config.paths.predictions_dir / f"{base_name}.csv"
        diagnostics_base_name = f"model_diagnostics_batch_{batch_index:04d}_{model_name}"
        latest_report_path = self.config.paths.reports_dir / self._LATEST_REPORT_NAME
        archive_report_path = (
            self.config.paths.reports_dir
            / self._ARCHIVE_REPORT_DIR
            / f"{diagnostics_base_name}.md"
        )
        model_figures_dir = self._model_figures_dir()
        archive_figures_dir = self._archive_model_figures_dir() / diagnostics_base_name
        archive_figures_dir.mkdir(parents=True, exist_ok=True)

        y_true_array = y_true.to_numpy(dtype=float)
        prediction_array = np.asarray(predictions, dtype=float)
        diagnostics = pd.DataFrame(
            {
                "actual": y_true_array,
                "prediction": prediction_array,
                "residual": y_true_array - prediction_array,
            }
        )
        if timeline_dates is not None:
            diagnostics.insert(
                0,
                "date",
                pd.to_datetime(timeline_dates, errors="coerce")
                .dt.strftime("%Y-%m-%d")
                .to_numpy(),
            )

        archive_actual_vs_pred_path = (
            archive_figures_dir / self._LATEST_PLOT_NAMES["actual_vs_prediction"]
        )
        archive_prediction_timeline_path = (
            archive_figures_dir / self._LATEST_PLOT_NAMES["prediction_timeline"]
        )
        archive_residuals_path = archive_figures_dir / self._LATEST_PLOT_NAMES["residuals"]

        actual_vs_pred_path = (
            model_figures_dir / self._LATEST_PLOT_NAMES["actual_vs_prediction"]
        )
        prediction_timeline_path = (
            model_figures_dir / self._LATEST_PLOT_NAMES["prediction_timeline"]
        )
        residuals_path = model_figures_dir / self._LATEST_PLOT_NAMES["residuals"]
        write_scatter_svg(
            actual_vs_pred_path,
            np.log1p(np.clip(y_true_array, 0.0, None)),
            np.log1p(np.clip(prediction_array, 0.0, None)),
            "Actual vs predicted sales, log1p",
            "actual log1p",
            "predicted log1p",
        )
        write_scatter_svg(
            archive_actual_vs_pred_path,
            np.log1p(np.clip(y_true_array, 0.0, None)),
            np.log1p(np.clip(prediction_array, 0.0, None)),
            "Actual vs predicted sales, log1p",
            "actual log1p",
            "predicted log1p",
        )
        write_histogram_svg(
            residuals_path,
            y_true_array - prediction_array,
            "Residual distribution",
            "actual - prediction",
        )
        write_histogram_svg(
            archive_residuals_path,
            y_true_array - prediction_array,
            "Residual distribution",
            "actual - prediction",
        )
        timeline_plot_path = self._write_prediction_timeline(
            prediction_timeline_path,
            timeline_dates,
            y_true_array,
            prediction_array,
            str(metrics["model_name"]),
        )
        archive_timeline_plot_path = self._write_prediction_timeline(
            archive_prediction_timeline_path,
            timeline_dates,
            y_true_array,
            prediction_array,
            str(metrics["model_name"]),
        )
        diagnostics.to_csv(predictions_path, index=False)
        self._write_diagnostics_markdown(
            latest_report_path,
            metrics,
            predictions_path,
            [
                actual_vs_pred_path,
                timeline_plot_path,
                residuals_path,
            ],
        )
        self._write_diagnostics_markdown(
            archive_report_path,
            metrics,
            predictions_path,
            [
                archive_actual_vs_pred_path,
                archive_timeline_plot_path,
                archive_residuals_path,
            ],
        )

        output_paths = {
            "diagnostics_report_path": str(latest_report_path),
            "archive_diagnostics_report_path": str(archive_report_path),
            "latest_diagnostics_report_path": str(latest_report_path),
            "predictions_path": str(predictions_path),
            "actual_vs_prediction_plot_path": str(actual_vs_pred_path),
            "latest_actual_vs_prediction_plot_path": str(actual_vs_pred_path),
            "residuals_plot_path": str(residuals_path),
            "latest_residuals_plot_path": str(residuals_path),
        }
        if timeline_plot_path is not None:
            output_paths.update(
                {
                    "prediction_timeline_plot_path": str(timeline_plot_path),
                    "latest_prediction_timeline_plot_path": str(timeline_plot_path),
                }
            )
        if archive_timeline_plot_path is not None:
            output_paths["archive_prediction_timeline_plot_path"] = str(
                archive_timeline_plot_path
            )
        return output_paths

    def _write_prediction_timeline(
        self,
        path: Path,
        timeline_dates: pd.Series | None,
        y_true: np.ndarray,
        predictions: np.ndarray,
        model_name: str,
    ) -> Path | None:
        if timeline_dates is None:
            return None

        daily = pd.DataFrame(
            {
                "date": pd.to_datetime(timeline_dates, errors="coerce"),
                "actual": y_true,
                "prediction": predictions,
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
            path,
            day_offsets,
            {
                "actual sales": (day_offsets, daily["actual"].to_numpy(dtype=float)),
                f"{model_name} prediction": (
                    day_offsets,
                    daily["prediction"].to_numpy(dtype=float),
                ),
            },
            "Training validation timeline: actual vs model prediction",
            "Date",
            "Daily sales",
            format_day_offset,
        )
        return path

    def write_metrics_history_plot(self, metrics_history_path: Path) -> None:
        if not metrics_history_path.exists():
            return
        history = pd.read_csv(metrics_history_path)
        if history.empty or "batch_index" not in history.columns:
            return

        self._history_figures_dir().mkdir(parents=True, exist_ok=True)
        plot_specs: list[tuple[str, Path, str]] = []
        for metric_name, metric_label in self._METRIC_HISTORY_SPECS:
            if metric_name not in history.columns:
                continue
            clean = history[["batch_index", metric_name]].dropna()
            if clean.empty:
                continue
            x_values = clean["batch_index"].to_numpy(dtype=float)
            y_values = clean[metric_name].to_numpy(dtype=float)
            y_min, y_max = self._robust_history_range(y_values)
            clipped_y_values = np.clip(y_values, y_min, y_max)
            plot_path = (
                self._history_figures_dir()
                / f"model_metrics_history_{metric_name}.svg"
            )
            write_line_chart_svg(
                plot_path,
                {metric_label: (x_values, clipped_y_values)},
                f"{metric_label} history",
                "batch",
                metric_label,
            )
            plot_specs.append((metric_label, plot_path, metric_name))

        if plot_specs:
            self._write_metrics_history_index(plot_specs)
        legacy_plot_path = self._history_figures_dir() / "model_metrics_history.svg"
        if legacy_plot_path.exists():
            legacy_plot_path.unlink()

    def _robust_history_range(self, values: np.ndarray) -> tuple[float, float]:
        finite_values = values[np.isfinite(values)]
        if len(finite_values) == 0:
            return 0.0, 1.0
        if len(finite_values) < 6:
            return self._range_with_padding(finite_values)

        lower, upper = np.quantile(finite_values, [0.05, 0.95])
        if np.isclose(lower, upper):
            return self._range_with_padding(finite_values)

        data_min = float(np.min(finite_values))
        data_max = float(np.max(finite_values))
        span = float(upper - lower)
        padding = max(span * 0.15, abs(lower) * 0.02, abs(upper) * 0.02, 1e-6)
        lower = max(data_min, float(lower - padding))
        upper = min(data_max, float(upper + padding))
        if np.isclose(lower, upper):
            return self._range_with_padding(finite_values)
        return lower, upper

    def _range_with_padding(self, values: np.ndarray) -> tuple[float, float]:
        value_min = float(np.min(values))
        value_max = float(np.max(values))
        if np.isclose(value_min, value_max):
            padding = max(abs(value_min) * 0.1, 1.0)
            return value_min - padding, value_max + padding
        padding = (value_max - value_min) * 0.05
        return value_min - padding, value_max + padding

    def _write_metrics_history_index(
        self,
        plot_specs: list[tuple[str, Path, str]],
    ) -> None:
        index_path = self._history_figures_dir() / "model_metrics_history.md"
        lines = [
            "# Model metrics history",
            "",
            "Metrics are split into separate charts so each one keeps its own scale.",
            "Each chart uses a robust y-range to keep a single extreme batch from flattening the rest of the trend.",
            "",
        ]
        for metric_label, plot_path, metric_name in plot_specs:
            relative_plot_path = os.path.relpath(plot_path, index_path.parent)
            lines.extend(
                [
                    f"## {metric_label}",
                    "",
                    f"![{metric_name}]({relative_plot_path})",
                    "",
                ]
            )
        index_path.write_text("\n".join(lines), encoding="utf-8")

    def _write_diagnostics_markdown(
        self,
        report_path: Path,
        metrics: dict[str, Any],
        predictions_path: Path,
        plot_paths: list[Path | None],
    ) -> None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = self._diagnostics_markdown_lines(metrics, predictions_path, plot_paths, report_path)
        report_path.write_text("\n".join(lines), encoding="utf-8")

    def _diagnostics_markdown_lines(
        self,
        metrics: dict[str, Any],
        predictions_path: Path,
        plot_paths: list[Path | None],
        report_path: Path,
    ) -> list[str]:
        metric_names = [
            "rmse",
            "mae",
            "r2",
            "smape",
            "actual_positive_rate",
            "predicted_positive_rate",
            "prediction_mean",
            "target_mean",
            "pearson_corr",
        ]
        lines = [
            f"# Model diagnostics: {metrics['model_name']}",
            "",
            f"- Batch index: {metrics['batch_index']}",
            f"- Predictions CSV: `{predictions_path}`",
            "",
            "## Metrics",
            "",
        ]
        for name in metric_names:
            if name in metrics:
                lines.append(f"- {name}: {metrics[name]}")
        lines.extend(["", "## Plots", ""])
        for plot_path in plot_paths:
            if plot_path is None:
                continue
            relative_plot_path = os.path.relpath(plot_path, report_path.parent)
            lines.append(f"![{plot_path.stem}]({relative_plot_path})")
            lines.append("")
        return lines
