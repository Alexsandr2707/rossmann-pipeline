from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import Config


class DataQualityAnalyzer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_and_clean_batch(
        self,
        batch_path: Path,
        batch_metadata: dict[str, Any],
    ) -> tuple[Path, dict[str, Any], Path]:
        batch = self._load_batch(batch_path)
        cleaned = self._clean_batch(batch)
        processed_path = self._processed_batch_path(batch_path)
        cleaned.to_csv(processed_path, index=False)

        metrics = self._build_quality_metrics(
            batch=batch,
            cleaned=cleaned,
            batch_path=batch_path,
            processed_path=processed_path,
            batch_metadata=batch_metadata,
        )
        report_path = self._report_path(metrics)
        metrics["eda_report_path"] = str(report_path)
        self._write_eda_report(batch, cleaned, metrics, report_path)
        self._append_quality_metrics(metrics)

        self.logger.info("Data quality metrics: %s", metrics)
        self.logger.info("Processed batch saved to %s", processed_path)
        self.logger.info("EDA report saved to %s", report_path)
        return processed_path, metrics, report_path

    def _load_batch(self, batch_path: Path) -> pd.DataFrame:
        dtype = {
            column: "string"
            for column in (
                *self.config.data_schema.categorical_columns,
                *self.config.data_schema.datetime_columns,
                *self.config.data_schema.id_columns,
                *self.config.data_schema.service_columns,
            )
        }
        batch = pd.read_csv(batch_path, dtype=dtype)

        for column in self.config.data_schema.numeric_columns:
            if column in batch.columns:
                batch[column] = pd.to_numeric(batch[column], errors="coerce")

        target_column = self.config.data.target_column
        if target_column in batch.columns:
            batch[target_column] = pd.to_numeric(batch[target_column], errors="coerce")

        indicator_column = self._target_missing_indicator_column
        if indicator_column in batch.columns:
            batch[indicator_column] = pd.to_numeric(
                batch[indicator_column],
                errors="coerce",
            ).fillna(0).astype(int)

        return batch

    @property
    def _target_missing_indicator_column(self) -> str:
        return (
            f"{self.config.data.target_column}"
            f"{self.config.target_preprocessing.missing_indicator_suffix}"
        )

    def _clean_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        cleaned = batch.drop_duplicates().copy()
        row_missing_part = cleaned.isna().mean(axis=1)
        cleaned = cleaned[row_missing_part <= self.config.quality.max_missing_part].copy()
        return cleaned

    def _build_quality_metrics(
        self,
        batch: pd.DataFrame,
        cleaned: pd.DataFrame,
        batch_path: Path,
        processed_path: Path,
        batch_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        duplicate_rows = int(batch.duplicated().sum())
        duplicate_part = float(duplicate_rows / len(batch)) if len(batch) else 0.0
        analysis_columns = self._analysis_columns(batch)
        analysis_batch = batch[analysis_columns]
        missing_by_column = analysis_batch.isna().mean()
        high_missing_columns = missing_by_column[
            missing_by_column > self.config.quality.max_missing_part
        ].index.tolist()
        constant_columns = [
            column
            for column in analysis_columns
            if batch[column].nunique(dropna=False) <= 1
        ]
        required_columns = [
            self.config.data.time_column,
            self.config.data.target_column,
        ]
        schema_missing_columns = [
            column for column in required_columns if column not in batch.columns
        ]

        numeric = analysis_batch.select_dtypes(include=["number"])
        categorical = analysis_batch.select_dtypes(
            include=["object", "category", "string"]
        )
        target = pd.to_numeric(
            batch.get(self.config.data.target_column, pd.Series(dtype="float64")),
            errors="coerce",
        )

        quality_passed = (
            float(analysis_batch.isna().mean().mean())
            <= self.config.quality.max_missing_part
            and duplicate_part <= self.config.quality.max_duplicate_part
            and not schema_missing_columns
        )

        metrics: dict[str, Any] = {
            "batch_index": int(batch_metadata.get("batch_index", -1)),
            "batch_path": str(batch_path),
            "processed_path": str(processed_path),
            "rows_before": int(len(batch)),
            "rows_after": int(len(cleaned)),
            "rows_removed": int(len(batch) - len(cleaned)),
            "columns": int(batch.shape[1]),
            "analysis_columns": int(len(analysis_columns)),
            "excluded_columns": ";".join(self._excluded_columns(batch)),
            "missing_part": float(analysis_batch.isna().mean().mean())
            if len(batch) and len(analysis_columns)
            else 0.0,
            "max_column_missing_part": float(missing_by_column.max())
            if len(missing_by_column)
            else 0.0,
            "duplicate_rows": duplicate_rows,
            "duplicate_part": duplicate_part,
            "constant_columns_count": int(len(constant_columns)),
            "constant_columns": ";".join(constant_columns),
            "high_missing_columns": ";".join(high_missing_columns),
            "schema_missing_columns": ";".join(schema_missing_columns),
            "numeric_columns": int(numeric.shape[1]),
            "categorical_columns": int(categorical.shape[1]),
            "outlier_part": self._outlier_part(numeric),
            "max_category_cardinality": self._max_category_cardinality(categorical),
            "target_mean": self._target_stat(target, "mean"),
            "target_median": self._target_stat(target, "median"),
            "target_std": self._target_stat(target, "std"),
            "target_min": self._target_stat(target, "min"),
            "target_max": self._target_stat(target, "max"),
            "target_q05": self._target_quantile(target, 0.05),
            "target_q95": self._target_quantile(target, 0.95),
            "target_zero_part": float((target == 0).mean()) if len(target) else 0.0,
            "quality_passed": bool(quality_passed),
            "eda_report_path": None,
        }
        return metrics

    def _processed_batch_path(self, batch_path: Path) -> Path:
        self.config.paths.processed_data_dir.mkdir(parents=True, exist_ok=True)
        return self.config.paths.processed_data_dir / f"{batch_path.stem}_processed.csv"

    def _excluded_columns(self, batch: pd.DataFrame) -> list[str]:
        excluded = {
            self.config.data.target_column,
            self._target_missing_indicator_column,
            *self.config.data_schema.datetime_columns,
            *self.config.data_schema.id_columns,
            *self.config.data_schema.service_columns,
        }
        return [column for column in batch.columns if column in excluded]

    def _analysis_columns(self, batch: pd.DataFrame) -> list[str]:
        excluded = set(self._excluded_columns(batch))
        return [column for column in batch.columns if column not in excluded]

    def _append_quality_metrics(self, metrics: dict[str, Any]) -> None:
        history_path = self.config.paths.data_quality_history_path
        history_path.parent.mkdir(parents=True, exist_ok=True)
        row = pd.DataFrame([metrics])
        if history_path.exists():
            history = pd.read_csv(history_path)
            history = pd.concat([history, row], ignore_index=True, sort=False)
            history.to_csv(history_path, index=False)
            return

        row.to_csv(history_path, index=False)

    def _report_path(self, metrics: dict[str, Any]) -> Path:
        return self.config.paths.reports_dir / (
            f"eda_batch_{int(metrics['batch_index']):04d}.md"
        )

    def _write_eda_report(
        self,
        batch: pd.DataFrame,
        cleaned: pd.DataFrame,
        metrics: dict[str, Any],
        report_path: Path,
    ) -> Path:
        report_path.parent.mkdir(parents=True, exist_ok=True)

        analysis_batch = batch[self._analysis_columns(batch)]
        target = pd.to_numeric(batch[self.config.data.target_column], errors="coerce")
        numeric_summary = (
            analysis_batch.select_dtypes(include=["number"]).describe().transpose()
        )
        categorical_summary = self._categorical_summary(analysis_batch)

        lines = [
            f"# EDA batch {int(metrics['batch_index']):04d}",
            "",
            "## Quality",
            "",
            f"- Rows before cleaning: {metrics['rows_before']}",
            f"- Rows after cleaning: {metrics['rows_after']}",
            f"- Analysis columns: {metrics['analysis_columns']}",
            f"- Excluded columns: {metrics['excluded_columns']}",
            f"- Missing part: {metrics['missing_part']:.6f}",
            f"- Duplicate part: {metrics['duplicate_part']:.6f}",
            f"- Outlier part: {metrics['outlier_part']:.6f}",
            f"- Quality passed: {metrics['quality_passed']}",
            "",
            "## Target",
            "",
            f"- Mean: {target.mean():.6f}",
            f"- Median: {target.median():.6f}",
            f"- Std: {target.std():.6f}",
            f"- Min: {target.min():.6f}",
            f"- Max: {target.max():.6f}",
            f"- 5% quantile: {target.quantile(0.05):.6f}",
            f"- 95% quantile: {target.quantile(0.95):.6f}",
            "",
            "## Numeric Summary",
            "",
            f"```text\n{numeric_summary.to_string()}\n```"
            if not numeric_summary.empty
            else "No numeric columns.",
            "",
            "## Categorical Summary",
            "",
            f"```text\n{categorical_summary.to_string(index=False)}\n```"
            if not categorical_summary.empty
            else "No categorical columns.",
            "",
            "## Processed Output",
            "",
            f"- Processed batch: `{metrics['processed_path']}`",
            f"- Cleaned rows preview source shape: {cleaned.shape[0]} rows, {cleaned.shape[1]} columns",
            "",
        ]
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    def _categorical_summary(self, batch: pd.DataFrame) -> pd.DataFrame:
        rows = []
        categorical = batch.select_dtypes(include=["object", "category", "string"])
        for column in categorical.columns:
            values = batch[column].dropna()
            top_value = values.mode().iloc[0] if not values.mode().empty else ""
            rows.append(
                {
                    "column": column,
                    "missing_part": float(batch[column].isna().mean()),
                    "unique_values": int(batch[column].nunique(dropna=True)),
                    "top_value": str(top_value),
                    "top_value_count": int((batch[column] == top_value).sum())
                    if top_value != ""
                    else 0,
                }
            )
        return pd.DataFrame(rows)

    def _outlier_part(self, numeric: pd.DataFrame) -> float:
        if numeric.empty:
            return 0.0
        outlier_flags = []
        for column in numeric.columns:
            series = numeric[column].dropna()
            if series.empty:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_flags.append((numeric[column] < lower) | (numeric[column] > upper))
        if not outlier_flags:
            return 0.0
        flags = pd.concat(outlier_flags, axis=1).any(axis=1)
        return float(flags.mean())

    def _max_category_cardinality(self, categorical: pd.DataFrame) -> int:
        if categorical.empty:
            return 0
        return int(categorical.nunique(dropna=True).max())

    def _target_stat(self, target: pd.Series, name: str) -> float | None:
        clean = target.dropna()
        if clean.empty:
            return None
        return float(getattr(clean, name)())

    def _target_quantile(self, target: pd.Series, quantile: float) -> float | None:
        clean = target.dropna()
        if clean.empty:
            return None
        return float(clean.quantile(quantile))
