from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from app.core.config import Config


class DataQualityAnalyzer:
    def __init__(self, config: Config) -> None:
        self.config = config

    def analyze_batch(
        self,
        dataset: pd.DataFrame,
        batch_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        batch_index = self._batch_index(batch_metadata)
        report_path = (
            self.config.paths.reports_dir
            / "archive"
            / "eda"
            / f"eda_batch_{batch_index:04d}.md"
        )
        latest_report_path = self.config.paths.reports_dir / "eda_latest.md"
        metrics = self._calculate_metrics(dataset, batch_metadata, report_path)
        metrics["latest_eda_report_path"] = str(latest_report_path)
        self._write_markdown_report(dataset, batch_metadata, metrics, report_path)
        latest_report_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(report_path, latest_report_path)
        self._append_history(metrics)
        return metrics

    def _calculate_metrics(
        self,
        dataset: pd.DataFrame,
        batch_metadata: Mapping[str, Any],
        report_path: Path,
    ) -> dict[str, Any]:
        expected_columns = self._expected_columns(batch_metadata)
        actual_columns = list(dataset.columns)
        duplicate_rows = int(dataset.duplicated().sum())
        constant_columns = [
            column
            for column in actual_columns
            if dataset[column].nunique(dropna=False) <= 1
        ]
        category_cardinality = self._category_cardinality(dataset)
        outliers = self._numeric_outliers(dataset)

        rows = int(len(dataset))
        metrics: dict[str, Any] = {
            "batch_index": self._batch_index(batch_metadata),
            "stream_batch_index": batch_metadata.get("stream_batch_index", ""),
            "period_type": batch_metadata.get("period_type", ""),
            "batch_path": batch_metadata.get("batch_path", ""),
            "eda_report_path": str(report_path),
            "rows": rows,
            "columns": int(dataset.shape[1]),
            "time_min": batch_metadata.get("time_min", ""),
            "time_max": batch_metadata.get("time_max", ""),
            "date_min": batch_metadata.get("date_min", ""),
            "date_max": batch_metadata.get("date_max", ""),
            "missing_part": self._safe_part(
                int(dataset.isna().sum().sum()),
                dataset.size,
            ),
            "duplicate_rows": duplicate_rows,
            "duplicate_part": self._safe_part(duplicate_rows, rows),
            "constant_columns": constant_columns,
            "schema_missing_columns": [
                column for column in expected_columns if column not in actual_columns
            ],
            "schema_extra_columns": [
                column for column in actual_columns if column not in expected_columns
            ],
            "numeric_outlier_part": outliers["numeric_outlier_part"],
            "numeric_outlier_columns": outliers["numeric_outlier_columns"],
            "category_cardinality": category_cardinality,
        }
        return metrics

    def _expected_columns(self, batch_metadata: Mapping[str, Any]) -> list[str]:
        schema_context = self._schema_context(batch_metadata)
        if schema_context == "raw_inference":
            return self._expected_raw_inference_columns()
        return self._expected_raw_training_columns()

    def _schema_context(self, batch_metadata: Mapping[str, Any]) -> str:
        explicit_context = batch_metadata.get("schema_context")
        if explicit_context:
            return str(explicit_context)

        period_type = str(batch_metadata.get("period_type", "")).lower()
        if period_type == "inference":
            return "raw_inference"
        return "raw_training"

    def _expected_raw_training_columns(self) -> list[str]:
        columns = [
            *self.config.data_schema.numeric_columns,
            *self.config.data_schema.categorical_columns,
            *self.config.data_schema.datetime_columns,
            *self.config.data_schema.service_columns,
            self.config.data.target_column,
        ]
        return list(dict.fromkeys(columns))

    def _expected_raw_inference_columns(self) -> list[str]:
        columns = [
            *self.config.data_schema.numeric_columns,
            *self.config.data_schema.categorical_columns,
            *self.config.data_schema.datetime_columns,
            *self.config.data_schema.id_columns,
            *self.config.data_schema.service_columns,
        ]
        return list(dict.fromkeys(columns))

    def _category_cardinality(self, dataset: pd.DataFrame) -> dict[str, int]:
        return {
            column: int(dataset[column].nunique(dropna=True))
            for column in self.config.data_schema.categorical_columns
            if column in dataset.columns
        }

    def _numeric_outliers(self, dataset: pd.DataFrame) -> dict[str, Any]:
        numeric_columns = list(
            dict.fromkeys(
                [
                    *self.config.data_schema.numeric_columns,
                    self.config.data.target_column,
                ]
            )
        )
        outlier_cells = 0
        numeric_cells = 0
        column_parts: dict[str, float] = {}

        for column in numeric_columns:
            if column not in dataset.columns:
                continue

            values = pd.to_numeric(dataset[column], errors="coerce").dropna()
            if values.empty:
                column_parts[column] = 0.0
                continue

            q1 = float(values.quantile(0.25))
            q3 = float(values.quantile(0.75))
            iqr = q3 - q1
            if iqr == 0:
                column_outliers = 0
            else:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                column_outliers = int(((values < lower) | (values > upper)).sum())

            outlier_cells += column_outliers
            numeric_cells += int(values.shape[0])
            column_parts[column] = self._safe_part(
                column_outliers, int(values.shape[0])
            )

        return {
            "numeric_outlier_part": self._safe_part(outlier_cells, numeric_cells),
            "numeric_outlier_columns": column_parts,
        }

    def _append_history(self, metrics: dict[str, Any]) -> None:
        history_path = self.config.paths.data_quality_history_path
        history_path.parent.mkdir(parents=True, exist_ok=True)
        row = pd.DataFrame([self._serialize_metrics(metrics)])
        row.to_csv(
            history_path,
            mode="a",
            header=not history_path.exists(),
            index=False,
        )

    def _write_markdown_report(
        self,
        dataset: pd.DataFrame,
        batch_metadata: Mapping[str, Any],
        metrics: dict[str, Any],
        report_path: Path,
    ) -> None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# EDA / Data Quality batch {int(metrics['batch_index']):04d}",
            "",
            "## Batch metadata",
            "",
        ]
        for field in (
            "batch_index",
            "stream_batch_index",
            "period_type",
            "batch_path",
            "rows",
            "columns",
            "time_min",
            "time_max",
            "date_min",
            "date_max",
        ):
            value = metrics.get(field, batch_metadata.get(field, ""))
            lines.append(f"- {field}: {self._format_value(value)}")

        lines.extend(
            [
                "",
                "## Data quality metrics",
                "",
                f"- missing_part: {metrics['missing_part']:.6g}",
                f"- duplicate_rows: {metrics['duplicate_rows']}",
                f"- duplicate_part: {metrics['duplicate_part']:.6g}",
                f"- numeric_outlier_part: {metrics['numeric_outlier_part']:.6g}",
                "- constant_columns: "
                + self._format_sequence(metrics["constant_columns"]),
                "- schema_missing_columns: "
                + self._format_sequence(metrics["schema_missing_columns"]),
                "- schema_extra_columns: "
                + self._format_sequence(metrics["schema_extra_columns"]),
                "",
                "## Category cardinality",
                "",
            ]
        )
        lines.extend(
            self._mapping_table(
                metrics["category_cardinality"],
                "column",
                "unique_values",
            )
        )
        lines.extend(["", "## Numeric outliers by column", ""])
        lines.extend(
            self._mapping_table(
                metrics["numeric_outlier_columns"],
                "column",
                "outlier_part",
            )
        )
        lines.extend(["", "## Column profile", ""])
        lines.extend(self._column_profile_table(dataset))

        report_path.write_text("\n".join(lines), encoding="utf-8")

    def _column_profile_table(self, dataset: pd.DataFrame) -> list[str]:
        if dataset.empty and not len(dataset.columns):
            return ["No columns available."]

        rows = []
        for column in dataset.columns:
            rows.append(
                {
                    "column": column,
                    "dtype": str(dataset[column].dtype),
                    "missing_part": self._safe_part(
                        int(dataset[column].isna().sum()),
                        len(dataset),
                    ),
                    "unique_values": int(dataset[column].nunique(dropna=True)),
                }
            )
        return [self._records_to_markdown(rows)]

    def _mapping_table(
        self,
        values: Mapping[str, Any],
        key_name: str,
        value_name: str,
    ) -> list[str]:
        if not values:
            return ["No values available."]

        rows = [{key_name: key, value_name: value} for key, value in values.items()]
        return [self._records_to_markdown(rows)]

    def _serialize_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        serialized: dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, (dict, list, tuple)):
                serialized[key] = json.dumps(value, ensure_ascii=True, sort_keys=True)
            else:
                serialized[key] = value
        return serialized

    def _batch_index(self, batch_metadata: Mapping[str, Any]) -> int:
        value = batch_metadata.get(
            "stream_batch_index",
            batch_metadata.get("batch_index", 0),
        )
        if value in (None, ""):
            return 0
        return int(value)

    def _safe_part(self, numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return float(numerator) / float(denominator)

    def _format_sequence(self, values: Any) -> str:
        if not values:
            return "none"
        return ", ".join(str(value) for value in values)

    def _format_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return self._escape_markdown(str(value))

    def _records_to_markdown(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return ""
        frame = pd.DataFrame(rows)
        return frame.to_markdown(index=False, floatfmt=".6g")

    def _escape_markdown(self, value: str) -> str:
        return value.replace("|", "\\|")
