from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import Config

DEFAULT_TAIL_ROWS = 5
SUMMARY_DIR_NAME = "summary"
SUMMARY_FILE_NAME = "summary_latest.md"


def generate_summary_report(config: Config) -> Path:
    report_path = config.paths.reports_dir / SUMMARY_DIR_NAME / SUMMARY_FILE_NAME
    report_path.parent.mkdir(parents=True, exist_ok=True)

    batch_history = _read_history(config.paths.batch_metadata_path)
    data_quality_history = _read_history(config.paths.data_quality_history_path)
    model_history = _read_history(config.paths.model_metrics_history_path)

    lines: list[str] = [
        "# Pipeline summary",
        "",
        "## Latest batch metadata",
        "",
    ]
    lines.extend(_latest_batch_section(batch_history, report_path.parent))
    lines.extend(["", "## Latest data quality metrics", ""])
    lines.extend(_latest_data_quality_section(data_quality_history, report_path.parent))
    lines.extend(["", "## Model metrics trend", ""])
    lines.extend(_model_metrics_trend_section(model_history, report_path.parent))
    lines.extend(["", "## Source artifacts", ""])
    lines.extend(_artifact_links(config, report_path.parent))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _read_history(path: Path) -> list[dict[str, str]] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    except Exception:
        return None
    if not rows:
        return None
    return rows


def _latest_batch_section(
    history: list[dict[str, str]] | None,
    report_dir: Path,
) -> list[str]:
    if history is None:
        return [
            "No batch metadata history is available yet.",
            "",
            f"- Expected file: `{_relative_path(report_dir, Path('artifacts/batch_metadata_history.csv'))}`",
        ]

    latest_row = history[-1]
    fields = [
        "batch_index",
        "batch_path",
        "start_row",
        "end_row",
        "rows",
        "columns",
        "time_min",
        "time_max",
        "missing_part",
        "numeric_features",
        "categorical_features",
        "target_missing",
        "target_mean",
        "target_min",
        "target_max",
        "target_missing_strategy",
        "target_missing_indicator_column",
    ]
    lines = [f"- History rows: {len(history)}"]
    for field in fields:
        if field not in latest_row:
            continue
        lines.append(f"- {field}: {_format_value(latest_row[field])}")
    return lines


def _model_metrics_trend_section(
    history: list[dict[str, str]] | None,
    report_dir: Path,
) -> list[str]:
    if history is None:
        return [
            "No model metrics history is available yet.",
            "",
            f"- Expected file: `{_relative_path(report_dir, Path('artifacts/model_metrics_history.csv'))}`",
        ]

    columns = [
        "batch_index",
        "model_name",
        "rmse",
        "mae",
        "r2",
        "is_best_in_update",
    ]
    table = _tail_markdown_table(history, columns, tail_rows=DEFAULT_TAIL_ROWS)
    if table is None:
        return [
            "Model metrics history exists, but it does not contain the expected columns.",
        ]

    latest = history[-1]
    lines = [f"- Entries available: {len(history)}"]
    metric_chart_index = report_dir / Path("figures/history/model_metrics_history.md")
    if metric_chart_index.exists():
        lines.append(
            f"- Metric charts: `{_relative_path(report_dir, metric_chart_index)}`"
        )
    for field in ("rmse", "mae", "r2", "smape", "pearson_corr"):
        if field in latest:
            lines.append(f"- Latest {field}: {_format_value(latest[field])}")
    lines.extend(["", table])
    return lines


def _latest_data_quality_section(
    history: list[dict[str, str]] | None,
    report_dir: Path,
) -> list[str]:
    if history is None:
        return [
            "No data quality history is available yet.",
            "",
            f"- Expected file: `{_relative_path(report_dir, Path('artifacts/data_quality_history.csv'))}`",
        ]

    latest_row = history[-1]
    fields = [
        "batch_index",
        "stream_batch_index",
        "period_type",
        "batch_path",
        "rows",
        "columns",
        "missing_part",
        "duplicate_rows",
        "duplicate_part",
        "constant_columns",
        "schema_missing_columns",
        "schema_extra_columns",
        "numeric_outlier_part",
        "category_cardinality",
    ]
    lines = [f"- History rows: {len(history)}"]
    for field in fields:
        if field not in latest_row:
            continue
        lines.append(f"- {field}: {_format_value(latest_row[field])}")

    report_path = latest_row.get("eda_report_path", "")
    if report_path:
        lines.append(f"- EDA report: `{_relative_path(report_dir, Path(report_path))}`")

    table = _tail_markdown_table(
        history,
        [
            "batch_index",
            "rows",
            "missing_part",
            "duplicate_part",
            "numeric_outlier_part",
        ],
        tail_rows=DEFAULT_TAIL_ROWS,
    )
    if table is not None:
        lines.extend(["", table])
    return lines


def _artifact_links(config: Config, report_dir: Path) -> list[str]:
    artifact_paths = [
        config.paths.batch_metadata_path,
        config.paths.data_quality_history_path,
        config.paths.model_metrics_history_path,
    ]
    lines: list[str] = []
    for artifact_path in artifact_paths:
        relative = _relative_path(report_dir, artifact_path)
        if artifact_path.exists():
            lines.append(f"- [{artifact_path.name}]({relative})")
        else:
            lines.append(f"- {artifact_path.name}: `{relative}` (missing)")
    latest_reports = [
        report_dir.parent / "model_diagnostics_latest.md",
    ]
    for report_path in latest_reports:
        relative = _relative_path(report_dir, report_path)
        if report_path.exists():
            lines.append(f"- [{report_path.name}]({relative})")
        else:
            lines.append(f"- {report_path.name}: `{relative}` (missing)")
    return lines


def _tail_markdown_table(
    history: list[dict[str, str]],
    columns: list[str],
    tail_rows: int,
) -> str | None:
    selected_columns = [
        column for column in columns if any(column in row for row in history)
    ]
    if not selected_columns:
        return None

    subset = history[-tail_rows:]
    if not subset:
        return None

    rows = [
        {column: _format_value(row.get(column, "")) for column in selected_columns}
        for row in subset
    ]
    return pd.DataFrame(rows).to_markdown(index=False)


def _format_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (int, float)):
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    text = str(value).strip()
    if not text:
        return ""

    lower_text = text.lower()
    if lower_text in {"true", "false"}:
        return lower_text

    try:
        if any(char in lower_text for char in (".", "e")):
            number = float(text)
            return f"{number:.6g}"
        if text.lstrip("-").isdigit():
            return str(int(text))
    except ValueError:
        pass

    return text


def _relative_path(report_dir: Path, target: Path) -> str:
    return os.path.relpath(target, report_dir)
