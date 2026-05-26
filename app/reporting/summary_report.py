from __future__ import annotations

import csv
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import Config
from app.reporting.prediction_history import (
    UPDATE_PREDICTION_TIMELINE_PATH,
    generate_update_prediction_timeline,
)

DEFAULT_TABLE_ROWS = 5
SUMMARY_FILE_NAME = "summary_latest.md"
SUMMARY_ARCHIVE_DIR_NAME = "archive/summary"


def generate_summary_report(config: Config, archive_context: str = "manual") -> Path:
    report_path = config.paths.reports_dir / SUMMARY_FILE_NAME
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        "\n".join(_summary_lines(config, report_path.parent)),
        encoding="utf-8",
    )

    archive_path = _summary_archive_path(config, archive_context)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(
        "\n".join(_summary_lines(config, archive_path.parent)),
        encoding="utf-8",
    )
    return report_path


def _summary_archive_path(config: Config, archive_context: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    context = re.sub(r"[^A-Za-z0-9_-]+", "_", archive_context).strip("_")
    suffix = f"_{context}" if context else ""
    return (
        config.paths.reports_dir
        / SUMMARY_ARCHIVE_DIR_NAME
        / f"summary_{timestamp}{suffix}.md"
    )


def _summary_lines(config: Config, report_dir: Path) -> list[str]:
    batch_history = _read_history(config.paths.batch_metadata_path)
    data_quality_history = _read_history(config.paths.data_quality_history_path)
    model_history = _read_history(config.paths.model_metrics_history_path)
    performance_history = _read_history(config.paths.performance_history_path)
    update_prediction_timeline_path = _resolve_update_prediction_timeline(config)

    lines: list[str] = ["# Pipeline summary", "", "## Project overview", ""]
    lines.extend(
        _project_overview_section(
            config,
            batch_history,
            data_quality_history,
            model_history,
            performance_history,
        )
    )
    lines.extend(["", "## Latest batch metadata", ""])
    lines.extend(_latest_batch_section(batch_history, report_dir))
    lines.extend(["", "## Latest data quality metrics", ""])
    lines.extend(_latest_data_quality_section(data_quality_history, report_dir))
    lines.extend(["", "## Model diagnostics", ""])
    lines.extend(_model_diagnostics_section(config, report_dir))
    lines.extend(["", "## Performance history", ""])
    lines.extend(_performance_section(performance_history, report_dir))
    lines.extend(["", "## Update prediction timeline", ""])
    lines.extend(
        _update_prediction_timeline_section(
            update_prediction_timeline_path,
            report_dir,
        )
    )
    lines.extend(["", "## Model metrics trend", ""])
    lines.extend(_model_metrics_trend_section(config, model_history, report_dir))
    lines.extend(["", "## Model interpretation", ""])
    lines.extend(_model_interpretation_section(config, report_dir))
    lines.extend(["", "## Model hyperparameters", ""])
    lines.extend(_hyperparameters_section(config))
    lines.extend(["", "## Offline evaluation", ""])
    lines.extend(_offline_evaluation_section(config, report_dir))
    lines.extend(["", "## Source artifacts", ""])
    lines.extend(_artifact_links(config, report_dir))
    return lines


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


def _resolve_update_prediction_timeline(config: Config) -> Path:
    generated_path = generate_update_prediction_timeline(config)
    if generated_path is not None:
        return generated_path
    return config.paths.reports_dir / UPDATE_PREDICTION_TIMELINE_PATH


def _project_overview_section(
    config: Config,
    batch_history: list[dict[str, str]] | None,
    data_quality_history: list[dict[str, str]] | None,
    model_history: list[dict[str, str]] | None,
    performance_history: list[dict[str, str]] | None,
) -> list[str]:
    latest_batch = batch_history[-1] if batch_history else {}
    latest_quality = data_quality_history[-1] if data_quality_history else {}
    latest_metrics = model_history[-1] if model_history else {}
    latest_performance = performance_history[-1] if performance_history else {}
    overview_fields = [
        ("project", config.project.name),
        ("latest_batch", latest_batch.get("batch_index", "not ready")),
        ("rows", latest_batch.get("rows", "not ready")),
        ("missing_part", latest_quality.get("missing_part", "not ready")),
        ("duplicate_part", latest_quality.get("duplicate_part", "not ready")),
        (
            "numeric_outlier_part",
            latest_quality.get("numeric_outlier_part", "not ready"),
        ),
        ("model", latest_metrics.get("model_name", config.model.selected_model)),
        ("rmse", latest_metrics.get("rmse", "not ready")),
        ("mae", latest_metrics.get("mae", "not ready")),
        ("r2", latest_metrics.get("r2", "not ready")),
        ("latest_operation", latest_performance.get("operation", "not ready")),
        ("latest_status", latest_performance.get("status", "not ready")),
    ]
    return [f"- {name}: {_format_value(value)}" for name, value in overview_fields]


def _latest_batch_section(
    history: list[dict[str, str]] | None,
    report_dir: Path,
) -> list[str]:
    if history is None:
        expected_path = Path("artifacts/batch_metadata_history.csv")
        return [
            "No batch metadata history is available yet.",
            "",
            f"- Expected file: `{_relative_path(report_dir, expected_path)}`",
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
        if field in latest_row:
            lines.append(f"- {field}: {_format_value(latest_row[field])}")
    return lines


def _latest_data_quality_section(
    history: list[dict[str, str]] | None,
    report_dir: Path,
) -> list[str]:
    if history is None:
        expected_path = Path("artifacts/data_quality_history.csv")
        return [
            "No data quality history is available yet.",
            "",
            f"- Expected file: `{_relative_path(report_dir, expected_path)}`",
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
        if field in latest_row:
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
        tail_rows=DEFAULT_TABLE_ROWS,
    )
    if table is not None:
        lines.extend(["", table])
    return lines


def _model_diagnostics_section(config: Config, report_dir: Path) -> list[str]:
    diagnostics_report_path = config.paths.reports_dir / "model_diagnostics_latest.md"
    lines = [
        _file_status_line("Diagnostics report", report_dir, diagnostics_report_path)
    ]
    image_lines = _image_lines(
        report_dir,
        [
            config.paths.reports_dir / "figures/model/prediction_timeline.svg",
            config.paths.reports_dir / "figures/model/actual_vs_prediction.svg",
            config.paths.reports_dir / "figures/model/residuals.svg",
        ],
    )
    if image_lines:
        lines.extend(["", *image_lines])
    else:
        lines.extend(["", "No model diagnostic figures are available yet."])
    return lines


def _performance_section(
    history: list[dict[str, str]] | None,
    report_dir: Path,
) -> list[str]:
    if history is None:
        expected_path = Path("artifacts/performance_history.csv")
        return [
            "No performance history is available yet.",
            "",
            f"- Expected file: `{_relative_path(report_dir, expected_path)}`",
        ]

    lines = [f"- Entries available: {len(history)}"]
    latest = history[-1]
    for field in ("operation", "status", "duration_seconds", "timestamp"):
        if field in latest:
            lines.append(f"- Latest {field}: {_format_value(latest[field])}")
    table = _tail_markdown_table(
        history,
        [
            "timestamp",
            "operation",
            "status",
            "duration_seconds",
            "input_rows",
            "output_rows",
            "model_name",
            "output_path",
            "error_message",
        ],
        tail_rows=DEFAULT_TABLE_ROWS,
    )
    if table is not None:
        lines.extend(["", table])
    return lines


def _update_prediction_timeline_section(
    timeline_path: Path,
    report_dir: Path,
) -> list[str]:
    if not timeline_path.exists():
        return [
            "Aggregate update prediction timeline is not available yet.",
            "",
            "Run update mode to write stream prediction CSV files.",
            "",
            f"- Expected file: `{_relative_path(report_dir, timeline_path)}`",
        ]
    relative_path = _relative_path(report_dir, timeline_path)
    return [
        "Daily actual sales and model predictions aggregated from all stream updates.",
        "",
        f"![Update prediction timeline]({relative_path})",
    ]


def _model_metrics_trend_section(
    config: Config,
    history: list[dict[str, str]] | None,
    report_dir: Path,
) -> list[str]:
    if history is None:
        expected_path = Path("artifacts/model_metrics_history.csv")
        return [
            "No model metrics history is available yet.",
            "",
            f"- Expected file: `{_relative_path(report_dir, expected_path)}`",
        ]

    table = _tail_markdown_table(
        history,
        ["batch_index", "model_name", "rmse", "mae", "r2", "is_best_in_update"],
        tail_rows=DEFAULT_TABLE_ROWS,
    )
    if table is None:
        return [
            "Model metrics history exists, but it does not contain "
            "the expected columns.",
        ]

    latest = history[-1]
    lines = [f"- Entries available: {len(history)}"]
    metric_chart_index = (
        config.paths.reports_dir / "figures/history/model_metrics_history.md"
    )
    if metric_chart_index.exists():
        lines.append(
            f"- Metric charts: `{_relative_path(report_dir, metric_chart_index)}`"
        )
    image_lines = _image_lines(
        report_dir,
        [
            config.paths.reports_dir / "figures/history/model_metrics_history_rmse.svg",
            config.paths.reports_dir / "figures/history/model_metrics_history_mae.svg",
            config.paths.reports_dir
            / "figures/history/model_metrics_history_smape.svg",
            config.paths.reports_dir / "figures/history/model_metrics_history_r2.svg",
        ],
    )
    for field in ("rmse", "mae", "r2", "smape", "pearson_corr"):
        if field in latest:
            lines.append(f"- Latest {field}: {_format_value(latest[field])}")
    interpretation_report_path = latest.get("latest_interpretation_report_path", "")
    if interpretation_report_path:
        lines.append(
            "- Latest interpretation report: "
            f"`{_relative_path(report_dir, Path(interpretation_report_path))}`"
        )
    if image_lines:
        lines.extend(["", *image_lines])
    lines.extend(["", table])
    return lines


def _model_interpretation_section(config: Config, report_dir: Path) -> list[str]:
    interpretation_path = config.paths.reports_dir / "model_interpretation_latest.md"
    if not interpretation_path.exists():
        return [
            "No model interpretation report is available yet.",
            "",
            f"- Expected file: `{_relative_path(report_dir, interpretation_path)}`",
        ]

    try:
        content = interpretation_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return [f"Could not read model interpretation report: {error}"]

    lines = [f"- Source file: `{_relative_path(report_dir, interpretation_path)}`"]
    top_features_path = _latest_top_features_path(config)
    if top_features_path is not None and top_features_path.exists():
        lines.append(
            f"- Top features CSV: `{_relative_path(report_dir, top_features_path)}`"
        )
        top_features_table = _top_rows_markdown_table(
            _read_history(top_features_path),
            ["feature", "value", "abs_value"],
            limit=DEFAULT_TABLE_ROWS,
        )
        if top_features_table is not None:
            lines.extend(["", "### Top features preview", "", top_features_table])

    lines.append("")
    for line in _without_top_features_section(content):
        if line.startswith("# "):
            lines.append(f"### {line[2:]}")
        elif line.startswith("## "):
            lines.append(f"### {line[3:]}")
        else:
            lines.append(line)
    return lines


def _without_top_features_section(lines: list[str]) -> list[str]:
    filtered: list[str] = []
    skip = False
    for line in lines:
        if line == "## Top features (by absolute value)":
            skip = True
            continue
        if skip and line.startswith("## "):
            skip = False
        if not skip:
            filtered.append(line)
    return filtered


def _latest_top_features_path(config: Config) -> Path | None:
    model_history = _read_history(config.paths.model_metrics_history_path)
    if not model_history:
        return None
    top_features_value = model_history[-1].get("interpretation_top_features_path", "")
    if not top_features_value:
        return None
    return Path(top_features_value)


def _hyperparameters_section(config: Config) -> list[str]:
    selected_model = config.model.selected_model
    params = config.model.model_parameters.get(selected_model, {})
    lines = [
        f"- selected_model: {selected_model}",
        f"- training_mode: {config.model.training_mode}",
        f"- update_strategy: {config.model.update_strategy}",
        f"- primary_metric: {config.model.primary_metric}",
        f"- model_parameters[{selected_model}]:",
    ]
    if not params:
        lines.append("  - (empty)")
        return lines
    for key, value in sorted(params.items()):
        lines.append(f"  - {key}: {_format_value(value)}")
    return lines


def _offline_evaluation_section(config: Config, report_dir: Path) -> list[str]:
    report_path = config.paths.reports_dir / "offline_model_evaluation.md"
    csv_path = config.paths.artifacts_dir / "offline_model_evaluation.csv"
    chart_path = (
        config.paths.reports_dir
        / "figures/offline_evaluation/actual_vs_prediction_timeline.svg"
    )
    lines = [
        _file_status_line("Evaluation report", report_dir, report_path),
        _file_status_line("Evaluation CSV", report_dir, csv_path),
    ]
    image_lines = _image_lines(report_dir, [chart_path])
    if image_lines:
        lines.extend(["", *image_lines])
    else:
        lines.extend(["", "No offline evaluation chart is available yet."])
    return lines


def _artifact_links(config: Config, report_dir: Path) -> list[str]:
    artifact_paths = [
        config.paths.batch_metadata_path,
        config.paths.data_quality_history_path,
        config.paths.performance_history_path,
        config.paths.model_metrics_history_path,
        config.paths.artifacts_dir / "offline_model_evaluation.csv",
    ]
    latest_reports = [
        config.paths.reports_dir / "eda_latest.md",
        config.paths.reports_dir / "model_diagnostics_latest.md",
        config.paths.reports_dir / "model_interpretation_latest.md",
        config.paths.reports_dir / "offline_model_evaluation.md",
    ]
    lines: list[str] = []
    for artifact_path in artifact_paths:
        relative = _relative_path(report_dir, artifact_path)
        if artifact_path.exists():
            lines.append(f"- [{artifact_path.name}]({relative})")
        else:
            lines.append(f"- {artifact_path.name}: `{relative}` (missing)")
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


def _top_rows_markdown_table(
    history: list[dict[str, str]] | None,
    columns: list[str],
    limit: int,
) -> str | None:
    if not history:
        return None
    selected_columns = [
        column for column in columns if any(column in row for row in history)
    ]
    if not selected_columns:
        return None
    rows = [
        {column: _format_value(row.get(column, "")) for column in selected_columns}
        for row in history[:limit]
    ]
    return pd.DataFrame(rows).to_markdown(index=False)


def _image_lines(report_dir: Path, image_paths: list[Path]) -> list[str]:
    lines: list[str] = []
    for image_path in image_paths:
        if not image_path.exists():
            continue
        title = image_path.stem.replace("_", " ").title()
        lines.append(f"![{title}]({_relative_path(report_dir, image_path)})")
    return lines


def _file_status_line(label: str, report_dir: Path, path: Path) -> str:
    relative_path = _relative_path(report_dir, path)
    if path.exists():
        return f"- {label}: `{relative_path}`"
    return f"- {label}: `{relative_path}` (missing)"


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
