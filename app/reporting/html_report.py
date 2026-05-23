from __future__ import annotations

import csv
import os
from pathlib import Path

import pandas as pd
from jinja2 import Environment, select_autoescape
from markupsafe import Markup

from app.config import Config


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pipeline report</title>
  <style>
    body {
      margin: 0;
      background: #f5f6f8;
      color: #222;
      font-family: Arial, sans-serif;
    }
    main {
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px;
    }
    section {
      background: white;
      border: 1px solid #ddd;
      border-radius: 8px;
      margin-bottom: 18px;
      padding: 18px;
    }
    h1, h2 {
      margin-top: 0;
    }
    .cards, .charts, .links {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
    }
    .card {
      min-width: 140px;
      border: 1px solid #ddd;
      border-radius: 6px;
      padding: 10px;
      background: #eef6f8;
    }
    .card b {
      display: block;
      margin-top: 5px;
      font-size: 18px;
    }
    .chart {
      width: calc(50% - 6px);
      min-width: 320px;
      border: 1px solid #ddd;
      border-radius: 6px;
      padding: 10px;
    }
    .chart img {
      width: 100%;
      height: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
    }
    th, td {
      border-bottom: 1px solid #ddd;
      padding: 8px;
      text-align: left;
      white-space: nowrap;
    }
    .table-wrapper {
      overflow-x: auto;
    }
    a, .missing {
      display: inline-block;
      border: 1px solid #ddd;
      border-radius: 6px;
      padding: 8px 10px;
      background: white;
    }
    a {
      color: #126e82;
      text-decoration: none;
      font-weight: bold;
    }
    .missing {
      color: #777;
    }
  </style>
</head>
<body>
<main>
  <section>
    <h1>{{ project_name }} report</h1>
    <div class="cards">
      {% for card in overview_cards %}
      <div class="card">{{ card.title }}<b>{{ card.value }}</b></div>
      {% endfor %}
    </div>
  </section>

  <section>
    <h2>Latest data quality</h2>
    <div class="cards">
      {% for card in data_quality_cards %}
      <div class="card">{{ card.title }}<b>{{ card.value }}</b></div>
      {% endfor %}
    </div>
    <div class="links">
      {% for link in data_quality_links %}
        {% if link.href %}
        <a href="{{ link.href }}">{{ link.label }}</a>
        {% else %}
        <span class="missing">{{ link.label }}</span>
        {% endif %}
      {% endfor %}
    </div>
    {{ data_quality_table }}
  </section>

  <section>
    <h2>Latest model diagnostics</h2>
    <div class="charts">
      {% for chart in model_charts %}
      <div class="chart"><b>{{ chart.title }}</b>
        {% if chart.href %}
        <img src="{{ chart.href }}" alt="{{ chart.title }}">
        {% else %}
        <p>File is not ready yet.</p>
        {% endif %}
      </div>
      {% endfor %}
    </div>
  </section>

  <section>
    <h2>Metric history</h2>
    <div class="charts">
      {% for chart in history_charts %}
      <div class="chart"><b>{{ chart.title }}</b>
        {% if chart.href %}
        <img src="{{ chart.href }}" alt="{{ chart.title }}">
        {% else %}
        <p>File is not ready yet.</p>
        {% endif %}
      </div>
      {% endfor %}
    </div>
    {{ metric_table }}
  </section>

  <section>
    <h2>Offline evaluation</h2>
    <div class="charts">
      {% for chart in offline_charts %}
      <div class="chart"><b>{{ chart.title }}</b>
        {% if chart.href %}
        <img src="{{ chart.href }}" alt="{{ chart.title }}">
        {% else %}
        <p>File is not ready yet.</p>
        {% endif %}
      </div>
      {% endfor %}
    </div>
    <div class="links">
      {% for link in offline_links %}
        {% if link.href %}
        <a href="{{ link.href }}">{{ link.label }}</a>
        {% else %}
        <span class="missing">{{ link.label }} is missing</span>
        {% endif %}
      {% endfor %}
    </div>
  </section>

  <section>
    <h2>Files</h2>
    <div class="links">
      {% for link in file_links %}
        {% if link.href %}
        <a href="{{ link.href }}">{{ link.label }}</a>
        {% else %}
        <span class="missing">{{ link.label }} is missing</span>
        {% endif %}
      {% endfor %}
    </div>
  </section>
</main>
</body>
</html>
"""

_JINJA_ENV = Environment(
    autoescape=select_autoescape(default=True, default_for_string=True),
)


def generate_html_report(config: Config) -> Path:
    """Create a small static dashboard that can be opened in a browser."""
    reports_dir = config.paths.reports_dir
    output_path = reports_dir / "index.html"
    reports_dir.mkdir(parents=True, exist_ok=True)

    batch_rows = read_csv(config.paths.batch_metadata_path)
    data_quality_rows = read_csv(config.paths.data_quality_history_path)
    metric_rows = read_csv(config.paths.model_metrics_history_path)
    latest_batch = batch_rows[-1] if batch_rows else {}
    latest_quality = data_quality_rows[-1] if data_quality_rows else {}
    latest_metrics = metric_rows[-1] if metric_rows else {}
    html_text = _JINJA_ENV.from_string(HTML_TEMPLATE).render(
        project_name=config.project.name,
        overview_cards=[
            card("Latest batch", latest_batch.get("batch_index", "not ready")),
            card("Rows", latest_batch.get("rows", "not ready")),
            card("Missing part", latest_quality.get("missing_part", "not ready")),
            card("Duplicate part", latest_quality.get("duplicate_part", "not ready")),
            card(
                "Outlier part",
                latest_quality.get("numeric_outlier_part", "not ready"),
            ),
            card("Model", latest_metrics.get("model_name", config.model.selected_model)),
            card("RMSE", latest_metrics.get("rmse", "not ready")),
            card("MAE", latest_metrics.get("mae", "not ready")),
            card("R2", latest_metrics.get("r2", "not ready")),
        ],
        data_quality_cards=[
            card("Duplicate rows", latest_quality.get("duplicate_rows", "not ready")),
            card(
                "Constant columns",
                latest_quality.get("constant_columns", "not ready"),
            ),
            card(
                "Missing schema",
                latest_quality.get("schema_missing_columns", "not ready"),
            ),
            card(
                "Extra schema",
                latest_quality.get("schema_extra_columns", "not ready"),
            ),
        ],
        data_quality_links=[
            link(reports_dir, config.paths.data_quality_history_path),
            link(
                reports_dir,
                Path(latest_quality["eda_report_path"]),
                missing_label="EDA batch report is missing",
            )
            if latest_quality.get("eda_report_path")
            else {"href": "", "label": "EDA batch report is missing"},
        ],
        data_quality_table=table(
            data_quality_rows[-8:],
            [
                "batch_index",
                "rows",
                "missing_part",
                "duplicate_part",
                "numeric_outlier_part",
            ],
        )
        if data_quality_rows
        else Markup("<p>No data quality history yet.</p>"),
        model_charts=[
            image(reports_dir, reports_dir / "figures/model/prediction_timeline.svg"),
            image(reports_dir, reports_dir / "figures/model/actual_vs_prediction.svg"),
            image(reports_dir, reports_dir / "figures/model/residuals.svg"),
        ],
        history_charts=[
            image(
                reports_dir,
                reports_dir / "figures/history/model_metrics_history_rmse.svg",
            ),
            image(
                reports_dir,
                reports_dir / "figures/history/model_metrics_history_mae.svg",
            ),
            image(
                reports_dir,
                reports_dir / "figures/history/model_metrics_history_smape.svg",
            ),
            image(
                reports_dir,
                reports_dir / "figures/history/model_metrics_history_r2.svg",
            ),
        ],
        metric_table=table(metric_rows[-8:])
        if metric_rows
        else Markup("<p>No metric history yet.</p>"),
        offline_charts=[
            image(
                reports_dir,
                reports_dir
                / "figures/offline_evaluation/actual_vs_prediction_timeline.svg",
            )
        ],
        offline_links=[
            link(reports_dir, reports_dir / "offline_model_evaluation.md"),
            link(reports_dir, config.paths.artifacts_dir / "offline_model_evaluation.csv"),
        ],
        file_links=[
            link(reports_dir, reports_dir / "summary/summary_latest.md"),
            link(reports_dir, reports_dir / "model_diagnostics_latest.md"),
            link(reports_dir, config.paths.batch_metadata_path),
            link(reports_dir, config.paths.data_quality_history_path),
            link(reports_dir, config.paths.model_metrics_history_path),
        ],
    )
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def card(title: str, value: object) -> dict[str, object]:
    return {"title": title, "value": value}


def image(report_dir: Path, path: Path) -> dict[str, str]:
    title = path.stem.replace("_", " ").title()
    if not path.exists():
        return {"title": title, "href": ""}
    return {"title": title, "href": os.path.relpath(path, report_dir)}


def link(
    report_dir: Path,
    path: Path,
    missing_label: str | None = None,
) -> dict[str, str]:
    label = path.name
    if not path.exists():
        return {"label": missing_label or label, "href": ""}
    return {"label": label, "href": os.path.relpath(path, report_dir)}


def table(rows: list[dict[str, str]], columns: list[str] | None = None) -> str:
    if columns is None:
        columns = ["batch_index", "model_name", "rmse", "mae", "r2", "smape"]
    columns = [column for column in columns if any(column in row for row in rows)]
    if not columns:
        return Markup("<p>No table data yet.</p>")

    table_html = pd.DataFrame(rows, columns=columns).to_html(
        index=False,
        escape=True,
        border=0,
    )
    return Markup(f'<div class="table-wrapper">{table_html}</div>')
