from __future__ import annotations

import csv
import html
import os
from pathlib import Path

from app.config import Config


def generate_html_report(config: Config) -> Path:
    """Create a small static dashboard that can be opened in a browser."""
    reports_dir = config.paths.reports_dir
    output_path = reports_dir / "index.html"
    reports_dir.mkdir(parents=True, exist_ok=True)

    batch_rows = read_csv(config.paths.batch_metadata_path)
    metric_rows = read_csv(config.paths.model_metrics_history_path)
    latest_batch = batch_rows[-1] if batch_rows else {}
    latest_metrics = metric_rows[-1] if metric_rows else {}

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pipeline report</title>
  <style>
    body {{
      margin: 0;
      background: #f5f6f8;
      color: #222;
      font-family: Arial, sans-serif;
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px;
    }}
    section {{
      background: white;
      border: 1px solid #ddd;
      border-radius: 8px;
      margin-bottom: 18px;
      padding: 18px;
    }}
    h1, h2 {{
      margin-top: 0;
    }}
    .cards, .charts, .links {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .card {{
      min-width: 140px;
      border: 1px solid #ddd;
      border-radius: 6px;
      padding: 10px;
      background: #eef6f8;
    }}
    .card b {{
      display: block;
      margin-top: 5px;
      font-size: 18px;
    }}
    .chart {{
      width: calc(50% - 6px);
      min-width: 320px;
      border: 1px solid #ddd;
      border-radius: 6px;
      padding: 10px;
    }}
    .chart img {{
      width: 100%;
      height: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
    }}
    th, td {{
      border-bottom: 1px solid #ddd;
      padding: 8px;
      text-align: left;
      white-space: nowrap;
    }}
    .table-wrapper {{
      overflow-x: auto;
    }}
    a, .missing {{
      display: inline-block;
      border: 1px solid #ddd;
      border-radius: 6px;
      padding: 8px 10px;
      background: white;
    }}
    a {{
      color: #126e82;
      text-decoration: none;
      font-weight: bold;
    }}
    .missing {{
      color: #777;
    }}
  </style>
</head>
<body>
<main>
  <section>
    <h1>{escape(config.project.name)} report</h1>
    <div class="cards">
      {card("Latest batch", latest_batch.get("batch_index", "not ready"))}
      {card("Rows", latest_batch.get("rows", "not ready"))}
      {card("Model", latest_metrics.get("model_name", config.model.selected_model))}
      {card("RMSE", latest_metrics.get("rmse", "not ready"))}
      {card("MAE", latest_metrics.get("mae", "not ready"))}
      {card("R2", latest_metrics.get("r2", "not ready"))}
    </div>
  </section>

  <section>
    <h2>Latest model diagnostics</h2>
    <div class="charts">
      {image(reports_dir, reports_dir / "figures/model/prediction_timeline.svg")}
      {image(reports_dir, reports_dir / "figures/model/actual_vs_prediction.svg")}
      {image(reports_dir, reports_dir / "figures/model/residuals.svg")}
    </div>
  </section>

  <section>
    <h2>Metric history</h2>
    <div class="charts">
      {image(reports_dir, reports_dir / "figures/history/model_metrics_history_rmse.svg")}
      {image(reports_dir, reports_dir / "figures/history/model_metrics_history_mae.svg")}
      {image(reports_dir, reports_dir / "figures/history/model_metrics_history_smape.svg")}
      {image(reports_dir, reports_dir / "figures/history/model_metrics_history_r2.svg")}
    </div>
    {table(metric_rows[-8:]) if metric_rows else "<p>No metric history yet.</p>"}
  </section>

  <section>
    <h2>Offline evaluation</h2>
    <div class="charts">
      {image(reports_dir, reports_dir / "figures/offline_evaluation/actual_vs_prediction_timeline.svg")}
    </div>
    <div class="links">
      {link(reports_dir, reports_dir / "offline_model_evaluation.md")}
      {link(reports_dir, config.paths.artifacts_dir / "offline_model_evaluation.csv")}
    </div>
  </section>

  <section>
    <h2>Files</h2>
    <div class="links">
      {link(reports_dir, reports_dir / "summary/summary_latest.md")}
      {link(reports_dir, reports_dir / "model_diagnostics_latest.md")}
      {link(reports_dir, config.paths.batch_metadata_path)}
      {link(reports_dir, config.paths.model_metrics_history_path)}
    </div>
  </section>
</main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def card(title: str, value: object) -> str:
    return f'<div class="card">{escape(title)}<b>{escape(value)}</b></div>'


def image(report_dir: Path, path: Path) -> str:
    title = path.stem.replace("_", " ").title()
    if not path.exists():
        return f'<div class="chart"><b>{escape(title)}</b><p>File is not ready yet.</p></div>'
    relative_path = os.path.relpath(path, report_dir)
    return (
        f'<div class="chart"><b>{escape(title)}</b>'
        f'<img src="{escape(relative_path)}" alt="{escape(title)}"></div>'
    )


def link(report_dir: Path, path: Path) -> str:
    if not path.exists():
        return f'<span class="missing">{escape(path.name)} is missing</span>'
    relative_path = os.path.relpath(path, report_dir)
    return f'<a href="{escape(relative_path)}">{escape(path.name)}</a>'


def table(rows: list[dict[str, str]]) -> str:
    columns = ["batch_index", "model_name", "rmse", "mae", "r2", "smape"]
    columns = [column for column in columns if any(column in row for row in rows)]
    if not columns:
        return "<p>No table data yet.</p>"

    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    lines = [f"<tr>{header}</tr>"]
    for row in rows:
        cells = "".join(f"<td>{escape(row.get(column, ''))}</td>" for column in columns)
        lines.append(f"<tr>{cells}</tr>")

    return f'<div class="table-wrapper"><table>{"".join(lines)}</table></div>'


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)
