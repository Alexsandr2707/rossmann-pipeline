from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import Config
from app.visualization import write_time_series_svg

UPDATE_PREDICTION_TIMELINE_PATH = Path("figures/history/update_prediction_timeline.svg")


def generate_update_prediction_timeline(config: Config) -> Path | None:
    if not config.paths.model_metrics_history_path.exists():
        return None

    history = pd.read_csv(config.paths.model_metrics_history_path)
    if history.empty or "predictions_path" not in history.columns:
        return None

    if "period_type" in history.columns:
        history = history.loc[history["period_type"] == "stream"]
    if history.empty:
        return None

    prediction_frames = []
    for predictions_path in history["predictions_path"].dropna():
        path = Path(str(predictions_path))
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=lambda column: column in _PREDICTION_COLUMNS)
        if _PREDICTION_COLUMNS.issubset(frame.columns):
            prediction_frames.append(frame)

    if not prediction_frames:
        return None

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
    predictions["actual"] = pd.to_numeric(predictions["actual"], errors="coerce")
    predictions["prediction"] = pd.to_numeric(
        predictions["prediction"],
        errors="coerce",
    )
    predictions = predictions.dropna(subset=["date", "actual", "prediction"])
    if predictions.empty:
        return None

    daily = (
        predictions.assign(date=predictions["date"].dt.floor("D"))
        .groupby("date", as_index=False)[["actual", "prediction"]]
        .sum()
        .sort_values("date")
    )
    if daily.empty:
        return None

    first_date = daily["date"].min()
    day_offsets = (daily["date"] - first_date).dt.days.to_numpy(dtype=float)

    def format_day_offset(value: float) -> str:
        return (first_date + pd.to_timedelta(round(value), unit="D")).strftime(
            "%Y-%m-%d"
        )

    output_path = config.paths.reports_dir / UPDATE_PREDICTION_TIMELINE_PATH
    write_time_series_svg(
        output_path,
        day_offsets,
        {
            "actual sales": (day_offsets, daily["actual"].to_numpy(dtype=float)),
            "model prediction": (
                day_offsets,
                daily["prediction"].to_numpy(dtype=float),
            ),
        },
        "Update period: actual vs model prediction",
        "Date",
        "Daily sales",
        format_day_offset,
    )
    return output_path


_PREDICTION_COLUMNS = {"date", "actual", "prediction"}
