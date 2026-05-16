from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.config import Config


@dataclass(frozen=True)
class DatePeriodSplit:
    initial_dates: pd.DatetimeIndex
    validation_dates: pd.DatetimeIndex
    stream_dates: pd.DatetimeIndex


def split_date_periods(dataset: pd.DataFrame, config: Config) -> DatePeriodSplit:
    dates = sorted_unique_dates(dataset, config.data.time_column)
    if len(dates) < 3:
        raise ValueError("At least three unique dates are required for period split.")

    initial_count = max(int(len(dates) * config.model.initial_train_ratio), 1)
    validation_count = max(int(len(dates) * config.model.validation_ratio), 1)
    if initial_count + validation_count >= len(dates):
        validation_count = max(len(dates) - initial_count - 1, 1)

    stream_start = initial_count + validation_count
    return DatePeriodSplit(
        initial_dates=dates[:initial_count],
        validation_dates=dates[initial_count:stream_start],
        stream_dates=dates[stream_start:],
    )


def sorted_unique_dates(dataset: pd.DataFrame, time_column: str) -> pd.DatetimeIndex:
    if time_column not in dataset.columns:
        raise ValueError(f"Time column is missing: {time_column}")
    parsed = pd.to_datetime(dataset[time_column], errors="coerce").dropna()
    dates = pd.DatetimeIndex(parsed.dt.floor("D").unique()).sort_values()
    if dates.empty:
        raise ValueError(f"No valid dates found in column: {time_column}")
    return dates


def rows_for_dates(
    dataset: pd.DataFrame,
    time_column: str,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    parsed = pd.to_datetime(dataset[time_column], errors="coerce").dt.floor("D")
    return dataset.loc[parsed.isin(dates)].copy().reset_index(drop=True)


def stream_date_batch(
    split: DatePeriodSplit,
    batch_index: int,
    batch_days: int,
) -> pd.DatetimeIndex:
    if batch_days <= 0:
        raise ValueError("stream_batch_days must be positive.")
    start = batch_index * batch_days
    end = start + batch_days
    return split.stream_dates[start:end]


def period_boundaries(split: DatePeriodSplit) -> dict[str, str | int]:
    return {
        "initial_date_min": _date_min(split.initial_dates),
        "initial_date_max": _date_max(split.initial_dates),
        "initial_date_count": int(len(split.initial_dates)),
        "validation_date_min": _date_min(split.validation_dates),
        "validation_date_max": _date_max(split.validation_dates),
        "validation_date_count": int(len(split.validation_dates)),
        "stream_date_min": _date_min(split.stream_dates),
        "stream_date_max": _date_max(split.stream_dates),
        "stream_date_count": int(len(split.stream_dates)),
    }


def _date_min(dates: pd.DatetimeIndex) -> str:
    return "" if dates.empty else dates.min().strftime("%Y-%m-%d")


def _date_max(dates: pd.DatetimeIndex) -> str:
    return "" if dates.empty else dates.max().strftime("%Y-%m-%d")
