from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

FIGURE_SIZE = (9.0, 5.0)
COLORS = ("tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange")


def write_scatter_svg(
    path: Path,
    x_values: np.ndarray,
    y_values: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    x_values, y_values = _clean_xy(x_values, y_values)
    if len(x_values) > 1000:
        sample_index = np.linspace(0, len(x_values) - 1, 1000).astype(int)
        x_values = x_values[sample_index]
        y_values = y_values[sample_index]

    figure, axis = _new_figure(title, x_label, y_label)
    if len(x_values):
        axis.scatter(x_values, y_values, s=14, alpha=0.3, color=COLORS[0], linewidths=0)
        line_min = float(min(x_values.min(), y_values.min()))
        line_max = float(max(x_values.max(), y_values.max()))
        axis.plot(
            [line_min, line_max],
            [line_min, line_max],
            linestyle="--",
            linewidth=1.2,
            color=COLORS[1],
            alpha=0.9,
        )
        axis.set_xlim(*_range_with_padding(x_values))
        axis.set_ylim(*_range_with_padding(y_values))
    _save_figure(figure, path)


def write_histogram_svg(
    path: Path,
    values: np.ndarray,
    title: str,
    x_label: str,
) -> None:
    values = _clean_values(values)
    figure, axis = _new_figure(title, x_label, "rows")
    if len(values):
        clipped = np.clip(values, np.quantile(values, 0.01), np.quantile(values, 0.99))
        axis.hist(clipped, bins=24, color=COLORS[0], alpha=0.75)
        axis.set_xlim(*_range_with_padding(clipped))
    _save_figure(figure, path)


def write_line_chart_svg(
    path: Path,
    series_by_label: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    figure, axis = _new_figure(title, x_label, y_label)
    cleaned_series = _clean_series(series_by_label)
    for index, (label, (x_values, y_values)) in enumerate(cleaned_series.items()):
        axis.plot(
            x_values,
            y_values,
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            color=COLORS[index % len(COLORS)],
            label=label,
        )
    if cleaned_series:
        axis.legend(loc="best", frameon=False)
        _set_series_limits(axis, cleaned_series)
    _save_figure(figure, path)


def write_time_series_svg(
    path: Path,
    x_values: np.ndarray,
    series_by_label: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    x_label: str,
    y_label: str,
    x_tick_formatter: Callable[[float], str],
) -> None:
    figure, axis = _new_figure(title, x_label, y_label)
    cleaned_series = _clean_time_series(x_values, series_by_label)
    for index, (label, (series_x, series_y)) in enumerate(cleaned_series.items()):
        axis.plot(
            series_x,
            series_y,
            marker="o",
            markersize=3.0,
            linewidth=1.8,
            color=COLORS[index % len(COLORS)],
            label=label,
        )
    if cleaned_series:
        axis.legend(loc="best", frameon=False)
        _set_series_limits(axis, cleaned_series)
        ticks = np.linspace(
            min(values[0].min() for values in cleaned_series.values()),
            max(values[0].max() for values in cleaned_series.values()),
            num=min(7, max(2, len(next(iter(cleaned_series.values()))[0]))),
        )
        axis.set_xticks(ticks)
        axis.set_xticklabels(
            [x_tick_formatter(float(value)) for value in ticks],
            rotation=30,
            ha="right",
        )
    _save_figure(figure, path)


def _new_figure(title: str, x_label: str, y_label: str) -> tuple[plt.Figure, plt.Axes]:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, color="lightgray", linewidth=0.8)
    axis.set_axisbelow(True)
    return figure, axis


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def _clean_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _clean_xy(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[valid], y_values[valid]


def _clean_series(
    series_by_label: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    cleaned = {}
    for label, (x_values, y_values) in series_by_label.items():
        x_clean, y_clean = _clean_xy(x_values, y_values)
        if len(x_clean):
            cleaned[label] = (x_clean, y_clean)
    return cleaned


def _clean_time_series(
    x_values: np.ndarray,
    series_by_label: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    shared_x = np.asarray(x_values, dtype=float)
    cleaned = {}
    for label, (_, y_values) in series_by_label.items():
        x_clean, y_clean = _clean_xy(shared_x, y_values)
        if len(x_clean):
            cleaned[label] = (x_clean, y_clean)
    return cleaned


def _set_series_limits(
    axis: plt.Axes,
    series_by_label: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    all_x = np.concatenate([series[0] for series in series_by_label.values()])
    all_y = np.concatenate([series[1] for series in series_by_label.values()])
    axis.set_xlim(*_range_with_padding(all_x))
    axis.set_ylim(*_range_with_padding(all_y))


def _range_with_padding(values: np.ndarray) -> tuple[float, float]:
    values = _clean_values(values)
    if len(values) == 0:
        return 0.0, 1.0
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    if np.isclose(value_min, value_max):
        padding = max(abs(value_min) * 0.1, 1.0)
        return value_min - padding, value_max + padding
    padding = (value_max - value_min) * 0.05
    return value_min - padding, value_max + padding
