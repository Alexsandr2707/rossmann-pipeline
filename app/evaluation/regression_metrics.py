from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.models.prediction_postprocessing import non_negative_predictions


def calculate_regression_metrics(
    y_true: pd.Series | np.ndarray,
    predictions: np.ndarray,
    *,
    include_prediction_range: bool = True,
) -> dict[str, Any]:
    y_true_array = np.asarray(y_true, dtype=float)
    prediction_array = non_negative_predictions(predictions)

    metrics: dict[str, Any] = {
        "rmse": float(np.sqrt(mean_squared_error(y_true_array, prediction_array))),
        "mae": float(mean_absolute_error(y_true_array, prediction_array)),
        "r2": float(r2_score(y_true_array, prediction_array)),
        "smape": _smape(y_true_array, prediction_array),
        "target_mean": float(y_true_array.mean()),
        "prediction_mean": float(prediction_array.mean()),
    }

    if include_prediction_range:
        metrics.update(
            {
                "prediction_min": float(prediction_array.min()),
                "prediction_max": float(prediction_array.max()),
                "actual_zero_rate": float((y_true_array == 0).mean()),
                "prediction_zero_rate": float((prediction_array <= 0).mean()),
            }
        )

    return metrics


def _smape(y_true: np.ndarray, predictions: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(predictions)
    smape_values = np.divide(
        2.0 * np.abs(predictions - y_true),
        denominator,
        out=np.zeros_like(predictions, dtype=float),
        where=denominator != 0,
    )
    return float(smape_values.mean())
