"""Data loading, validation, preprocessing and feature utilities."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CollectorState": "app.data.data_collection",
    "DataCollector": "app.data.data_collection",
    "DataPreprocessor": "app.data.preprocessing",
    "DataQualityAnalyzer": "app.data.data_quality",
    "DatePeriodSplit": "app.data.period_splitting",
    "build_features": "app.data.feature_engineering",
    "build_features_and_target": "app.data.feature_engineering",
    "excluded_feature_columns": "app.data.feature_engineering",
    "load_source_dataset": "app.data.dataset_loading",
    "merge_store_metadata": "app.data.dataset_loading",
    "period_boundaries": "app.data.period_splitting",
    "rows_for_dates": "app.data.period_splitting",
    "sorted_unique_dates": "app.data.period_splitting",
    "split_date_periods": "app.data.period_splitting",
    "stream_date_batch": "app.data.period_splitting",
}

__all__ = sorted(_EXPORTS)  # type: ignore


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
