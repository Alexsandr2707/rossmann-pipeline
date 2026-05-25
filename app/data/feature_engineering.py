from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.config import Config


def build_features(dataset: pd.DataFrame, config: Config) -> pd.DataFrame:
    features = pd.DataFrame(index=dataset.index)
    excluded_columns = excluded_feature_columns(config)

    for column in config.data_schema.numeric_columns:
        if column in dataset.columns and column not in excluded_columns:
            features[column] = pd.to_numeric(dataset[column], errors="coerce")

    for column in config.data_schema.categorical_columns:
        if column in dataset.columns and column not in excluded_columns:
            categorical = dataset[column]
            features[column] = (
                categorical.astype("string")
                .astype("object")
                .where(categorical.notna(), np.nan)
            )

    time_column = config.data.time_column
    if time_column in dataset.columns:
        parsed_time = pd.to_datetime(dataset[time_column], errors="coerce")
        features[f"{time_column}_year"] = parsed_time.dt.year
        features[f"{time_column}_month"] = parsed_time.dt.month
        features[f"{time_column}_quarter"] = parsed_time.dt.quarter
        features[f"{time_column}_day"] = parsed_time.dt.day
        features[f"{time_column}_weekofyear"] = (
            parsed_time.dt.isocalendar().week.astype("float64")
        )

    return features.reset_index(drop=True)


def build_features_and_target(
    dataset: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, pd.Series]:
    target_column = config.data.target_column
    if target_column not in dataset.columns:
        raise ValueError(f"Target column is missing: {target_column}")

    target = pd.to_numeric(dataset[target_column], errors="coerce")
    training = dataset[target.notna()].copy()
    target = target[target.notna()]

    return build_features(training, config), target.reset_index(drop=True)


def excluded_feature_columns(config: Config) -> set[str]:
    return {
        config.data.target_column,
        "Customers",
        *config.data_schema.id_columns,
        *config.data_schema.service_columns,
    }
