from __future__ import annotations

import pandas as pd

from app.config import Config


class DataPreprocessor:
    def __init__(self, config: Config) -> None:
        self.config = config

    @property
    def target_missing_indicator_column(self) -> str:
        return (
            f"{self.config.data.target_column}"
            f"{self.config.target_preprocessing.missing_indicator_suffix}"
        )

    def transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        transformed = dataset.copy()
        transformed = self._transform_time_column(transformed)
        transformed = self._transform_target_column(transformed)
        transformed = transformed.sort_values(
            [self.config.data.time_column, "_source_file"]
        ).reset_index(drop=True)
        return transformed

    def _transform_time_column(self, dataset: pd.DataFrame) -> pd.DataFrame:
        time_column = self.config.data.time_column
        if time_column not in dataset.columns:
            raise ValueError(f"Time column is missing: {time_column}")

        dataset[time_column] = pd.to_datetime(
            dataset[time_column],
            format="%d-%b-%y",
            errors="coerce",
        )
        return dataset.dropna(subset=[time_column]).copy()

    def _transform_target_column(self, dataset: pd.DataFrame) -> pd.DataFrame:
        target_column = self.config.data.target_column
        if target_column not in dataset.columns:
            raise ValueError(f"Target column is missing: {target_column}")

        target = pd.to_numeric(dataset[target_column], errors="coerce")

        if self.config.target_preprocessing.add_missing_indicator:
            dataset[self.target_missing_indicator_column] = target.isna().astype(int)

        strategy = self.config.target_preprocessing.missing_strategy
        if strategy == "fill_zero":
            dataset[target_column] = target.fillna(0.0)
        elif strategy == "fill_value":
            dataset[target_column] = target.fillna(
                self.config.target_preprocessing.missing_fill_value
            )
        elif strategy == "drop":
            dataset[target_column] = target
            dataset = dataset.dropna(subset=[target_column]).copy()
        elif strategy == "keep":
            dataset[target_column] = target
        else:
            raise ValueError(f"Unsupported target missing strategy: {strategy}")

        return dataset
