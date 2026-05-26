from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

FEATURE_PREPROCESSING_VERSION = "standard_scale_frequency_v7"


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        unknown_value: float = 0.0,
        unknown_category_value: float = -1.0,
    ) -> None:
        self.unknown_value = unknown_value
        self.unknown_category_value = unknown_category_value

    def fit(self, x: Any, y: Any = None) -> "FrequencyEncoder":
        data = self._to_frame(x)
        self.feature_names_in_ = list(data.columns)
        self.category_maps_: list[dict[Any, float]] = []
        self.frequency_maps_: list[dict[Any, float]] = []

        for column in self.feature_names_in_:
            categories = pd.unique(data[column])
            self.category_maps_.append(
                {category: float(index) for index, category in enumerate(categories)}
            )
            frequencies = data[column].value_counts(normalize=True, dropna=False)
            self.frequency_maps_.append(
                {
                    category: float(frequency)
                    for category, frequency in frequencies.items()
                }
            )
        return self

    def transform(self, x: Any) -> np.ndarray:
        data = self._to_frame(x)
        encoded_columns = []
        for index, column in enumerate(self.feature_names_in_):
            category_codes = data[column].map(self.category_maps_[index])
            frequencies = data[column].map(self.frequency_maps_[index])
            encoded_columns.append(
                category_codes.fillna(self.unknown_category_value).to_numpy()
            )
            encoded_columns.append(frequencies.fillna(self.unknown_value).to_numpy())
        return np.column_stack(encoded_columns)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            columns = self.feature_names_in_
        else:
            columns = np.asarray(input_features).astype(str).tolist()
        feature_names = []
        for column in columns:
            feature_names.extend([column, f"{column}_frequency"])
        return np.asarray(feature_names)

    def _to_frame(self, x: Any) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            return x.reset_index(drop=True)
        return pd.DataFrame(x)


def make_feature_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        column for column in features.columns if column not in numeric_features
    ]

    numeric_pipeline = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = SklearnPipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__MISSING__"),
            ),
            ("encoder", FrequencyEncoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
