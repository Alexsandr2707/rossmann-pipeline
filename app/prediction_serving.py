from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.config import Config
from app.dataset_loading import merge_store_metadata
from app.feature_engineering import build_features
from app.preprocessing import DataPreprocessor


class PredictionServing:
    def __init__(self, config: Config) -> None:
        self.config = config

    def predict_file(self, input_path: Path) -> Path:
        if not input_path.exists():
            raise FileNotFoundError(f"Inference input not found: {input_path}")

        raw_input = pd.read_csv(input_path, low_memory=False)
        serving_dataset = self._prepare_raw_dataset(raw_input, input_path)
        processed = DataPreprocessor(self.config).transform_features(serving_dataset)
        if len(processed) != len(raw_input):
            raise ValueError(
                "Inference preprocessing changed row count: "
                f"input_rows={len(raw_input)} processed_rows={len(processed)}"
            )

        features = build_features(processed, self.config)
        model = self._load_model()
        predictions = model.predict(features)

        output = raw_input.copy()
        output["predict"] = predictions
        output_path = self._next_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)
        return output_path

    def _prepare_raw_dataset(
        self,
        dataset: pd.DataFrame,
        input_path: Path,
    ) -> pd.DataFrame:
        prepared = dataset.copy()
        if "_source_file" not in prepared.columns:
            prepared["_source_file"] = input_path.name
        return merge_store_metadata(prepared, self.config)

    def _load_model(self) -> Any:
        model_path = self.config.paths.best_model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Best model not found: {model_path}")

        payload = joblib.load(model_path)
        if isinstance(payload, dict) and "pipeline" in payload:
            return payload["pipeline"]
        return payload

    def _next_output_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = (
            self.config.paths.predictions_dir / f"inference_{timestamp}.csv"
        )
        if not base_path.exists():
            return base_path

        for index in range(1, 1000):
            candidate = self.config.paths.predictions_dir / (
                f"inference_{timestamp}_{index:03d}.csv"
            )
            if not candidate.exists():
                return candidate

        raise RuntimeError("Cannot allocate a unique inference output path.")
