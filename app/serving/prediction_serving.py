from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.core.config import Config
from app.data.dataset_loading import merge_store_metadata
from app.data.feature_engineering import build_features
from app.monitoring.performance_monitoring import PerformanceMonitor, PerformanceRecord
from app.data.preprocessing import DataPreprocessor
from app.models import (
    FEATURE_PREPROCESSING_VERSION,
    canonical_model_name,
    model_signature,
)
from app.models.prediction_postprocessing import non_negative_predictions


class PredictionServing:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.performance_monitor = PerformanceMonitor(config)

    def predict_file(self, input_path: Path) -> Path:
        start_time = self.performance_monitor.start()
        input_rows = 0
        output_path: Path | None = None
        status = "failure"
        error_message: str | None = None
        try:
            if not input_path.exists():
                raise FileNotFoundError(f"Inference input not found: {input_path}")

            raw_input = pd.read_csv(input_path, low_memory=False)
            input_rows = int(len(raw_input))
            serving_dataset = self._prepare_raw_dataset(raw_input, input_path)
            processed = DataPreprocessor(self.config).transform_features(
                serving_dataset
            )
            if len(processed) != len(raw_input):
                raise ValueError(
                    "Inference preprocessing changed row count: "
                    f"input_rows={len(raw_input)} processed_rows={len(processed)}"
                )

            features = build_features(processed, self.config)
            model = self._load_model()
            predictions = non_negative_predictions(model.predict(features))

            output = raw_input.copy()
            output["predict"] = predictions
            output_path = self._next_output_path()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output.to_csv(output_path, index=False)
            status = "success"
            return output_path
        except Exception as error:
            error_message = str(error)
            raise
        finally:
            self.performance_monitor.record(
                PerformanceRecord(
                    operation="inference",
                    status=status,
                    duration_seconds=0.0,
                    input_rows=input_rows or None,
                    output_rows=input_rows if status == "success" else None,
                    model_name=self.config.model.selected_model,
                    model_path=str(self.config.paths.best_model_path),
                    input_path=str(input_path),
                    output_path=str(output_path) if output_path else "",
                    artifact_path=str(output_path) if output_path else "",
                    error_message=error_message,
                ),
                start_time=start_time,
            )

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
            error = self._model_compatibility_error(payload)
            if error is not None:
                raise ValueError(f"Best model is incompatible: {error}. Reset/retrain.")
            return payload["pipeline"]
        raise ValueError(
            "Best model is incompatible: missing model payload metadata. Reset/retrain."
        )

    def _model_compatibility_error(self, payload: dict[str, Any]) -> str | None:
        expected_model_name = canonical_model_name(self.config.model.selected_model)
        payload_model_name = payload.get("model_name")
        if payload_model_name is None and isinstance(payload.get("metrics"), dict):
            payload_model_name = payload["metrics"].get("model_name")
        if canonical_model_name(str(payload_model_name)) != expected_model_name:
            return (
                f"model_name={payload_model_name!r}, "
                f"expected {expected_model_name!r}"
            )

        version = payload.get("feature_preprocessing_version")
        if version != FEATURE_PREPROCESSING_VERSION:
            return (
                f"feature_preprocessing_version={version!r}, "
                f"expected {FEATURE_PREPROCESSING_VERSION!r}"
            )

        signature = payload.get("model_signature")
        expected_signature = model_signature(
            expected_model_name,
            self._model_parameters(expected_model_name),
        )
        if signature != expected_signature:
            return "model_signature does not match selected model configuration"

        return None

    def _model_parameters(self, model_name: str) -> dict[str, Any]:
        canonical_name = canonical_model_name(model_name)
        parameters = self.config.model.model_parameters
        if canonical_name in parameters:
            return dict(parameters[canonical_name])
        for configured_name, configured_parameters in parameters.items():
            if canonical_model_name(configured_name) == canonical_name:
                return dict(configured_parameters)
        return {}

    def _next_output_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self.config.paths.predictions_dir / f"inference_{timestamp}.csv"
        if not base_path.exists():
            return base_path

        for index in range(1, 1000):
            candidate = self.config.paths.predictions_dir / (
                f"inference_{timestamp}_{index:03d}.csv"
            )
            if not candidate.exists():
                return candidate

        raise RuntimeError("Cannot allocate a unique inference output path.")
