from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import Config
from app.preprocessing import DataPreprocessor


@dataclass
class CollectorState:
    current_batch_index: int
    processed_batches: list[str]
    total_rows_seen: int


class DataCollector:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.preprocessor = DataPreprocessor(config)

    def collect_next_batch(self) -> tuple[Path, dict[str, Any]] | None:
        dataset = self._load_dataset()
        state = self._load_state()

        start = state.current_batch_index * self.config.data.batch_size
        end = start + self.config.data.batch_size

        if start >= len(dataset):
            self.logger.info("No new data: start=%s total_rows=%s", start, len(dataset))
            return None

        batch = dataset.iloc[start:end].copy()
        batch_path = self.config.paths.raw_data_dir / f"batch_{state.current_batch_index:04d}.csv"
        batch.to_csv(batch_path, index=False)

        metadata = self._build_metadata(batch, batch_path, state.current_batch_index, start, end)
        self._append_metadata(metadata)

        state.processed_batches.append(str(batch_path))
        state.total_rows_seen += len(batch)
        state.current_batch_index += 1
        self._save_state(state)

        self.logger.info("Collected batch %s with %s rows", batch_path, len(batch))
        return batch_path, metadata

    def validate_source_dataset(self) -> dict[str, Any]:
        dataset = self._load_dataset()
        categorical_count = int(dataset.select_dtypes(include=["object", "category"]).shape[1])
        return {
            "rows": int(len(dataset)),
            "features": int(dataset.shape[1]),
            "categorical_features": categorical_count,
            "has_time_column": self.config.data.time_column in dataset.columns,
            "has_target_column": self.config.data.target_column in dataset.columns,
            "missing_part": float(dataset.isna().mean().mean()),
            "meets_min_rows": len(dataset) >= self.config.data.min_rows,
            "meets_min_features": dataset.shape[1] >= self.config.data.min_features,
            "meets_min_categorical_features": categorical_count
            >= self.config.data.min_categorical_features,
        }

    def _load_dataset(self) -> pd.DataFrame:
        frames = []
        for source_path in self.config.data.source_paths:
            if not source_path.exists():
                raise FileNotFoundError(f"Source dataset not found: {source_path}")
            frame = pd.read_csv(source_path)
            frame["_source_file"] = source_path.name
            frames.append(frame)

        dataset = pd.concat(frames, ignore_index=True)
        return self._prepare_dataset(dataset)

    def _prepare_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return self.preprocessor.transform(dataset)

    def _load_state(self) -> CollectorState:
        state_path = self.config.paths.collector_state_path
        if not state_path.exists():
            return CollectorState(current_batch_index=0, processed_batches=[], total_rows_seen=0)

        with state_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)

        return CollectorState(
            current_batch_index=int(raw["current_batch_index"]),
            processed_batches=list(raw["processed_batches"]),
            total_rows_seen=int(raw["total_rows_seen"]),
        )

    def _save_state(self, state: CollectorState) -> None:
        state_path = self.config.paths.collector_state_path
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with state_path.open("w", encoding="utf-8") as file:
            json.dump(asdict(state), file, indent=2)

    def _build_metadata(
        self,
        batch: pd.DataFrame,
        batch_path: Path,
        batch_index: int,
        start: int,
        end: int,
    ) -> dict[str, Any]:
        time_column = self.config.data.time_column
        target_column = self.config.data.target_column
        target = pd.to_numeric(batch[target_column], errors="coerce")

        return {
            "batch_index": batch_index,
            "batch_path": str(batch_path),
            "start_row": start,
            "end_row": min(end, start + len(batch)),
            "rows": int(len(batch)),
            "columns": int(batch.shape[1]),
            "time_min": str(batch[time_column].min()),
            "time_max": str(batch[time_column].max()),
            "missing_part": float(batch.isna().mean().mean()),
            "numeric_features": int(batch.select_dtypes(include=["number"]).shape[1]),
            "categorical_features": int(batch.select_dtypes(include=["object", "category"]).shape[1]),
            "target_missing": int(target.isna().sum()),
            "target_mean": float(target.mean()) if target.notna().any() else None,
            "target_min": float(target.min()) if target.notna().any() else None,
            "target_max": float(target.max()) if target.notna().any() else None,
            "target_missing_strategy": self.config.target_preprocessing.missing_strategy,
            "target_missing_indicator_column": self.preprocessor.target_missing_indicator_column
            if self.config.target_preprocessing.add_missing_indicator
            else "",
        }

    def _append_metadata(self, metadata: dict[str, Any]) -> None:
        metadata_path = self.config.paths.batch_metadata_path
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        row = pd.DataFrame([metadata])
        row.to_csv(
            metadata_path,
            mode="a",
            header=not metadata_path.exists(),
            index=False,
        )
