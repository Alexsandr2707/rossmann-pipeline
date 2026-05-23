from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import Config
from app.dataset_loading import load_source_dataset
from app.period_splitting import (
    period_boundaries,
    rows_for_dates,
    split_date_periods,
    stream_date_batch,
)
from app.preprocessing import DataPreprocessor


@dataclass
class CollectorState:
    stream_batch_index: int = 0
    period_boundaries: dict[str, Any] | None = None


class DataCollector:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.preprocessor = DataPreprocessor(config)

    def current_stream_batch_index(self) -> int:
        return self._load_state().stream_batch_index

    def load_sorted_source_dataset(self) -> pd.DataFrame:
        return self._load_dataset()

    def initialize_stream_state(self, dataset: pd.DataFrame) -> CollectorState:
        state = self._load_state()
        state.stream_batch_index = 0
        state.period_boundaries = period_boundaries(
            split_date_periods(dataset, self.config)
        )
        self._save_state(state)
        return state

    def collect_next_stream_batch(self) -> tuple[Path, dict[str, Any]] | None:
        dataset = self._load_dataset()
        split = split_date_periods(dataset, self.config)
        state = self._load_state()
        batch_dates = stream_date_batch(
            split,
            state.stream_batch_index,
            self.config.model.stream_batch_days,
        )
        if batch_dates.empty:
            self.logger.info(
                "No stream data: stream_batch_index=%s stream_dates=%s",
                state.stream_batch_index,
                len(split.stream_dates),
            )
            return None

        batch = rows_for_dates(dataset, self.config.data.time_column, batch_dates)
        batch_path = (
            self.config.paths.raw_data_dir
            / f"batch_{state.stream_batch_index:04d}.csv"
        )
        batch.to_csv(batch_path, index=False)

        metadata = self._build_metadata(
            batch,
            batch_path,
            state.stream_batch_index,
            start=0,
            end=len(batch),
        )
        metadata.update(
            {
                "period_type": "stream",
                "stream_batch_index": int(state.stream_batch_index),
                "date_min": batch_dates.min().strftime("%Y-%m-%d"),
                "date_max": batch_dates.max().strftime("%Y-%m-%d"),
                "date_count": int(len(batch_dates)),
                **period_boundaries(split),
            }
        )
        self._append_metadata(metadata)

        state.stream_batch_index += 1
        state.period_boundaries = period_boundaries(split)
        self._save_state(state)

        self.logger.info(
            "Collected stream batch %s with %s rows across %s dates",
            batch_path,
            len(batch),
            len(batch_dates),
        )
        return batch_path, metadata

    def validate_source_dataset(self) -> dict[str, Any]:
        dataset = self._load_dataset()
        categorical_count = self._count_existing_columns(
            dataset,
            self.config.data_schema.categorical_columns,
        )
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
        dataset = load_source_dataset(self.config)
        return self._sort_raw_dataset(dataset)

    def _sort_raw_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        time_column = self.config.data.time_column
        if time_column not in dataset.columns:
            raise ValueError(f"Time column is missing: {time_column}")

        sortable = dataset.copy()
        sortable["_parsed_stream_time"] = pd.to_datetime(
            sortable[time_column],
            errors="coerce",
        )
        sortable = sortable.dropna(subset=["_parsed_stream_time"]).copy()
        sortable = sortable.sort_values(
            ["_parsed_stream_time", "_source_file"],
        ).reset_index(drop=True)
        return sortable.drop(columns=["_parsed_stream_time"])

    def _load_state(self) -> CollectorState:
        state_path = self.config.paths.collector_state_path
        if not state_path.exists():
            return CollectorState()

        with state_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)

        return CollectorState(
            stream_batch_index=int(raw.get("stream_batch_index", 0)),
            period_boundaries=raw.get("period_boundaries"),
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
        parsed_time = pd.to_datetime(
            batch[time_column],
            errors="coerce",
        )

        return {
            "batch_index": batch_index,
            "batch_path": str(batch_path),
            "start_row": start,
            "end_row": min(end, start + len(batch)),
            "rows": int(len(batch)),
            "columns": int(batch.shape[1]),
            "time_min": str(parsed_time.min()),
            "time_max": str(parsed_time.max()),
            "missing_part": float(batch.isna().mean().mean()),
            "numeric_features": self._count_existing_columns(
                batch,
                self.config.data_schema.numeric_columns,
            ),
            "categorical_features": self._count_existing_columns(
                batch,
                self.config.data_schema.categorical_columns,
            ),
            "target_missing": int(target.isna().sum()),
            "target_mean": float(target.mean()) if target.notna().any() else None,
            "target_min": float(target.min()) if target.notna().any() else None,
            "target_max": float(target.max()) if target.notna().any() else None,
            "target_missing_strategy": self.config.target_preprocessing.missing_strategy,
            "target_missing_indicator_column": self.preprocessor.target_missing_indicator_column
            if self.config.target_preprocessing.add_missing_indicator
            else "",
        }

    def _count_existing_columns(
        self,
        dataset: pd.DataFrame,
        columns: tuple[str, ...],
    ) -> int:
        return sum(1 for column in columns if column in dataset.columns)

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
