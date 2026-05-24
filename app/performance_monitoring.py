from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from app.config import Config


@dataclass(frozen=True)
class PerformanceRecord:
    operation: str
    status: str
    duration_seconds: float
    input_rows: int | None = None
    output_rows: int | None = None
    model_name: str | None = None
    model_path: str | None = None
    input_path: str | None = None
    output_path: str | None = None
    artifact_path: str | None = None
    error_message: str | None = None


class PerformanceMonitor:
    def __init__(self, config: Config) -> None:
        self._history_path = config.paths.performance_history_path

    def start(self) -> float:
        return perf_counter()

    def record(self, record: PerformanceRecord, start_time: float | None = None) -> None:
        duration = record.duration_seconds
        if start_time is not None:
            duration = perf_counter() - start_time

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": record.operation,
            "status": record.status,
            "duration_seconds": f"{duration:.6f}",
            "input_rows": self._to_text(record.input_rows),
            "output_rows": self._to_text(record.output_rows),
            "model_name": self._to_text(record.model_name),
            "model_path": self._to_text(record.model_path),
            "input_path": self._to_text(record.input_path),
            "output_path": self._to_text(record.output_path),
            "artifact_path": self._to_text(record.artifact_path),
            "error_message": self._to_text(record.error_message),
        }
        self._append_row(row)

    def _append_row(self, row: dict[str, str]) -> None:
        history_path = self._history_path
        history_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(row.keys())
        with history_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if history_path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _to_text(value: object) -> str:
        if value is None:
            return ""
        return str(value)
