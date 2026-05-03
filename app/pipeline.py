from __future__ import annotations

import logging
from pathlib import Path

from app.config import Config
from app.data_collection import DataCollector


class Pipeline:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._ensure_directories()
        self.collector = DataCollector(config)

    def update(self) -> bool:
        self.logger.info("Update mode requested")
        validation = self.collector.validate_source_dataset()
        self.logger.info("Source dataset validation: %s", validation)

        collected = self.collector.collect_next_batch()
        if collected is None:
            return False

        batch_path, metadata = collected
        self.logger.info("Batch metadata: %s", metadata)
        self.logger.info("Batch saved to %s", batch_path)
        return True

    def inference(self, input_path: Path) -> Path:
        self.logger.info("Inference mode requested for %s", input_path)
        raise NotImplementedError(
            "Inference pipeline will be implemented after model training."
        )

    def summary(self) -> Path:
        self.logger.info("Summary mode requested")
        raise NotImplementedError(
            "Summary report generation will be implemented after metrics storage."
        )

    def _ensure_directories(self) -> None:
        for directory in (
            self.config.paths.external_data_dir,
            self.config.paths.raw_data_dir,
            self.config.paths.processed_data_dir,
            self.config.paths.artifacts_dir,
            self.config.paths.models_dir,
            self.config.paths.logs_dir,
            self.config.paths.reports_dir,
            self.config.paths.predictions_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
