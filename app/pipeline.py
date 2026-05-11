from __future__ import annotations

import logging
from pathlib import Path

from app.config import Config
from app.data_collection import DataCollector
from app.data_quality import DataQualityAnalyzer
from app.model_training import ModelTrainer


class Pipeline:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._ensure_directories()
        self.collector = DataCollector(config)
        self.quality_analyzer = DataQualityAnalyzer(config)
        self.model_trainer = ModelTrainer(config)

    def update(self) -> bool:
        self.logger.info("Update mode requested")
        validation = self.collector.validate_source_dataset()
        self.logger.info("Source dataset validation: %s", validation)

        collected_batches = self._collect_and_prepare_update_batches()
        if not collected_batches:
            return False

        processed_path, metadata = collected_batches[-1]

        model_path, model_metrics = self.model_trainer.train_on_processed_data(
            latest_processed_path=processed_path,
            batch_metadata=metadata,
        )
        self.logger.info("Model saved to %s", model_path)
        self.logger.info("Best model metrics: %s", model_metrics)
        return True

    def _collect_and_prepare_update_batches(self) -> list[tuple[Path, dict]]:
        batch_count = self._update_batch_count()
        prepared_batches: list[tuple[Path, dict]] = []

        for _ in range(batch_count):
            collected = self.collector.collect_next_batch()
            if collected is None:
                break

            batch_path, metadata = collected
            self.logger.info("Batch metadata: %s", metadata)
            self.logger.info("Batch saved to %s", batch_path)

            processed_path, quality_metrics, report_path = (
                self.quality_analyzer.analyze_and_clean_batch(batch_path, metadata)
            )
            self.logger.info("Data quality history updated for %s", processed_path)
            self.logger.info("EDA report saved to %s", report_path)
            self.logger.info("Quality passed: %s", quality_metrics["quality_passed"])
            prepared_batches.append((processed_path, metadata))

        return prepared_batches

    def _update_batch_count(self) -> int:
        if (
            self.config.model.update_strategy == "incremental"
            and not self.model_trainer.current_model_path.exists()
        ):
            return max(self.config.model.initial_training_batches, 1)
        return 1

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

    def reset(self) -> dict[str, int]:
        self.logger.info("Reset mode requested")
        cleanup_targets = {
            "raw_batches": list(self.config.paths.raw_data_dir.glob("batch_*.csv")),
            "processed_batches": list(
                self.config.paths.processed_data_dir.glob("batch_*_processed.csv")
            ),
            "eda_reports": list(self.config.paths.reports_dir.glob("eda_batch_*.md")),
            "predictions": list(self.config.paths.predictions_dir.glob("*.csv")),
            "models": list(self.config.paths.models_dir.glob("*.pkl")),
            "history_files": [
                self.config.paths.collector_state_path,
                self.config.paths.batch_metadata_path,
                self.config.paths.data_quality_history_path,
                self.config.paths.model_metrics_history_path,
            ],
        }

        removed: dict[str, int] = {}
        for group, paths in cleanup_targets.items():
            removed[group] = self._remove_files(paths)

        self.logger.info("Reset completed: %s", removed)
        return removed

    def _remove_files(self, paths: list[Path]) -> int:
        removed = 0
        for path in paths:
            if not path.exists() or not path.is_file():
                continue
            path.unlink()
            removed += 1
        return removed

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
