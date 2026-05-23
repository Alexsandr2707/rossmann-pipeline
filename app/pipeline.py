from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from app.config import Config
from app.reporting import generate_html_report, generate_summary_report


class Pipeline:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._ensure_directories()
        self.collector = None
        self.data_quality_analyzer = None
        self.model_trainer = None
        self.offline_evaluator = None

    def update(self) -> int:
        self.logger.info("Update mode requested")
        collector = self._get_collector()
        validation = collector.validate_source_dataset()
        self.logger.info("Source dataset validation: %s", validation)

        model_trainer = self._get_model_trainer()
        if not model_trainer.current_model_path.exists():
            self.logger.info(
                "Current model is missing. Running pretrain before stream update."
            )
            self.pretrain()

        collected = collector.collect_next_stream_batch()
        if collected is None:
            self.logger.info("Update completed: no new batches were available.")
            return 0

        from app.preprocessing import DataPreprocessor

        batch_path, metadata = collected
        self.logger.info("Stream batch metadata: %s", metadata)
        raw_batch = self._read_batch(batch_path)
        quality_metrics = self._get_data_quality_analyzer().analyze_batch(
            raw_batch,
            metadata,
        )
        self.logger.info("Data quality metrics: %s", quality_metrics)
        processed_dataset = DataPreprocessor(self.config).transform(raw_batch)
        processed_path = (
            self.config.paths.processed_data_dir
            / f"batch_{int(metadata['stream_batch_index']):04d}_processed.csv"
        )
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        processed_dataset.to_csv(processed_path, index=False)

        model_path, model_metrics = model_trainer.update_on_stream_batch(
            latest_processed_path=processed_path,
            raw_batch_path=batch_path,
            batch_metadata=metadata,
        )
        self.logger.info("Updated model saved to %s", model_path)
        self.logger.info("Stream update metrics: %s", model_metrics)
        self.logger.info("Update completed: processed 1 stream batch.")
        return 1

    def pretrain(self) -> Path:
        self.logger.info("Pretrain mode requested")
        collector = self._get_collector()
        validation = collector.validate_source_dataset()
        self.logger.info("Source dataset validation: %s", validation)

        from app.preprocessing import DataPreprocessor

        raw_dataset = collector.load_sorted_source_dataset()
        if raw_dataset.empty:
            raise ValueError("No rows available for pretraining.")

        processed_dataset = DataPreprocessor(self.config).transform(raw_dataset)
        processed_path = self.config.paths.processed_data_dir / "pretrain_processed.csv"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        processed_dataset.to_csv(processed_path, index=False)

        parsed_time = processed_dataset[self.config.data.time_column]
        batch_metadata = {
            "batch_index": -1,
            "batch_path": "pretrain",
            "latest_processed_path": str(processed_path),
            "rows": int(len(processed_dataset)),
            "pretrain_rows": int(len(processed_dataset)),
            "time_min": str(parsed_time.min()),
            "time_max": str(parsed_time.max()),
        }
        model_path, model_metrics = self._get_model_trainer().pretrain_on_dataset(
            processed_dataset,
            processed_path,
            batch_metadata,
        )
        if self.config.model.pretrain_mark_collector_state:
            state = collector.initialize_stream_state(processed_dataset)
            self.logger.info(
                "Stream state initialized after pretrain: stream_batch_index=%s",
                state.stream_batch_index,
            )

        self.logger.info("Pretrain model saved to %s", model_path)
        self.logger.info("Pretrain metrics: %s", model_metrics)
        return model_path

    def _read_batch(self, batch_path: Path):
        import pandas as pd

        return pd.read_csv(batch_path, low_memory=False)

    def inference(self, input_path: Path) -> Path:
        self.logger.info("Inference mode requested for %s", input_path)
        from app.prediction_serving import PredictionServing

        output_path = PredictionServing(self.config).predict_file(input_path)
        self.logger.info("Inference completed: predictions saved to %s", output_path)
        return output_path

    def summary(self) -> Path:
        self.logger.info("Summary mode requested")
        report_path = generate_summary_report(self.config)
        dashboard_path = generate_html_report(self.config)
        self.logger.info(
            "Summary completed: report saved to %s; dashboard saved to %s",
            report_path,
            dashboard_path,
        )
        return report_path

    def evaluate(self) -> Path:
        self.logger.info("Offline evaluation mode requested")
        report_path = self._get_offline_evaluator().evaluate()
        self.logger.info("Evaluation completed: report saved to %s", report_path)
        return report_path

    def reset(self) -> dict[str, int]:
        self.logger.info("Reset mode requested")
        cleanup_targets = {
            "raw_batches": self.config.paths.raw_data_dir.glob("batch_*.csv"),
            "processed_stream_batches": self.config.paths.processed_data_dir.glob(
                "batch_*_processed.csv"
            ),
            "pretrain_processed": [
                self.config.paths.processed_data_dir / "pretrain_processed.csv"
            ],
            "reports": self.config.paths.reports_dir.rglob("*"),
            "logs": self.config.paths.logs_dir.rglob("*"),
            "predictions": self.config.paths.predictions_dir.glob("*.csv"),
            "models": self.config.paths.models_dir.glob("*.pkl"),
            "offline_evaluation": [
                self.config.paths.artifacts_dir / "offline_model_evaluation.csv"
            ],
            "history_files": [
                self.config.paths.collector_state_path,
                self.config.paths.batch_metadata_path,
                self.config.paths.data_quality_history_path,
                self.config.paths.model_metrics_history_path,
            ],
        }

        removed: dict[str, int] = {}
        for group, paths in cleanup_targets.items():
            removed[group] = self._remove_paths(paths)

        self._remove_empty_directories(
            [
                self.config.paths.reports_dir / "figures" / "model",
                self.config.paths.reports_dir / "figures" / "history",
                self.config.paths.reports_dir / "figures",
                self.config.paths.reports_dir / "figures" / "archive" / "model_diagnostics",
                self.config.paths.reports_dir / "figures" / "archive",
                self.config.paths.reports_dir / "archive" / "model_diagnostics",
                self.config.paths.reports_dir / "archive",
                self.config.paths.reports_dir / "summary",
                self.config.paths.reports_dir,
                self.config.paths.logs_dir,
                self.config.paths.predictions_dir,
                self.config.paths.models_dir,
            ]
        )

        self.logger.info(
            "Reset completed: removed %s items (%s).",
            sum(removed.values()),
            ", ".join(
                f"{group}={count}" for group, count in removed.items() if count > 0
            )
            or "nothing to remove",
        )
        return removed

    def _remove_paths(self, paths: Iterable[Path]) -> int:
        removed = 0
        directories: list[Path] = []
        for path in paths:
            if path.name == ".gitkeep" or not path.exists():
                continue
            if path.is_file():
                path.unlink()
                removed += 1
                continue
            if path.is_dir():
                directories.append(path)
                continue
            path.unlink()
            removed += 1
        for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            try:
                path.rmdir()
                removed += 1
            except OSError:
                pass
        return removed

    def _remove_empty_directories(self, paths: list[Path]) -> None:
        for path in paths:
            if not path.exists() or not path.is_dir():
                continue
            try:
                path.rmdir()
            except OSError:
                pass

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

    def _get_collector(self):
        if self.collector is None:
            from app.data_collection import DataCollector

            self.collector = DataCollector(self.config)
        return self.collector

    def _get_data_quality_analyzer(self):
        if self.data_quality_analyzer is None:
            from app.data_quality import DataQualityAnalyzer

            self.data_quality_analyzer = DataQualityAnalyzer(self.config)
        return self.data_quality_analyzer

    def _get_model_trainer(self):
        if self.model_trainer is None:
            from app.model_training import ModelTrainer

            self.model_trainer = ModelTrainer(self.config)
        return self.model_trainer

    def _get_offline_evaluator(self):
        if self.offline_evaluator is None:
            from app.offline_evaluation import OfflineModelEvaluator

            self.offline_evaluator = OfflineModelEvaluator(self.config)
        return self.offline_evaluator
