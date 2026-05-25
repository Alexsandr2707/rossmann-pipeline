from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    random_seed: int


@dataclass(frozen=True)
class DataConfig:
    source_paths: tuple[Path, ...]
    store_path: Path | None
    time_column: str
    target_column: str
    min_rows: int
    min_features: int
    min_categorical_features: int


@dataclass(frozen=True)
class TargetPreprocessingConfig:
    missing_strategy: str
    missing_fill_value: float
    add_missing_indicator: bool
    missing_indicator_suffix: str


@dataclass(frozen=True)
class DataSchemaConfig:
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    datetime_columns: tuple[str, ...]
    id_columns: tuple[str, ...]
    service_columns: tuple[str, ...]


@dataclass(frozen=True)
class PathConfig:
    external_data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    artifacts_dir: Path
    models_dir: Path
    logs_dir: Path
    reports_dir: Path
    predictions_dir: Path
    collector_state_path: Path
    batch_metadata_path: Path
    data_quality_history_path: Path
    performance_history_path: Path
    model_metrics_history_path: Path
    best_model_path: Path
    pipeline_log_path: Path


@dataclass(frozen=True)
class ModelConfig:
    primary_metric: str
    candidate_models: tuple[str, ...]
    training_mode: str
    selected_model: str
    update_strategy: str
    stream_batch_days: int
    initial_train_ratio: float
    validation_ratio: float
    stream_ratio: float
    rolling_train_period_days: int
    pretrain_mark_collector_state: bool
    model_parameters: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class Config:
    project: ProjectConfig
    data: DataConfig
    target_preprocessing: TargetPreprocessingConfig
    data_schema: DataSchemaConfig
    paths: PathConfig
    model: ModelConfig


def _require_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid config section: {name}")
    return section


def _path(value: str) -> Path:
    return Path(value)


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file.read())

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config file: {config_path}")

    project = _require_section(raw, "project")
    data = _require_section(raw, "data")
    target_preprocessing = _require_section(raw, "target_preprocessing")
    data_schema = _require_section(raw, "data_schema")
    paths = _require_section(raw, "paths")
    model = _require_section(raw, "model")

    return Config(
        project=ProjectConfig(
            name=str(project["name"]),
            random_seed=int(project["random_seed"]),
        ),
        data=DataConfig(
            source_paths=tuple(_path(item) for item in data["source_paths"]),
            store_path=_path(data["store_path"]) if data.get("store_path") else None,
            time_column=str(data["time_column"]),
            target_column=str(data["target_column"]),
            min_rows=int(data["min_rows"]),
            min_features=int(data["min_features"]),
            min_categorical_features=int(data["min_categorical_features"]),
        ),
        target_preprocessing=TargetPreprocessingConfig(
            missing_strategy=str(target_preprocessing["missing_strategy"]),
            missing_fill_value=float(target_preprocessing["missing_fill_value"]),
            add_missing_indicator=bool(target_preprocessing["add_missing_indicator"]),
            missing_indicator_suffix=str(
                target_preprocessing["missing_indicator_suffix"]
            ),
        ),
        data_schema=DataSchemaConfig(
            numeric_columns=tuple(str(item) for item in data_schema["numeric_columns"]),
            categorical_columns=tuple(
                str(item) for item in data_schema["categorical_columns"]
            ),
            datetime_columns=tuple(
                str(item) for item in data_schema["datetime_columns"]
            ),
            id_columns=tuple(str(item) for item in data_schema["id_columns"]),
            service_columns=tuple(str(item) for item in data_schema["service_columns"]),
        ),
        paths=PathConfig(
            external_data_dir=_path(paths["external_data_dir"]),
            raw_data_dir=_path(paths["raw_data_dir"]),
            processed_data_dir=_path(paths["processed_data_dir"]),
            artifacts_dir=_path(paths["artifacts_dir"]),
            models_dir=_path(paths["models_dir"]),
            logs_dir=_path(paths["logs_dir"]),
            reports_dir=_path(paths["reports_dir"]),
            predictions_dir=_path(paths["predictions_dir"]),
            collector_state_path=_path(paths["collector_state_path"]),
            batch_metadata_path=_path(paths["batch_metadata_path"]),
            data_quality_history_path=_path(
                paths.get(
                    "data_quality_history_path",
                    "artifacts/data_quality_history.csv",
                )
            ),
            performance_history_path=_path(
                paths.get(
                    "performance_history_path",
                    "artifacts/performance_history.csv",
                )
            ),
            model_metrics_history_path=_path(paths["model_metrics_history_path"]),
            best_model_path=_path(paths["best_model_path"]),
            pipeline_log_path=_path(paths["pipeline_log_path"]),
        ),
        model=ModelConfig(
            primary_metric=str(model["primary_metric"]),
            candidate_models=tuple(model["candidate_models"]),
            training_mode=str(model.get("training_mode", "all")),
            selected_model=str(
                model.get("selected_model", model["candidate_models"][0])
            ),
            update_strategy=str(model.get("update_strategy", "full_refit")),
            stream_batch_days=int(model.get("stream_batch_days", 7)),
            initial_train_ratio=float(model.get("initial_train_ratio", 0.50)),
            validation_ratio=float(model.get("validation_ratio", 0.20)),
            stream_ratio=float(model.get("stream_ratio", 0.30)),
            rolling_train_period_days=int(model.get("rolling_train_period_days", 365)),
            pretrain_mark_collector_state=bool(
                model.get("pretrain_mark_collector_state", True)
            ),
            model_parameters=_model_parameters(raw.get("model_parameters", {})),
        ),
    )


def _model_parameters(raw: Any) -> dict[str, dict[str, Any]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Config section model_parameters must be a mapping.")

    parameters: dict[str, dict[str, Any]] = {}
    for model_name, model_params in raw.items():
        if model_params is None:
            parameters[str(model_name)] = {}
            continue
        if not isinstance(model_params, dict):
            raise ValueError(f"Model parameters for {model_name} must be a mapping.")
        parameters[str(model_name)] = {
            str(parameter_name): parameter_value
            for parameter_name, parameter_value in model_params.items()
        }
    return parameters
