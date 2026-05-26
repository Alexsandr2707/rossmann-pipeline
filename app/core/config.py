from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.models.factory import MODEL_NAMES, canonical_model_name


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


def _bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ValueError(f"{field_name} must be a boolean.")


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
    model_parameters = _model_parameters(raw.get("model_parameters", {}))

    target_missing_strategy = str(target_preprocessing["missing_strategy"])
    _validate_target_preprocessing(target_missing_strategy)

    model_values = _validated_model_values(model)

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
        ),
        target_preprocessing=TargetPreprocessingConfig(
            missing_strategy=target_missing_strategy,
            missing_fill_value=float(target_preprocessing["missing_fill_value"]),
            add_missing_indicator=_bool(
                target_preprocessing["add_missing_indicator"],
                "target_preprocessing.add_missing_indicator",
            ),
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
            primary_metric=model_values["primary_metric"],
            candidate_models=model_values["candidate_models"],
            training_mode=model_values["training_mode"],
            selected_model=model_values["selected_model"],
            update_strategy=model_values["update_strategy"],
            stream_batch_days=model_values["stream_batch_days"],
            initial_train_ratio=model_values["initial_train_ratio"],
            validation_ratio=model_values["validation_ratio"],
            stream_ratio=model_values["stream_ratio"],
            rolling_train_period_days=model_values["rolling_train_period_days"],
            pretrain_mark_collector_state=model_values["pretrain_mark_collector_state"],
            model_parameters=model_parameters,
        ),
    )


def _validate_target_preprocessing(missing_strategy: str) -> None:
    allowed = {"drop", "fill_zero", "fill_value", "keep"}
    if missing_strategy not in allowed:
        raise ValueError(
            "target_preprocessing.missing_strategy must be one of "
            f"{sorted(allowed)}."
        )


def _validated_model_values(model: dict[str, Any]) -> dict[str, Any]:
    primary_metric = str(model["primary_metric"])
    allowed_primary_metrics = {"rmse", "mae", "smape"}
    if primary_metric not in allowed_primary_metrics:
        raise ValueError(
            f"primary_metric must be one of {sorted(allowed_primary_metrics)}."
        )

    candidate_models_raw = model.get("candidate_models")
    if (
        not isinstance(candidate_models_raw, (list, tuple))
        or len(candidate_models_raw) == 0
    ):
        raise ValueError("candidate_models must not be empty.")
    candidate_models = tuple(str(item) for item in candidate_models_raw)
    canonical_candidates = tuple(
        canonical_model_name(item) for item in candidate_models
    )
    for model_name in canonical_candidates:
        if model_name not in MODEL_NAMES:
            raise ValueError(
                f"candidate_models contains unsupported model: {model_name}"
            )

    training_mode = str(model.get("training_mode", "all"))
    if training_mode not in {"all", "single"}:
        raise ValueError("training_mode must be 'all' or 'single'.")

    selected_model = str(model.get("selected_model", candidate_models[0]))
    canonical_selected_model = canonical_model_name(selected_model)
    if canonical_selected_model not in canonical_candidates:
        raise ValueError(
            f"selected_model must be listed in candidate_models: {selected_model}"
        )

    update_strategy = str(model.get("update_strategy", "full_refit"))
    if update_strategy not in {"full_refit", "rolling_refit", "incremental"}:
        raise ValueError(
            "update_strategy must be one of "
            "['full_refit', 'incremental', 'rolling_refit']."
        )

    stream_batch_days = int(model.get("stream_batch_days", 7))
    if stream_batch_days <= 0:
        raise ValueError("stream_batch_days must be positive.")

    rolling_train_period_days = int(model.get("rolling_train_period_days", 365))
    if rolling_train_period_days <= 0:
        raise ValueError("rolling_train_period_days must be positive.")

    initial_train_ratio = float(model.get("initial_train_ratio", 0.50))
    validation_ratio = float(model.get("validation_ratio", 0.20))
    stream_ratio = float(model.get("stream_ratio", 0.30))
    if initial_train_ratio <= 0 or validation_ratio <= 0 or stream_ratio <= 0:
        raise ValueError("split ratios must be positive.")
    if initial_train_ratio + validation_ratio >= 1.0:
        raise ValueError("split ratios must leave a stream period.")

    return {
        "primary_metric": primary_metric,
        "candidate_models": candidate_models,
        "training_mode": training_mode,
        "selected_model": selected_model,
        "update_strategy": update_strategy,
        "stream_batch_days": stream_batch_days,
        "initial_train_ratio": initial_train_ratio,
        "validation_ratio": validation_ratio,
        "stream_ratio": stream_ratio,
        "rolling_train_period_days": rolling_train_period_days,
        "pretrain_mark_collector_state": _bool(
            model.get("pretrain_mark_collector_state", True),
            "pretrain_mark_collector_state",
        ),
    }


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
