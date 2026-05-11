from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    random_seed: int


@dataclass(frozen=True)
class DataConfig:
    source_paths: tuple[Path, ...]
    time_column: str
    target_column: str
    batch_size: int
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
    model_metrics_history_path: Path
    best_model_path: Path
    pipeline_log_path: Path


@dataclass(frozen=True)
class QualityConfig:
    max_missing_part: float
    max_duplicate_part: float


@dataclass(frozen=True)
class ModelConfig:
    primary_metric: str
    incremental_model: str
    candidate_models: tuple[str, ...]
    training_mode: str
    selected_model: str
    update_strategy: str
    initial_training_batches: int
    initial_training_max_rows: int
    model_parameters: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class Config:
    project: ProjectConfig
    data: DataConfig
    target_preprocessing: TargetPreprocessingConfig
    data_schema: DataSchemaConfig
    paths: PathConfig
    quality: QualityConfig
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
        raw = _load_yaml(file.read())

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config file: {config_path}")

    project = _require_section(raw, "project")
    data = _require_section(raw, "data")
    target_preprocessing = _require_section(raw, "target_preprocessing")
    data_schema = _require_section(raw, "data_schema")
    paths = _require_section(raw, "paths")
    quality = _require_section(raw, "quality")
    model = _require_section(raw, "model")

    return Config(
        project=ProjectConfig(
            name=str(project["name"]),
            random_seed=int(project["random_seed"]),
        ),
        data=DataConfig(
            source_paths=tuple(_path(item) for item in data["source_paths"]),
            time_column=str(data["time_column"]),
            target_column=str(data["target_column"]),
            batch_size=int(data["batch_size"]),
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
            datetime_columns=tuple(str(item) for item in data_schema["datetime_columns"]),
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
            data_quality_history_path=_path(paths["data_quality_history_path"]),
            model_metrics_history_path=_path(paths["model_metrics_history_path"]),
            best_model_path=_path(paths["best_model_path"]),
            pipeline_log_path=_path(paths["pipeline_log_path"]),
        ),
        quality=QualityConfig(
            max_missing_part=float(quality["max_missing_part"]),
            max_duplicate_part=float(quality["max_duplicate_part"]),
        ),
        model=ModelConfig(
            primary_metric=str(model["primary_metric"]),
            incremental_model=str(model["incremental_model"]),
            candidate_models=tuple(model["candidate_models"]),
            training_mode=str(model.get("training_mode", "all")),
            selected_model=str(model.get("selected_model", model["incremental_model"])),
            update_strategy=str(model.get("update_strategy", "refit")),
            initial_training_batches=int(model.get("initial_training_batches", 1)),
            initial_training_max_rows=int(model.get("initial_training_max_rows", 0)),
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
            raise ValueError(
                f"Model parameters for {model_name} must be a mapping."
            )
        parameters[str(model_name)] = {
            str(parameter_name): parameter_value
            for parameter_name, parameter_value in model_params.items()
        }
    return parameters


def _load_yaml(content: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return _load_simple_yaml(content)

    loaded = yaml.safe_load(content)
    if not isinstance(loaded, dict):
        raise ValueError("Config file must contain a mapping.")
    return loaded


def _load_simple_yaml(content: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_section: str | None = None
    current_list_key: str | None = None
    current_nested_key: str | None = None

    for raw_line in content.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))

        if indent == 0:
            if not line.endswith(":"):
                raise ValueError(f"Invalid top-level YAML line: {raw_line}")
            current_section = line[:-1]
            result[current_section] = {}
            current_list_key = None
            current_nested_key = None
            continue

        if current_section is None:
            raise ValueError(f"YAML key without section: {raw_line}")

        stripped = line.strip()
        section = result[current_section]

        if stripped.startswith("- "):
            if current_list_key is None:
                raise ValueError(f"YAML list item without list key: {raw_line}")
            section[current_list_key].append(_parse_scalar(stripped[2:]))
            continue

        key, separator, value = stripped.partition(":")
        if not separator:
            raise ValueError(f"Invalid YAML key-value line: {raw_line}")

        if indent == 4:
            if current_nested_key is None:
                raise ValueError(f"Nested YAML key without parent: {raw_line}")
            if not isinstance(section[current_nested_key], dict):
                section[current_nested_key] = {}
            section[current_nested_key][key] = _parse_scalar(value.strip())
            continue

        if value.strip() == "":
            section[key] = []
            current_list_key = key
            current_nested_key = key
        else:
            section[key] = _parse_scalar(value.strip())
            current_list_key = None
            current_nested_key = None

    return result


def _parse_scalar(value: str) -> str | int | float | bool | None:
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "false"}:
        return value == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")
