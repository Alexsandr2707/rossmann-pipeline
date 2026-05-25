"""Core orchestration, configuration and logging utilities."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "Config": "app.core.config",
    "DataConfig": "app.core.config",
    "DataSchemaConfig": "app.core.config",
    "ModelConfig": "app.core.config",
    "PathConfig": "app.core.config",
    "Pipeline": "app.core.pipeline",
    "ProjectConfig": "app.core.config",
    "TargetPreprocessingConfig": "app.core.config",
    "configure_logging": "app.core.logging_utils",
    "load_config": "app.core.config",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
