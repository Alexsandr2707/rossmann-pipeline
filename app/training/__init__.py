"""Model training, diagnostics and interpretation workflows."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ModelDiagnosticsWriter": "app.training.model_diagnostics",
    "ModelInterpretationWriter": "app.training.model_interpretation",
    "ModelTrainer": "app.training.model_training",
}

__all__ = sorted(_EXPORTS)  # type: ignore


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
