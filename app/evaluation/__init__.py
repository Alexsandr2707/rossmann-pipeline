"""Offline model evaluation workflows."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "OfflineModelEvaluator": "app.evaluation.offline_evaluation",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
