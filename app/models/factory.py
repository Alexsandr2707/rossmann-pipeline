from __future__ import annotations

from typing import Any

from app.models.sklearn_regression import (
    PartialFitSGDRegressor,
    RefitDecisionTreeRegressor,
    RefitKNeighborsRegressor,
    RefitRandomForestRegressor,
    RefitRidgeRegressor,
)

DECISION_TREE_REGRESSION_MODEL_NAME = "decision_tree_regression"
KNN_REGRESSION_MODEL_NAME = "knn_regression"
RANDOM_FOREST_REGRESSION_MODEL_NAME = "random_forest_regression"
RIDGE_REGRESSION_MODEL_NAME = "ridge_regression"
SGD_REGRESSION_MODEL_NAME = "sgd_regression"
MODEL_ALIASES: dict[str, str] = {
    "decision_tree": DECISION_TREE_REGRESSION_MODEL_NAME,
    "knn": KNN_REGRESSION_MODEL_NAME,
    "random_forest": RANDOM_FOREST_REGRESSION_MODEL_NAME,
    "ridge": RIDGE_REGRESSION_MODEL_NAME,
    "sgd": SGD_REGRESSION_MODEL_NAME,
}
MODEL_NAMES = {
    DECISION_TREE_REGRESSION_MODEL_NAME,
    KNN_REGRESSION_MODEL_NAME,
    RANDOM_FOREST_REGRESSION_MODEL_NAME,
    RIDGE_REGRESSION_MODEL_NAME,
    SGD_REGRESSION_MODEL_NAME,
}


def canonical_model_name(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


def make_model(
    model_name: str,
    model_parameters: dict[str, Any],
    random_seed: int,
) -> Any:
    name = canonical_model_name(model_name)
    parameters = dict(model_parameters)

    if name == DECISION_TREE_REGRESSION_MODEL_NAME:
        return RefitDecisionTreeRegressor(**_with_random_state(parameters, random_seed))
    if name == KNN_REGRESSION_MODEL_NAME:
        return RefitKNeighborsRegressor(**parameters)
    if name == RANDOM_FOREST_REGRESSION_MODEL_NAME:
        return RefitRandomForestRegressor(**_with_random_state(parameters, random_seed))
    if name == RIDGE_REGRESSION_MODEL_NAME:
        return RefitRidgeRegressor(**parameters)
    if name == SGD_REGRESSION_MODEL_NAME:
        return PartialFitSGDRegressor(**_with_random_state(parameters, random_seed))
    raise ValueError(f"Unsupported model name: {model_name}")


def supports_incremental_update(model: Any) -> bool:
    return bool(getattr(model, "supports_incremental_update", False))


def model_signature(
    model_name: str, model_parameters: dict[str, Any]
) -> dict[str, Any]:
    return {
        "model_name": canonical_model_name(model_name),
        "model_parameters": model_parameters,
    }


def _with_random_state(
    model_parameters: dict[str, Any],
    random_seed: int,
) -> dict[str, Any]:
    parameters = dict(model_parameters)
    parameters.setdefault("random_state", random_seed)
    return parameters
