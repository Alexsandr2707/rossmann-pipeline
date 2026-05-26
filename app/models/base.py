from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class RegressionModel(Protocol):
    supports_incremental_update: bool

    def fit(self, x: Any, y: Any) -> Any: ...

    def update(self, x: Any, y: Any) -> Any: ...

    def predict(self, x: Any) -> np.ndarray: ...


class RefitUpdateMixin:
    supports_incremental_update = False

    def update(self, x: Any, y: Any) -> Any:
        return self.fit(x, y)  # type: ignore


class PartialFitUpdateMixin:
    supports_incremental_update = True

    def update(self, x: Any, y: Any) -> Any:
        partial_fit = getattr(self, "partial_fit", None)
        if partial_fit is None:
            raise TypeError(
                f"{self.__class__.__name__} does not implement partial_fit."
            )
        return partial_fit(x, y)
