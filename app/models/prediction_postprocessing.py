from __future__ import annotations

import numpy as np


def non_negative_predictions(predictions) -> np.ndarray:
    return np.clip(np.asarray(predictions, dtype=float), 0.0, None)
