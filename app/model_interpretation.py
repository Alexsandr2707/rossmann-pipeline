from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import Config


class ModelInterpretationWriter:
    _LATEST_REPORT_NAME = "model_interpretation_latest.md"
    _ARCHIVE_REPORT_DIR = "archive/model_interpretation"

    def __init__(self, config: Config) -> None:
        self.config = config

    def write_model_interpretation(
        self,
        metrics: dict[str, Any],
        preprocessor: Any,
        estimator: Any,
    ) -> dict[str, str]:
        self.config.paths.reports_dir.mkdir(parents=True, exist_ok=True)
        self.config.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

        model_name = str(metrics["model_name"])
        batch_index = int(metrics["batch_index"])
        interpretation_base = (
            f"model_interpretation_batch_{batch_index:04d}_{model_name}"
        )
        latest_report_path = self.config.paths.reports_dir / self._LATEST_REPORT_NAME
        archive_report_path = (
            self.config.paths.reports_dir
            / self._ARCHIVE_REPORT_DIR
            / f"{interpretation_base}.md"
        )
        archive_report_path.parent.mkdir(parents=True, exist_ok=True)

        feature_names = self._feature_names(preprocessor)
        interpretation_kind, importances = self._importance_values(estimator)
        top_features_path = (
            self.config.paths.artifacts_dir / f"{interpretation_base}_top_features.csv"
        )
        top_features_written = self._write_top_features_csv(
            top_features_path,
            feature_names,
            interpretation_kind,
            importances,
        )

        self._write_markdown(
            latest_report_path,
            metrics,
            feature_names,
            interpretation_kind,
            importances,
            top_features_path if top_features_written else None,
        )
        self._write_markdown(
            archive_report_path,
            metrics,
            feature_names,
            interpretation_kind,
            importances,
            top_features_path if top_features_written else None,
        )

        output = {
            "interpretation_report_path": str(latest_report_path),
            "archive_interpretation_report_path": str(archive_report_path),
            "latest_interpretation_report_path": str(latest_report_path),
        }
        if top_features_written:
            output["interpretation_top_features_path"] = str(top_features_path)
        return output

    def _feature_names(self, preprocessor: Any) -> list[str]:
        if preprocessor is None or not hasattr(preprocessor, "get_feature_names_out"):
            return []
        try:
            names = preprocessor.get_feature_names_out()
        except Exception:
            return []
        values = np.asarray(names).astype(str).tolist()
        return [value for value in values if value]

    def _importance_values(
        self, estimator: Any
    ) -> tuple[str | None, np.ndarray | None]:
        if estimator is None:
            return None, None
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(getattr(estimator, "feature_importances_"), dtype=float)
            return "feature_importances", values
        if hasattr(estimator, "coef_"):
            raw = np.asarray(getattr(estimator, "coef_"), dtype=float)
            return "coefficients", np.ravel(raw)
        return None, None

    def _write_top_features_csv(
        self,
        output_path: Path,
        feature_names: list[str],
        interpretation_kind: str | None,
        importances: np.ndarray | None,
    ) -> bool:
        if interpretation_kind is None or importances is None:
            return False
        if len(feature_names) != len(importances):
            return False
        frame = pd.DataFrame(
            {
                "feature": feature_names,
                "value": importances,
                "abs_value": np.abs(importances),
            }
        ).sort_values("abs_value", ascending=False)
        frame.to_csv(output_path, index=False)
        return True

    def _write_markdown(
        self,
        output_path: Path,
        metrics: dict[str, Any],
        feature_names: list[str],
        interpretation_kind: str | None,
        importances: np.ndarray | None,
        top_features_path: Path | None,
    ) -> None:
        lines = [
            "# Model interpretation",
            "",
            f"- model_name: {metrics.get('model_name', '')}",
            f"- batch_index: {metrics.get('batch_index', '')}",
            f"- period_type: {metrics.get('period_type', '')}",
            "",
        ]
        if interpretation_kind is None or importances is None:
            lines.extend(
                [
                    "Interpretation unavailable: estimator does not provide "
                    "`feature_importances_` or `coef_`.",
                ]
            )
            output_path.write_text("\n".join(lines), encoding="utf-8")
            return

        lines.append(f"- interpretation_type: {interpretation_kind}")
        lines.append(f"- feature_count: {len(importances)}")
        if top_features_path is not None:
            lines.append(f"- top_features_csv: {top_features_path}")
        lines.extend(["", "## Top features (by absolute value)", ""])

        if len(feature_names) == len(importances):
            frame = pd.DataFrame(
                {
                    "feature": feature_names,
                    "value": importances,
                    "abs_value": np.abs(importances),
                }
            ).sort_values("abs_value", ascending=False)
            preview = frame.head(20)
            lines.append(preview.to_markdown(index=False, floatfmt=".6g"))
        else:
            lines.append(
                "Interpretation values were produced, but feature names are unavailable "
                "or have mismatched size after preprocessing."
            )

        output_path.write_text("\n".join(lines), encoding="utf-8")
