from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.core.config import Config


STORE_KEY = "Store"
STORE_METADATA_FILE = "store.csv"


def load_source_dataset(config: Config) -> pd.DataFrame:
    frames = []
    for source_path in config.data.source_paths:
        if not source_path.exists():
            raise FileNotFoundError(f"Source dataset not found: {source_path}")
        frame = pd.read_csv(source_path, low_memory=False)
        frame["_source_file"] = source_path.name
        frames.append(frame)

    if not frames:
        raise ValueError("No source datasets configured.")

    dataset = pd.concat(frames, ignore_index=True)
    return merge_store_metadata(dataset, config)


def merge_store_metadata(dataset: pd.DataFrame, config: Config) -> pd.DataFrame:
    if STORE_KEY not in dataset.columns:
        return dataset
    if "StoreType" in dataset.columns:
        return dataset

    store_path = _store_metadata_path(config)
    if store_path is None:
        return dataset

    store = pd.read_csv(store_path, low_memory=False)
    if STORE_KEY not in store.columns:
        raise ValueError(f"Store metadata key is missing: {store_path}:{STORE_KEY}")

    merged = dataset.merge(store, on=STORE_KEY, how="left", validate="many_to_one")
    missing_store_rows = int(merged["StoreType"].isna().sum())
    if missing_store_rows:
        raise ValueError(
            f"Store metadata merge left {missing_store_rows} rows without StoreType."
        )
    return merged


def _store_metadata_path(config: Config) -> Path | None:
    if config.data.store_path is not None:
        return config.data.store_path

    candidates = [
        config.paths.external_data_dir / STORE_METADATA_FILE,
        *(path.parent / STORE_METADATA_FILE for path in config.data.source_paths),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
