"""Cleaning, train/test split and reusable preprocessing pipelines.

All fitted transformations (imputers, scaler, encoder) live inside sklearn
Pipelines, so they are fitted on training rows only and applied identically
at inference time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


class FarmFeatureBuilder(BaseEstimator, TransformerMixin):
    """Add three agronomy-inspired features. Uses only the row's own inputs."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        frame["temp_gap_c"] = (frame["avg_temp_c"] - config.OPTIMUM_TEMP_C).abs()
        with np.errstate(divide="ignore", invalid="ignore"):
            frame["fertilizer_per_ha"] = frame["fertilizer_kg"] / frame["field_size_ha"]
        frame["water_index"] = frame["rainfall_mm"] * frame["soil_moisture_pct"] / 1000.0
        return frame


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicate rows (before splitting, so copies cannot leak)."""
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    return cleaned, {
        "rows_before": int(before),
        "rows_after": int(len(cleaned)),
        "duplicates_removed": int(before - len(cleaned)),
        "missing_by_column_after_dedup": {c: int(v) for c, v in cleaned.isna().sum().items() if v},
        "missing_handling": "median imputation (numeric) and most-frequent (binary), fitted on training rows only inside the Pipeline",
    }


@dataclass
class SplitData:
    train: pd.DataFrame
    test: pd.DataFrame


def split_dataset(df: pd.DataFrame, seed: int) -> SplitData:
    """Fixed 80/20 split, random_state = LLLL, stratified on the risk label."""
    train, test = train_test_split(
        df,
        test_size=config.TEST_FRACTION,
        random_state=seed,
        stratify=df[config.CLF_TARGET],
    )
    return SplitData(train=train.reset_index(drop=True), test=test.reset_index(drop=True))


def features_of(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs only (never targets, never the ID)."""
    return df[config.BASE_FEATURES]


def build_supervised_preprocessor() -> ColumnTransformer:
    numeric = config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             config.CATEGORICAL_FEATURES),
            ("bin", SimpleImputer(strategy="most_frequent"), config.BINARY_FEATURES),
        ]
    )


def make_model_pipeline(estimator) -> Pipeline:
    """features -> preprocessing -> estimator, one object for train and serve."""
    return Pipeline(
        [
            ("features", FarmFeatureBuilder()),
            ("prep", build_supervised_preprocessor()),
            ("model", estimator),
        ]
    )


def build_cluster_preprocessor() -> ColumnTransformer:
    """Scaled raw inputs only: clustering never sees the targets."""
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             config.CATEGORICAL_FEATURES),
            ("bin", SimpleImputer(strategy="most_frequent"), config.BINARY_FEATURES),
        ]
    )
