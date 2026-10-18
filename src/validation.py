"""Data validation: schema, ranges, missing values, duplicates, leakage."""
from __future__ import annotations

import pandas as pd

from src import config


class DataValidationError(ValueError):
    """Raised when the dataset breaks a hard rule (schema, range, leakage)."""


def assert_no_leakage(feature_names) -> None:
    """Refuse to train if a target-derived column is among the model inputs."""
    leaked = sorted(set(feature_names) & config.FORBIDDEN_INPUTS)
    if leaked:
        raise DataValidationError(f"Target leakage: {leaked} cannot be model inputs.")


def validate_raw_dataframe(df: pd.DataFrame) -> dict:
    """Check the raw CSV and return an evidence report.

    Missing values and duplicates are *expected* (they are deliberate) so they
    are reported, not rejected. Wrong schema, impossible values and bad labels
    are rejected.
    """
    missing_cols = [c for c in config.RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    problems: list[str] = []
    for col, (lo, hi) in config.VALUE_RANGES.items():
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().sum() > df[col].isna().sum():
            problems.append(f"{col}: contains non-numeric text")
            continue
        bad = int(((values < lo) | (values > hi)).sum())
        if bad:
            problems.append(f"{col}: {bad} value(s) outside [{lo}, {hi}]")

    if not df["season"].dropna().isin(config.ALLOWED_SEASONS).all():
        problems.append("season: values other than A/B found")
    if not df["cooperative_support"].dropna().isin([0, 1]).all():
        problems.append("cooperative_support: values other than 0/1 found")
    if not df[config.CLF_TARGET].isin([0, 1]).all():
        problems.append("high_risk: values other than 0/1 found")
    if (df[config.REG_TARGET] < config.MIN_YIELD_TONS - 1e-9).any():
        problems.append(f"yield_tons: values below {config.MIN_YIELD_TONS}")
    if df[config.REG_TARGET].isna().any() or df[config.CLF_TARGET].isna().any():
        problems.append("targets contain missing values")
    if problems:
        raise DataValidationError("Dataset rejected: " + "; ".join(problems))

    assert_no_leakage(config.BASE_FEATURES)
    duplicates = int(df.duplicated().sum())
    return {
        "rows_total": int(len(df)),
        "rows_unique": int(len(df) - duplicates),
        "duplicate_rows": duplicates,
        "missing_by_column": {c: int(v) for c, v in df.isna().sum().items() if v},
        "missing_percent_by_column": {
            c: round(100 * v / len(df), 2) for c, v in df.isna().sum().items() if v
        },
        "range_check": "passed",
        "leakage_check": "passed: no target-derived column is an input feature",
        "high_risk_share": round(float(df[config.CLF_TARGET].mean()), 4),
    }
