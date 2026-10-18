"""Project-wide constants for the Musanze Harvest Intelligence Service.

Everything that several modules must agree on (column names, valid ranges,
split fraction, risk bands) lives here so that training, the CLI and the API
can never drift apart.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- column definitions ------------------------------------------------------
ID_COLUMN = "farm_id"
NUMERIC_FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
]
CATEGORICAL_FEATURES = ["season"]
BINARY_FEATURES = ["cooperative_support"]
BASE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
ENGINEERED_FEATURES = ["temp_gap_c", "fertilizer_per_ha", "water_index"]

REG_TARGET = "yield_tons"
CLF_TARGET = "high_risk"
RAW_COLUMNS = [ID_COLUMN] + BASE_FEATURES + [REG_TARGET, CLF_TARGET]

# Columns that must never be used as model inputs (target leakage).
FORBIDDEN_INPUTS = frozenset({"yield_tons", "yield_per_ha", "high_risk", "farm_id"})

# Inclusive valid ranges taken from the generator specification.
VALUE_RANGES = {
    "rainfall_mm": (600.0, 1400.0),
    "avg_temp_c": (12.0, 26.0),
    "soil_moisture_pct": (20.0, 80.0),
    "fertilizer_kg": (0.0, 250.0),
    "field_size_ha": (0.2, 4.0),
}
ALLOWED_SEASONS = ("A", "B")
MIN_YIELD_TONS = 0.5
OPTIMUM_TEMP_C = 18.0  # agronomic optimum stated in the yield formula

# ---- experiment settings -----------------------------------------------------
TEST_FRACTION = 0.20
CV_FOLDS = 5
K_VALUES = (2, 3, 4, 5, 6)
RISK_LOW_BELOW = 0.35   # probability below this  -> "low"
RISK_HIGH_FROM = 0.65   # probability from this   -> "high"

SYNTHETIC_NOTICE = (
    "Synthetic classroom data. Decision-support prototype only: it does not "
    "replace an agronomist and must not decide access to finance, insurance "
    "or public services. A human must review every result."
)


@dataclass(frozen=True)
class ProjectPaths:
    """All filesystem locations used by the system, relative to one root."""

    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def plots_dir(self) -> Path:
        return self.artifacts_dir / "plots"

    @property
    def fingerprint_path(self) -> Path:
        return self.artifacts_dir / "data_fingerprint.txt"

    def csv_path(self, reg_digits: str) -> Path:
        return self.data_dir / f"AI_A2_{reg_digits}.csv"

    def reset_outputs(self) -> None:
        """Delete and recreate generated folders so every run starts clean."""
        for folder in (self.models_dir, self.artifacts_dir):
            if folder.exists():
                shutil.rmtree(folder)
        for folder in (self.data_dir, self.models_dir, self.artifacts_dir, self.plots_dir):
            folder.mkdir(parents=True, exist_ok=True)
