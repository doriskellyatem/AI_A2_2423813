"""Typed request validation shared by predict.py and the FastAPI service."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import VALUE_RANGES


def _bounded(name: str):
    lo, hi = VALUE_RANGES[name]
    return Annotated[float, Field(strict=True, ge=lo, le=hi, allow_inf_nan=False,
                                  description=f"Allowed range {lo} to {hi}")]


class FarmRecord(BaseModel):
    """One farm. Every field is required; unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    rainfall_mm: _bounded("rainfall_mm")
    avg_temp_c: _bounded("avg_temp_c")
    soil_moisture_pct: _bounded("soil_moisture_pct")
    fertilizer_kg: _bounded("fertilizer_kg")
    field_size_ha: _bounded("field_size_ha")
    season: Literal["A", "B"]
    cooperative_support: Literal[0, 1]


def describe_errors(exc) -> list[dict]:
    """Turn a pydantic ValidationError into short, readable problems."""
    problems = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"] if p != "body") or "(body)"
        problems.append({"field": field, "problem": err["msg"], "type": err["type"]})
    return problems
