"""Load saved artifacts and score one validated farm record."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src import config
from src.schemas import FarmRecord


class ArtifactsMissingError(FileNotFoundError):
    """Raised when run_all.py has not been executed yet."""


@dataclass
class ModelBundle:
    regression: object
    classification: object
    clustering: object
    metadata: dict
    regression_report: dict
    classification_report: dict
    clustering_report: dict


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_bundle(root: Path = config.PROJECT_ROOT) -> ModelBundle:
    paths = config.ProjectPaths(Path(root))
    needed = [paths.models_dir / f for f in
              ("regression_model.joblib", "classification_model.joblib", "cluster_model.joblib")]
    needed += [paths.artifacts_dir / f for f in
               ("run_metadata.json", "regression_metrics.json",
                "classification_metrics.json", "clustering_report.json")]
    absent = [str(p.name) for p in needed if not p.exists()]
    if absent:
        raise ArtifactsMissingError(
            f"Missing {absent}. Run `python run_all.py --student-id ... --seed ...` first.")
    return ModelBundle(
        regression=joblib.load(needed[0]),
        classification=joblib.load(needed[1]),
        clustering=joblib.load(needed[2]),
        metadata=_read_json(needed[3]),
        regression_report=_read_json(needed[4]),
        classification_report=_read_json(needed[5]),
        clustering_report=_read_json(needed[6]),
    )


def risk_band(probability: float) -> str:
    if probability < config.RISK_LOW_BELOW:
        return "low"
    if probability >= config.RISK_HIGH_FROM:
        return "high"
    return "moderate"


def predict_record(bundle: ModelBundle, record: dict) -> dict:
    """Validate (raises pydantic.ValidationError) and predict for one farm."""
    farm = FarmRecord.model_validate(record)
    frame = pd.DataFrame([farm.model_dump()])[config.BASE_FEATURES]

    yield_estimate = max(float(bundle.regression.predict(frame)[0]), config.MIN_YIELD_TONS)
    probability = float(bundle.classification.predict_proba(frame)[0][
        list(bundle.classification.classes_).index(1)])
    cluster = int(bundle.clustering.predict(frame)[0])
    profile = next(p["profile"] for p in bundle.clustering_report["profiles"] if p["cluster"] == cluster)

    return {
        "verification_code": bundle.metadata["verification_code"],
        "yield_estimate_tons": round(yield_estimate, 3),
        "high_risk_prediction": bool(probability >= 0.5),
        "high_risk_probability": round(probability, 4),
        "risk_band": risk_band(probability),
        "cluster": cluster,
        "cluster_profile": profile,
        "human_review_required": True,
        "notice": config.SYNTHETIC_NOTICE,
    }
