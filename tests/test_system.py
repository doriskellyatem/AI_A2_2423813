"""Automated tests: happy paths AND failure paths.

EDIT the two constants below to your own details before submitting.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) 
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import create_app
from data.generate_data import generate_dataset
from src import config
from src.identity import IdentityError, build_identity
from src.inference import load_bundle, predict_record
from src.pipeline import run_pipeline
from src.validation import DataValidationError, assert_no_leakage

STUDENT_ID = "ug2527389@ines.ac.rw"  
SEED = 7389                           

VALID = {"rainfall_mm": 1000.0, "avg_temp_c": 18.0, "soil_moisture_pct": 50.0,
         "fertilizer_kg": 100.0, "field_size_ha": 1.5, "season": "A", "cooperative_support": 1}
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def run_a(tmp_path_factory):
    root = tmp_path_factory.mktemp("run_a")
    result = run_pipeline(STUDENT_ID, SEED, root=root, quiet=True)
    return root, result


@pytest.fixture(scope="session")
def bundle(run_a):
    return load_bundle(run_a[0])


# ---- inference: happy path and three failure paths ---------------------------
def test_valid_inference_returns_every_field(bundle):
    out = predict_record(bundle, VALID)
    assert out["verification_code"] == f"AI-A2-{SEED:04d}"
    assert out["yield_estimate_tons"] >= config.MIN_YIELD_TONS
    assert 0.0 <= out["high_risk_probability"] <= 1.0
    assert isinstance(out["cluster"], int)
    assert out["human_review_required"] is True


def test_missing_field_is_rejected(bundle):
    bad = {k: v for k, v in VALID.items() if k != "fertilizer_kg"}
    with pytest.raises(ValidationError) as err:
        predict_record(bundle, bad)
    assert "fertilizer_kg" in str(err.value)


def test_wrong_type_is_rejected(bundle):
    with pytest.raises(ValidationError):
        predict_record(bundle, {**VALID, "rainfall_mm": "lots"})


@pytest.mark.parametrize("field,value", [("rainfall_mm", 1400.01), ("avg_temp_c", 11.9),
                                          ("field_size_ha", 0.0), ("season", "C"),
                                          ("cooperative_support", 2)])
def test_range_and_category_errors_are_rejected(bundle, field, value):
    with pytest.raises(ValidationError):
        predict_record(bundle, {**VALID, field: value})


def test_unknown_field_is_rejected(bundle):
    with pytest.raises(ValidationError):
        predict_record(bundle, {**VALID, "yield_tons": 9.0})


# ---- identity / seed rule ------------------------------------------------------
def test_wrong_seed_is_rejected():
    with pytest.raises(IdentityError):
        build_identity(STUDENT_ID, SEED + 1)


# ---- reproducibility -------------------------------------------------------------
def test_reproducibility_same_fingerprint_and_metrics(run_a, tmp_path):
    root_a, result_a = run_a
    result_b = run_pipeline(STUDENT_ID, SEED, root=tmp_path, quiet=True)
    assert result_a["fingerprint"] == result_b["fingerprint"]
    fp_file = (root_a / "artifacts" / "data_fingerprint.txt").read_text()
    assert result_a["fingerprint"] in fp_file
    ra, rb = result_a["regression"]["candidates"], result_b["regression"]["candidates"]
    for a, b in zip(ra, rb):
        assert a["test"]["rmse"] == pytest.approx(b["test"]["rmse"], rel=1e-9)
    ca, cb = result_a["classification"]["candidates"], result_b["classification"]["candidates"]
    for a, b in zip(ca, cb):
        assert a["test"]["f1"] == pytest.approx(b["test"]["f1"], rel=1e-9)


def test_generator_row_count_follows_seed_rule(tmp_path):
    info = generate_dataset(SEED, tmp_path / "d.csv")
    assert info["base_rows"] == 240 + SEED % 41
    assert info["total_rows_with_duplicates"] == info["base_rows"] + 3


# ---- data problems are present and handled -------------------------------------------
def test_deliberate_data_problems_are_detected_and_cleaned(run_a):
    root, _ = run_a
    report = json.loads((root / "artifacts" / "data_quality_report.json").read_text())
    m = report["generation"]["missing_per_column_injected"]
    # after removing duplicates, exactly m values are missing in each of the two columns
    assert report["cleaning"]["missing_by_column_after_dedup"] == {"soil_moisture_pct": m, "fertilizer_kg": m}
    # raw file: a duplicated row can copy a blank, so raw counts are >= m
    raw_missing = report["validation"]["missing_by_column"]
    assert raw_missing["soil_moisture_pct"] >= m and raw_missing["fertilizer_kg"] >= m
    assert report["validation"]["duplicate_rows"] == 3
    assert report["cleaning"]["duplicates_removed"] == 3


def test_leakage_guard_blocks_target_columns():
    with pytest.raises(DataValidationError):
        assert_no_leakage(config.BASE_FEATURES + ["high_risk"])
    assert_no_leakage(config.BASE_FEATURES)   # the real feature list is clean


# ---- API --------------------------------------------------------------------------------
def test_api_health_and_model_info(run_a):
    with TestClient(create_app(run_a[0])) as client:
        assert client.get("/health").json()["status"] == "ok"
        info = client.get("/model-info").json()
        assert info["verification_code"] == f"AI-A2-{SEED:04d}"
        assert info["synthetic_data"] is True


def test_api_predict_valid_response(run_a):
    with TestClient(create_app(run_a[0])) as client:
        resp = client.post("/predict", json=VALID)
        assert resp.status_code == 200
        body = resp.json()
        assert {"yield_estimate_tons", "high_risk_probability", "cluster",
                "verification_code"} <= body.keys()


def test_api_invalid_request_gives_422_with_field_names(run_a):
    with TestClient(create_app(run_a[0])) as client:
        resp = client.post("/predict", json={**VALID, "rainfall_mm": 5000, "season": "Z"})
        assert resp.status_code == 422
        fields = {p["field"] for p in resp.json()["problems"]}
        assert {"rainfall_mm", "season"} <= fields


def test_api_without_models_returns_503(tmp_path):
    with TestClient(create_app(tmp_path)) as client:   # empty folder: nothing trained
        assert client.post("/predict", json=VALID).status_code == 503


# ---- CLI failure path ----------------------------------------------------------------------
def test_cli_rejects_invalid_record_with_nonzero_exit(run_a, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**VALID, "rainfall_mm": 9999}))
    proc = subprocess.run([sys.executable, str(ROOT / "predict.py"), "--json", str(bad),
                           "--root", str(run_a[0])], capture_output=True, text=True) 
    assert proc.returncode == 2
    assert "rainfall_mm" in proc.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))