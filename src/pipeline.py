"""End-to-end orchestration: data -> validation -> models -> artifacts."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.generate_data import generate_dataset  # noqa: E402
from src import config  # noqa: E402
from src.clustering import train_clustering  # noqa: E402
from src.evaluation import plot_clusters, plot_confusion, plot_residuals  # noqa: E402
from src.identity import build_identity  # noqa: E402
from src.modeling import (  # noqa: E402
    largest_regression_errors, most_confident_mistakes,
    train_classification, train_regression,
)
from src.preprocessing import clean_dataset, features_of, split_dataset  # noqa: E402
from src.validation import assert_no_leakage, validate_raw_dataframe  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def _strip_model(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "model"}


def run_pipeline(student_id: str, seed: int, root: Path = ROOT, quiet: bool = False) -> dict:
    """Run the whole system. Raises IdentityError / DataValidationError on bad input."""
    say = (lambda *a: None) if quiet else print
    identity = build_identity(student_id, seed)        # rejects any seed != LLLL
    code = identity.verification_code
    paths = config.ProjectPaths(Path(root))
    paths.reset_outputs()

    say("=" * 66)
    say(" Musanze Harvest Intelligence Service")
    say(f" Student ID        : {identity.student_id}")
    say(f" Verification code : {code}")
    say(f" Seed (LLLL)       : {seed}")
    say("=" * 66)

    say("[1/6] Generating dataset ...")
    gen = generate_dataset(seed, paths.csv_path(identity.reg_digits), paths.fingerprint_path)
    say(f"      base rows {gen['base_rows']} (+3 duplicates = {gen['total_rows_with_duplicates']})")
    say(f"      SHA-256 fingerprint: {gen['sha256']}")

    say("[2/6] Validating data ...")
    raw = pd.read_csv(gen["csv_path"])
    quality = validate_raw_dataframe(raw)
    assert_no_leakage(config.BASE_FEATURES)
    say(f"      missing: {quality['missing_by_column']}  duplicates: {quality['duplicate_rows']}")

    say("[3/6] Cleaning and splitting (80/20, random_state=%d) ..." % seed)
    clean, clean_report = clean_dataset(raw)
    split = split_dataset(clean, seed)
    say(f"      train {len(split.train)} rows | test {len(split.test)} rows")

    say("[4/6] Regression ...")
    reg = train_regression(split, seed)
    say(f"      selected: {reg['selected_model']}")
    say("[5/6] Classification ...")
    clf = train_classification(split, seed)
    say(f"      selected: {clf['selected_model']}")
    say("[6/6] Clustering ...")
    clu = train_clustering(split, seed)
    say(f"      selected K = {clu['report']['selected_k']}")

    # ---- save models --------------------------------------------------------
    joblib.dump(reg["model"], paths.models_dir / "regression_model.joblib")
    joblib.dump(clf["model"], paths.models_dir / "classification_model.joblib")
    joblib.dump(clu["model"], paths.models_dir / "cluster_model.joblib")

    # ---- metric files -------------------------------------------------------
    header = {"verification_code": code, "seed": seed}
    reg_report = {**header, **_strip_model(reg),
                  "largest_test_errors": largest_regression_errors(split, reg["model"])}
    clf_report = {**header, **_strip_model(clf),
                  "positive_class": "high_risk = 1",
                  "confusion_matrix_layout": "[[TN, FP], [FN, TP]]",
                  "most_confident_mistakes": most_confident_mistakes(split, clf["model"])}
    clu_report = {**header, **clu["report"]}
    _write_json(paths.artifacts_dir / "regression_metrics.json", reg_report)
    _write_json(paths.artifacts_dir / "classification_metrics.json", clf_report)
    _write_json(paths.artifacts_dir / "clustering_report.json", clu_report)
    _write_json(paths.artifacts_dir / "data_quality_report.json",
                {**header, "validation": quality, "cleaning": clean_report,
                 "generation": {k: gen[k] for k in
                                ("base_rows", "total_rows_with_duplicates",
                                 "missing_per_column_injected", "duplicate_rows_appended")}})

    # ---- per-row test predictions ------------------------------------------
    X_te = features_of(split.test)
    pred_y = reg["model"].predict(X_te)
    prob = clf["model"].predict_proba(X_te)[:, list(clf["model"].classes_).index(1)]
    out = pd.DataFrame({
        "farm_id": split.test[config.ID_COLUMN],
        "actual_yield_tons": split.test[config.REG_TARGET],
        "predicted_yield_tons": pred_y,
        "yield_error_tons": split.test[config.REG_TARGET] - pred_y,
        "abs_yield_error_tons": np.abs(split.test[config.REG_TARGET] - pred_y),
        "actual_high_risk": split.test[config.CLF_TARGET],
        "predicted_high_risk": (prob >= 0.5).astype(int),
        "probability_high_risk": prob,
        "cluster": clu["model"].predict(X_te),
    })
    out.round(4).to_csv(paths.artifacts_dir / "test_predictions.csv", index=False,
                        float_format="%.4f", lineterminator="\n")

    # ---- plots --------------------------------------------------------------
    best_clf = next(c for c in clf["candidates"] if c["name"] == clf["selected_model"])
    plot_residuals(split.test[config.REG_TARGET], pred_y, paths.plots_dir / "residuals.png",
                   code, reg["selected_model"])
    plot_confusion(best_clf["test"]["confusion_matrix"], paths.plots_dir / "confusion_matrix.png",
                   code, clf["selected_model"])
    plot_clusters(clu["Z_train"], clu["labels_train"], clu["centers"],
                  paths.plots_dir / "clusters.png", code, seed, clu["report"]["selected_k"])

    # ---- run metadata (used by predict.py and the API) -----------------------
    _write_json(paths.artifacts_dir / "run_metadata.json", {
        **header,
        "student_id": identity.student_id,
        "registration_digits": identity.reg_digits,
        "dataset_file": Path(gen["csv_path"]).name,
        "dataset_sha256": gen["sha256"],
        "rows_total": gen["total_rows_with_duplicates"],
        "rows_after_cleaning": clean_report["rows_after"],
        "train_rows": len(split.train),
        "test_rows": len(split.test),
        "random_state_everywhere": seed,
        "synthetic_data": True,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    })

    say("-" * 66)
    say(f" Regression     : {reg['selected_model']}  RMSE={next(c for c in reg['candidates'] if c['name']==reg['selected_model'])['test']['rmse']:.3f}")
    say(f" Classification : {clf['selected_model']}  F1={best_clf['test']['f1']:.3f}  AUC={best_clf['test']['roc_auc']:.3f}")
    say(f" Clustering     : K={clu['report']['selected_k']}")
    say(f" Fingerprint    : {gen['sha256']}")
    say(f" Verification   : {code}")
    return {"identity": identity, "fingerprint": gen["sha256"], "regression": reg_report,
            "classification": clf_report, "clustering": clu_report, "paths": paths}
