"""Regression and classification: baseline + candidates, CV-based selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from src import config
from src.evaluation import classification_metrics, regression_metrics
from src.preprocessing import SplitData, features_of, make_model_pipeline


def _regression_models(seed: int) -> dict:
    return {
        "ridge_regression": Ridge(alpha=1.0, random_state=seed),
        "random_forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=2,
                                               random_state=seed, n_jobs=1),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
    }


def _classification_models(seed: int) -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
                                                random_state=seed, n_jobs=1),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }


def train_regression(split: SplitData, seed: int) -> dict:
    """Mean baseline vs 3 candidates. Selection uses CV RMSE on TRAIN only."""
    X_tr, y_tr = features_of(split.train), split.train[config.REG_TARGET]
    X_te, y_te = features_of(split.test), split.test[config.REG_TARGET]
    cv = KFold(config.CV_FOLDS, shuffle=True, random_state=seed)

    base = make_model_pipeline(DummyRegressor(strategy="mean")).fit(X_tr, y_tr)
    baseline = {"name": "mean_baseline", "test": regression_metrics(y_te, base.predict(X_te))}

    fitted, candidates = {}, []
    for name, est in _regression_models(seed).items():
        pipe = make_model_pipeline(est)
        cv_rmse = -cross_val_score(pipe, X_tr, y_tr, cv=cv,
                                   scoring="neg_root_mean_squared_error").mean()
        pipe.fit(X_tr, y_tr)
        fitted[name] = pipe
        candidates.append({"name": name, "cv_rmse_train": float(cv_rmse),
                           "test": regression_metrics(y_te, pipe.predict(X_te))})

    best = min(candidates, key=lambda c: c["cv_rmse_train"])
    reason = (
        f"{best['name']} had the lowest {config.CV_FOLDS}-fold cross-validated RMSE on the "
        f"training split ({best['cv_rmse_train']:.3f} tons) among the candidates, and it beats "
        f"the mean baseline on the held-out test set (RMSE {best['test']['rmse']:.3f} vs "
        f"{baseline['test']['rmse']:.3f}). Selection used training-only evidence; the test set "
        f"was kept for the final report."
    )
    return {"baseline": baseline, "candidates": candidates, "selected_model": best["name"],
            "selection_reason": reason, "model": fitted[best["name"]]}


def train_classification(split: SplitData, seed: int) -> dict:
    """Majority baseline vs 3 candidates. Selection uses CV F1 on TRAIN only."""
    X_tr, y_tr = features_of(split.train), split.train[config.CLF_TARGET]
    X_te, y_te = features_of(split.test), split.test[config.CLF_TARGET]
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=seed)

    def evaluate(pipe):
        prob = pipe.predict_proba(X_te)[:, list(pipe.classes_).index(1)]
        return classification_metrics(y_te, pipe.predict(X_te), prob)

    base = make_model_pipeline(DummyClassifier(strategy="most_frequent")).fit(X_tr, y_tr)
    baseline = {"name": "majority_baseline", "test": evaluate(base)}

    fitted, candidates = {}, []
    for name, est in _classification_models(seed).items():
        pipe = make_model_pipeline(est)
        cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1").mean()
        pipe.fit(X_tr, y_tr)
        fitted[name] = pipe
        candidates.append({"name": name, "cv_f1_train": float(cv_f1), "test": evaluate(pipe)})

    best = max(candidates, key=lambda c: c["cv_f1_train"])
    reason = (
        f"{best['name']} had the highest {config.CV_FOLDS}-fold cross-validated F1 on the "
        f"training split ({best['cv_f1_train']:.3f}). On the held-out test set it reached F1 "
        f"{best['test']['f1']:.3f} and ROC-AUC {best['test']['roc_auc']:.3f}, versus F1 "
        f"{baseline['test']['f1']:.3f} for the majority baseline."
    )
    return {"baseline": baseline, "candidates": candidates, "selected_model": best["name"],
            "selection_reason": reason, "model": fitted[best["name"]]}


def largest_regression_errors(split: SplitData, model, n: int = 3) -> list[dict]:
    """The n worst test predictions, to help write the error interpretation."""
    test = split.test.copy()
    test["predicted"] = model.predict(features_of(test))
    test["error"] = test[config.REG_TARGET] - test["predicted"]
    worst = test.reindex(test["error"].abs().sort_values(ascending=False).index).head(n)
    cols = [config.ID_COLUMN] + config.BASE_FEATURES + [config.REG_TARGET, "predicted", "error"]
    return [
    {
        k: (
            None if pd.isna(v)
            else round(float(v), 4) if isinstance(v, (float, np.floating))
            else v
        )
        for k, v in row.items()
    }
    for row in worst[cols].to_dict("records")
]


def most_confident_mistakes(split: SplitData, model, n: int = 3) -> list[dict]:
    """The n misclassified test farms where the model was most sure (and wrong)."""
    test = split.test.copy()
    test["probability_high_risk"] = model.predict_proba(features_of(test))[:, list(model.classes_).index(1)]
    test["predicted"] = (test["probability_high_risk"] >= 0.5).astype(int)
    wrong = test[test["predicted"] != test[config.CLF_TARGET]].copy()
    wrong["confidence_in_wrong_answer"] = np.where(
        wrong["predicted"] == 1, wrong["probability_high_risk"], 1 - wrong["probability_high_risk"])
    wrong = wrong.sort_values("confidence_in_wrong_answer", ascending=False).head(n)
    cols = [config.ID_COLUMN] + config.BASE_FEATURES + [config.REG_TARGET, config.CLF_TARGET,
            "predicted", "probability_high_risk"]
    return [
    {
        k: (
            None if pd.isna(v)
            else round(float(v), 4) if isinstance(v, (float, np.floating))
            else v
        )
        for k, v in row.items()
    }
    for row in wrong[cols].to_dict("records")
]