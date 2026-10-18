"""K-Means farm grouping: K=2..6, silhouette-based choice, plain-language profiles."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

from src import config
from src.preprocessing import SplitData, build_cluster_preprocessor, features_of

_WORDS = {
    "rainfall_mm": ("drier", "wetter"),
    "avg_temp_c": ("cooler", "warmer"),
    "soil_moisture_pct": ("lower soil moisture", "higher soil moisture"),
    "fertilizer_kg": ("light fertilizer use", "heavy fertilizer use"),
    "field_size_ha": ("smaller fields", "larger fields"),
}


def _describe(cluster_rows, all_rows) -> tuple[str, dict]:
    """Compare a cluster with the whole training set, feature by feature."""
    phrases, z_scores = [], {}
    for col in config.NUMERIC_FEATURES:
        z = (cluster_rows[col].mean() - all_rows[col].mean()) / all_rows[col].std()
        z_scores[col] = round(float(z), 3)
        if abs(z) >= 0.5:
            phrases.append(_WORDS[col][1 if z > 0 else 0])
    support = cluster_rows["cooperative_support"].mean()
    if support >= 0.65:
        phrases.append("mostly cooperative-supported")
    elif support <= 0.35:
        phrases.append("mostly unsupported")
    season_a = (cluster_rows["season"] == "A").mean()
    if season_a >= 0.65:
        phrases.append("mainly season A")
    elif season_a <= 0.35:
        phrases.append("mainly season B")
    text = ", ".join(phrases) if phrases else "close to the cooperative average on every input"
    return text, z_scores


def train_clustering(split: SplitData, seed: int) -> dict:
    X_tr = features_of(split.train)
    prep = build_cluster_preprocessor().fit(X_tr)
    Z_tr = prep.transform(X_tr)

    scores = {}
    for k in config.K_VALUES:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z_tr)
        scores[k] = float(silhouette_score(Z_tr, labels))
    best_k = max(scores, key=scores.get)

    model = Pipeline([
        ("prep", build_cluster_preprocessor()),
        ("kmeans", KMeans(n_clusters=best_k, n_init=10, random_state=seed)),
    ]).fit(X_tr)
    labels = model.predict(X_tr)
    Z_fit = model.named_steps["prep"].transform(X_tr)
    centers = model.named_steps["kmeans"].cluster_centers_

    train = split.train.assign(cluster=labels)
    profiles = []
    for c in range(best_k):
        rows = train[train["cluster"] == c]
        text, z = _describe(rows, train)
        risk_share = float(rows[config.CLF_TARGET].mean())
        if risk_share >= 0.60:
            meaning = "Prioritise for field-officer visits: most farms here are high production risk."
        elif risk_share <= 0.40:
            meaning = "Routine monitoring: most farms here are not high production risk."
        else:
            meaning = "Mixed group: review farms individually."
        profiles.append({
            "cluster": c,
            "farms_in_training": int(len(rows)),
            "profile": f"Cluster {c}: {text}.",
            "feature_means": {col: round(float(rows[col].mean()), 3) for col in config.NUMERIC_FEATURES},
            "share_cooperative_support": round(float(rows["cooperative_support"].mean()), 3),
            "share_season_A": round(float((rows["season"] == "A").mean()), 3),
            "z_vs_overall": z,
            "after_the_fact_mean_yield_tons": round(float(rows[config.REG_TARGET].mean()), 3),
            "after_the_fact_high_risk_share": round(risk_share, 3),
            "operational_meaning": meaning,
        })

    weak = scores[best_k] < 0.25
    reason = (
        f"K={best_k} has the highest silhouette score ({scores[best_k]:.3f}) among K=2..6."
        + (" The score is low, so the groups are only weak tendencies, not sharp natural "
           "clusters (expected for independently drawn synthetic features)." if weak else "")
    )
    return {
        "model": model,
        "report": {
            "silhouette_by_k": {str(k): round(v, 4) for k, v in scores.items()},
            "selected_k": best_k,
            "selection_reason": reason,
            "clustering_uses_targets": False,
            "profiles": profiles,
        },
        "Z_train": Z_fit,
        "labels_train": labels,
        "centers": centers,
    }
