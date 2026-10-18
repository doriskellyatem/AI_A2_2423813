"""Metrics, error tables and plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)

PALETTE = {"soil": "#6b4f2a", "leaf": "#3f7d4e", "clay": "#c0693b", "sky": "#3b6ea5"}


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred, y_prob) -> dict:
    """Precision/recall/F1 for class 1 (high risk), confusion matrix [[TN,FP],[FN,TP]]."""
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) == 2 else None
    except ValueError:
        auc = None
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def plot_residuals(y_true, y_pred, path: Path, code: str, model_name: str) -> None:
    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    a1.scatter(y_pred, resid, color=PALETTE["leaf"], alpha=0.8, edgecolor="white")
    a1.axhline(0, color=PALETTE["clay"], lw=1.5)
    a1.set(xlabel="Predicted yield (tons)", ylabel="Residual = actual - predicted",
           title="Residuals vs prediction")
    a2.hist(resid, bins=12, color=PALETTE["soil"], edgecolor="white")
    a2.set(xlabel="Residual (tons)", ylabel="Farms", title="Residual distribution")
    fig.suptitle(f"{model_name} on held-out test set  |  {code}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, path: Path, code: str, model_name: str) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.imshow(cm, cmap="YlGn")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center", fontsize=16, color="black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["not high", "high"],
           yticklabels=["not high", "high"], xlabel="Predicted", ylabel="Actual",
           title=f"{model_name}\n{code}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_clusters(Z_train, labels, centers, path: Path, code: str, seed: int, k: int) -> None:
    pca = PCA(n_components=2, random_state=seed).fit(Z_train)
    pts = pca.transform(Z_train)
    ctr = pca.transform(centers)
    fig, ax = plt.subplots(figsize=(6.4, 5))
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=labels, cmap="viridis", alpha=0.8, edgecolor="white")
    ax.scatter(ctr[:, 0], ctr[:, 1], marker="X", s=200, c="red", edgecolor="black", label="centre")
    ax.set(xlabel="PCA 1", ylabel="PCA 2", title=f"Farm clusters (K={k})  |  {code}")
    ax.legend(*sc.legend_elements(), title="cluster", loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
