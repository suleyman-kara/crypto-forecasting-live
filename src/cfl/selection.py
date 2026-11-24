"""Feature selection inside one training window: MRMR, RFE and ordered forward SFS.

Mirrors the ranking step of retail-sales-forecasting, with two fixes:
- SFS records the order in which features are added, so a prefix of length K is the
  K features SFS picked first (scikit-learn's selector returns an unordered mask).
- The (method, K) pair is chosen by time-series cross-validation on the training
  window, not on a test set.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from sklearn.feature_selection import RFE
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

from .model import make_classifier

# Small, fast model used only for ranking / choosing K; tuning happens later
SELECTION_PARAMS = {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.1,
                    "subsample": 0.8, "colsample_bytree": 1.0, "min_child_weight": 5}
SFS_MAX = 8                  # SFS adds features in order until this many
K_CANDIDATES = (3, 5, 8)     # prefix lengths compared by CV
CV_SPLITS = 3                # cheaper CV for selection than for tuning (4)


def cv_log_loss(X: pd.DataFrame, y: pd.Series, cols: list[str], params: dict,
                n_splits: int = CV_SPLITS, n_jobs: int = 4) -> float:
    """Mean log loss over a TimeSeriesSplit with gap=1."""
    losses = []
    for tr, va in TimeSeriesSplit(n_splits=n_splits, gap=1).split(X):
        model = make_classifier(params, n_jobs=n_jobs)
        model.fit(X.iloc[tr][cols], y.iloc[tr])
        p = model.predict_proba(X.iloc[va][cols])[:, 1]
        losses.append(log_loss(y.iloc[va], p, labels=[0, 1]))
    return float(np.mean(losses))


def rank_mrmr(X: pd.DataFrame, y: pd.Series) -> list[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return list(mrmr_classif(X=X, y=y, K=X.shape[1], show_progress=False, n_jobs=1))


def rank_rfe(X: pd.DataFrame, y: pd.Series, n_jobs: int = 4) -> list[str]:
    # Selecting 1 feature gives a full ranking (same trick as the retail pipeline)
    selector = RFE(estimator=make_classifier(SELECTION_PARAMS, n_jobs=n_jobs),
                   n_features_to_select=1).fit(X, y)
    return [col for _, col in sorted(zip(selector.ranking_, X.columns))]


def rank_sfs(X: pd.DataFrame, y: pd.Series, max_features: int = SFS_MAX,
             n_jobs: int = 4) -> list[str]:
    """Forward selection that keeps the order of addition (CV log loss, ties by name)."""
    chosen: list[str] = []
    remaining = list(X.columns)
    while remaining and len(chosen) < max_features:
        scores = {c: cv_log_loss(X, y, chosen + [c], SELECTION_PARAMS, n_jobs=n_jobs)
                  for c in remaining}
        best = min(remaining, key=lambda c: (scores[c], c))
        chosen.append(best)
        remaining.remove(best)
    return chosen


def select_features(X: pd.DataFrame, y: pd.Series, n_jobs: int = 4) -> dict:
    """Rank with all three methods, then pick (method, K) by CV log loss.

    Returns the chosen ordered feature list plus every ranking and score, so the
    report can show selection stability.
    """
    rankings = {
        "MRMR": rank_mrmr(X, y),
        "RFE": rank_rfe(X, y, n_jobs=n_jobs),
        "SFS": rank_sfs(X, y, n_jobs=n_jobs),
    }
    scores = {}
    for name, ranking in rankings.items():
        for k in K_CANDIDATES:
            if k <= len(ranking):
                scores[(name, k)] = cv_log_loss(X, y, ranking[:k], SELECTION_PARAMS, n_jobs=n_jobs)
    # Lowest loss wins; ties go to fewer features, then method name (deterministic)
    method, k = min(scores, key=lambda key: (round(scores[key], 10), key[1], key[0]))
    return {
        "method": method,
        "k": k,
        "features": rankings[method][:k],
        "rankings": rankings,
        "cv_log_loss": {f"{m}@{kk}": v for (m, kk), v in scores.items()},
    }

