"""XGBClassifier, Bayesian tuning with TimeSeriesSplit, and the decision threshold.

Keeps the BayesSearchCV structure of ``optimize_best_model`` in
retail-sales-forecasting, but with ``TimeSeriesSplit(n_splits=4, gap=1)`` instead of a
random ``cv=3`` and ``neg_log_loss`` because the output is a probability.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Integer, Real

from .portfolio import COST_PER_SIDE, strategy_log_return

SEED = 42
DEFAULT_PARAMS = {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05,
                  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5}
SEARCH_SPACE = {
    "n_estimators": Integer(50, 300),
    "max_depth": Integer(2, 6),
    "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
    "subsample": Real(0.5, 1.0),
    "colsample_bytree": Real(0.5, 1.0),
    "min_child_weight": Integer(1, 50),
}
N_ITER = 20
TUNE_SPLITS = 4
# m in "long if p > 0.5 + m": candidate grid, chosen on training data only
THRESHOLD_GRID = (0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1)


def make_classifier(params: dict | None = None, n_jobs: int = 4) -> xgb.XGBClassifier:
    params = {**DEFAULT_PARAMS, **(params or {})}
    return xgb.XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=n_jobs,
    )


def tune(X: pd.DataFrame, y: pd.Series, n_iter: int = N_ITER, n_jobs: int = 4) -> dict:
    """BayesSearchCV over SEARCH_SPACE, scored by log loss on a gapped TimeSeriesSplit."""
    search = BayesSearchCV(
        estimator=make_classifier(n_jobs=n_jobs),
        search_spaces=SEARCH_SPACE,
        n_iter=n_iter,
        cv=TimeSeriesSplit(n_splits=TUNE_SPLITS, gap=1),
        scoring="neg_log_loss",
        random_state=SEED,
        n_jobs=1,
        refit=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y)
    params = {k: (int(v) if isinstance(v, (np.integer, int)) else float(v))
              for k, v in search.best_params_.items()}
    return {"params": params, "cv_log_loss": float(-search.best_score_)}


def oof_probabilities(X: pd.DataFrame, y: pd.Series, params: dict,
                      n_splits: int = TUNE_SPLITS, n_jobs: int = 4) -> pd.Series:
    """Out-of-fold probabilities on the training window (gapped TimeSeriesSplit)."""
    out = pd.Series(np.nan, index=X.index)
    for tr, va in TimeSeriesSplit(n_splits=n_splits, gap=1).split(X):
        model = make_classifier(params, n_jobs=n_jobs).fit(X.iloc[tr], y.iloc[tr])
        out.iloc[va] = model.predict_proba(X.iloc[va])[:, 1]
    return out.dropna()


def choose_threshold(p: pd.Series, next_ret: pd.Series,
                     grid=THRESHOLD_GRID, cost: float = COST_PER_SIDE) -> float:
    """Pick m maximising the net-of-cost log return of 'long if p > 0.5 + m'.

    Uses only out-of-fold predictions from the training window. Ties go to the
    smaller m.
    """
    best_m, best_val = grid[0], -np.inf
    for m in grid:
        pos = (p.to_numpy() > 0.5 + m).astype(float)
        val = strategy_log_return(pos, next_ret.loc[p.index].to_numpy(), cost)
        if val > best_val + 1e-12:
            best_m, best_val = m, val
    return float(best_m)
