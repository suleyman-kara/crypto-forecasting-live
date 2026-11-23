"""Classification and trading metrics, baselines, and feature-selection stability."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .features import FEATURES, NEXT_RET, TARGET
from .model import SEED
from .portfolio import COST_PER_SIDE, START_BALANCE, count_trades, simulate

PERIODS_PER_YEAR = 6 * 365  # 4h bars, crypto trades every day


def classification(y_true, y_pred, score=None, prob=None) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    hits = int((y_true == y_pred).sum())
    out = {
        "n": n,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, score)) if score is not None else None,
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])) if prob is not None else None,
        "brier": float(brier_score_loss(y_true, prob)) if prob is not None else None,
        # Is accuracy different from a coin flip?
        "binom_p_vs_50": float(binomtest(hits, n, 0.5).pvalue),
    }
    return out


def trading(positions, next_ret, cost: float = COST_PER_SIDE) -> dict:
    sim = simulate(positions, next_ret, cost=cost)
    growth = sim["growth"].to_numpy()
    per_bar = growth - 1
    equity = sim["equity"].to_numpy()
    peak = np.maximum.accumulate(np.concatenate([[START_BALANCE], equity]))[1:]
    drawdown = equity / peak - 1
    sd = per_bar.std(ddof=1)
    return {
        "return_pct": float((equity[-1] / START_BALANCE - 1) * 100),
        "sharpe": float(per_bar.mean() / sd * np.sqrt(PERIODS_PER_YEAR)) if sd > 0 else 0.0,
        "max_drawdown_pct": float(drawdown.min() * 100),
        "exposure_pct": float(np.mean(sim["position"]) * 100),
        "trades": count_trades(sim["position"]),
        "equity": sim["equity"],
    }


def baselines(preds: pd.DataFrame) -> dict[str, dict]:
    """Positions / predictions / scores for every baseline on the prediction rows.

    - Buy & hold: long the whole time (one entry).
    - Always up: predicts 'up' on every bar; trades exactly like buy & hold.
    - Momentum: predicts the sign of the last bar's return; long when it is up.
    - Random: seeded coin flips.
    """
    y = preds[TARGET].to_numpy()
    n = len(y)
    rng = np.random.default_rng(SEED)
    coin = rng.random(n)
    last_ret = preds["ret_lag_1"].to_numpy()
    ones = np.ones(n, dtype=int)
    return {
        "Buy & hold": {"pred": ones, "score": None, "position": ones},
        "Always up": {"pred": ones, "score": np.full(n, 0.5), "position": ones},
        "Momentum": {"pred": (last_ret > 0).astype(int), "score": last_ret,
                     "position": (last_ret > 0).astype(int)},
        "Random": {"pred": (coin > 0.5).astype(int), "score": coin,
                   "position": (coin > 0.5).astype(int)},
    }


def evaluate(preds: pd.DataFrame) -> dict:
    """All metrics for the model and the baselines on the same rows."""
    y = preds[TARGET].to_numpy()
    r = preds[NEXT_RET].to_numpy()
    rows = {}
    model_pred = (preds["p"] > 0.5).astype(int)
    rows["Model"] = {
        **classification(y, model_pred, score=preds["p"], prob=preds["p"]),
        **trading(preds["position"], r),
    }
    for name, b in baselines(preds).items():
        cls = classification(y, b["pred"], score=b["score"]) if name != "Buy & hold" else {}
        rows[name] = {**cls, **trading(b["position"], r)}
    # Accuracy of the model vs the share of up bars (what 'always up' scores)
    hits = int((model_pred.to_numpy() == y).sum())
    up_rate = float(y.mean())
    rows["Model"]["binom_p_vs_up_rate"] = float(binomtest(hits, len(y), up_rate,
                                                          alternative="greater").pvalue)
    rows["Model"]["up_rate"] = up_rate
    return rows


def rolling_accuracy(preds: pd.DataFrame, window: int = 6 * 90) -> pd.Series:
    hit = ((preds["p"] > 0.5).astype(int) == preds[TARGET]).astype(float)
    return hit.rolling(window, min_periods=window).mean()


def selection_stability(window_log: list[dict]) -> pd.DataFrame:
    """How often each feature was selected across selection rounds."""
    rounds = [w["selection"] for w in window_log if "selection" in w]
    n = len(rounds)
    chosen = Counter(f for s in rounds for f in s["features"])
    first = Counter(s["features"][0] for s in rounds)
    method = {m: Counter() for m in ("MRMR", "RFE", "SFS")}
    for s in rounds:
        for m, ranking in s["rankings"].items():
            # A feature 'appears' in a method's top-K for the K that round selected
            method[m].update(ranking[: s["k"]])
    df = pd.DataFrame({
        "selected": [chosen[f] for f in FEATURES],
        "selected_pct": [chosen[f] / n * 100 for f in FEATURES],
        "ranked_first": [first[f] for f in FEATURES],
        **{f"in_{m}_topK": [method[m][f] for f in FEATURES] for m in method},
    }, index=FEATURES)
    df.index.name = "feature"
    return df.sort_values(["selected", "ranked_first"], ascending=False)
