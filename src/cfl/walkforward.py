"""Holdout split and expanding-window walk-forward backtest.

- The last 6 months are set aside as a holdout before anything else happens. The
  walk-forward never reads a holdout row, and its last training label ends before the
  holdout starts.
- Walk-forward: train on every row before the window, predict the next 168 bars
  (4 weeks), move on. Feature selection and tuning run every 6th window on that
  window's training rows only; the windows in between reuse them. The decision
  threshold m is chosen in every window, on out-of-fold training predictions.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import BAR
from .features import FEATURES, NEXT_RET, TARGET
from .model import (DEFAULT_PARAMS, N_ITER, SEED, choose_threshold, make_classifier,
                    oof_probabilities, tune)
from .selection import select_features

# Frozen data cut so a rerun reproduces the same report even after the cache grows
DATA_END = pd.Timestamp("2026-11-23 00:00", tz="UTC")
HOLDOUT_MONTHS = 6
HOLDOUT_START = DATA_END - pd.DateOffset(months=HOLDOUT_MONTHS)

STEP = 168            # bars per window (4 weeks of 4h bars)
MIN_TRAIN = 4380      # rows in the first training window (~2 years)
TUNE_EVERY = 6        # select features + tune every 6th window


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def split_holdout(ds: pd.DataFrame, holdout_start: pd.Timestamp = HOLDOUT_START,
                  data_end: pd.Timestamp = DATA_END) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a feature dataset into development rows and holdout rows.

    A row at time t has a label from bar t+1 (open t+4h, close t+8h). Development rows
    need that bar to open before ``holdout_start``, so no development label uses a
    holdout price. Holdout rows predict bars that open in [holdout_start, data_end).
    """
    dev = ds[ds.index + BAR < holdout_start]
    hold = ds[(ds.index + BAR >= holdout_start) & (ds.index + 2 * BAR <= data_end)]
    return dev, hold


@dataclass(frozen=True)
class Window:
    number: int
    train_end: int    # training rows are [0, train_end)
    test_start: int   # == train_end
    test_end: int     # test rows are [test_start, test_end)


def make_windows(n_rows: int, min_train: int = MIN_TRAIN, step: int = STEP,
                 first_test: int | None = None) -> list[Window]:
    start = min_train if first_test is None else first_test
    windows = []
    while start < n_rows:
        end = min(start + step, n_rows)
        windows.append(Window(len(windows), start, start, end))
        start = end
    return windows


def run_walkforward(ds: pd.DataFrame, min_train: int = MIN_TRAIN, step: int = STEP,
                    tune_every: int = TUNE_EVERY, n_iter: int = N_ITER,
                    first_test: int | None = None, max_windows: int | None = None,
                    n_jobs: int = 4, verbose: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """Walk forward over ``ds`` (already holdout-free unless evaluating the holdout).

    Returns per-row predictions and a per-window log.
    """
    seed_everything()
    X, y, r = ds[FEATURES], ds[TARGET], ds[NEXT_RET]
    windows = make_windows(len(ds), min_train, step, first_test)
    if max_windows is not None:
        windows = windows[:max_windows]

    preds, log = [], []
    selection, params = None, dict(DEFAULT_PARAMS)
    t0 = time.time()
    for w in windows:
        Xtr, ytr, rtr = X.iloc[:w.train_end], y.iloc[:w.train_end], r.iloc[:w.train_end]
        Xte = X.iloc[w.test_start:w.test_end]
        assert Xtr.index.max() < Xte.index.min()

        tuned = w.number % tune_every == 0
        entry = {"window": w.number, "train_rows": w.train_end,
                 "train_from": Xtr.index.min().isoformat(), "train_to": Xtr.index.max().isoformat(),
                 "test_from": Xte.index.min().isoformat(), "test_to": Xte.index.max().isoformat(),
                 "tuned": tuned}
        if tuned:
            selection = select_features(Xtr, ytr, n_jobs=n_jobs)
            tuning = tune(Xtr[selection["features"]], ytr, n_iter=n_iter, n_jobs=n_jobs)
            params = tuning["params"]
            entry.update({"selection": selection, "tuning_cv_log_loss": tuning["cv_log_loss"]})
        cols = selection["features"]

        oof = oof_probabilities(Xtr[cols], ytr, params, n_jobs=n_jobs)
        m = choose_threshold(oof, rtr)

        model = make_classifier(params, n_jobs=n_jobs).fit(Xtr[cols], ytr)
        p = model.predict_proba(Xte[cols])[:, 1]
        preds.append(pd.DataFrame({
            "p": p, "m": m, "position": (p > 0.5 + m).astype(int),
            TARGET: y.iloc[w.test_start:w.test_end].to_numpy(),
            NEXT_RET: r.iloc[w.test_start:w.test_end].to_numpy(),
            "ret_lag_1": Xte["ret_lag_1"].to_numpy(),  # for the momentum baseline
            "window": w.number,
        }, index=Xte.index))
        entry.update({"features": cols, "method": selection["method"], "k": selection["k"],
                      "params": params, "m": m})
        log.append(entry)
        if verbose:
            print(f"window {w.number + 1:>3}/{len(windows)}  train={w.train_end:>5}  "
                  f"{Xte.index.min():%Y-%m-%d}  {'tuned ' if tuned else '      '}"
                  f"{selection['method']}@{selection['k']}  m={m:.3f}  "
                  f"elapsed={time.time() - t0:,.0f}s", flush=True)
    return pd.concat(preds), log


def run_holdout(ds_all: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, list[dict]]:
    """Walk forward through the holdout only (rows before it are training data).

    Gated behind ``scripts/backtest.py --evaluate-holdout``. Do not call this during
    development: the holdout is opened once, after the walk-forward results are final.
    """
    dev, hold = split_holdout(ds_all)
    ds = pd.concat([dev, hold])
    return run_walkforward(ds, first_test=len(dev), **kwargs)
