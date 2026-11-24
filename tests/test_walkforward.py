import numpy as np
import pandas as pd
import pytest

import cfl.walkforward as wf
from cfl.data import BAR
from cfl.features import FEATURES, NEXT_RET, TARGET
from cfl.walkforward import make_windows, split_holdout


def _fake_dataset(start="2023-01-01", end="2024-01-01", seed=0) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="4h", tz="UTC", inclusive="left", name="time")
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(len(idx), len(FEATURES))), index=idx, columns=FEATURES)
    df[NEXT_RET] = rng.normal(0, 0.01, len(idx))
    df[TARGET] = (df[NEXT_RET] > 0).astype(int)
    return df


def test_windows_are_contiguous_and_disjoint():
    windows = make_windows(1000, min_train=300, step=168)
    assert windows[0].test_start == 300
    for a, b in zip(windows, windows[1:]):
        assert a.test_end == b.test_start
        assert b.train_end == b.test_start
    assert windows[-1].test_end == 1000
    covered = np.concatenate([np.arange(w.test_start, w.test_end) for w in windows])
    assert len(covered) == len(set(covered)) == 700
    for w in windows:
        assert w.train_end <= w.test_start  # training strictly before testing


def test_holdout_split_has_no_overlap_or_leak():
    ds = _fake_dataset()
    holdout_start = pd.Timestamp("2023-07-01", tz="UTC")
    data_end = pd.Timestamp("2024-01-01", tz="UTC")
    dev, hold = split_holdout(ds, holdout_start, data_end)
    assert dev.index.max() < hold.index.min()
    # every development label (bar t+1 closes at t + 8h) is known by holdout_start
    assert (dev.index + 2 * BAR <= holdout_start).all()
    assert dev.index.max() == holdout_start - 2 * BAR
    # holdout predicts bars that open at/after holdout_start and close by data_end
    assert (hold.index + BAR >= holdout_start).all()
    assert (hold.index + 2 * BAR <= data_end).all()


def test_walkforward_never_touches_holdout(monkeypatch):
    ds = _fake_dataset()
    holdout_start = pd.Timestamp("2023-10-01", tz="UTC")
    dev, hold = split_holdout(ds, holdout_start, ds.index.max() + 2 * BAR)

    seen = []

    def fake_select(X, y, n_jobs=1):
        seen.append(X.index.max())
        return {"method": "MRMR", "k": 2, "features": FEATURES[:2], "rankings": {}, "cv_log_loss": {}}

    def fake_tune(X, y, n_iter=1, n_jobs=1):
        seen.append(X.index.max())
        return {"params": {"n_estimators": 5}, "cv_log_loss": 0.69}

    monkeypatch.setattr(wf, "select_features", fake_select)
    monkeypatch.setattr(wf, "tune", fake_tune)
    preds, log = wf.run_walkforward(dev, min_train=600, step=168, tune_every=3, n_jobs=1, verbose=False)

    assert preds.index.max() <= dev.index.max() < holdout_start
    assert not preds.index.isin(hold.index).any()
    assert max(seen) < holdout_start
    assert preds.index.is_unique and preds.index.is_monotonic_increasing
    assert len(preds) == len(dev) - 600
    for w in log:  # each window trains strictly before it predicts
        assert w["train_to"] < w["test_from"]
    assert [w["tuned"] for w in log][:4] == [True, False, False, True]
