import numpy as np
import pandas as pd
import pytest

import cfl.features as features
from cfl.data import resample_4h
from cfl.features import FEATURES, GAP_LOOKBACK, TARGET, build_features, dataset

from conftest import make_1h


@pytest.fixture
def bars():
    return resample_4h(make_1h(4 * 400, seed=1))


def _perturb_after(bars: pd.DataFrame, t: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = bars.copy()
    n = len(out) - (t + 1)
    out.iloc[t + 1:, out.columns.get_indexer(["open", "high", "low", "close", "volume"])] *= \
        rng.uniform(0.5, 1.5, size=(n, 5))
    return out


@pytest.mark.parametrize("t", [60, 150, 300])
def test_no_lookahead(bars, t):
    """Changing every candle after t must not change any feature at t (or before)."""
    a = build_features(bars)[FEATURES]
    b = build_features(_perturb_after(bars, t))[FEATURES]
    pd.testing.assert_frame_equal(a.iloc[: t + 1], b.iloc[: t + 1])
    # The perturbation is real: the target at t (next bar) and later features change
    assert not a.iloc[t + 1:].equals(b.iloc[t + 1:])


def test_no_lookahead_detects_a_leak(bars, monkeypatch):
    """Sanity check of the test itself: a centred rolling window must be caught."""
    original = features.build_features

    def leaky(b):
        out = original(b)
        out["vol_6"] = np.log(b["close"]).diff().rolling(6, center=True).std()
        return out

    t = 150
    a = leaky(bars)[FEATURES]
    b = leaky(_perturb_after(bars, t))[FEATURES]
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(a.iloc[: t + 1], b.iloc[: t + 1])


def test_target_is_next_bar_direction(bars):
    df = build_features(bars)
    nxt = np.log(bars["close"]).diff().shift(-1)
    expected = (nxt > 0).astype(float).where(nxt.notna())
    pd.testing.assert_series_equal(df[TARGET], expected, check_names=False)
    assert np.isnan(df[TARGET].iloc[-1])  # last bar's outcome is unknown


def test_ret_lag_1_is_current_bar_return(bars):
    df = build_features(bars)
    i = 100
    assert df["ret_lag_1"].iloc[i] == pytest.approx(np.log(bars["close"].iloc[i] / bars["close"].iloc[i - 1]))
    assert df["ret_lag_42"].iloc[i] == pytest.approx(np.log(bars["close"].iloc[i - 41] / bars["close"].iloc[i - 42]))


def test_dataset_drops_nan_rows_and_never_fills(bars):
    ds = dataset(bars)
    assert not ds[FEATURES + [TARGET]].isna().any().any()
    assert ds.index.min() > bars.index[0]       # warm-up rows dropped
    assert ds.index.max() < bars.index[-1]      # last row (no target) dropped


def test_rows_near_an_empty_bar_are_dropped(bars):
    gap = 200
    holed = bars.copy()
    holed.iloc[gap, holed.columns.get_indexer(["open", "high", "low", "close", "volume"])] = np.nan
    ds = dataset(holed)
    t_gap = bars.index[gap]
    dropped = bars.index[gap - 1: gap + GAP_LOOKBACK + 1]  # gap-1 loses its target
    assert not ds.index.isin(dropped).any()
    assert t_gap + (GAP_LOOKBACK + 1) * pd.Timedelta(hours=4) in ds.index
