import numpy as np
import pandas as pd
import pytest


def make_1h(n_hours: int, start: str = "2020-01-01", seed: int = 0) -> pd.DataFrame:
    """Synthetic random-walk 1h candles (no network)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_hours, freq="h", tz="UTC", name="time")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_hours)))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n_hours))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n_hours))
    vol = rng.uniform(10, 100, n_hours)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


@pytest.fixture
def h1():
    return make_1h(4 * 600)
