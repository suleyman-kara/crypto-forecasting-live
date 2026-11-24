import numpy as np
import pandas as pd
import pytest

from cfl.metrics import trading
from cfl.model import choose_threshold
from cfl.portfolio import COST_PER_SIDE, count_trades, simulate, strategy_log_return


def test_cost_constant():
    assert COST_PER_SIDE == pytest.approx(0.0015)


def test_always_flat_keeps_balance():
    sim = simulate([0, 0, 0], [0.1, -0.2, 0.3], cost=0.01, start=100)
    assert sim["equity"].tolist() == [100, 100, 100]


def test_enter_hold_exit():
    # enter at row 0 (pay 1%), earn +10%; hold, earn -5%; exit at row 2 (pay 1%), flat
    sim = simulate([1, 1, 0], [0.10, -0.05, 0.50], cost=0.01, start=100)
    e0 = 100 * 0.99 * 1.10
    e1 = e0 * 0.95
    e2 = e1 * 0.99
    np.testing.assert_allclose(sim["equity"], [e0, e1, e2])
    assert sim["turnover"].tolist() == [1, 0, 1]
    assert count_trades([1, 1, 0]) == 1


def test_round_trips_counted_and_charged():
    pos = [1, 0, 1, 0]
    r = [0.0, 0.0, 0.0, 0.0]
    sim = simulate(pos, r, cost=0.001, start=1000)
    assert sim["equity"].iloc[-1] == pytest.approx(1000 * 0.999 ** 4)
    assert count_trades(pos) == 2


def test_log_return_matches_simulation():
    rng = np.random.default_rng(3)
    pos = rng.integers(0, 2, 50)
    r = rng.normal(0, 0.02, 50)
    sim = simulate(pos, r, start=1.0)
    assert strategy_log_return(pos, r) == pytest.approx(np.log(sim["equity"].iloc[-1]))


def test_trading_metrics_buy_and_hold():
    r = pd.Series([0.10, -0.20, 0.10])
    out = trading(pd.Series([1, 1, 1]), r, cost=0.0)
    assert out["return_pct"] == pytest.approx((1.1 * 0.8 * 1.1 - 1) * 100)
    assert out["max_drawdown_pct"] == pytest.approx(-20.0)
    assert out["exposure_pct"] == 100
    assert out["trades"] == 1


def test_threshold_prefers_not_trading_noise():
    # predictions just above 0.5 are wrong, the ones above 0.6 are right
    p = pd.Series([0.51, 0.52, 0.51, 0.65, 0.66, 0.52, 0.51, 0.7])
    r = pd.Series([-0.01, -0.01, -0.01, 0.02, 0.02, -0.01, -0.01, 0.02])
    # every m in [0.02, 0.1] keeps exactly the good trades; ties go to the smallest m
    assert choose_threshold(p, r) == pytest.approx(0.02)
    assert choose_threshold(p, r, grid=(0.0, 0.01)) == pytest.approx(0.01)
    # if every trade loses, the largest m (trade least) wins
    assert choose_threshold(p, -r.abs(), grid=(0.0, 0.1, 0.3)) == pytest.approx(0.3)
