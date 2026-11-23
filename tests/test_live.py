import json

import numpy as np
import pandas as pd
import pytest

from cfl import live
from cfl.data import BAR, resample_4h
from cfl.features import dataset
from cfl.model import make_classifier
from cfl.portfolio import COST_PER_SIDE

from conftest import make_1h

FEATS = ["ret_lag_1", "ret_lag_2", "hour"]
T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


@pytest.fixture(scope="module")
def candles():
    return make_1h(4 * 400, start="2024-01-01")  # 1h candles through 2024-03-08 15:00


@pytest.fixture
def env(tmp_path, candles):
    """A tiny trained model, a backtest summary and an empty live/ dir (no network)."""
    models = tmp_path / "models"
    models.mkdir()
    ds = dataset(resample_4h(candles))
    clf = make_classifier({"n_estimators": 5}, n_jobs=1).fit(ds[FEATS], ds["target"])
    clf.save_model(models / "model.json")
    (models / "meta.json").write_text(json.dumps({
        "version": "test-1", "trainedThrough": "2024-01-01T00:00:00+00:00", "features": FEATS,
        "threshold": 0.0}))
    bt = tmp_path / "backtest.json"
    bt.write_text(json.dumps({"period": {"from": "a", "to": "b"}, "accuracy": 0.53, "auc": 0.55,
                              "returnPct": -9.0, "buyHoldReturnPct": 20.0, "maxDrawdownPct": -30.0,
                              "sharpe": -0.2, "trades": 7, "extra": 1}))
    return {"live_dir": tmp_path / "live", "models_dir": models, "backtest_json": bt}


def _run(candles, env, now):
    now = T(now)
    h1 = candles[candles.index + pd.Timedelta(hours=1) <= now.floor("h")]  # only closed hours exist
    return live.run(h1, now, **env)


def _log(env):
    return live.read_log(env["live_dir"] / "predictions.jsonl")


def _status(env):
    return json.loads((env["live_dir"] / "status.json").read_text())


def test_latest_closed_open():
    assert live.latest_closed_open(T("2024-02-01 08:07")) == T("2024-02-01 04:00")
    assert live.latest_closed_open(T("2024-02-01 07:59")) == T("2024-02-01 00:00")


def test_prediction_targets_next_candle_and_is_idempotent(candles, env):
    res = _run(candles, env, "2024-02-01 08:07")
    assert res.changed
    log = _log(env)
    assert len(log) == 1
    r = log[0]
    assert r["candle_open"] == "2024-02-01T04:00:00Z"
    assert r["target_open"] == "2024-02-01T08:00:00Z"
    assert r["target_close"] == "2024-02-01T12:00:00Z"
    assert r["predicted_at"] == "2024-02-01T08:07:00Z"
    assert 0 < r["p"] < 1 and r["direction"] == ("up" if r["p"] > 0.5 else "down")
    assert r["action"] == ("long" if r["p"] > 0.5 else "flat")  # threshold 0 in this model
    assert r["outcome"] == "pending" and r["model_version"] == "test-1"

    before = {p.name: p.read_bytes() for p in env["live_dir"].iterdir()}
    res2 = _run(candles, env, "2024-02-01 08:40")  # same slot, later in the hour
    assert not res2.changed
    assert {p.name: p.read_bytes() for p in env["live_dir"].iterdir()} == before


def test_features_use_only_data_through_the_closed_candle(candles, env):
    _run(candles, env, "2024-02-01 08:07")
    p1 = _log(env)[0]["p"]
    # Corrupting everything after the predicted-from candle's close must not change p
    bad = candles.copy()
    after = bad.index >= T("2024-02-01 08:00")
    bad.loc[after, ["open", "high", "low", "close"]] *= 3
    env2 = {**env, "live_dir": env["live_dir"].parent / "live2"}
    live.run(bad, T("2024-02-01 08:07"), **env2)
    assert _log(env2)[0]["p"] == p1


def test_scoring_hit_miss_and_realized_return(candles, env):
    _run(candles, env, "2024-02-01 08:07")
    _run(candles, env, "2024-02-01 12:07")
    log = _log(env)
    assert [r["outcome"] for r in log] == [log[0]["outcome"], "pending"]
    r = log[0]
    bars = resample_4h(candles)
    ret = (bars.at[T("2024-02-01 08:00"), "close"] / bars.at[T("2024-02-01 04:00"), "close"] - 1) * 100
    assert r["realized_return_pct"] == pytest.approx(ret, abs=1e-6)
    assert r["outcome"] == ("hit" if (ret > 0) == (r["direction"] == "up") else "miss")
    st = _status(env)
    assert st["stats"]["scored"] == 1 and st["stats"]["hits"] == (r["outcome"] == "hit")
    assert st["recent"][1]["realizedReturnPct"] == pytest.approx(ret, abs=1e-4)
    assert st["recent"][0]["outcome"] == "pending" and st["recent"][0]["realizedReturnPct"] is None


def test_missed_slots_are_skipped_not_backfilled(candles, env):
    _run(candles, env, "2024-02-01 08:07")
    _run(candles, env, "2024-02-02 00:05")  # runs at 12:07, 16:07 and 20:07 never happened
    log = _log(env)
    assert [r["candle_open"][11:16] for r in log] == ["04:00", "08:00", "12:00", "16:00", "20:00"]
    assert [r["outcome"] for r in log[1:4]] == ["skipped"] * 3
    assert all(r["p"] is None and r["predicted_at"] is None for r in log[1:4])
    assert log[4]["outcome"] == "pending" and log[4]["p"] is not None
    st = _status(env)
    assert st["stats"]["predictions"] == 2
    skipped = [x for x in st["recent"] if x["outcome"] == "skipped"]
    assert len(skipped) == 3 and all(x["realizedReturnPct"] is None for x in skipped)
    # A rerun does not touch the skipped slots
    assert not _run(candles, env, "2024-02-02 00:30").changed


def test_late_run_records_skip_instead_of_predicting(candles, env):
    res = _run(candles, env, "2024-02-01 10:30")  # 2.5h into the target candle
    log = _log(env)
    assert len(log) == 1 and log[0]["outcome"] == "skipped" and log[0]["p"] is None
    assert "after the target candle opened" in log[0]["reason"]
    assert res.changed
    assert _status(env)["current"] is None


def test_portfolio_math(candles, env):
    for hour in range(0, 24 * 3, 4):
        _run(candles, env, str(T("2024-02-01 00:07") + pd.Timedelta(hours=hour)))
    log = _log(env)
    done = [r for r in log if r["resolved"]]
    assert len(done) == len(log) - 1 and log[-1]["outcome"] == "pending"
    eq, bh, prev = 10_000.0, 10_000.0 * (1 - COST_PER_SIDE), 0
    for r in done:
        pos = 1 if r["action"] == "long" else 0
        eq *= (1 - COST_PER_SIDE * abs(pos - prev)) * (1 + pos * r["realized_return_pct"] / 100)
        bh *= 1 + r["realized_return_pct"] / 100
        prev = pos
    st = _status(env)
    assert st["stats"]["equity"] == pytest.approx(eq, abs=0.01)
    assert st["stats"]["returnPct"] == pytest.approx((eq / 1e4 - 1) * 100, abs=1e-3)
    assert st["stats"]["buyHoldReturnPct"] == pytest.approx((bh / 1e4 - 1) * 100, abs=1e-3)
    curve = st["equity"]
    assert len(curve) == len(done) + 1 and curve[0] == {"t": log[0]["target_open"], "model": 1e4, "buyHold": 1e4}
    assert curve[-1]["model"] == pytest.approx(eq, abs=0.01)
    peak = np.maximum.accumulate([c["model"] for c in curve])
    dd = min(c["model"] / p - 1 for c, p in zip(curve, peak)) * 100
    assert st["stats"]["maxDrawdownPct"] == pytest.approx(dd, abs=1e-3) and dd <= 0


def test_skipped_slot_keeps_position():
    recs = [
        {"candle_open": "2024-01-01T00:00:00Z", "target_open": "2024-01-01T04:00:00Z",
         "target_close": "2024-01-01T08:00:00Z", "p": 0.7, "action": "long", "outcome": "hit",
         "realized_return_pct": 10.0, "resolved": True},
        {"candle_open": "2024-01-01T04:00:00Z", "target_open": "2024-01-01T08:00:00Z",
         "target_close": "2024-01-01T12:00:00Z", "p": None, "action": None, "outcome": "skipped",
         "realized_return_pct": -5.0, "resolved": True},
    ]
    pf = live.portfolio(recs)
    assert pf["position"].tolist() == [1.0, 1.0]
    assert pf["model"].iloc[-1] == pytest.approx(1e4 * (1 - COST_PER_SIDE) * 1.10 * 0.95)


def test_status_schema(candles, env):
    for now in ("2024-02-01 08:07", "2024-02-01 12:07", "2024-02-01 20:07"):
        _run(candles, env, now)
    st = _status(env)
    assert set(st) == {"schema", "generatedAt", "asset", "interval", "mode", "startingBalance", "model",
                       "current", "stats", "equity", "recent", "backtest"}
    assert (st["schema"], st["asset"], st["interval"], st["mode"], st["startingBalance"]) == \
        (1, "BTC-USD", "4h", "paper", 10000)
    assert st["model"] == {"version": "test-1", "trainedThrough": "2024-01-01T00:00:00+00:00"}
    assert set(st["current"]) == {"candleOpen", "targetClose", "p", "direction", "action", "threshold"}
    assert st["current"]["candleOpen"] == "2024-02-01T20:00:00Z"  # the TARGET candle
    assert st["current"]["targetClose"] == "2024-02-02T00:00:00Z"
    assert set(st["stats"]) == {"predictions", "scored", "hits", "accuracy", "accuracy30d", "equity",
                                "returnPct", "buyHoldReturnPct", "maxDrawdownPct", "since"}
    s = st["stats"]
    assert s["predictions"] == 3 and s["scored"] == 2 and 0 <= s["accuracy"] <= 1
    assert s["since"] == "2024-02-01T08:00:00Z"
    for e in st["equity"]:
        assert set(e) == {"t", "model", "buyHold"}
    for r in st["recent"]:
        assert set(r) == {"candleOpen", "p", "direction", "action", "outcome", "realizedReturnPct"}
        assert r["outcome"] in {"hit", "miss", "pending", "skipped"}
    times = [r["candleOpen"] for r in st["recent"]]
    assert times == sorted(times, reverse=True)  # newest first
    assert st["backtest"] == {"period": {"from": "a", "to": "b"}, "accuracy": 0.53, "auc": 0.55,
                              "returnPct": -9.0, "buyHoldReturnPct": 20.0, "maxDrawdownPct": -30.0,
                              "sharpe": -0.2, "trades": 7}


def test_empty_status_before_any_prediction(env, candles):
    st = live.build_status([], None, None, T("2024-02-01"))
    assert st["current"] is None and st["equity"] == [] and st["stats"]["accuracy"] is None
    assert st["stats"]["returnPct"] == 0 and st["stats"]["since"] is None


def test_recent_and_equity_are_capped(candles, env):
    for hour in range(0, 4 * 30, 4):
        _run(candles, env, str(T("2024-02-01 00:07") + pd.Timedelta(hours=hour)))
    st = _status(env)
    assert len(st["recent"]) == live.RECENT_MAX
