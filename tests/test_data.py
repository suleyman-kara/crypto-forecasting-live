import numpy as np
import pandas as pd

from cfl.data import _to_frame, resample_4h

from conftest import make_1h


def test_to_frame_parses_coinbase_order():
    rows = [[7200, 1.0, 5.0, 2.0, 4.0, 10.0], [3600, 0.5, 3.0, 1.0, 2.0, 20.0]]  # newest first
    df = _to_frame(rows)
    assert [int(t.timestamp()) for t in df.index] == [3600, 7200]
    assert df.iloc[1].to_dict() == {"open": 2.0, "high": 5.0, "low": 1.0, "close": 4.0, "volume": 10.0}


def test_resample_ohlcv_and_alignment():
    h1 = make_1h(48, start="2021-03-01 00:00")
    bars = resample_4h(h1)
    assert len(bars) == 12
    assert set(bars.index.hour) == {0, 4, 8, 12, 16, 20}
    first = h1.iloc[:4]
    b = bars.iloc[0]
    assert b["open"] == first["open"].iloc[0]
    assert b["close"] == first["close"].iloc[-1]
    assert b["high"] == first["high"].max()
    assert b["low"] == first["low"].min()
    assert np.isclose(b["volume"], first["volume"].sum())
    assert b["n_hours"] == 4


def test_resample_aligns_when_data_starts_mid_bucket():
    h1 = make_1h(30, start="2021-03-01 02:00")
    bars = resample_4h(h1)
    assert bars.index[0] == pd.Timestamp("2021-03-01 00:00", tz="UTC")
    assert bars["n_hours"].iloc[0] == 2  # 02:00 and 03:00 only


def test_resample_drops_unclosed_bucket():
    h1 = make_1h(10, start="2021-03-01 00:00")  # 00:00..09:00, last 1h closes 10:00
    bars = resample_4h(h1)
    assert list(bars.index.hour) == [0, 4]  # 08:00 bucket closes at 12:00 -> excluded
    bars = resample_4h(h1, as_of=pd.Timestamp("2021-03-01 04:00", tz="UTC"))
    assert list(bars.index.hour) == [0]


def test_resample_as_of_ignores_later_hours():
    h1 = make_1h(24, start="2021-03-01 00:00")
    bars = resample_4h(h1, as_of=pd.Timestamp("2021-03-01 08:00", tz="UTC"))
    assert bars.index.max() == pd.Timestamp("2021-03-01 04:00", tz="UTC")
    assert bars["close"].iloc[-1] == h1["close"].iloc[7]


def test_resample_partial_and_empty_buckets():
    h1 = make_1h(24, start="2021-03-01 00:00")
    ts = lambda hour: pd.Timestamp(f"2021-03-01 {hour:02d}:00", tz="UTC")  # noqa: E731
    h1 = h1.drop([ts(5)])                          # one hour missing in the 04:00 bucket
    h1 = h1.drop([ts(12), ts(13), ts(14), ts(15)])  # 12:00 bucket fully missing
    bars = resample_4h(h1)
    assert len(bars) == 6                  # grid stays regular
    assert bars.loc["2021-03-01 04:00", "n_hours"].item() == 3
    empty = bars[bars["n_hours"] == 0]
    assert len(empty) == 1
    assert empty[["open", "high", "low", "close", "volume"]].isna().all().all()
