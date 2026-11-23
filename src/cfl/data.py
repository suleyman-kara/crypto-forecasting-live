"""Coinbase 1h candles, incremental parquet cache, and 4h resampling.

Coinbase Exchange has no 4h granularity, so 1h candles are fetched and resampled to
4h candles aligned to 00/04/08/12/16/20 UTC. Only closed candles are ever stored or used.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
PRODUCT = "BTC-USD"
HOUR = pd.Timedelta(hours=1)
BAR = pd.Timedelta(hours=4)
MAX_ROWS = 300  # Coinbase returns at most 300 candles per request
EARLIEST = pd.Timestamp("2015-07-01", tz="UTC")  # BTC-USD history starts on 2015-07-20
COLUMNS = ["open", "high", "low", "close", "volume"]

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data" / "candles_1h.parquet"
LIVE_PATH = ROOT / "data" / "candles_1h_live.csv"  # appended by the live job


def _get(session: requests.Session, url: str, params: dict, retries: int = 6) -> list:
    """GET with retry on 429 / 5xx / network errors and exponential backoff."""
    delay = 1.0
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=30)
        except requests.RequestException:
            if attempt == retries - 1:
                raise
        else:
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code != 429 and resp.status_code < 500:
                resp.raise_for_status()
            if attempt == retries - 1:
                resp.raise_for_status()
        time.sleep(delay)
        delay *= 2
    raise RuntimeError("unreachable")


def _to_frame(rows: list) -> pd.DataFrame:
    """Coinbase rows are [time, low, high, open, close, volume], newest first."""
    if not rows:
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="time"), dtype=float)
    arr = np.asarray(rows, dtype=float)
    df = pd.DataFrame(
        {"open": arr[:, 3], "high": arr[:, 2], "low": arr[:, 1], "close": arr[:, 4], "volume": arr[:, 5]},
        index=pd.to_datetime(arr[:, 0].astype("int64"), unit="s", utc=True),
    )
    df.index.name = "time"
    return df.sort_index()


def fetch_1h(start: pd.Timestamp, end: pd.Timestamp, product: str = PRODUCT,
             sleep: float = 0.25, verbose: bool = True) -> pd.DataFrame:
    """Fetch closed 1h candles whose open time is in [start, end)."""
    url = API_URL.format(product=product)
    session = requests.Session()
    session.headers["User-Agent"] = "cfl-backtest"
    frames = []
    cursor = start
    n = 0
    while cursor < end:
        chunk_end = min(cursor + MAX_ROWS * HOUR, end)
        # Coinbase treats `end` as inclusive, so ask for one hour less
        params = {"granularity": 3600, "start": cursor.isoformat(),
                  "end": (chunk_end - HOUR).isoformat()}
        frames.append(_to_frame(_get(session, url, params)))
        cursor = chunk_end
        n += 1
        if verbose and n % 25 == 0:
            print(f"  fetched through {cursor:%Y-%m-%d} ({n} requests)", flush=True)
        time.sleep(sleep)
    if not frames:
        return _to_frame([])
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[(df.index >= start) & (df.index < end)]


def _read_live(live_path: Path) -> pd.DataFrame | None:
    if live_path is None or not live_path.exists():
        return None
    df = pd.read_csv(live_path)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("time"), utc=True), name="time")
    return df[COLUMNS].astype(float)


def _append_live(live_path: Path, new: pd.DataFrame) -> None:
    """Append rows to the text cache (one line per hour, git-diff friendly)."""
    live_path.parent.mkdir(parents=True, exist_ok=True)
    out = new[COLUMNS].copy()
    out.index = out.index.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.index.name = "time"
    header = not live_path.exists()
    with live_path.open("a", newline="") as fh:
        out.to_csv(fh, header=header)


def update_cache(path: Path = CACHE_PATH, now: pd.Timestamp | None = None,
                 verbose: bool = True, live_path: Path | None = LIVE_PATH) -> pd.DataFrame:
    """Fetch newly closed 1h candles and return the full cache.

    The parquet file holds the bulk history and is only written when it does not exist
    yet. Later candles are appended to ``live_path`` (CSV), so the scheduled live job
    adds a few small lines per run instead of rewriting a 4 MB binary file in git.
    Pass ``live_path=None`` to append to the parquet file instead.
    """
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    last_closed_end = now.floor("h")  # the candle opened at now.floor('h') is still open
    cache = load_1h(path, live_path) if path.exists() else None
    start = EARLIEST if cache is None else cache.index.max() + HOUR
    if start < last_closed_end:
        if verbose:
            print(f"Fetching 1h candles {start} -> {last_closed_end}", flush=True)
        new = fetch_1h(start, last_closed_end, verbose=verbose)
        new = new[new.index >= start]
        if len(new):
            if cache is not None and live_path is not None:
                _append_live(live_path, new)
                cache = pd.concat([cache, new])
            else:
                cache = new if cache is None else pd.concat([cache, new])
                cache = cache[~cache.index.duplicated(keep="last")].sort_index()
                path.parent.mkdir(parents=True, exist_ok=True)
                cache.to_parquet(path, compression="zstd")
    return cache


def load_1h(path: Path = CACHE_PATH, live_path: Path | None = LIVE_PATH) -> pd.DataFrame:
    """The parquet history plus any candles appended to the live CSV."""
    df = pd.read_parquet(path)
    live = _read_live(live_path)
    if live is not None and len(live):
        df = pd.concat([df, live])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def resample_4h(h1: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Resample 1h candles to 4h candles on a regular 00/04/.../20 UTC grid.

    - A 4h candle is kept only if it is fully closed: open time + 4h <= ``as_of``.
      ``as_of`` defaults to the close time of the last 1h candle.
    - Hours missing inside a bucket (Coinbase omits hours without trades, and had a
      few outages) are skipped: open = first available open, close = last available
      close, high/low = max/min, volume = sum. ``n_hours`` records how many were present.
    - Buckets with no 1h candle at all stay on the grid as NaN rows. They are never
      filled; every feature or target that touches them becomes NaN and is dropped.
    """
    if as_of is None:
        as_of = h1.index.max() + HOUR
    h1 = h1[h1.index + HOUR <= as_of]
    g = h1.resample("4h", origin="epoch", label="left", closed="left")
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(min_count=1),
        "n_hours": g["close"].count(),
    })
    out = out[out.index + BAR <= as_of]
    grid = pd.date_range(out.index.min(), out.index.max(), freq="4h", tz="UTC", name="time")
    out = out.reindex(grid)
    out["n_hours"] = out["n_hours"].fillna(0).astype(int)
    out.loc[out["n_hours"] == 0, COLUMNS] = np.nan
    return out


def gap_report(h1: pd.DataFrame, bars: pd.DataFrame) -> dict:
    """Counts of missing 1h candles and partial / empty 4h buckets."""
    full = pd.date_range(h1.index.min(), h1.index.max(), freq="h", tz="UTC")
    missing_h = full.difference(h1.index)
    by_year = pd.Series(1, index=missing_h).groupby(missing_h.year).sum().to_dict() if len(missing_h) else {}
    return {
        "hours_expected": int(len(full)),
        "hours_present": int(len(h1)),
        "hours_missing": int(len(missing_h)),
        "hours_missing_by_year": {int(k): int(v) for k, v in by_year.items()},
        "bars_total": int(len(bars)),
        "bars_empty": int((bars["n_hours"] == 0).sum()),
        "bars_partial": int(((bars["n_hours"] > 0) & (bars["n_hours"] < 4)).sum()),
    }
