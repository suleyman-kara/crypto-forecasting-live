"""Live predictions: publish, score, paper-trade and summarise in live/status.json.

One run (``scripts/predict.py``, every 4 hours in GitHub Actions):

1. New 1h candles are fetched (``data.update_cache``).
2. A prediction is made for the latest fully closed 4h candle L: features use data
   through L's close, the target is the next candle (open L+4h, close L+8h), which is
   still open when the prediction is published. Slots between the previous record
   and L that never got a prediction are recorded as ``skipped`` and never
   back-filled. A run that starts more than ``MAX_LATENESS`` after the target candle
   opened also records ``skipped``: by then part of the outcome is already visible.
3. Records whose target candle has closed are scored: ``hit`` when the predicted
   direction matches the sign of the target candle's close-to-close return.
4. A long/flat paper portfolio ($10,000, same costs as the backtest) and buy & hold are
   computed from the first live prediction, and ``live/status.json`` is written.

Records are keyed by ``candle_open`` (the candle whose close the features use), so a
second run over the same candles changes nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .data import BAR, PRODUCT, resample_4h
from .features import FEATURES, build_features
from .portfolio import START_BALANCE, simulate

ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = ROOT / "live"
MODELS_DIR = ROOT / "models"
BACKTEST_JSON = ROOT / "reports" / "backtest.json"

MAX_LATENESS = pd.Timedelta(hours=1)
RECENT_MAX = 20
EQUITY_MAX = 500
SCHEMA = 1


# ---------------------------------------------------------------- helpers

def iso(ts: pd.Timestamp | None) -> str | None:
    if ts is None:
        return None
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(s: str) -> pd.Timestamp:
    return pd.Timestamp(s).tz_convert("UTC") if pd.Timestamp(s).tzinfo else pd.Timestamp(s, tz="UTC")


def latest_closed_open(now: pd.Timestamp) -> pd.Timestamp:
    """Open time of the latest 4h candle that has closed by ``now``."""
    return now.floor("4h") - BAR


@dataclass
class LiveModel:
    clf: xgb.XGBClassifier
    version: str
    trained_through: str
    features: list[str]
    threshold: float

    def predict(self, row: pd.DataFrame) -> float:
        return float(self.clf.predict_proba(row[self.features])[:, 1][0])


def load_model(models_dir: Path = MODELS_DIR) -> LiveModel:
    meta = json.loads((models_dir / "meta.json").read_text())
    clf = xgb.XGBClassifier()
    clf.load_model(models_dir / "model.json")
    return LiveModel(clf, meta["version"], meta["trainedThrough"], list(meta["features"]),
                     float(meta["threshold"]))


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dump_log(records: list[dict]) -> str:
    return "".join(json.dumps(r, separators=(", ", ": ")) + "\n" for r in records)


# ---------------------------------------------------------------- predict

def _skipped(candle: pd.Timestamp, reason: str) -> dict:
    return {"candle_open": iso(candle), "target_open": iso(candle + BAR), "target_close": iso(candle + 2 * BAR),
            "predicted_at": None, "p": None, "direction": None, "action": None, "threshold": None,
            "model_version": None, "outcome": "skipped", "reason": reason,
            "realized_return_pct": None, "resolved": False}


def add_predictions(records: list[dict], bars: pd.DataFrame, model: LiveModel,
                    now: pd.Timestamp) -> list[dict]:
    """Return records plus skipped slots and (if due) a prediction for the latest closed candle."""
    records = [dict(r) for r in records]
    latest = latest_closed_open(now)
    have = {r["candle_open"] for r in records}
    if records:
        c = max(parse(r["candle_open"]) for r in records) + BAR
        while c < latest:
            if iso(c) not in have:
                records.append(_skipped(c, "no prediction was made while the target candle was open"))
            c += BAR
    if iso(latest) in have or (records and parse(records[-1]["candle_open"]) > latest):
        return _sorted(records)

    target_open = latest + BAR
    late = now - target_open
    if late > MAX_LATENESS:
        records.append(_skipped(latest, f"run started {late.total_seconds() / 60:.0f} min after the "
                                        f"target candle opened (limit {MAX_LATENESS.total_seconds() / 60:.0f})"))
        return _sorted(records)
    if latest not in bars.index or pd.isna(bars.at[latest, "close"]):
        # Candle not in the data yet (exchange lag): leave the slot open; if no later
        # run fills it in time it becomes 'skipped'.
        return _sorted(records)
    feats = build_features(bars)[FEATURES]
    row = feats.loc[[latest]]
    if row[model.features].isna().any(axis=None):
        records.append(_skipped(latest, "features unavailable (gap in the price data)"))
        return _sorted(records)

    p = model.predict(row)
    records.append({
        "candle_open": iso(latest), "target_open": iso(target_open), "target_close": iso(target_open + BAR),
        "predicted_at": iso(now.floor("s")), "p": round(p, 6), "direction": "up" if p > 0.5 else "down",
        "action": "long" if p > 0.5 + model.threshold else "flat", "threshold": model.threshold,
        "model_version": model.version, "close_at_prediction": float(bars.at[latest, "close"]),
        "outcome": "pending", "realized_return_pct": None, "resolved": False,
    })
    return _sorted(records)


def _sorted(records: list[dict]) -> list[dict]:
    return sorted(records, key=lambda r: parse(r["candle_open"]))


# ---------------------------------------------------------------- score

def resolve(records: list[dict], bars: pd.DataFrame) -> list[dict]:
    """Fill the realized return of every record whose target candle has closed."""
    out = []
    last_open = bars.index.max()
    for r in records:
        r = dict(r)
        target = parse(r["target_open"])
        if not r.get("resolved") and target <= last_open:
            base = parse(r["candle_open"])
            c0 = bars["close"].get(base, np.nan)
            c1 = bars["close"].get(target, np.nan)
            r["resolved"] = True
            if pd.notna(c0) and pd.notna(c1):
                ret = (c1 / c0 - 1) * 100
                r["realized_return_pct"] = round(float(ret), 6)
                r["close_at_target"] = float(c1)
                if r["outcome"] == "pending":
                    r["outcome"] = "hit" if (ret > 0) == (r["direction"] == "up") else "miss"
            elif r["outcome"] == "pending":
                r["outcome"] = "skipped"
                r["reason"] = "no price data for the target candle"
        out.append(r)
    return out


# ---------------------------------------------------------------- portfolio

def portfolio(records: list[dict]) -> pd.DataFrame:
    """Paper equity of the model (long/flat) and buy & hold from the first prediction.

    One row per resolved target candle, starting with the first record that has a
    prediction. Skipped slots keep the previous position (nobody acted); a slot with
    no price data earns 0. Costs and timing are those of ``portfolio.simulate``.
    """
    first = next((i for i, r in enumerate(records) if r.get("p") is not None), None)
    if first is None:
        return pd.DataFrame(columns=["t", "position", "ret", "model", "buyHold"])
    rows, pos = [], 0.0
    for r in records[first:]:
        if not r.get("resolved"):
            break
        if r["action"] is not None:
            pos = 1.0 if r["action"] == "long" else 0.0
        ret = (r.get("realized_return_pct") or 0.0) / 100
        rows.append({"t": parse(r["target_close"]), "position": pos, "ret": ret})
    df = pd.DataFrame(rows, columns=["t", "position", "ret"])
    if df.empty:
        return df.assign(model=[], buyHold=[])
    df["model"] = simulate(df["position"].to_numpy(), df["ret"].to_numpy())["equity"].to_numpy()
    df["buyHold"] = simulate(np.ones(len(df)), df["ret"].to_numpy())["equity"].to_numpy()
    return df


def _max_drawdown_pct(equity: np.ndarray) -> float:
    eq = np.concatenate([[START_BALANCE], equity])
    return float((eq / np.maximum.accumulate(eq) - 1).min() * 100)


# ---------------------------------------------------------------- status

def backtest_summary(path: Path = BACKTEST_JSON) -> dict | None:
    if not path.exists():
        return None
    b = json.loads(path.read_text())
    return {"period": {"from": b["period"]["from"], "to": b["period"]["to"]},
            "accuracy": b["accuracy"], "auc": b["auc"], "returnPct": b["returnPct"],
            "buyHoldReturnPct": b["buyHoldReturnPct"], "maxDrawdownPct": b["maxDrawdownPct"],
            "sharpe": b["sharpe"], "trades": b["trades"]}


def build_status(records: list[dict], model: dict | None, backtest: dict | None,
                 now: pd.Timestamp) -> dict:
    predicted = [r for r in records if r.get("p") is not None]
    scored = [r for r in records if r["outcome"] in ("hit", "miss")]
    hits = sum(r["outcome"] == "hit" for r in scored)
    cutoff = now - pd.Timedelta(days=30)
    recent30 = [r for r in scored if parse(r["target_close"]) >= cutoff]

    pf = portfolio(records)
    equity = START_BALANCE if pf.empty else float(pf["model"].iloc[-1])
    bh = START_BALANCE if pf.empty else float(pf["buyHold"].iloc[-1])
    since = parse(predicted[0]["target_open"]) if predicted else None
    curve = []
    if since is not None:
        curve = [{"t": iso(since), "model": START_BALANCE, "buyHold": START_BALANCE}]
        curve += [{"t": iso(row.t), "model": round(row.model, 2), "buyHold": round(row.buyHold, 2)}
                  for row in pf.itertuples()]

    last = records[-1] if records else None
    current = None
    if last is not None and last["outcome"] == "pending":
        current = {"candleOpen": last["target_open"], "targetClose": last["target_close"], "p": last["p"],
                   "direction": last["direction"], "action": last["action"],
                   # Site shows the p(up) level above which the model goes long: 0.5 + m.
                   "threshold": round(0.5 + last["threshold"], 6)}

    recent = [{"candleOpen": r["target_open"], "p": r["p"], "direction": r["direction"], "action": r["action"],
               "outcome": r["outcome"],
               "realizedReturnPct": (round(r["realized_return_pct"], 4)
                                     if r["outcome"] in ("hit", "miss") else None)}
              for r in reversed(records[-RECENT_MAX:])]

    return {
        "schema": SCHEMA,
        "generatedAt": iso(now.floor("s")),
        "asset": PRODUCT,
        "interval": "4h",
        "mode": "paper",
        "startingBalance": START_BALANCE,
        "model": model,
        "current": current,
        "stats": {
            "predictions": len(predicted),
            "scored": len(scored),
            "hits": hits,
            "accuracy": round(hits / len(scored), 6) if scored else None,
            "accuracy30d": (round(sum(r["outcome"] == "hit" for r in recent30) / len(recent30), 6)
                            if recent30 else None),
            "equity": round(equity, 2),
            "returnPct": round((equity / START_BALANCE - 1) * 100, 4),
            "buyHoldReturnPct": round((bh / START_BALANCE - 1) * 100, 4),
            "maxDrawdownPct": round(_max_drawdown_pct(pf["model"].to_numpy(dtype=float)), 4) if not pf.empty else 0.0,
            "since": iso(since),
        },
        "equity": curve[-EQUITY_MAX:],
        "recent": recent,
        "backtest": backtest,
    }


def _same_status(a: dict, b: dict) -> bool:
    strip = lambda d: {k: v for k, v in d.items() if k != "generatedAt"}  # noqa: E731
    return strip(a) == strip(b)


# ---------------------------------------------------------------- run

@dataclass
class RunResult:
    changed: bool
    log_changed: bool
    status_changed: bool
    latest: dict | None


def run(h1: pd.DataFrame, now: pd.Timestamp, live_dir: Path = LIVE_DIR, models_dir: Path = MODELS_DIR,
        backtest_json: Path = BACKTEST_JSON) -> RunResult:
    """Predict, score and write live/ from a 1h candle frame. No network access."""
    model = load_model(models_dir)
    bars = resample_4h(h1, as_of=min(now.floor("h"), h1.index.max() + pd.Timedelta(hours=1)))
    log_path, status_path = live_dir / "predictions.jsonl", live_dir / "status.json"

    old = read_log(log_path)
    records = resolve(add_predictions(old, bars, model, now), bars)
    log_changed = dump_log(records) != dump_log(old)

    status = build_status(records, {"version": model.version, "trainedThrough": model.trained_through},
                          backtest_summary(backtest_json), now)
    old_status = json.loads(status_path.read_text()) if status_path.exists() else None
    status_changed = old_status is None or not _same_status(status, old_status)

    live_dir.mkdir(parents=True, exist_ok=True)
    if log_changed:
        log_path.write_text(dump_log(records))
    if status_changed:
        status_path.write_text(json.dumps(status, indent=1) + "\n")
    return RunResult(log_changed or status_changed, log_changed, status_changed, records[-1] if records else None)
