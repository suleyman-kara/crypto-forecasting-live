"""Walk-forward backtest: data -> features -> walk-forward -> metrics -> reports/.

    python scripts/backtest.py                 # uses the committed 1h cache
    python scripts/backtest.py --update-data   # fetch new 1h candles first

The holdout (last 6 months before DATA_END) is never touched unless
``--evaluate-holdout`` is passed. That flag is for the single, final evaluation after
the walk-forward results have been reviewed; it is off by default.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from cfl import report
from cfl.data import BAR, gap_report, load_1h, resample_4h, update_cache
from cfl.features import NEXT_RET, dataset
from cfl.metrics import evaluate, selection_stability, trading
from cfl.model import N_ITER, SEED, THRESHOLD_GRID
from cfl.portfolio import FEE, SLIPPAGE
from cfl.walkforward import (DATA_END, HOLDOUT_START, MIN_TRAIN, STEP, TUNE_EVERY,
                             run_holdout, run_walkforward, split_holdout)

REPORTS = report.REPORTS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update-data", action="store_true", help="fetch new 1h candles before running")
    ap.add_argument("--max-windows", type=int, default=None, help="stop after N windows (smoke test)")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--out", default=str(REPORTS), help="output directory")
    ap.add_argument("--evaluate-holdout", action="store_true",
                    help="ONE-TIME final evaluation on the holdout. Off by default; do not use during development.")
    args = ap.parse_args()

    h1 = update_cache() if args.update_data else load_1h()
    if h1.index.max() + pd.Timedelta(hours=1) < DATA_END:
        print(f"1h cache ends before DATA_END={DATA_END}; run with --update-data", file=sys.stderr)
        return 1
    bars = resample_4h(h1, as_of=DATA_END)
    gaps = gap_report(h1[h1.index < DATA_END], bars)
    ds = dataset(bars)
    dev, hold = split_holdout(ds)
    assert dev.index.max() + BAR < HOLDOUT_START

    if args.evaluate_holdout:
        return evaluate_holdout(ds, args)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"development rows: {len(dev):,} ({dev.index.min():%Y-%m-%d} -> {dev.index.max():%Y-%m-%d}); "
          f"holdout rows set aside: {len(hold):,} (from {HOLDOUT_START:%Y-%m-%d})", flush=True)

    t0 = time.time()
    preds, log = run_walkforward(dev, max_windows=args.max_windows, n_jobs=args.n_jobs)
    runtime = time.time() - t0

    results = evaluate(preds)
    no_threshold = trading((preds["p"] > 0.5).astype(int), preds[NEXT_RET])
    results_table = {"Model": results["Model"],
                     "Model, m = 0 (no threshold)": {**{k: None for k in ("accuracy", "auc", "log_loss", "brier")},
                                                    **no_threshold},
                     **{k: v for k, v in results.items() if k != "Model"}}
    stab = selection_stability(log)
    n_rounds = sum(1 for w in log if "selection" in w)

    model, bh = results["Model"], results["Buy & hold"]
    period = {"from": (preds.index.min() + BAR).isoformat(), "to": (preds.index.max() + 2 * BAR).isoformat()}
    summary = {
        "period": period,
        "accuracy": round(model["accuracy"], 6),
        "auc": round(model["auc"], 6),
        "returnPct": round(model["return_pct"], 4),
        "buyHoldReturnPct": round(bh["return_pct"], 4),
        "maxDrawdownPct": round(model["max_drawdown_pct"], 4),
        "sharpe": round(model["sharpe"], 4),
        "trades": int(model["trades"]),
        "predictions": int(len(preds)),
        "logLoss": round(model["log_loss"], 6),
        "brier": round(model["brier"], 6),
        "exposurePct": round(model["exposure_pct"], 4),
        "buyHoldMaxDrawdownPct": round(bh["max_drawdown_pct"], 4),
        "buyHoldSharpe": round(bh["sharpe"], 4),
        "binomialPValueVs50": model["binom_p_vs_50"],
        "binomialPValueVsUpRate": model["binom_p_vs_up_rate"],
        "upRate": round(model["up_rate"], 6),
        "baselines": {name: {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()
                             if k != "equity"} for name, r in results_table.items() if name != "Model"},
        "featureSelection": {f: int(row["selected"]) for f, row in stab.iterrows()},
        "selectionRounds": n_rounds,
        "holdout": load_holdout(out) or {"from": HOLDOUT_START.isoformat(), "to": DATA_END.isoformat(),
                                         "evaluated": False},
        "config": {"step": STEP, "minTrain": MIN_TRAIN, "tuneEvery": TUNE_EVERY, "nIter": N_ITER,
                   "fee": FEE, "slippage": SLIPPAGE, "thresholdGrid": list(THRESHOLD_GRID),
                   "dataEnd": DATA_END.isoformat()},
        "runtimeSeconds": round(runtime, 1),
    }
    report.write_json(summary, out / "backtest.json")
    (out / "windows.json").write_text(json.dumps(log, indent=1) + "\n")
    preds.to_csv(out / "walkforward_predictions.csv.gz", float_format="%.6g")

    report.plot_equity(model["equity"], bh["equity"], out / "equity.png")
    report.plot_rolling_accuracy(preds, out / "rolling_accuracy.png")
    (out / "backtest.md").write_text(render_markdown(summary, results_table, stab, n_rounds, preds, log, gaps, len(dev), len(hold)))
    print(f"\nwalk-forward finished in {runtime / 60:.1f} min; reports written to {out}")
    print(report.headline_table(results_table))
    return 0


def verdict(results: dict) -> str:
    m, bh = results["Model"], results["Buy & hold"]
    lines = []
    sig = m["binom_p_vs_50"] < 0.05
    lines.append(f"- Accuracy is {m['accuracy'] * 100:.2f}% on {m['n']:,} predictions; the binomial test against "
                 f"a coin flip gives p = {m['binom_p_vs_50']:.3g} "
                 f"({'different from 50% at the 5% level' if sig else 'not distinguishable from 50% at the 5% level'}).")
    lines.append(f"- {m['up_rate'] * 100:.2f}% of bars closed up, so 'always up' scores {m['up_rate'] * 100:.2f}%. "
                 f"One-sided test that the model beats that rate: p = {m['binom_p_vs_up_rate']:.3g}.")
    lines.append(f"- AUC {m['auc']:.3f} (0.5 = no ranking skill).")
    better_ret = m["return_pct"] > bh["return_pct"]
    better_sharpe = m["sharpe"] > bh["sharpe"]
    lines.append(f"- After costs the model returned {m['return_pct']:,.1f}% vs {bh['return_pct']:,.1f}% for buy & hold "
                 f"({'higher' if better_ret else 'lower'}), Sharpe {m['sharpe']:.2f} vs {bh['sharpe']:.2f} "
                 f"({'higher' if better_sharpe else 'lower'}), max drawdown {m['max_drawdown_pct']:.1f}% vs "
                 f"{bh['max_drawdown_pct']:.1f}%, in the market {m['exposure_pct']:.1f}% of the time.")
    return "\n".join(lines)


def render_markdown(summary, results, stab, n_rounds, preds, log, gaps, n_dev, n_hold) -> str:
    ms = [w["m"] for w in log]
    m_counts = pd.Series(ms).value_counts().sort_index()
    methods = pd.Series([w["method"] for w in log if w["tuned"]]).value_counts()
    ks = pd.Series([w["k"] for w in log if w["tuned"]]).value_counts().sort_index()
    h = summary["holdout"]
    return f"""# Walk-forward backtest: BTC-USD, next 4h candle

Generated by `python scripts/backtest.py`. {holdout_intro(h, n_hold)}

- Period traded: {summary['period']['from']} → {summary['period']['to']}
- Predictions: {summary['predictions']:,} (one per closed 4h candle, {len(log)} windows of {STEP} bars)
- Costs: {FEE * 100:.2f}% fee + {SLIPPAGE * 100:.2f}% slippage per side; long/flat, $10,000 start
- Runtime: {summary['runtimeSeconds'] / 60:.1f} min

## Results

{report.headline_table(results)}

Accuracy / AUC / log loss / Brier use `p > 0.5` as the prediction. Trading uses the
decision rule `long if p > 0.5 + m`. "Trades" counts entries (flat → long); every
entry and exit pays the per-side cost. "Always up" trades exactly like buy & hold;
"Model, m = 0" shows what the threshold changed. Sharpe is annualised from 4h
returns (2,190 bars a year, risk-free rate 0).

{verdict(results)}

![Equity vs buy & hold](equity.png)

![Rolling accuracy](rolling_accuracy.png)

### By year

{report.yearly_table(preds)}

{holdout_markdown(h) if h.get("evaluated") else ""}## Feature-selection stability

Selection ran {n_rounds} times (every {TUNE_EVERY}th window, on that window's
training rows only). The chosen ranker was {', '.join(f'{k} {v}×' for k, v in methods.items())};
chosen K was {', '.join(f'{k}: {v}×' for k, v in ks.items())}.

{report.stability_table(stab, n_rounds)}

"In X top-K" counts the rounds in which the feature was inside that ranker's first K,
for the K chosen in that round, whether or not that ranker won.

### Selection rounds

{report.windows_table(log)}

Threshold m chosen per window: {', '.join(f'{k:.3f}: {v}×' for k, v in m_counts.items())}.

## Data

- Source: Coinbase Exchange public API, 1h candles, resampled to 4h (00/04/…/20 UTC).
- 1h candles: {gaps['hours_present']:,} of {gaps['hours_expected']:,} expected hours
  ({gaps['hours_missing']} missing: {', '.join(f'{y}: {n}' for y, n in gaps['hours_missing_by_year'].items())}).
- 4h candles: {gaps['bars_total']:,}; {gaps['bars_partial']} built from fewer than 4 hours,
  {gaps['bars_empty']} with no data at all (left empty, never filled).
- Development rows after dropping NaN: {n_dev:,}. The first {MIN_TRAIN:,} rows are only ever training data.

## Decisions

| Decision | Reason |
| --- | --- |
| Last 6 months held out before anything else runs | Model and threshold choices made by looking at a test set give optimistic scores (Known Limitation 1 of the retail project). The holdout is opened once, later, with `--evaluate-holdout`. |
| The last development row is dropped | Its label is the first holdout candle; keeping it would leak a holdout price into training. |
| Expanding walk-forward, retrain every 168 bars (4 weeks) | Every prediction is made by a model that only saw earlier candles, the way the live system will run. |
| `TimeSeriesSplit(n_splits=4, gap=1)` for tuning, 3 splits for selection | Random K-fold on time-ordered rows trains on the future (Known Limitation 2). Selection uses 3 splits to keep the SFS cost down. |
| `neg_log_loss` scoring | The model outputs a probability and the trading rule thresholds it, so calibration matters more than accuracy. |
| MRMR, RFE and forward SFS compete; (ranker, K) picked by CV log loss on the training window | Same rankers as the retail project, but the winner is chosen without the test set. |
| SFS keeps the order features were added | scikit-learn's selector returns an unordered mask, so "first K" meant column order (Known Limitation 3). |
| Selection and tuning every {TUNE_EVERY}th window, reused in between | Keeps the full run at about {summary['runtimeSeconds'] / 60:.0f} minutes on 4 CPU cores. Both still only see the training rows of the window they run in. |
| BayesSearchCV `n_iter={N_ITER}`; SFS adds at most 8 features; K in {{3, 5, 8}} | Runtime budget. The search space is wider than the retail one (adds `colsample_bytree`, `min_child_weight`, log-uniform learning rate). |
| Threshold m chosen per window from out-of-fold training predictions | m has to cover 0.30% round-trip costs; picking it on the walk-forward results would be tuning on the test set. Grid: {', '.join(str(x) for x in THRESHOLD_GRID)}. |
| NaN rows are dropped, never filled with 0 | Filling invents data (the retail pipeline filled lag NaNs with 0). |
| Empty 4h buckets stay empty; rows whose 42-bar look-back touches one are dropped | Forward-filling an outage would create flat fake candles and fake 0 returns. Partial buckets (missing hours with no trades) keep the hours that exist. |
| Only fully closed candles | A 4h candle is used only when all four hours have closed; the open candle never enters a feature. |
| Frozen `DATA_END` ({DATA_END:%Y-%m-%d %H:%M} UTC) | The cache grows over time; freezing the cut keeps the report reproducible. |
| Fixed seed ({SEED}) for numpy, XGBoost, BayesSearchCV and the random baseline | Two runs give the same report. |

## Known Limitations

| Limitation | Why it is left |
| --- | --- |
| Selection and tuning run every 6th window, not every window | Runtime; in-between windows reuse a choice made on older (still strictly past) data. |
| One asset, one exchange, one bar size | The live system trades exactly this, so the backtest matches it. |
| Costs are a flat 0.15% per side | Real slippage depends on size and liquidity; for a $10,000 paper account it is small. |
| No short side | Long/flat only, as in the live paper portfolio. |
| The design was fixed before the run and not iterated on the results | Re-tuning features or thresholds after seeing these numbers would overfit the walk-forward period, the problem this project exists to fix. Any change would need the holdout to stay closed until it is final. |
| Bars dropped around the five data outages are skipped by every strategy alike | Buy & hold return here is computed on the same rows as the model, so it can differ slightly from the raw price change. |
"""


HOLDOUT_INTRO_OPEN = "Every number below comes from the\nwalk-forward period only. **The holdout"
SECTION_ANCHOR = "## Feature-selection stability"


def holdout_intro(h: dict, n_hold: int) -> str:
    if h.get("evaluated"):
        return (f"Every number below except the\nholdout section comes from the walk-forward period only. "
                f"**The holdout ({h['from'][:10]} → {h['to'][:10]})\nwas evaluated once, on "
                f"{h['evaluatedAt'][:10]}, after the design above was frozen.**")
    return (f"Every number below comes from the\nwalk-forward period only. **The holdout ({h['from'][:10]} →\n"
            f"{h['to'][:10]}, {n_hold:,} rows) has not been evaluated.**")


def load_holdout(out: Path) -> dict | None:
    path = Path(out) / "holdout.json"
    return json.loads(path.read_text()) if path.exists() else None


def holdout_markdown(h: dict) -> str:
    """The holdout section, rendered from reports/holdout.json alone."""
    rows = h["table"]
    lines = ["| Strategy | Accuracy | AUC | Log loss | Brier | Return % | Max DD % | Sharpe | Exposure % | Trades |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    f = report._f
    for name, r in rows.items():
        lines.append(
            f"| {name} | {f(r.get('accuracy') and r['accuracy'] * 100, '{:.2f}%')} | {f(r.get('auc'), '{:.3f}')} "
            f"| {f(r.get('log_loss'), '{:.4f}')} | {f(r.get('brier'), '{:.4f}')} "
            f"| {r['return_pct']:,.1f} | {r['max_drawdown_pct']:.1f} | {r['sharpe']:.2f} "
            f"| {r['exposure_pct']:.1f} | {r['trades']:,} |")
    m, bh = rows["Model"], rows["Buy & hold"]
    sig = h["binomialPValueVs50"] < 0.05
    rounds = "; ".join(f"window {w['window']} (training through {w['train_to'][:10]}): "
                       f"{w['method']}@{w['k']} " + ", ".join(f"`{c}`" for c in w["features"])
                       for w in h["selectionRounds"])
    ms = ", ".join(f"{x:.3f}" for x in h["thresholds"])
    return f"""## Holdout (evaluated once, {h['evaluatedAt'][:10]})

The last six months were set aside before any development run and opened exactly
once, after the walk-forward results above were reviewed and the design was frozen.
Nothing was changed after seeing these numbers. The holdout runs the same pipeline as
the walk-forward: windows of {STEP} bars, the first trained on every development row,
feature selection + tuning in its first window and every {TUNE_EVERY}th after that,
threshold m chosen in every window on out-of-fold training predictions.

- Period: {h['period']['from']} → {h['period']['to']} ({h['predictions']:,} predictions, {h['windows']} windows)

{chr(10).join(lines)}

- Accuracy {m['accuracy'] * 100:.2f}% on {h['predictions']:,} predictions; binomial test vs a coin flip
  p = {h['binomialPValueVs50']:.3g} ({'different from' if sig else 'not distinguishable from'} 50% at the 5% level).
  {h['upRate'] * 100:.2f}% of holdout bars closed up; one-sided test that the model beats
  that rate: p = {h['binomialPValueVsUpRate']:.3g}.
- After costs the model returned {m['return_pct']:,.1f}% vs {bh['return_pct']:,.1f}% for buy & hold
  (max drawdown {m['max_drawdown_pct']:.1f}% vs {bh['max_drawdown_pct']:.1f}%), in the market
  {m['exposure_pct']:.1f}% of the time.
- Selection rounds: {rounds}. Thresholds m per window: {ms}.

"""


def evaluate_holdout(ds: pd.DataFrame, args) -> int:
    """One-time holdout evaluation.

    Writes reports/holdout.json (refuses to overwrite it), the holdout predictions, the
    holdout section of reports/backtest.md and the ``holdout`` object of
    reports/backtest.json. The walk-forward numbers are not touched.
    """
    out_dir = Path(args.out)
    out = out_dir / "holdout.json"
    if out.exists():
        print(f"{out} already exists: the holdout has been evaluated once. Refusing to rerun.", file=sys.stderr)
        return 1
    print("WARNING: evaluating the holdout. This should happen exactly once.", flush=True)
    evaluated_at = pd.Timestamp.now(tz="UTC").floor("s")
    preds, log = run_holdout(ds, n_jobs=args.n_jobs)
    # Persist the raw output first, before any formatting can fail
    preds.to_csv(out_dir / "holdout_predictions.csv.gz", float_format="%.6g")
    (out_dir / "holdout_windows.json").write_text(json.dumps(log, indent=1) + "\n")

    results = evaluate(preds)
    no_threshold = trading((preds["p"] > 0.5).astype(int), preds[NEXT_RET])
    table = {"Model": results["Model"],
             "Model, m = 0 (no threshold)": {**{k: None for k in ("accuracy", "auc", "log_loss", "brier")},
                                            **no_threshold},
             **{k: v for k, v in results.items() if k != "Model"}}
    m, bh = results["Model"], results["Buy & hold"]
    clean = {name: {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if k != "equity"}
             for name, r in table.items()}
    h = {
        "from": HOLDOUT_START.isoformat(), "to": DATA_END.isoformat(), "evaluated": True,
        "evaluatedAt": evaluated_at.isoformat(),
        "period": {"from": (preds.index.min() + BAR).isoformat(), "to": (preds.index.max() + 2 * BAR).isoformat()},
        "predictions": int(len(preds)), "windows": len(log),
        "accuracy": round(m["accuracy"], 6), "auc": round(m["auc"], 6),
        "logLoss": round(m["log_loss"], 6), "brier": round(m["brier"], 6),
        "returnPct": round(m["return_pct"], 4), "buyHoldReturnPct": round(bh["return_pct"], 4),
        "maxDrawdownPct": round(m["max_drawdown_pct"], 4), "buyHoldMaxDrawdownPct": round(bh["max_drawdown_pct"], 4),
        "sharpe": round(m["sharpe"], 4), "buyHoldSharpe": round(bh["sharpe"], 4),
        "exposurePct": round(m["exposure_pct"], 4), "trades": int(m["trades"]),
        "alwaysUpAccuracy": round(results["Always up"]["accuracy"], 6),
        "upRate": round(m["up_rate"], 6),
        "binomialPValueVs50": m["binom_p_vs_50"], "binomialPValueVsUpRate": m["binom_p_vs_up_rate"],
        "thresholds": [w["m"] for w in log],
        "selectionRounds": [{"window": w["window"], "train_to": w["train_to"], "method": w["method"],
                             "k": w["k"], "features": w["features"], "params": w["params"]}
                            for w in log if w["tuned"]],
        "table": clean,
    }
    report.write_json(h, out)

    bt_json = out_dir / "backtest.json"
    if bt_json.exists():
        summary = json.loads(bt_json.read_text())
        summary["holdout"] = h
        report.write_json(summary, bt_json)
    bt_md = out_dir / "backtest.md"
    if bt_md.exists():
        md = bt_md.read_text()
        i = md.index(HOLDOUT_INTRO_OPEN)
        j = md.index("has not been evaluated.**", i) + len("has not been evaluated.**")
        md = md[:i] + holdout_intro(h, h["predictions"]) + md[j:]
        md = md.replace(SECTION_ANCHOR, holdout_markdown(h) + SECTION_ANCHOR, 1)
        bt_md.write_text(md)
    print(report.headline_table(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
