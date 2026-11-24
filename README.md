# Crypto Forecasting Live

An XGBoost model that predicts the direction of the next 4-hour BTC-USD candle and
publishes each prediction **before** the outcome is known. Every prediction is committed
to this repository, so the git history is a timestamped record that cannot be edited
after the fact. A paper portfolio (virtual balance, real prices) trades on the
predictions and is compared with buy-and-hold.

This project continues
[retail-sales-forecasting](https://github.com/suleyman-kara/retail-sales-forecasting):
the same lag features, MRMR / RFE / SFS feature selection and Bayesian tuning, with the
limitations listed in that README fixed:

| Limitation there | Fix here |
| --- | --- |
| The best model is selected on the test set | Walk-forward validation plus an untouched holdout, and live predictions nobody can overfit |
| `cv=3` splits time-ordered rows randomly | `TimeSeriesSplit` with a gap |
| SFS returns an unordered subset | Ordered selection with a stability report |

> Paper trading only. Not investment advice.

## Results

Full report: [reports/backtest.md](reports/backtest.md). Walk-forward over
2017-08 → 2026-05 (19,198 predictions), then a six-month holdout opened once.

| | Walk-forward | Holdout (2026-05-23 → 2026-11-23) |
| --- | ---: | ---: |
| Model accuracy | 53.66% | 53.08% |
| "Always up" accuracy | 51.27% | 49.46% |
| Model AUC | 0.554 | 0.547 |
| Binomial p vs coin flip | 3.5e-24 | 0.044 |
| Model return after costs | -93.6% | -6.3% |
| Buy & hold return | +2,730.4% | -19.1% |
| Model max drawdown | -97.8% | -7.2% |

**The finding:** the classifier has a small but statistically significant edge in
direction (about 53.5% against a 51% base rate, held up on the holdout). On a 4-hour
long/flat strategy, 0.15% per side in fees and slippage erases that edge: the model
lost money in the walk-forward while buy & hold gained 28×. In the holdout it lost less
than buy & hold only because the cost-covering threshold kept it in cash 97% of the
time. The design was frozen before any of these numbers were seen and was not tuned
afterwards; the live log below tests it on candles that did not exist when it was built.

## Live

Every 4 hours, 7 minutes after a candle closes, [`predict.yml`](.github/workflows/predict.yml)
runs `scripts/predict.py`:

1. Fetch the new 1h candles from Coinbase.
2. Score every earlier prediction whose target candle has now closed: **hit** when the
   predicted direction matches the sign of that candle's close-to-close return.
3. Predict the candle that has just opened, using data through the close of the one
   that just closed, and append it to [`live/predictions.jsonl`](live/predictions.jsonl)
   with `predicted_at`, `p` = P(up), direction, action (`long` if `p > 0.5 + m`, else
   `flat`) and the model version.
4. Update the paper portfolio ($10,000, long/flat, the backtest's costs) and buy & hold,
   both starting at the first live prediction, and write
   [`live/status.json`](live/status.json).
5. Commit `data/` and `live/` if anything changed.

Missed slots are never filled in afterwards. If no prediction was made while a target
candle was open, or a run starts more than an hour after it opened, the slot is
recorded as `skipped` with the reason. Running the script twice for the same candle
changes nothing. [`retrain.yml`](.github/workflows/retrain.yml) refits the model on all
committed candles every Sunday and commits `models/` with a new version.

**Why the log can be trusted:** each prediction reaches GitHub in a commit made while
its target candle is still open, and the outcome is written by a later commit. Git
history and GitHub's push timestamps show when every line appeared; changing a past
prediction would mean rewriting public history, which anyone with a clone would see.

The website reads the summary straight from
`https://raw.githubusercontent.com/suleyman-kara/crypto-forecasting-live/main/live/status.json`
(schema 1: current prediction, hit rate overall and over 30 days, paper equity vs buy &
hold, the last 20 predictions, and the walk-forward backtest numbers).

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e . --no-deps

python -m pytest -q                # tests, no network
python scripts/backtest.py         # walk-forward report into reports/ (~9 min, 4 cores)
python scripts/train.py            # live model into models/ (~30 s)
python scripts/predict.py          # one live step (fetches candles, writes live/)
```

`--update-data` on the first two fetches new 1h candles from the Coinbase Exchange
public API (no key) into `data/candles_1h.parquet` first. The backtest stops at a frozen
`DATA_END`, so a larger cache does not change the report. `backtest.py
--evaluate-holdout` was the one-time holdout evaluation; it refuses to run again once
`reports/holdout.json` exists.

## Files

| Path | What it is |
| --- | --- |
| `src/cfl/data.py` | Coinbase 1h candles, incremental parquet cache, 4h resampling |
| `src/cfl/features.py` | Return lags, volatility, RSI, EMA spread, range, volume z-score, calendar |
| `src/cfl/selection.py` | MRMR, RFE and ordered forward SFS inside one training window |
| `src/cfl/model.py` | XGBClassifier, BayesSearchCV with `TimeSeriesSplit(gap=1)`, threshold choice |
| `src/cfl/walkforward.py` | Holdout split and expanding-window walk-forward |
| `src/cfl/portfolio.py` | Long/flat paper portfolio with fees and slippage |
| `src/cfl/metrics.py` | Classification and trading metrics, baselines, selection stability |
| `src/cfl/report.py` | Charts and tables for `reports/` |
| `scripts/backtest.py` | Walk-forward backtest and the one-time holdout evaluation |
| `src/cfl/live.py` | Live prediction, scoring, paper portfolio, `status.json` |
| `scripts/train.py` | Fits the live model on every labelled candle |
| `scripts/predict.py` | One live step (run by `predict.yml`) |
| `data/candles_1h.parquet` | BTC-USD 1h candles since 2015-07-20 (bulk history) |
| `data/candles_1h_live.csv` | Newer 1h candles, appended by the live job |
| `live/predictions.jsonl` | Every live prediction and its outcome, one line per 4h slot |
| `live/status.json` | Summary the website reads |
| `models/model.json`, `meta.json` | Live model (XGBoost JSON) and its features, threshold and parameters |
| `reports/` | Backtest and holdout report, charts, per-row predictions |

## How it works

1. Fetch closed 1h candles and resample to 4h candles on a 00/04/…/20 UTC grid.
2. Build features at the close of each 4h candle from that candle and earlier ones;
   the target is whether the next candle's log return is positive.
3. Set aside the last six months as a holdout.
4. Walk forward: train on everything before a 4-week window, predict the window,
   move on. Feature selection and tuning run on the training rows only.
5. Trade `long if p > 0.5 + m`, otherwise cash, paying 0.10% fee + 0.05% slippage per side.
6. Compare with buy & hold, always-up, momentum and random baselines.

## Decisions

| Decision | Reason |
| --- | --- |
| Last 6 months held out, opened once after the design was frozen | Choosing a model by looking at the test set gives optimistic scores (Known Limitation 1 of the retail project). |
| Results published as they came out, no re-tuning | Changing features or thresholds after seeing the walk-forward numbers would overfit it, the problem this project exists to fix. |
| The live model is trained on every labelled candle, holdout included | The holdout has served its one evaluation; leaving the most recent six months out of the live model would only make it staler. |
| Expanding walk-forward, retrain every 168 bars (4 weeks) | Each prediction comes from a model that only saw earlier candles, as in live use. |
| `TimeSeriesSplit(n_splits=4, gap=1)` and `neg_log_loss` for tuning | Random K-fold trains on the future (Known Limitation 2); the output is a probability, so calibration matters. |
| MRMR, RFE and SFS compete; SFS keeps its order | The winner and K are picked by CV on the training window, and "first K" means the first K chosen (Known Limitation 3). |
| Threshold m chosen on out-of-fold training predictions | m has to cover 0.30% round-trip costs; choosing it on test results would be tuning on the test set. |
| NaN rows dropped, empty 4h buckets left empty | Filling with 0 or forward-filling invents prices and returns. |
| Live slots that were missed or started late are `skipped`, never back-filled | A prediction made after its candle opened has seen part of the outcome. |
| A skipped slot keeps the previous paper position | Nobody acted, so whatever was held stays held; the portfolio is marked to that candle's return. |
| New candles go to an append-only CSV, not the parquet file | Rewriting a 4 MB binary every 4 hours would add gigabytes to the git history within a year. |
| Coinbase, not Binance | Binance blocks GitHub's US runners; Coinbase's public candles need no key. |
| Fixed seed (42) | Two runs give the same report. |

## Known Limitations

| Limitation | Why it is left |
| --- | --- |
| The edge does not survive trading costs at this bar size | That is the result; the project reports it rather than searching for a variant that looks better. |
| Holdout accuracy rests on 1,104 predictions (p = 0.044) | Six months of 4h candles is all the holdout has; the live log keeps adding out-of-sample evidence. |
| Selection and tuning every 6th window, not every window | Runtime; windows in between reuse a choice made on older, still strictly past, data. |
| One asset, one exchange, one bar size, long/flat only | The live paper portfolio trades exactly this, so the backtest matches it. |
| Costs are a flat 0.15% per side | Real slippage depends on size and liquidity; for a $10,000 paper account it is small. |
