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

**Status:** work in progress.
