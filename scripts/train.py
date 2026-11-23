"""Fit the final model on every non-holdout row and save it to models/.

    python scripts/train.py

Runs feature selection, Bayesian tuning and the threshold choice exactly as one
walk-forward selection window does, but on all development rows (everything before
the holdout). Writes models/model.json (XGBoost native JSON) and models/meta.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

from cfl.data import BAR, load_1h, resample_4h
from cfl.features import FEATURES, NEXT_RET, TARGET, dataset
from cfl.model import N_ITER, SEED, choose_threshold, make_classifier, oof_probabilities, tune
from cfl.portfolio import FEE, SLIPPAGE
from cfl.selection import select_features
from cfl.walkforward import DATA_END, HOLDOUT_START, seed_everything, split_holdout

MODELS = Path(__file__).resolve().parents[1] / "models"


def main() -> int:
    seed_everything()
    t0 = time.time()
    ds = dataset(resample_4h(load_1h(), as_of=DATA_END))
    dev, _ = split_holdout(ds)
    X, y, r = dev[FEATURES], dev[TARGET], dev[NEXT_RET]
    print(f"training on {len(dev):,} rows ({dev.index.min():%Y-%m-%d} -> {dev.index.max():%Y-%m-%d})", flush=True)

    sel = select_features(X, y)
    cols = sel["features"]
    print(f"selected {sel['method']}@{sel['k']}: {cols}", flush=True)
    tuning = tune(X[cols], y, n_iter=N_ITER)
    params = tuning["params"]
    m = choose_threshold(oof_probabilities(X[cols], y, params), r)
    model = make_classifier(params).fit(X[cols], y)

    MODELS.mkdir(exist_ok=True)
    model.save_model(MODELS / "model.json")
    trained_through = dev.index.max() + 2 * BAR  # close of the last bar whose label was used
    meta = {
        "version": f"{trained_through:%Y%m%d%H%M}",
        "trainedThrough": trained_through.isoformat(),
        "trainRows": int(len(dev)),
        "trainFrom": dev.index.min().isoformat(),
        "holdoutStart": HOLDOUT_START.isoformat(),
        "features": cols,
        "selection": {"method": sel["method"], "k": sel["k"], "rankings": sel["rankings"],
                      "cvLogLoss": sel["cv_log_loss"]},
        "threshold": m,
        "decisionRule": "long if p > 0.5 + threshold, else flat",
        "params": params,
        "tuningCvLogLoss": tuning["cv_log_loss"],
        "seed": SEED,
        "costPerSide": {"fee": FEE, "slippage": SLIPPAGE},
        "bar": "4h, UTC-aligned (00/04/08/12/16/20)",
        "product": "BTC-USD",
    }
    (MODELS / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved models/model.json and models/meta.json (m={m}, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
