"""Fit the live model on every labelled row and save it to models/.

    python scripts/train.py                 # uses the committed 1h cache
    python scripts/train.py --update-data   # fetch new 1h candles first (weekly retrain)

Runs feature selection, Bayesian tuning and the threshold choice exactly as one
walk-forward selection window does, on every row whose label is known: all closed
4h candles up to the latest one. The holdout is included: it has been used for its
single evaluation (reports/holdout.json) and is now ordinary training data.
Writes models/model.json (XGBoost native JSON) and models/meta.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from cfl.data import BAR, load_1h, resample_4h, update_cache
from cfl.features import FEATURES, NEXT_RET, TARGET, dataset
from cfl.model import N_ITER, SEED, choose_threshold, make_classifier, oof_probabilities, tune
from cfl.portfolio import FEE, SLIPPAGE
from cfl.selection import select_features
from cfl.walkforward import seed_everything

MODELS = Path(__file__).resolve().parents[1] / "models"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update-data", action="store_true", help="fetch new 1h candles before training")
    ap.add_argument("--out", default=str(MODELS), help="output directory")
    args = ap.parse_args()

    seed_everything()
    t0 = time.time()
    h1 = update_cache() if args.update_data else load_1h()
    ds = dataset(resample_4h(h1))  # only closed 4h candles; the last one has no label yet and is dropped
    X, y, r = ds[FEATURES], ds[TARGET], ds[NEXT_RET]
    print(f"training on {len(ds):,} rows ({ds.index.min():%Y-%m-%d} -> {ds.index.max():%Y-%m-%d %H:%M})", flush=True)

    sel = select_features(X, y)
    cols = sel["features"]
    print(f"selected {sel['method']}@{sel['k']}: {cols}", flush=True)
    tuning = tune(X[cols], y, n_iter=N_ITER)
    params = tuning["params"]
    m = choose_threshold(oof_probabilities(X[cols], y, params), r)
    model = make_classifier(params).fit(X[cols], y)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(out / "model.json")
    trained_through = ds.index.max() + 2 * BAR  # close of the last bar whose label was used
    meta = {
        "version": f"{trained_through:%Y-%m-%d-%H%M}",
        "trainedThrough": trained_through.isoformat(),
        "trainRows": int(len(ds)),
        "trainFrom": ds.index.min().isoformat(),
        "includesHoldout": True,
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
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved {out}/model.json and meta.json (version {meta['version']}, m={m}, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
