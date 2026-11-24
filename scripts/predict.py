"""Publish the live prediction for the latest closed 4h candle and update live/.

    python scripts/predict.py              # fetch new candles, predict, score, write live/
    python scripts/predict.py --no-fetch   # use the committed candles only

Idempotent: running it again before the next candle closes changes nothing.
Exits 0 whether or not anything changed. When run in GitHub Actions it writes
``changed`` and ``candle`` to $GITHUB_OUTPUT for the commit step.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from cfl import live
from cfl.data import load_1h, update_cache


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-fetch", action="store_true", help="do not call the Coinbase API")
    ap.add_argument("--now", default=None, help="override the clock (ISO, UTC); for testing")
    args = ap.parse_args()

    now = live.parse(args.now) if args.now else pd.Timestamp.now(tz="UTC")
    h1 = load_1h() if args.no_fetch else update_cache(now=now)
    res = live.run(h1, now)

    last = res.latest or {}
    if not res.changed:
        print("nothing new")
    else:
        print(f"predictions.jsonl {'updated' if res.log_changed else 'unchanged'}, "
              f"status.json {'updated' if res.status_changed else 'unchanged'}")
    if last:
        print(f"latest: candle {last['candle_open']} -> target {last['target_open']}  "
              f"outcome={last['outcome']}  p={last['p']}  action={last['action']}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"changed={'true' if res.changed else 'false'}\n")
            fh.write(f"candle={last.get('target_open', '')}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
