"""Long/flat paper portfolio with fees and slippage.

Timing: the position decided at the close of bar t is held during bar t+1 and earns
that bar's simple return (``next_ret`` at row t). Changing position costs
``COST_PER_SIDE`` of equity per side (0.10% fee + 0.05% slippage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEE = 0.0010
SLIPPAGE = 0.0005
COST_PER_SIDE = FEE + SLIPPAGE
START_BALANCE = 10_000.0


def simulate(positions, next_ret, cost: float = COST_PER_SIDE,
             start: float = START_BALANCE, initial_position: float = 0.0) -> pd.DataFrame:
    """Equity curve for a sequence of 0/1 positions.

    Row i: pay ``cost * |pos_i - pos_{i-1}|`` when the position changes at the close
    of bar i, then earn ``pos_i * next_ret_i`` over the next bar. The equity at row i
    is the value *after* that next bar. The final position is not closed.
    """
    pos = np.asarray(positions, dtype=float)
    r = np.asarray(next_ret, dtype=float)
    prev = np.concatenate([[initial_position], pos[:-1]])
    turnover = np.abs(pos - prev)
    growth = (1 - cost * turnover) * (1 + pos * r)
    equity = start * np.cumprod(growth)
    index = positions.index if isinstance(positions, pd.Series) else None
    return pd.DataFrame({"position": pos, "turnover": turnover,
                         "growth": growth, "equity": equity}, index=index)


def strategy_log_return(positions, next_ret, cost: float = COST_PER_SIDE) -> float:
    pos = np.asarray(positions, dtype=float)
    r = np.asarray(next_ret, dtype=float)
    prev = np.concatenate([[0.0], pos[:-1]])
    return float(np.sum(np.log1p(-cost * np.abs(pos - prev)) + np.log1p(pos * r)))


def count_trades(positions, initial_position: float = 0.0) -> int:
    """Number of entries (flat -> long). Each is one trade; exits are part of it."""
    pos = np.asarray(positions, dtype=float)
    prev = np.concatenate([[initial_position], pos[:-1]])
    return int(np.sum((pos > prev)))
