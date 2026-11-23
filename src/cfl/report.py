"""reports/backtest.md, reports/backtest.json and PNG charts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import rolling_accuracy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"   # model
SERIES_2 = "#eb6834"   # buy & hold
NEUTRAL = "#8a8983"    # reference lines


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_equity(model_eq: pd.Series, bh_eq: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=120)
    _style(ax, "Paper portfolio vs buy & hold (log scale, $10,000 start, after costs)")
    ax.plot(bh_eq.index, bh_eq.to_numpy(), color=SERIES_2, linewidth=1.6, label="Buy & hold")
    ax.plot(model_eq.index, model_eq.to_numpy(), color=SERIES_1, linewidth=1.6, label="Model (long/flat)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    for eq, color in ((bh_eq, SERIES_2), (model_eq, SERIES_1)):
        ax.annotate(f"${eq.iloc[-1]:,.0f}", (eq.index[-1], eq.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_rolling_accuracy(preds: pd.DataFrame, path: Path, window: int = 540) -> None:
    acc = rolling_accuracy(preds, window) * 100
    fig, ax = plt.subplots(figsize=(10, 4.0), dpi=120)
    _style(ax, f"Rolling accuracy over the last {window} predictions (~90 days)")
    ax.axhline(50, color=NEUTRAL, linewidth=1, linestyle="--")
    ax.annotate("coin flip", (acc.index[0], 50), xytext=(0, 4), textcoords="offset points",
                fontsize=9, color=INK_2)
    ax.plot(acc.index, acc.to_numpy(), color=SERIES_1, linewidth=1.6)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _f(v, fmt="{:.2f}") -> str:
    return "–" if v is None else fmt.format(v)


def headline_table(results: dict) -> str:
    lines = ["| Strategy | Accuracy | AUC | Log loss | Brier | Return % | Max DD % | Sharpe | Exposure % | Trades |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in results.items():
        lines.append(
            f"| {name} | {_f(r.get('accuracy') and r['accuracy'] * 100, '{:.2f}%')} | {_f(r.get('auc'), '{:.3f}')} "
            f"| {_f(r.get('log_loss'), '{:.4f}')} | {_f(r.get('brier'), '{:.4f}')} "
            f"| {r['return_pct']:,.1f} | {r['max_drawdown_pct']:.1f} | {r['sharpe']:.2f} "
            f"| {r['exposure_pct']:.1f} | {r['trades']:,} |")
    return "\n".join(lines)


def stability_table(stab: pd.DataFrame, n_rounds: int) -> str:
    lines = [f"| Feature | Selected (of {n_rounds}) | Share | Ranked first | In MRMR top-K | In RFE top-K | In SFS top-K |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for f, row in stab.iterrows():
        lines.append(f"| `{f}` | {row['selected']} | {row['selected_pct']:.0f}% | {row['ranked_first']} "
                     f"| {row['in_MRMR_topK']} | {row['in_RFE_topK']} | {row['in_SFS_topK']} |")
    return "\n".join(lines)


def yearly_table(preds: pd.DataFrame) -> str:
    from .metrics import trading
    lines = ["| Year | Predictions | Accuracy | Model return % | Buy & hold return % | Exposure % |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for year, g in preds.groupby(preds.index.year):
        acc = ((g["p"] > 0.5).astype(int) == g["target"]).mean() * 100
        m = trading(g["position"], g["next_ret"])
        b = trading(pd.Series(1, index=g.index), g["next_ret"])
        lines.append(f"| {year} | {len(g):,} | {acc:.2f}% | {m['return_pct']:,.1f} | {b['return_pct']:,.1f} "
                     f"| {m['exposure_pct']:.1f} |")
    return "\n".join(lines)


def windows_table(window_log: list[dict]) -> str:
    lines = ["| Window | Training rows | Training through | Method@K | Selected features (in order) | m |",
             "| ---: | ---: | --- | --- | --- | ---: |"]
    for w in window_log:
        if not w["tuned"]:
            continue
        lines.append(f"| {w['window']} | {w['train_rows']:,} | {w['train_to'][:10]} | {w['method']}@{w['k']} "
                     f"| {', '.join(f'`{c}`' for c in w['features'])} | {w['m']:.3f} |")
    return "\n".join(lines)


def write_json(summary: dict, path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2) + "\n")
