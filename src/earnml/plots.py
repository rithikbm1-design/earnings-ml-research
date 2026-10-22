"""Static result figures (PNG) for the README."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, evaluation

# Reference categorical palette (fixed order) and text/grid tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MUTED = "#8a8984"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
SEQ = ["#c6dcf5", "#94bdec", "#5e9be0", "#2a78d6", "#1b4f8f"]  # one hue, light -> dark

LABELS = {"persistence": "Persistence", "ridge": "Ridge", "random_forest": "Random forest", "gbrt": "Gradient boosting"}


def _style(ax, title: str, ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=12, color=TEXT, pad=10)
    ax.set_ylabel(ylabel, color=TEXT2, fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def quintile_car(preds: pd.DataFrame, signal: str, outcome: str, title: str, path) -> None:
    port = evaluation.quantile_portfolios(preds, signal, outcome)
    mean = port.mean() * 100
    se = port.std() / np.sqrt(port.count()) * 100
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.bar(mean.index, mean.values, width=0.6, color=SEQ, yerr=1.96 * se.values,
           error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 3})
    ax.axhline(0, color=TEXT2, linewidth=0.8)
    for x, v, e in zip(mean.index, mean.values, 1.96 * se.values):
        tip = v + e if v >= 0 else v - e
        ax.annotate(f"{v:+.2f}%", (x, tip), xytext=(0, 4 if v >= 0 else -12), textcoords="offset points",
                    ha="center", fontsize=8.5, color=TEXT)
    ax.margins(y=0.12)
    ax.set_xticks(mean.index, [f"Q{i}" + (" (low)" if i == 1 else " (high)" if i == 5 else "") for i in mean.index])
    _style(ax, title, "Mean market-adjusted CAR (%)")
    ax.set_xlabel("Quintile of signal (sorted each calendar quarter); bars ±95% CI", color=TEXT2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cumulative_long_short(preds: pd.DataFrame, models: list[str], path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ends = {}
    for i, m in enumerate(models):
        port = evaluation.quantile_portfolios(preds, f"pred_{m}", "car_m1p1")
        ls = (port[5] - port[1]).cumsum() * 100
        x = ls.index.to_timestamp()
        ax.plot(x, ls.values, color=SERIES[i], linewidth=2, label=LABELS[m])
        ends[m] = (x[-1], ls.values[-1])
    # Direct end labels, nudged apart so close series stay legible.
    span = np.ptp(ax.get_ylim())
    placed: list[float] = []
    for m, (x, y) in sorted(ends.items(), key=lambda kv: kv[1][1]):
        y_lab = max(y, placed[-1] + 0.06 * span) if placed else y
        placed.append(y_lab)
        ax.annotate(LABELS[m], (x, y), xytext=(x + pd.Timedelta(days=60), y_lab), textcoords="data",
                    fontsize=8.5, color=TEXT2, va="center")
    ax.axhline(0, color=TEXT2, linewidth=0.8)
    _style(ax, "Cumulative Q5–Q1 spread in CAR[-1,+1] (out of sample)", "Cumulative spread (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_xlim(right=max(v[0] for v in ends.values()) + pd.Timedelta(days=700))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def accuracy_bars(acc: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    names = [LABELS[m] for m in acc["model"]]
    for ax, col, title in [
        (axes[0], "oos_r2_vs_seasonal_rw", "OOS R² vs seasonal random walk"),
        (axes[1], "rank_ic_mean", "Rank IC with realized change"),
    ]:
        vals = acc[col].to_numpy()
        ax.barh(names, vals, color=SERIES[0], height=0.55)
        ax.axvline(0, color=TEXT2, linewidth=0.8)
        for y, v in enumerate(vals):
            ax.annotate(f"{v:.3f}", (v, y), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                        ha="left" if v >= 0 else "right", va="center", fontsize=8.5, color=TEXT)
        _style(ax, title)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.invert_yaxis()
        lo, hi = min(0, vals.min()), max(0, vals.max())
        pad = (hi - lo) * 0.25 or 0.05
        ax.set_xlim(lo - (pad if lo < 0 else 0), hi + pad)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def importance_bars(imp: pd.Series, path) -> None:
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.barh(imp.index, imp.values, color=SERIES[0], height=0.6)
    ax.axvline(0, color=TEXT2, linewidth=0.8)
    _style(ax, "Gradient boosting: permutation importance (Δ R², OOS)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all(preds: pd.DataFrame, acc: pd.DataFrame, imp: pd.Series, models: list[str]) -> None:
    F = config.FIGURES_DIR
    F.mkdir(parents=True, exist_ok=True)
    accuracy_bars(acc, F / "forecast_accuracy.png")
    quintile_car(preds, "pred_gbrt", "car_m1p1",
                 "CAR[-1,+1] by GBRT forecast quintile", F / "quintiles_gbrt_car.png")
    quintile_car(preds, "pred_gbrt", "car_p2p21",
                 "Drift CAR[+2,+21] by GBRT forecast quintile", F / "quintiles_gbrt_drift.png")
    quintile_car(preds, "y", "car_m1p1",
                 "Sanity check: CAR[-1,+1] by realized change\n(look-ahead, not tradable)",
                 F / "quintiles_realized_car.png")
    cumulative_long_short(preds, models, F / "cumulative_long_short.png")
    importance_bars(imp, F / "feature_importance.png")
