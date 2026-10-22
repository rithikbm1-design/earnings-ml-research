"""Forecast-accuracy metrics and return-predictability tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

N_QUANTILES = 5


def oos_r2(y: np.ndarray, pred: np.ndarray, benchmark: np.ndarray | None = None) -> float:
    """1 - SSE(model)/SSE(benchmark). Default benchmark is the zero forecast
    (seasonal random walk: earnings equal those of the same quarter last year)."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    bench = np.zeros_like(y) if benchmark is None else np.asarray(benchmark, float)
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - bench) ** 2))


def nw_tstat(x: pd.Series | np.ndarray, lags: int = 2) -> tuple[float, float]:
    """Mean and Newey-West t-statistic of a time series."""
    x = pd.Series(x).dropna().to_numpy(float)
    if len(x) < 3:
        return float("nan"), float("nan")
    fit = sm.OLS(x, np.ones_like(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.params[0]), float(fit.tvalues[0])


def _period(df: pd.DataFrame) -> pd.Series:
    return df["event_day"].dt.to_period("Q")


def forecast_accuracy(pred: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    y = pred["y_w"].to_numpy()
    for m in models:
        p = pred[f"pred_{m}"].to_numpy()
        ic = pred.groupby(_period(pred)).apply(
            lambda g: g["y_w"].corr(g[f"pred_{m}"], method="spearman"), include_groups=False
        )
        ic_mean, ic_t = nw_tstat(ic)
        rows.append({
            "model": m,
            "oos_r2_vs_seasonal_rw": oos_r2(y, p),
            "oos_r2_vs_persistence": oos_r2(y, p, pred["pred_persistence"]),
            "mae": float(np.mean(np.abs(y - p))),
            "rank_ic_mean": ic_mean,
            "rank_ic_t": ic_t,
            "n_obs": len(pred),
        })
    return pd.DataFrame(rows)


def quantile_portfolios(pred: pd.DataFrame, signal: str, outcome: str,
                        q: int = N_QUANTILES) -> pd.DataFrame:
    """Equal-weighted mean outcome per (calendar quarter, signal quantile)."""
    df = pred.dropna(subset=[signal, outcome]).copy()
    df["period"] = _period(df)
    df = df[df.groupby("period")[signal].transform("count") >= 2 * q]
    df["bin"] = df.groupby("period")[signal].transform(
        lambda s: pd.qcut(s.rank(method="first"), q, labels=False) + 1
    )
    return df.groupby(["period", "bin"])[outcome].mean().unstack("bin")


def long_short_test(pred: pd.DataFrame, signal: str, outcome: str, q: int = N_QUANTILES) -> dict:
    port = quantile_portfolios(pred, signal, outcome, q)
    ls = port[q] - port[1]
    mean, t = nw_tstat(ls)
    return {
        "signal": signal, "outcome": outcome,
        **{f"Q{i}": float(port[i].mean()) for i in range(1, q + 1)},
        "long_short": mean, "long_short_t": t, "hit_rate": float((ls > 0).mean()),
        "n_quarters": int(ls.notna().sum()),
    }


def _rank_scale(s: pd.Series) -> pd.Series:
    """Cross-sectional rank scaled to [-0.5, 0.5] (robust to outliers)."""
    r = s.rank()
    return (r - 1) / (r.count() - 1) - 0.5 if r.count() > 1 else r * np.nan


def fama_macbeth(pred: pd.DataFrame, outcome: str, regressors: list[str]) -> pd.DataFrame:
    """Quarterly cross-sectional OLS of the outcome on rank-scaled regressors.

    A coefficient is the return spread between the highest and lowest-ranked firm.
    """
    df = pred.dropna(subset=[outcome, *regressors]).copy()
    df["period"] = _period(df)
    coefs = []
    for _, g in df.groupby("period"):
        if len(g) < 20:
            continue
        X = sm.add_constant(pd.DataFrame({r: _rank_scale(g[r]) for r in regressors}))
        coefs.append(sm.OLS(g[outcome].to_numpy(float), X).fit().params)
    coefs = pd.DataFrame(coefs)
    rows = []
    for c in coefs.columns:
        mean, t = nw_tstat(coefs[c])
        rows.append({"term": c, "coef": mean, "nw_t": t, "n_quarters": len(coefs)})
    return pd.DataFrame(rows)
