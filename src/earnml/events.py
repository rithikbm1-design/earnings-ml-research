"""Earnings-announcement event dates and market-adjusted abnormal returns."""
from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_CLOSE_HOUR = 16
MAX_DAYS_AFTER_PERIOD_END = 75


def match_announcements(period_ends: pd.Series, releases: pd.Series) -> pd.Series:
    """For each fiscal period end, the first Item 2.02 release within 75 days after it.

    Returns a Series of release timestamps aligned with ``period_ends`` (NaT if none).
    """
    rel = np.sort(pd.to_datetime(releases).dropna().to_numpy())
    out = []
    for pe in pd.to_datetime(period_ends):
        i = np.searchsorted(rel, np.datetime64(pe + pd.Timedelta(days=1)))
        if i < len(rel) and (pd.Timestamp(rel[i]) - pe).days <= MAX_DAYS_AFTER_PERIOD_END:
            out.append(pd.Timestamp(rel[i]))
        else:
            out.append(pd.NaT)
    return pd.Series(out, index=period_ends.index, dtype="datetime64[ns]")


def event_trading_day(accepted_et: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """First trading session in which the market can react to the release.

    Releases accepted at or after the 16:00 close (or on non-trading days) are
    assigned to the next trading day.
    """
    td = trading_days.sort_values().to_numpy()
    out = []
    for ts in pd.to_datetime(accepted_et):
        if pd.isna(ts):
            out.append(pd.NaT)
            continue
        day = ts.normalize()
        if ts.hour >= MARKET_CLOSE_HOUR:
            day += pd.Timedelta(days=1)
        i = np.searchsorted(td, np.datetime64(day))
        out.append(pd.Timestamp(td[i]) if i < len(td) else pd.NaT)
    return pd.Series(out, index=accepted_et.index, dtype="datetime64[ns]")


def cumulative_abnormal_return(
    ret: pd.Series, mkt: pd.Series, event_day: pd.Timestamp, start: int, end: int
) -> float:
    """Sum of (r_i - r_m) over trading days [event+start, event+end] (inclusive).

    Returns NaN if the window is incomplete or any return is missing.
    """
    idx = ret.index
    pos = idx.searchsorted(event_day)
    if pos >= len(idx) or idx[pos] != event_day:
        return np.nan
    lo, hi = pos + start, pos + end
    if lo < 1 or hi >= len(idx):
        return np.nan
    ar = (ret.iloc[lo : hi + 1] - mkt.iloc[lo : hi + 1]).to_numpy()
    if np.isnan(ar).any():
        return np.nan
    return float(ar.sum())
