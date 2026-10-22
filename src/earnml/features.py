"""Build the firm-quarter event panel: point-in-time features, target, and returns.

For each firm and fiscal quarter q, the forecasting problem is posed at the
moment just before quarter q's earnings release:

* target  ``y`` = (NI_q - NI_{q-4}) / Assets_{q-1}  -- the seasonal change in
  quarterly earnings scaled by assets (a profitability-change measure; the
  benchmark seasonal random walk forecasts zero).
* features use only fiscal quarters <= q-1 whose filings were accepted before
  the release, plus prices up to two trading days before the event.
* outcomes are market-adjusted cumulative abnormal returns around the release.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .events import cumulative_abnormal_return, event_trading_day, match_announcements

FEATURES = [
    "ni_l1", "ni_l2", "ni_l3", "ni_l4", "ni_l5",
    "de_l1", "de_l2", "de_l4",
    "rev_g_l1", "opm_l1", "lev_l1", "log_assets", "loss_l1",
    "ret_3m", "ret_12m", "vol_3m", "prior_car",
]
FUNDAMENTAL_FEATURES = FEATURES[:13]
PRICE_FEATURES = FEATURES[13:]


def _lag(df: pd.DataFrame, col: str, k: int) -> pd.Series:
    """Value of ``col`` k fiscal quarters earlier, matched on the fiscal-quarter key."""
    s = df.set_index("fq")[col]
    s = s[~s.index.duplicated(keep="last")]
    return pd.Series(s.reindex(df["fq"] - 3 * k).to_numpy(), index=df.index)


MIN_ASSETS = 1e8  # every firm in the universe is a large cap; less means a shell registrant
MAX_ASSET_JUMP = 5.0


def clean_assets(assets: pd.Series, assets_prev: pd.Series) -> pd.Series:
    """Drop implausible scaling denominators (shell entities around mergers/reorgs).

    Uses only the current and *previous* quarter, so the filter adds no look-ahead.
    """
    ratio = assets / assets_prev
    jump = (ratio > MAX_ASSET_JUMP) | (ratio < 1 / MAX_ASSET_JUMP)
    return assets.where((assets >= MIN_ASSETS) & ~jump.fillna(False))


def fundamental_panel(fund: pd.DataFrame, releases: pd.Series) -> pd.DataFrame:
    """Per-quarter target and lagged accounting features for one firm."""
    df = fund.sort_values("fq").reset_index(drop=True).copy()
    df["accepted_et"] = match_announcements(df["period_end"], releases)

    A = clean_assets(_lag(df, "assets", 1), _lag(df, "assets", 2))
    ni = {k: _lag(df, "ni", k) for k in range(1, 7)}
    df["assets_l1"] = A
    df["y"] = (df["ni"] - ni[4]) / A
    for k in range(1, 6):
        df[f"ni_l{k}"] = ni[k] / A
    df["de_l1"] = (ni[1] - ni[5]) / A
    df["de_l2"] = (ni[2] - ni[6]) / A
    df["de_l4"] = (ni[4] - _lag(df, "ni", 8)) / A
    rev1, rev5 = _lag(df, "rev", 1), _lag(df, "rev", 5)
    df["rev_g_l1"] = (rev1 / rev5.where(rev5 > 0)) - 1
    df["opm_l1"] = _lag(df, "oi", 1) / rev1.where(rev1 > 0)
    df["lev_l1"] = 1 - _lag(df, "equity", 1) / A
    df["log_assets"] = np.log(A)
    df["loss_l1"] = (ni[1] < 0).astype(float).where(ni[1].notna())

    # Point-in-time guard: last quarter's numbers must have been filed before
    # this quarter's release, otherwise the forecaster could not have used them.
    filed_ni_l1 = _lag(df, "filed_ni", 1)
    filed_a_l1 = _lag(df, "filed_assets", 1)
    release_day = df["accepted_et"].dt.normalize()
    ok = (filed_ni_l1 < release_day) & (filed_a_l1 < release_day)
    for col in FUNDAMENTAL_FEATURES:
        df.loc[~ok, col] = np.nan
    df["label_date"] = df["filed_ni"]
    return df


def add_market_data(
    panel: pd.DataFrame, ret: pd.Series, mkt: pd.Series, price: pd.Series
) -> pd.DataFrame:
    """Attach event day, pre-event price features and abnormal-return outcomes."""
    df = panel.copy()
    days = ret.dropna().index
    df["event_day"] = event_trading_day(df["accepted_et"], days)
    idx = ret.index
    cols = {c: [] for c in ("ret_3m", "ret_12m", "vol_3m", "car_m1p1", "car_p2p21")}
    for ev in df["event_day"]:
        p = idx.searchsorted(ev) if pd.notna(ev) else -1
        if p < 0 or p >= len(idx) or idx[p] != ev or p < 255:
            for v in cols.values():
                v.append(np.nan)
            continue
        cols["ret_3m"].append(price.iloc[p - 2] / price.iloc[p - 65] - 1)
        cols["ret_12m"].append(price.iloc[p - 2] / price.iloc[p - 254] - 1)
        cols["vol_3m"].append(float(ret.iloc[p - 64 : p - 1].std()))
        cols["car_m1p1"].append(cumulative_abnormal_return(ret, mkt, ev, -1, 1))
        cols["car_p2p21"].append(cumulative_abnormal_return(ret, mkt, ev, 2, 21))
    for k, v in cols.items():
        df[k] = v
    # Previous quarter's announcement reaction is public before this release.
    prev = df.set_index("fq")["car_m1p1"]
    prev = prev[~prev.index.duplicated(keep="last")]
    df["prior_car"] = prev.reindex(df["fq"] - 3).to_numpy()
    return df
