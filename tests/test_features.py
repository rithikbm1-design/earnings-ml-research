import numpy as np
import pandas as pd
import pytest

from earnml import features, sec


def _fund(n=12, filing_lag_days=30):
    ends = pd.date_range("2020-03-31", periods=n, freq="QE")
    ni = np.arange(1, n + 1, dtype=float) * 10
    df = pd.DataFrame({
        "period_end": ends, "ni": ni, "filed_ni": ends + pd.Timedelta(days=filing_lag_days),
        "rev": ni * 5, "oi": ni * 2, "assets": np.full(n, 1e9), "equity": np.full(n, 4e8),
    })
    for c in ("rev", "oi", "assets", "equity"):
        df[f"filed_{c}"] = df["filed_ni"]
    df["fq"] = sec.fiscal_quarter_key(df["period_end"])
    return df


def _releases(fund, lag_days=25):
    return pd.Series(fund["period_end"] + pd.Timedelta(days=lag_days) + pd.Timedelta(hours=16, minutes=5))


def test_target_is_seasonal_change_scaled_by_lagged_assets():
    f = _fund()
    p = features.fundamental_panel(f, _releases(f))
    row = p.iloc[8]
    assert row["y"] == pytest.approx((f["ni"].iloc[8] - f["ni"].iloc[4]) / 1e9)
    assert row["ni_l1"] == pytest.approx(f["ni"].iloc[7] / 1e9)
    assert row["de_l1"] == pytest.approx((f["ni"].iloc[7] - f["ni"].iloc[3]) / 1e9)
    assert row["lev_l1"] == pytest.approx(0.6)


def test_features_blanked_when_prior_quarter_not_yet_filed():
    f = _fund()
    f["filed_ni"] = f["period_end"] + pd.Timedelta(days=120)  # 10-Q filed after next release
    p = features.fundamental_panel(f, _releases(f, lag_days=25))
    assert p[features.FUNDAMENTAL_FEATURES].isna().all().all()


def test_clean_assets_drops_shell_entities_and_jumps():
    a = pd.Series([46_000.0, 1e11, 1e11, 9e11])
    prev = pd.Series([1e11, np.nan, 1e11, 1e11])
    out = features.clean_assets(a, prev)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == 1e11
    assert out.iloc[2] == 1e11
    assert np.isnan(out.iloc[3])


def test_price_features_use_only_pre_event_prices():
    days = pd.bdate_range("2019-01-01", periods=400)
    price = pd.Series(np.linspace(100, 200, len(days)), index=days)
    ret = price.pct_change()
    mkt = ret * 0
    ev_pos = 300
    panel = pd.DataFrame({"fq": [1], "accepted_et": [days[ev_pos] + pd.Timedelta(hours=9)]})
    base = features.add_market_data(panel, ret, mkt, price)
    shocked_price = price.copy()
    shocked_price.iloc[ev_pos - 1 :] *= 3  # a shock from day -1 on must not move the features
    shocked = features.add_market_data(panel, shocked_price.pct_change(), mkt, shocked_price)
    for col in ("ret_3m", "ret_12m", "vol_3m"):
        assert shocked[col].iloc[0] == pytest.approx(base[col].iloc[0])
    assert shocked["car_m1p1"].iloc[0] != pytest.approx(base["car_m1p1"].iloc[0])
