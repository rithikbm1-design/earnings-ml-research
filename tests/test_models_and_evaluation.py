import numpy as np
import pandas as pd
import pytest

from earnml import evaluation, models
from earnml.features import FEATURES


def _synthetic_panel(n_firms=60, years=range(2010, 2018), seed=0):
    """Panel where y depends on two features and CAR depends on y (not on features)."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        for q, month in enumerate([2, 5, 8, 11]):
            day = pd.Timestamp(year=year, month=month, day=15)
            for f in range(n_firms):
                x = rng.normal(size=len(FEATURES))
                y = 0.01 * x[FEATURES.index("ni_l4")] - 0.005 * x[FEATURES.index("de_l1")] + rng.normal(0, 0.002)
                rows.append({"ticker": f"F{f}", "fq": year * 12 + month, "period_end": day - pd.Timedelta(days=45),
                             "event_day": day, "label_date": day + pd.Timedelta(days=10), "y": y,
                             "car_m1p1": 2 * y + rng.normal(0, 0.01), "car_p2p21": rng.normal(0, 0.01),
                             **dict(zip(FEATURES, x))})
    return pd.DataFrame(rows)


def test_walk_forward_is_out_of_sample_and_learns_signal():
    panel = _synthetic_panel()
    preds = models.walk_forward(panel, first_test_year=2013)
    assert preds["test_year"].min() == 2013
    # Training set for year Y contains only labels filed before 1 Jan Y.
    for year, g in preds.groupby("test_year"):
        expected = (panel["label_date"] < pd.Timestamp(year=year, month=1, day=1)).sum()
        assert (g["n_train"] == expected).all()
    acc = evaluation.forecast_accuracy(preds, ["ridge", "gbrt"]).set_index("model")
    assert acc.loc["ridge", "oos_r2_vs_seasonal_rw"] > 0.8
    assert acc.loc["gbrt", "rank_ic_mean"] > 0.7


def test_oos_r2():
    y = np.array([1.0, -1.0, 2.0])
    assert evaluation.oos_r2(y, y) == 1.0
    assert evaluation.oos_r2(y, np.zeros(3)) == 0.0
    assert evaluation.oos_r2(y, y, benchmark=y + 1) == 1.0


def test_long_short_detects_monotone_signal():
    rng = np.random.default_rng(1)
    days = pd.to_datetime([f"20{y}-0{m}-15" for y in range(10, 20) for m in (2, 5, 8)])
    df = pd.DataFrame([{"event_day": d, "sig": s, "out": s * 0.01 + rng.normal(0, 0.001)}
                       for d in days for s in rng.normal(size=50)])
    res = evaluation.long_short_test(df, "sig", "out")
    assert res["long_short"] > 0 and res["long_short_t"] > 5
    assert res["Q1"] < res["Q3"] < res["Q5"]


def test_fama_macbeth_recovers_positive_slope():
    rng = np.random.default_rng(2)
    days = pd.to_datetime([f"20{y}-0{m}-15" for y in range(10, 20) for m in (2, 5, 8)])
    df = pd.DataFrame([{"event_day": d, "a": a, "b": rng.normal(), "out": 0.02 * a + rng.normal(0, 0.005)}
                       for d in days for a in rng.normal(size=40)])
    fm = evaluation.fama_macbeth(df, "out", ["a", "b"]).set_index("term")
    assert fm.loc["a", "coef"] > 0 and fm.loc["a", "nw_t"] > 5
    assert abs(fm.loc["b", "nw_t"]) < 3


def test_winsorizer_uses_training_bounds():
    w = models.Winsorizer().fit(np.arange(101, dtype=float).reshape(-1, 1))
    out = w.transform(np.array([[-50.0], [50.0], [500.0]]))
    assert out[0, 0] == pytest.approx(1.0) and out[1, 0] == 50 and out[2, 0] == pytest.approx(99.0)
