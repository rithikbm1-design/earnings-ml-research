"""Walk-forward (expanding window) profitability forecasts.

Every calendar year Y the models are refit on all firm-quarters whose label
(the realized earnings) had been *filed* before 1 January of Y, then used to
forecast every announcement in Y. Winsorization bounds, imputation medians and
scaling are estimated on the training window only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import FEATURES

WINSOR_Q = (0.01, 0.99)


class Persistence(BaseEstimator, RegressorMixin):
    """Benchmark: next seasonal earnings change equals the latest one (de_l1)."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.nan_to_num(np.asarray(X[:, FEATURES.index("de_l1")], dtype=float))


def model_zoo() -> dict[str, BaseEstimator]:
    rs = config.RANDOM_STATE
    return {
        "persistence": Persistence(),
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            RidgeCV(alphas=np.logspace(-3, 3, 13)),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=300, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=rs
            ),
        ),
        "gbrt": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=40,
            l2_regularization=1.0, random_state=rs,
        ),
    }


class Winsorizer:
    """Clip columns at quantiles estimated on training data."""

    def fit(self, X: np.ndarray) -> "Winsorizer":
        self.lo_, self.hi_ = np.nanquantile(X, WINSOR_Q[0], axis=0), np.nanquantile(X, WINSOR_Q[1], axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.lo_, self.hi_)


def walk_forward(panel: pd.DataFrame, first_test_year: int = config.FIRST_TEST_YEAR,
                 last_test_year: int | None = None) -> pd.DataFrame:
    """Out-of-sample forecasts for every event in [first_test_year, last_test_year]."""
    data = panel.dropna(subset=["event_day", "y"]).copy()
    data = data[data[FEATURES].notna().sum(axis=1) >= len(FEATURES) // 2]
    last_test_year = last_test_year or int(data["event_day"].dt.year.max())
    zoo = model_zoo()
    out = []
    for year in range(first_test_year, last_test_year + 1):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        train = data[(data["label_date"] < cutoff) & (data["event_day"] < cutoff)]
        test = data[data["event_day"].dt.year == year]
        if len(train) < 200 or test.empty:
            continue
        wx = Winsorizer().fit(train[FEATURES].to_numpy(float))
        wy = Winsorizer().fit(train[["y"]].to_numpy(float))
        Xtr = wx.transform(train[FEATURES].to_numpy(float))
        ytr = wy.transform(train[["y"]].to_numpy(float)).ravel()
        Xte = wx.transform(test[FEATURES].to_numpy(float))
        res = test[["ticker", "fq", "period_end", "event_day", "y", "car_m1p1", "car_p2p21",
                    "de_l1", "ret_3m", "log_assets"]].copy()
        res["y_w"] = wy.transform(test[["y"]].to_numpy(float)).ravel()
        res["test_year"] = year
        res["n_train"] = len(train)
        for name, est in zoo.items():
            m = clone(est).fit(Xtr, ytr)
            res[f"pred_{name}"] = m.predict(Xte)
        out.append(res)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def feature_importance(panel: pd.DataFrame, cutoff_year: int) -> pd.Series:
    """Permutation importance of the GBRT fit on all data labelled before ``cutoff_year``.

    Measured on the following (out-of-sample) years to avoid in-sample optimism.
    """
    from sklearn.inspection import permutation_importance

    data = panel.dropna(subset=["event_day", "y"])
    cutoff = pd.Timestamp(year=cutoff_year, month=1, day=1)
    train = data[(data["label_date"] < cutoff) & (data["event_day"] < cutoff)]
    test = data[data["event_day"] >= cutoff]
    wx = Winsorizer().fit(train[FEATURES].to_numpy(float))
    wy = Winsorizer().fit(train[["y"]].to_numpy(float))
    m = clone(model_zoo()["gbrt"]).fit(
        wx.transform(train[FEATURES].to_numpy(float)), wy.transform(train[["y"]].to_numpy(float)).ravel()
    )
    pi = permutation_importance(
        m, wx.transform(test[FEATURES].to_numpy(float)), wy.transform(test[["y"]].to_numpy(float)).ravel(),
        n_repeats=10, random_state=config.RANDOM_STATE, scoring="r2",
    )
    return pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
