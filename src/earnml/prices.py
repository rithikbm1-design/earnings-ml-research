"""Daily adjusted prices from Yahoo Finance (via yfinance), cached to parquet."""
from __future__ import annotations

import pandas as pd

from . import config


def download_prices(tickers: list[str], start: str = config.PRICE_START, refresh: bool = False) -> pd.DataFrame:
    """Wide frame of split/dividend-adjusted closes (index=date, columns=ticker)."""
    cache = config.RAW_DIR / "prices" / "adj_close.parquet"
    if cache.exists() and not refresh:
        px = pd.read_parquet(cache)
        if set(tickers) <= set(px.columns) and px[tickers].notna().any().all():
            return px[tickers]
    px = _yahoo_close(tickers, start)
    empty = [t for t in tickers if t not in px or px[t].isna().all()]
    if empty:  # Yahoo occasionally drops tickers from a batch request; retry them
        retry = _yahoo_close(empty, start)
        px = px.drop(columns=[t for t in empty if t in px]).join(retry, how="outer")
    cache.parent.mkdir(parents=True, exist_ok=True)
    px.to_parquet(cache)
    return px


def _yahoo_close(tickers: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf

    yahoo = {t: t.replace(".", "-") for t in tickers}
    raw = yf.download(list(yahoo.values()), start=start, auto_adjust=True, progress=False,
                      threads=True, multi_level_index=True)
    px = raw["Close"].rename(columns={v: k for k, v in yahoo.items()})
    px.index = pd.to_datetime(px.index).tz_localize(None)
    return px.sort_index().dropna(how="all")


def daily_returns(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change(fill_method=None)
