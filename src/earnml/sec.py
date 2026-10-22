"""SEC EDGAR access: ticker->CIK map, 8-K Item 2.02 earnings releases, XBRL facts.

All values are kept *point-in-time*: for every fiscal period we keep the value as
it was first filed, together with that filing date, so later restatements can
never leak into a forecast.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config

_LAST_REQUEST = [0.0]

# Concept priority lists. The first tag that reports a given period wins.
NET_INCOME_TAGS = ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"]
REVENUE_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueGoodsNet",
]
OPERATING_INCOME_TAGS = ["OperatingIncomeLoss"]
ASSETS_TAGS = ["Assets"]
EQUITY_TAGS = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]

QUARTER_DAYS = (70, 110)
NINE_MONTH_DAYS = (250, 290)
ANNUAL_DAYS = (340, 390)


def _get_json(url: str, cache: Path, max_age_days: float | None = None) -> dict:
    """GET a JSON document from SEC with on-disk caching and polite rate limiting."""
    if cache.exists():
        age = (time.time() - cache.stat().st_mtime) / 86400
        if max_age_days is None or age < max_age_days:
            return json.loads(cache.read_text())
    wait = 0.15 - (time.time() - _LAST_REQUEST[0])  # SEC fair-access limit is 10 req/s
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, headers={"User-Agent": config.SEC_USER_AGENT}, timeout=60)
    _LAST_REQUEST[0] = time.time()
    resp.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(resp.text)
    return resp.json()


def ticker_to_cik() -> dict[str, int]:
    data = _get_json(
        "https://www.sec.gov/files/company_tickers.json",
        config.RAW_DIR / "sec" / "company_tickers.json",
        max_age_days=30,
    )
    return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}


def fetch_submissions(cik: int) -> dict:
    """Submission history, merging the paginated older-filings files."""
    base = config.RAW_DIR / "sec" / "submissions"
    sub = _get_json(
        f"https://data.sec.gov/submissions/CIK{cik:010d}.json", base / f"CIK{cik:010d}.json"
    )
    recent = {k: list(v) for k, v in sub["filings"]["recent"].items()}
    for f in sub["filings"].get("files", []):
        older = _get_json(f"https://data.sec.gov/submissions/{f['name']}", base / f["name"])
        for k in recent:
            recent[k].extend(older.get(k, []))
    sub["filings"]["all"] = recent
    return sub


def fetch_companyfacts(cik: int) -> dict:
    return _get_json(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
        config.RAW_DIR / "sec" / "companyfacts" / f"CIK{cik:010d}.json",
    )


def earnings_release_times(submissions: dict) -> pd.DataFrame:
    """8-K filings that include Item 2.02 (Results of Operations and Financial Condition).

    Returns one row per filing with the EDGAR acceptance timestamp in US/Eastern.
    """
    f = submissions["filings"].get("all", submissions["filings"]["recent"])
    df = pd.DataFrame(
        {"form": f["form"], "items": f["items"], "accepted": f["acceptanceDateTime"]}
    )
    df = df[(df["form"] == "8-K") & df["items"].fillna("").str.contains(r"\b2\.02\b", regex=True)]
    out = pd.DataFrame({"accepted_et": api_time_to_eastern(df["accepted"])})
    out = out.dropna().drop_duplicates()
    return out.sort_values("accepted_et").reset_index(drop=True)


def api_time_to_eastern(values: pd.Series) -> pd.Series:
    """Convert submissions-API ``acceptanceDateTime`` strings to naive US/Eastern times.

    The API value is shifted by *twice* the Eastern UTC offset relative to the
    "Accepted" time on the EDGAR filing index page (verified by hand, e.g. Apple
    accession 0000320193-26-000018: API 2026-07-31T00:30:28Z, index page
    2026-07-30 16:30:28). Applying the UTC->Eastern conversion twice undoes it.
    """
    ts = pd.to_datetime(values.str.slice(0, 19), format="%Y-%m-%dT%H:%M:%S", errors="coerce")
    for _ in range(2):
        ts = ts.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
    return ts


def _facts_frame(facts: dict, tags: list[str], unit: str = "USD") -> pd.DataFrame:
    """Flatten the given us-gaap tags into one frame with a tag-priority column."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    rows = []
    for prio, tag in enumerate(tags):
        for e in gaap.get(tag, {}).get("units", {}).get(unit, []):
            if not str(e.get("form", "")).startswith(("10-Q", "10-K")):
                continue
            rows.append(
                {"prio": prio, "start": e.get("start"), "end": e["end"],
                 "val": e["val"], "filed": e["filed"]}
            )
    if not rows:
        return pd.DataFrame(columns=["prio", "start", "end", "val", "filed"])
    df = pd.DataFrame(rows)
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    df["val"] = df["val"].astype(float)
    return df


def _first_filed(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Keep, per key, the value from the highest-priority tag, as first filed."""
    df = df.sort_values(["prio", "filed"])
    best_prio = df.groupby(keys)["prio"].transform("min")
    df = df[df["prio"] == best_prio]
    return df.groupby(keys, as_index=False).first()


def quarterly_flow(facts: dict, tags: list[str]) -> pd.DataFrame:
    """Discrete fiscal-quarter values of a flow variable (income statement item).

    Fourth quarters are rarely reported directly; they are derived as
    FY minus the nine-month year-to-date value, dated at the 10-K filing date.
    Returns columns: period_end, value, filed.
    """
    df = _facts_frame(facts, tags)
    if df.empty or df["start"].isna().all():
        return pd.DataFrame(columns=["period_end", "value", "filed"])
    df = df.dropna(subset=["start"]).copy()
    df["days"] = (df["end"] - df["start"]).dt.days

    q = df[df["days"].between(*QUARTER_DAYS)]
    q = _first_filed(q, ["end"])[["end", "val", "filed"]]

    fy = _first_filed(df[df["days"].between(*ANNUAL_DAYS)], ["start", "end"])
    nm = _first_filed(df[df["days"].between(*NINE_MONTH_DAYS)], ["start", "end"])
    derived = fy.merge(nm, on="start", suffixes=("_fy", "_9m"))
    derived = derived[(derived["end_fy"] - derived["end_9m"]).dt.days.between(80, 100)]
    derived = pd.DataFrame(
        {
            "end": derived["end_fy"],
            "val": derived["val_fy"] - derived["val_9m"],
            "filed": derived[["filed_fy", "filed_9m"]].max(axis=1),
        }
    )
    have = set(q["end"])
    derived = derived[~derived["end"].isin(have)]
    out = pd.concat([q, derived], ignore_index=True)
    out = out.rename(columns={"end": "period_end", "val": "value"})
    return out.sort_values("period_end").drop_duplicates("period_end").reset_index(drop=True)


def quarterly_stock(facts: dict, tags: list[str]) -> pd.DataFrame:
    """Point-in-time balance-sheet (instant) values. Columns: period_end, value, filed."""
    df = _facts_frame(facts, tags)
    if df.empty:
        return pd.DataFrame(columns=["period_end", "value", "filed"])
    df = df[df["start"].isna()]
    df = _first_filed(df, ["end"])[["end", "val", "filed"]]
    df = df.rename(columns={"end": "period_end", "val": "value"})
    return df.sort_values("period_end").reset_index(drop=True)


def fiscal_quarter_key(period_end: pd.Series) -> pd.Series:
    """Map a fiscal period end to a monthly integer key robust to 52/53-week calendars.

    Ends such as 2023-07-01 and 2025-06-28 both map to July of their year, so
    the same fiscal quarter one year earlier is always ``key - 12``.
    """
    shifted = pd.to_datetime(period_end) + pd.Timedelta(days=15)
    return (shifted.dt.year * 12 + shifted.dt.month - 1).astype("int64")


def firm_fundamentals(facts: dict) -> pd.DataFrame:
    """Merge the quarterly series we need into one frame keyed by fiscal quarter."""
    series = {
        "ni": quarterly_flow(facts, NET_INCOME_TAGS),
        "rev": quarterly_flow(facts, REVENUE_TAGS),
        "oi": quarterly_flow(facts, OPERATING_INCOME_TAGS),
        "assets": quarterly_stock(facts, ASSETS_TAGS),
        "equity": quarterly_stock(facts, EQUITY_TAGS),
    }
    base = series["ni"].rename(columns={"value": "ni", "filed": "filed_ni"})
    if base.empty:
        return base
    base["fq"] = fiscal_quarter_key(base["period_end"])
    for name in ("rev", "oi", "assets", "equity"):
        s = series[name]
        if s.empty:
            base[name] = np.nan
            base[f"filed_{name}"] = pd.NaT
            continue
        s = s.assign(fq=fiscal_quarter_key(s["period_end"]))
        s = s.drop_duplicates("fq", keep="last")[["fq", "value", "filed"]]
        base = base.merge(
            s.rename(columns={"value": name, "filed": f"filed_{name}"}), on="fq", how="left"
        )
    return base.drop_duplicates("fq", keep="last").sort_values("fq").reset_index(drop=True)
