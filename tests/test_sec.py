import pandas as pd

from earnml import sec


def _fact(start, end, val, filed, form="10-Q"):
    e = {"end": end, "val": val, "filed": filed, "form": form}
    if start:
        e["start"] = start
    return e


def _facts(tag, entries):
    return {"facts": {"us-gaap": {tag: {"units": {"USD": entries}}}}}


def test_api_time_to_eastern_matches_edgar_index_pages():
    # Ground truth copied by hand from the EDGAR filing index pages ("Accepted").
    api = pd.Series(["2026-07-31T00:30:28.000Z", "2026-01-30T02:30:33.000Z", "2021-07-28T00:35:44.000Z"])
    got = sec.api_time_to_eastern(api)
    expected = pd.to_datetime(["2026-07-30 16:30:28", "2026-01-29 16:30:33", "2021-07-27 16:35:44"])
    assert list(got) == list(expected)


def test_earnings_release_times_filters_item_202():
    sub = {"filings": {"recent": {
        "form": ["8-K", "8-K", "10-Q", "8-K"],
        "items": ["2.02,9.01", "5.02", "", "7.01,2.02"],
        "acceptanceDateTime": ["2024-02-01T21:30:00.000Z", "2024-02-02T21:30:00.000Z",
                               "2024-02-03T21:30:00.000Z", "2024-05-01T12:00:00.000Z"],
    }}}
    out = sec.earnings_release_times(sub)
    assert len(out) == 2
    assert out["accepted_et"].iloc[0] == pd.Timestamp("2024-02-01 11:30:00")


def test_quarterly_flow_keeps_first_filed_value_and_derives_q4():
    entries = [
        _fact("2023-01-01", "2023-03-31", 10, "2023-05-01"),
        _fact("2023-01-01", "2023-03-31", 99, "2024-05-01"),  # later restatement: ignored
        _fact("2023-04-01", "2023-06-30", 20, "2023-08-01"),
        _fact("2023-07-01", "2023-09-30", 30, "2023-11-01"),
        _fact("2023-01-01", "2023-09-30", 60, "2023-11-01"),  # nine-month YTD
        _fact("2023-01-01", "2023-12-31", 100, "2024-02-20", "10-K"),
    ]
    q = sec.quarterly_flow(_facts("NetIncomeLoss", entries), ["NetIncomeLoss"])
    assert q["value"].tolist() == [10, 20, 30, 40]
    q4 = q.iloc[-1]
    assert q4["period_end"] == pd.Timestamp("2023-12-31")
    assert q4["filed"] == pd.Timestamp("2024-02-20")


def test_quarterly_flow_prefers_higher_priority_tag():
    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [_fact("2023-01-01", "2023-03-31", 5, "2023-05-01")]}},
        "SalesRevenueNet": {"units": {"USD": [_fact("2023-01-01", "2023-03-31", 7, "2023-05-01"),
                                              _fact("2023-04-01", "2023-06-30", 8, "2023-08-01")]}},
    }}}
    q = sec.quarterly_flow(facts, ["Revenues", "SalesRevenueNet"])
    assert q["value"].tolist() == [5, 8]


def test_quarterly_stock_uses_instant_values():
    entries = [_fact(None, "2023-03-31", 1000, "2023-05-01"), _fact(None, "2023-03-31", 1100, "2024-05-01")]
    s = sec.quarterly_stock(_facts("Assets", entries), ["Assets"])
    assert s["value"].tolist() == [1000]


def test_fiscal_quarter_key_handles_52_53_week_years():
    ends = pd.Series(pd.to_datetime(["2023-07-01", "2024-06-29", "2025-06-28", "2025-03-29"]))
    k = sec.fiscal_quarter_key(ends)
    assert k.iloc[1] - k.iloc[0] == 12
    assert k.iloc[2] - k.iloc[1] == 12
    assert k.iloc[2] - k.iloc[3] == 3
