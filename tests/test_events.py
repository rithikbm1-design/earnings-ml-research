import numpy as np
import pandas as pd
import pytest

from earnml import events

DAYS = pd.bdate_range("2024-01-01", "2024-03-29")


def test_event_day_after_close_moves_to_next_session():
    acc = pd.Series(pd.to_datetime([
        "2024-02-01 16:30",  # after close Thursday -> Friday
        "2024-02-01 07:00",  # pre-market -> same day
        "2024-02-02 17:00",  # after close Friday -> Monday
        "2024-02-03 10:00",  # Saturday -> Monday
    ]))
    got = events.event_trading_day(acc, DAYS)
    assert list(got) == list(pd.to_datetime(["2024-02-02", "2024-02-01", "2024-02-05", "2024-02-05"]))


def test_match_announcements_takes_first_release_within_window():
    ends = pd.Series(pd.to_datetime(["2023-12-31", "2024-03-31"]))
    rel = pd.Series(pd.to_datetime(["2023-12-15 16:05", "2024-01-25 16:05", "2024-02-10 08:00",
                                    "2024-08-30 16:05"]))
    got = events.match_announcements(ends, rel)
    assert got.iloc[0] == pd.Timestamp("2024-01-25 16:05")
    assert pd.isna(got.iloc[1])  # next release is > 75 days after the quarter end


def test_cumulative_abnormal_return_sums_window():
    r = pd.Series(np.full(len(DAYS), 0.01), index=DAYS)
    m = pd.Series(np.full(len(DAYS), 0.004), index=DAYS)
    ev = DAYS[10]
    assert events.cumulative_abnormal_return(r, m, ev, -1, 1) == pytest.approx(3 * 0.006)
    assert np.isnan(events.cumulative_abnormal_return(r, m, DAYS[-1], -1, 1))  # window incomplete
    r.iloc[11] = np.nan
    assert np.isnan(events.cumulative_abnormal_return(r, m, ev, -1, 1))
