"""
Tests for src/utils/time_conventions.py.

Focus: DST boundary behaviour. The entire point of this module is to keep
the IV project's DEVIATIONS.md §4 bug (naive datetimes silently losing an
hour during BST) from coming back in Project F.

Every test here uses tz-aware UTC timestamps at construction. The module's
helpers are expected to refuse naive input and to behave correctly across
the March and October DST transitions.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.errors import TimeConventionError
from src.utils.time_conventions import (
    ensure_utc,
    evening_window_filter,
    filter_to_sps,
    gb_trading_date,
    overnight_window_filter,
    require_start_time,
    utc_to_sp_index,
)


# ---------------------------------------------------------------------------
# require_start_time
# ---------------------------------------------------------------------------

class TestRequireStartTime:
    def test_accepts_tz_aware_utc(self):
        df = pd.DataFrame({
            "start_time": pd.to_datetime(
                ["2024-06-15T10:00Z", "2024-06-15T10:30Z"], utc=True
            )
        })
        # Should not raise.
        require_start_time(df)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"something_else": [1, 2]})
        with pytest.raises(TimeConventionError, match="missing"):
            require_start_time(df)

    def test_naive_datetime_raises(self):
        df = pd.DataFrame({
            "start_time": pd.to_datetime(["2024-06-15T10:00", "2024-06-15T10:30"])
        })
        with pytest.raises(TimeConventionError, match="naive"):
            require_start_time(df)

    def test_non_utc_tz_raises(self):
        df = pd.DataFrame({
            "start_time": pd.to_datetime(
                ["2024-06-15T10:00", "2024-06-15T10:30"]
            ).tz_localize("Europe/London")
        })
        with pytest.raises(TimeConventionError, match="UTC"):
            require_start_time(df)

    def test_non_datetime_dtype_raises(self):
        df = pd.DataFrame({"start_time": ["2024-06-15", "2024-06-16"]})
        with pytest.raises(TimeConventionError, match="dtype"):
            require_start_time(df)

    def test_error_message_includes_name(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(TimeConventionError, match="Elexon system prices"):
            require_start_time(df, name="Elexon system prices")


# ---------------------------------------------------------------------------
# ensure_utc
# ---------------------------------------------------------------------------

class TestEnsureUtc:
    def test_already_utc_returns_copy(self):
        s = pd.to_datetime(["2024-01-01T00:00Z", "2024-01-01T00:30Z"], utc=True)
        result = ensure_utc(s)
        assert str(result.dtype.tz) == "UTC"
        assert (result == s).all()

    def test_converts_from_other_tz(self):
        s = pd.to_datetime(
            ["2024-06-15T11:00", "2024-06-15T11:30"]
        ).tz_localize("Europe/London")
        # 11:00 BST == 10:00 UTC
        result = ensure_utc(s)
        assert str(result.dtype.tz) == "UTC"
        assert result.iloc[0] == pd.Timestamp("2024-06-15T10:00Z")

    def test_rejects_naive_datetime(self):
        s = pd.to_datetime(["2024-06-15T10:00", "2024-06-15T10:30"])
        with pytest.raises(TimeConventionError, match="naive"):
            ensure_utc(s)

    def test_parses_object_dtype_as_utc(self):
        s = pd.Series(["2024-06-15T10:00Z", "2024-06-15T10:30Z"])
        result = ensure_utc(s)
        assert str(result.dtype.tz) == "UTC"

    def test_rejects_numeric(self):
        s = pd.Series([1, 2, 3])
        with pytest.raises(TimeConventionError, match="dtype"):
            ensure_utc(s)


# ---------------------------------------------------------------------------
# utc_to_sp_index
# ---------------------------------------------------------------------------

class TestUtcToSpIndex:
    def test_winter_midnight_is_sp_1(self):
        # 2024-01-15 00:00 UTC == 00:00 GMT local → SP 1
        s = pd.to_datetime(["2024-01-15T00:00Z"], utc=True)
        assert utc_to_sp_index(s).iloc[0] == 1

    def test_winter_noon_is_sp_25(self):
        # 2024-01-15 12:00 UTC == 12:00 GMT local → 12*60/30 + 1 = 25
        s = pd.to_datetime(["2024-01-15T12:00Z"], utc=True)
        assert utc_to_sp_index(s).iloc[0] == 25

    def test_winter_evening_sp_34(self):
        # 2024-01-15 16:30 UTC == 16:30 GMT local → SP 34 (start of pre-reg
        # evening window)
        s = pd.to_datetime(["2024-01-15T16:30Z"], utc=True)
        assert utc_to_sp_index(s).iloc[0] == 34

    def test_summer_noon_is_sp_25_local(self):
        # This is the exact case the IV bug hit. 2024-06-15 11:00 UTC ==
        # 12:00 BST local → SP 25 in local (trading-day) time, NOT SP 23
        # (which is what the buggy formula would produce from the UTC
        # calendar-date math).
        s = pd.to_datetime(["2024-06-15T11:00Z"], utc=True)
        assert utc_to_sp_index(s).iloc[0] == 25

    def test_summer_evening_sp_34(self):
        # 2024-06-15 15:30 UTC == 16:30 BST local → SP 34
        s = pd.to_datetime(["2024-06-15T15:30Z"], utc=True)
        assert utc_to_sp_index(s).iloc[0] == 34

    def test_rejects_naive(self):
        s = pd.to_datetime(["2024-06-15T10:00"])
        with pytest.raises(TimeConventionError, match="UTC"):
            utc_to_sp_index(s)

    def test_rejects_non_utc_tz(self):
        s = pd.to_datetime(["2024-06-15T10:00"]).tz_localize("Europe/London")
        with pytest.raises(TimeConventionError, match="UTC"):
            utc_to_sp_index(s)


# ---------------------------------------------------------------------------
# gb_trading_date
# ---------------------------------------------------------------------------

class TestGbTradingDate:
    def test_winter_utc_matches_local_date(self):
        s = pd.to_datetime(["2024-01-15T12:00Z"], utc=True)
        assert gb_trading_date(s).iloc[0] == pd.Timestamp("2024-01-15").date()

    def test_summer_late_night_utc_is_next_local_date(self):
        # 2024-06-15 23:30 UTC == 2024-06-16 00:30 BST local.
        # The GB trading day belongs to the local date, so this should
        # be the 16th, not the 15th.
        s = pd.to_datetime(["2024-06-15T23:30Z"], utc=True)
        assert gb_trading_date(s).iloc[0] == pd.Timestamp("2024-06-16").date()

    def test_summer_early_morning_utc_is_same_local_date(self):
        # 2024-06-15 01:00 UTC == 02:00 BST local → still the 15th.
        s = pd.to_datetime(["2024-06-15T01:00Z"], utc=True)
        assert gb_trading_date(s).iloc[0] == pd.Timestamp("2024-06-15").date()

    def test_rejects_naive(self):
        s = pd.to_datetime(["2024-06-15T12:00"])
        with pytest.raises(TimeConventionError):
            gb_trading_date(s)


# ---------------------------------------------------------------------------
# DST boundary: specifically exercise the day that broke the IV project
# ---------------------------------------------------------------------------

class TestDstBoundary:
    """
    On 2024-03-31 the UK switched to BST at 01:00 UTC (01:00 → 02:00 local).
    On 2024-10-27 it switched back at 01:00 UTC (02:00 → 01:00 local).
    """

    def test_march_dst_day_sp_index_continuous(self):
        # The trading day has 46 SPs on spring DST. SPs 1-2 are at
        # 00:00 and 00:30 UTC. SP 3 (local 02:00 BST = 01:00 UTC) is the
        # first SP after the switch. There is no SP corresponding to the
        # local hour 01:00-02:00 because that hour doesn't exist in BST.
        #
        # The key invariant we care about: two SPs that look like "same
        # settlement period number" on the naive formula actually fall in
        # different trading days if UTC arithmetic is used carelessly.
        # This test locks in that utc_to_sp_index returns sensible values
        # on the tricky day.

        # 00:30 UTC on the DST-transition date is SP 2 (00:30 GMT local,
        # since BST has not yet kicked in)
        pre_switch = pd.to_datetime(["2024-03-31T00:30Z"], utc=True)
        assert utc_to_sp_index(pre_switch).iloc[0] == 2

        # 01:00 UTC == 02:00 BST local → SP index 5 local-time
        # (02*60/30 + 1 = 5). The "missing" SPs 3 and 4 are the skipped
        # hour.
        post_switch = pd.to_datetime(["2024-03-31T01:00Z"], utc=True)
        assert utc_to_sp_index(post_switch).iloc[0] == 5

    def test_october_dst_day_sp_index(self):
        # On 2024-10-27 the clocks go back at 01:00 UTC: 02:00 BST → 01:00 GMT.
        # 00:30 UTC == 01:30 BST local → SP 4
        pre_switch = pd.to_datetime(["2024-10-27T00:30Z"], utc=True)
        assert utc_to_sp_index(pre_switch).iloc[0] == 4

        # 01:00 UTC == 01:00 GMT local (BST just ended) → SP 3.
        # This is the only day of the year where the SP index "goes back".
        # The trading-day concept handles this via the 50-SP day convention;
        # our helper returns the local-hour-indexed SP directly, which is
        # what downstream code expects.
        post_switch = pd.to_datetime(["2024-10-27T01:00Z"], utc=True)
        assert utc_to_sp_index(post_switch).iloc[0] == 3


# ---------------------------------------------------------------------------
# filter helpers
# ---------------------------------------------------------------------------

class TestFilters:
    def test_filter_to_sps_selects_given_sps(self):
        df = pd.DataFrame({
            "sp": list(range(1, 49)),
            "v": list(range(1, 49)),
        })
        result = filter_to_sps(df, "sp", (34, 35, 36))
        assert list(result["sp"]) == [34, 35, 36]

    def test_filter_missing_column_raises(self):
        df = pd.DataFrame({"other": [1, 2]})
        with pytest.raises(TimeConventionError, match="not in DataFrame"):
            filter_to_sps(df, "sp", (1,))

    def test_evening_filter_matches_pre_reg(self):
        df = pd.DataFrame({"sp": list(range(1, 49))})
        result = evening_window_filter(df, "sp")
        assert list(result["sp"]) == list(range(34, 45))
        assert len(result) == 11

    def test_overnight_filter_matches_pre_reg(self):
        df = pd.DataFrame({"sp": list(range(1, 49))})
        result = overnight_window_filter(df, "sp")
        assert list(result["sp"]) == list(range(1, 15))
        assert len(result) == 14

    def test_filter_returns_copy(self):
        df = pd.DataFrame({"sp": [34, 35], "v": [1, 2]})
        result = filter_to_sps(df, "sp", (34,))
        result["v"] = 99
        # Original should be unchanged
        assert df.loc[df["sp"] == 34, "v"].iloc[0] == 1
