"""
UTC-safe time handling for Elexon data.
=======================================

Background: the DST offset bug
------------------------------
Elexon's Balancing Mechanism data indexes rows by
(settlement_date, settlement_period). There are 48 settlement periods in a
normal day, 46 on the spring DST transition day, and 50 on the autumn one.
The tempting formula

    datetime = settlement_date + (settlement_period - 1) * 30 minutes

is WRONG during British Summer Time. Elexon's `settlement_date` is a local
calendar date anchored to the British trading day, which runs midnight
local time. The formula assumes UTC throughout. During BST the two differ
by one hour. The bug is silent — it produces plausible-looking numbers,
passes visual inspection, and only corrupts joins that cross DST
boundaries.

This bug was identified mid-project in the Depletion Multiplier IV study
(see that project's DEVIATIONS.md §4). By that point, rewriting every
loader was scoped out of the remaining work and the IV project accepted
the bug within the within-Elexon subset (where the offset cancels on
both sides of a join). Project F does NOT inherit this compromise.

Rule
----
Every datetime that appears in any Project F DataFrame must be
**tz-aware UTC**. Elexon API responses include an authoritative UTC
`start_time` column. Use that. Never compute datetime from
(settlement_date, settlement_period).

This module provides helpers that enforce the rule. Functions raise
TimeConventionError rather than silently computing, so the bug cannot be
introduced by accident. Any code that needs SP-level time arithmetic must
pass through these helpers.

Naive datetimes are accepted by some pandas operations but treated here as
an error — it's impossible to tell at runtime whether a naive timestamp
was supposed to be UTC, local, or British-trading-day-local, so we refuse
them at the boundary.
"""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from ..errors import TimeConventionError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def require_start_time(df: pd.DataFrame, *, name: str = "DataFrame") -> None:
    """
    Assert that `df` has a `start_time` column of tz-aware UTC datetimes.

    Raises TimeConventionError with a specific message if:
      - the column is missing entirely,
      - it is not a datetime dtype,
      - it is datetime but naive (no tz),
      - it has a tz but that tz is not UTC.

    This is the single function every loader should call at the top of its
    body, with the incoming DataFrame and a name like "Elexon system prices"
    so the error message is informative.

    Parameters
    ----------
    df : DataFrame to check
    name : human-readable name for error messages

    Raises
    ------
    TimeConventionError
    """
    if "start_time" not in df.columns:
        raise TimeConventionError(
            f"{name}: required column 'start_time' is missing. "
            f"All Elexon-derived DataFrames must include the API's "
            f"native UTC start_time column. Do NOT compute datetime from "
            f"(settlement_date, settlement_period); see module docstring."
        )

    s = df["start_time"]
    if not pd.api.types.is_datetime64_any_dtype(s):
        raise TimeConventionError(
            f"{name}: column 'start_time' has dtype {s.dtype}, "
            f"expected a datetime dtype."
        )

    tz = getattr(s.dtype, "tz", None)
    if tz is None:
        raise TimeConventionError(
            f"{name}: column 'start_time' is datetime but naive (no tz). "
            f"Expected tz-aware UTC. Run `start_time = pd.to_datetime("
            f"start_time, utc=True)` on the raw source before passing through."
        )

    # pandas reports tz as a tzinfo object; UTC appears as either
    # `datetime.timezone.utc` or `pytz.UTC`. Both compare equal to UTC via str.
    if str(tz) != "UTC":
        raise TimeConventionError(
            f"{name}: column 'start_time' has tz={tz}, expected UTC. "
            f"Convert with `df['start_time'].dt.tz_convert('UTC')`."
        )


def ensure_utc(series, *, name: str = "series") -> pd.Series:
    """
    Return a copy of `series` guaranteed to be tz-aware UTC, as a Series.

    Accepts:
      - pd.Series: already tz-aware UTC (returned unchanged, as a copy)
      - pd.Series: tz-aware non-UTC (converted)
      - pd.Series: object dtype containing ISO strings (parsed with utc=True)
      - pd.DatetimeIndex: coerced to Series, then the above rules apply

    Rejects naive datetime series — see module docstring. If you genuinely
    need to handle a naive series, you must first establish the intended
    timezone explicitly in your own code; this helper will not guess.

    Parameters
    ----------
    series : pd.Series or pd.DatetimeIndex
    name : human-readable name for error messages

    Returns
    -------
    pd.Series : tz-aware UTC datetimes

    Raises
    ------
    TimeConventionError : if the series is naive datetime or unparseable
    """
    # Coerce DatetimeIndex to Series so .dt accessor works. This is a no-op
    # for an already-Series input.
    if isinstance(series, pd.DatetimeIndex):
        series = pd.Series(series)

    if pd.api.types.is_datetime64_any_dtype(series):
        tz = getattr(series.dtype, "tz", None)
        if tz is None:
            raise TimeConventionError(
                f"ensure_utc({name}): series is naive datetime. "
                f"Refusing to assume a timezone. See module docstring."
            )
        if str(tz) == "UTC":
            return series.copy()
        return series.dt.tz_convert("UTC")

    # String dtype (object or pandas 2.x "str") — parse with utc=True so
    # naive strings are localised as UTC. This is the one place we accept
    # naive-looking input, because string parsing is unambiguous once we
    # commit to utc=True.
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return pd.to_datetime(series, utc=True)

    raise TimeConventionError(
        f"ensure_utc({name}): dtype {series.dtype} is not datetime or object. "
        f"Cannot coerce."
    )


# ---------------------------------------------------------------------------
# SP ↔ time helpers (one-way only)
# ---------------------------------------------------------------------------
# Deliberate design: we provide helpers that GO FROM `start_time` TO a
# settlement period index, but NOT the reverse. Going SP → time is the
# direction that requires knowing local-vs-UTC, which is exactly where
# the DST bug lives. Going time → SP is derivable purely from UTC
# arithmetic given that the UK trading day starts at 23:00 UTC the
# previous evening (winter) or 23:00 UTC the same evening (summer,
# accounting for BST). We sidestep the question entirely by computing
# SP as an index within the trading day based on start_time directly.


def utc_to_sp_index(start_time) -> pd.Series:
    """
    Compute the settlement-period index (1..48, 1..46, or 1..50 on DST days)
    from a tz-aware UTC start_time series or DatetimeIndex.

    The SP index is simply the count of 30-minute intervals since the start
    of the GB trading day that contains `start_time`. We do not need to
    know what local date the trading day is anchored to — the index is
    zero at the start of the trading day and increments every 30 minutes.

    For downstream joins that need (date, SP) tuples for reporting
    purposes, use `gb_trading_date()` to get the anchored date and this
    function to get the index. DO NOT re-derive start_time from those
    two values; if you have start_time already, keep it.

    Parameters
    ----------
    start_time : pd.Series or pd.DatetimeIndex of tz-aware UTC datetimes

    Returns
    -------
    pd.Series of int : SP index, 1-based
    """
    if isinstance(start_time, pd.DatetimeIndex):
        start_time = pd.Series(start_time)
    if not pd.api.types.is_datetime64_any_dtype(start_time):
        raise TimeConventionError(
            "utc_to_sp_index: input must be a datetime series."
        )
    tz = getattr(start_time.dtype, "tz", None)
    if tz is None or str(tz) != "UTC":
        raise TimeConventionError(
            "utc_to_sp_index: input must be tz-aware UTC. "
            f"Got tz={tz}."
        )

    # Convert to GB local time for SP-index arithmetic. This is the ONE place
    # local time appears, and only as a scratch variable — never written back.
    local = start_time.dt.tz_convert("Europe/London")

    # SP index = floor((hour*60 + minute) / 30) + 1
    minutes_since_midnight_local = local.dt.hour * 60 + local.dt.minute
    return (minutes_since_midnight_local // 30 + 1).astype(int)


def gb_trading_date(start_time) -> pd.Series:
    """
    Return the GB trading-day date (as pd.Timestamp.date) that each
    start_time belongs to. The GB trading day begins at 00:00 LOCAL time
    (Europe/London) and contains 48 SPs (46 or 50 on DST transition days).

    Use this when you need the `settlement_date` equivalent for reporting
    or for joining to external sources that key on (date, SP).

    Parameters
    ----------
    start_time : pd.Series or pd.DatetimeIndex of tz-aware UTC datetimes

    Returns
    -------
    pd.Series of object (datetime.date) : GB trading date
    """
    if isinstance(start_time, pd.DatetimeIndex):
        start_time = pd.Series(start_time)
    if not pd.api.types.is_datetime64_any_dtype(start_time):
        raise TimeConventionError(
            "gb_trading_date: input must be a datetime series."
        )
    tz = getattr(start_time.dtype, "tz", None)
    if tz is None or str(tz) != "UTC":
        raise TimeConventionError(
            "gb_trading_date: input must be tz-aware UTC."
        )

    return start_time.dt.tz_convert("Europe/London").dt.date


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def filter_to_sps(
    df: pd.DataFrame,
    sp_column: str,
    sps: Iterable[int],
) -> pd.DataFrame:
    """
    Return rows of `df` where `sp_column` is in the given SP range.

    Tiny helper but worth centralising: EVENING_SPS and OVERNIGHT_SPS from
    constants.py are tuples, and we want a single idiom for filtering by
    them rather than a mix of `.isin()` and range-based comparisons.

    Parameters
    ----------
    df : DataFrame to filter
    sp_column : name of the SP column
    sps : iterable of SP integers (e.g. constants.EVENING_SPS)

    Returns
    -------
    pd.DataFrame : filtered copy
    """
    if sp_column not in df.columns:
        raise TimeConventionError(
            f"filter_to_sps: column '{sp_column}' not in DataFrame."
        )
    return df[df[sp_column].isin(tuple(sps))].copy()


def evening_window_filter(
    df: pd.DataFrame,
    sp_column: str,
) -> pd.DataFrame:
    """Shorthand for filter_to_sps(df, sp_column, EVENING_SPS)."""
    from ..constants import EVENING_SPS
    return filter_to_sps(df, sp_column, EVENING_SPS)


def overnight_window_filter(
    df: pd.DataFrame,
    sp_column: str,
) -> pd.DataFrame:
    """Shorthand for filter_to_sps(df, sp_column, OVERNIGHT_SPS)."""
    from ..constants import OVERNIGHT_SPS
    return filter_to_sps(df, sp_column, OVERNIGHT_SPS)
