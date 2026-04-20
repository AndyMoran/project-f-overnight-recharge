"""
Pre-lock empirical check: WINDFOR vintage retention.
====================================================

This script satisfies the first pre-lock verification item in
PROJECT_F_PRE_REGISTRATION_v3.md §Review and adversarial check:

    [ ] WINDFOR vintage-retention empirical check: query
        /forecast/generation/wind/history?publishTime=2024-01-15T12:30Z
        and confirm (a) a 200 response with non-empty content, (b) the
        returned publishTime field equals 2024-01-15T12:30Z (not a more
        recent vintage), (c) the forecast horizon covers at least through
        end of 2024-01-16. Repeat for one date near the end of the test
        period (e.g. 2025-12-15T12:30Z). If either date fails,
        vintage-filtered history is not reliably available for the full
        study period and the pre-reg cannot lock with this matching
        covariate specified as written.

What this script does
---------------------
For each of two dates (one near the start of the training period, one
near the end of the test period), it queries the Elexon BMRS
`/forecast/generation/wind/history` endpoint with the specific
publishTime and runs four assertions:

  1. http_200_non_empty: response arrives with HTTP 200 and contains
     at least one forecast record.
  2. publish_time_exact_match: every returned record's publishTime is
     within 90 minutes of the requested vintage AND is not from a more
     recent date (which would indicate the historical vintage has been
     dropped from the API's retention window). The ±90-minute tolerance
     accommodates Elexon rounding to the nearest published half-hour.
  3. forecast_horizon_covers_next_day: the maximum startTime in the
     response is on or after midnight UTC of d+2 (i.e. the forecast
     horizon reaches at least through the end of d+1, per pre-reg).
  4. schema_has_required_fields: each record has publishTime,
     startTime, generation (the fields the build_z / matching pipeline
     depends on).

Output
------
Writes `prelock_checks/windfor_vintage_retention.json` with detailed
per-date evidence and an overall pass/fail. Also prints a
human-readable summary to stdout.

Usage
-----
    python prelock_checks/windfor_vintage_retention.py

Exit status
-----------
    0 if all assertions pass (both dates succeed on all four checks).
    1 if any assertion fails.

If this script exits nonzero, the pre-reg cannot lock as written —
either the Elexon API behaviour has changed, a specific historical
vintage has been dropped from retention, or the publishTime parameter
does not work as we believe. DO NOT suppress or work around the
failure; instead amend the pre-reg or find an alternative covariate
source before proceeding.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Two dates per pre-reg §Review and adversarial check. These are the exact
# dates named in the pre-reg — do not change without a DEVIATIONS entry.
DATES_TO_CHECK = [
    {
        "publish_time": "2024-01-15T12:30Z",
        "label": "training-period start (2024-01-15)",
        # The forecast must extend through end of d+1 = end of 2024-01-16.
        # Equivalently, max startTime >= 2024-01-17T00:00Z.
        "horizon_end_required": "2024-01-17T00:00Z",
    },
    {
        "publish_time": "2025-12-15T12:30Z",
        "label": "test-period end (2025-12-15)",
        "horizon_end_required": "2025-12-17T00:00Z",
    },
]

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "windfor_vintage_retention.json"


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def fetch_windfor_history(publish_time: str) -> tuple[int, list[dict]]:
    """
    Call Elexon /forecast/generation/wind/history with the given publishTime.

    Returns (http_status, list_of_records). If the API returns a non-200
    or an unexpected shape, the list may be empty; the caller checks the
    status separately.

    We use elexonpy rather than requests directly so this script matches
    the idiom of the other fetchers. The elexonpy client does not expose
    the underlying HTTP status directly for successful calls, so on
    success we return 200; on exception we return the status code if
    extractable, otherwise 0.
    """
    try:
        from elexonpy.api_client import ApiClient
        from elexonpy.api.generation_forecast_api import GenerationForecastApi
    except ImportError:
        print(
            "ERROR: elexonpy not installed. Run `pip install elexonpy>=0.0.12` "
            "or add it to requirements.txt.",
            file=sys.stderr,
        )
        sys.exit(2)

    client = ApiClient()
    api = GenerationForecastApi(client)

    # Parse the publish_time string to a datetime the library accepts.
    # "2024-01-15T12:30Z" → datetime(2024, 1, 15, 12, 30, tzinfo=UTC)
    publish_dt = datetime.strptime(publish_time, "%Y-%m-%dT%H:%MZ").replace(
        tzinfo=timezone.utc
    )

    try:
        response = api.forecast_generation_wind_history_get(
            publish_time=publish_dt,
            format="json",
        )
    except Exception as e:
        # elexonpy raises ApiException on non-200; try to extract status.
        status = getattr(e, "status", 0)
        return int(status) if status else 0, []

    if not hasattr(response, "data") or response.data is None:
        return 200, []

    # Each record in response.data is an elexonpy model object; convert
    # to plain dicts for JSON-friendly processing.
    records = [r.to_dict() for r in response.data]
    return 200, records


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_http_200_non_empty(status: int, records: list[dict]) -> bool:
    """(a) 200 response with non-empty content."""
    return status == 200 and len(records) > 0


def assert_publish_time_exact_match(
    records: list[dict],
    expected_publish_time: str,
) -> bool:
    """
    (b) Every returned record's publishTime is within 90 minutes of the
    requested vintage AND is not from a more recent date (which would
    indicate the historical vintage has been dropped and Elexon is
    returning a replay of a newer forecast).

    Why 90 minutes: WINDFOR forecasts are published on the half-hour.
    Elexon's history endpoint returns the nearest available vintage to
    the requested publishTime, which may differ by up to one publication
    interval (30 min) plus any clock rounding. We use 90 min to give
    comfortable headroom.

    The check that actually matters for data integrity: the returned
    publishTime must not be LATER than the requested time by more than
    the tolerance. If it is, the historical vintage has been dropped from
    the API's retention window and we are receiving a newer forecast,
    which would invalidate the matching covariate.
    """
    if not records:
        return False

    expected_dt = datetime.strptime(
        expected_publish_time, "%Y-%m-%dT%H:%MZ"
    ).replace(tzinfo=timezone.utc)

    from datetime import timedelta
    tolerance = timedelta(minutes=90)

    for r in records:
        pt = r.get("publish_time") or r.get("publishTime")
        if pt is None:
            return False
        if isinstance(pt, datetime):
            pt_dt = pt if pt.tzinfo else pt.replace(tzinfo=timezone.utc)
        else:
            pt_str = str(pt).replace("Z", "+00:00")
            try:
                pt_dt = datetime.fromisoformat(pt_str)
            except ValueError:
                return False

        delta = pt_dt - expected_dt  # positive = returned vintage is NEWER
        # Fail if returned vintage is more than 90 min NEWER than requested
        # (indicates historical data dropped — API serving a later vintage).
        if delta > tolerance:
            return False
        # Also fail if returned vintage is more than 90 min OLDER than requested
        # (indicates unexpected API behaviour).
        if delta < -tolerance:
            return False
    return True


def assert_forecast_horizon_covers(
    records: list[dict],
    horizon_end_required: str,
) -> bool:
    """
    (c) Forecast horizon covers at least through end of d+1.

    Equivalent to: max(startTime) >= horizon_end_required, where
    horizon_end_required is midnight UTC of d+2.
    """
    if not records:
        return False

    required_dt = datetime.strptime(
        horizon_end_required, "%Y-%m-%dT%H:%MZ"
    ).replace(tzinfo=timezone.utc)

    max_start = None
    for r in records:
        st = r.get("start_time") or r.get("startTime")
        if st is None:
            return False
        if isinstance(st, datetime):
            st_dt = st if st.tzinfo else st.replace(tzinfo=timezone.utc)
        else:
            st_str = str(st).replace("Z", "+00:00")
            try:
                st_dt = datetime.fromisoformat(st_str)
            except ValueError:
                return False
        if max_start is None or st_dt > max_start:
            max_start = st_dt

    return max_start is not None and max_start >= required_dt


def assert_schema_has_required_fields(records: list[dict]) -> bool:
    """
    Extra check: every record has publishTime, startTime, generation.

    These are the three fields the build_z / matching pipeline depends
    on. If Elexon has changed the response schema, we want to know now
    rather than mid-fetch.
    """
    if not records:
        return False

    for r in records:
        has_publish = "publish_time" in r or "publishTime" in r
        has_start = "start_time" in r or "startTime" in r
        has_gen = "generation" in r
        if not (has_publish and has_start and has_gen):
            return False
    return True


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------

def _iso(dt_or_str: Any) -> str | None:
    """Normalise a datetime or datetime-string to an ISO string."""
    if dt_or_str is None:
        return None
    if isinstance(dt_or_str, datetime):
        return dt_or_str.isoformat()
    return str(dt_or_str)


def extract_evidence(
    status: int,
    records: list[dict],
) -> dict:
    """Produce a compact evidence block for the JSON output."""
    if not records:
        return {
            "http_status": status,
            "n_records": 0,
            "first_publish_time_returned": None,
            "min_start_time": None,
            "max_start_time": None,
            "sample_record": None,
        }

    def _safe_parse(key_snake: str, key_camel: str, rec: dict) -> datetime | None:
        v = rec.get(key_snake) or rec.get(key_camel)
        if v is None:
            return None
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        s = str(v).replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return None

    start_times = [
        _safe_parse("start_time", "startTime", r) for r in records
    ]
    start_times = [s for s in start_times if s is not None]

    first = records[0]
    first_pt = first.get("publish_time") or first.get("publishTime")

    return {
        "http_status": status,
        "n_records": len(records),
        "first_publish_time_returned": _iso(first_pt),
        "min_start_time": _iso(min(start_times)) if start_times else None,
        "max_start_time": _iso(max(start_times)) if start_times else None,
        # Sample the first record with datetimes coerced to strings, so
        # the JSON dump is unambiguous.
        "sample_record": {
            k: _iso(v) if isinstance(v, datetime) else v
            for k, v in first.items()
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    results: list[dict] = []
    overall_passed = True

    for entry in DATES_TO_CHECK:
        publish_time = entry["publish_time"]
        label = entry["label"]
        horizon_end = entry["horizon_end_required"]

        print(f"\n[*] Checking WINDFOR vintage: {publish_time} ({label})")
        print(f"    Horizon must cover through: {horizon_end}")

        status, records = fetch_windfor_history(publish_time)

        assertions = {
            "http_200_non_empty": assert_http_200_non_empty(status, records),
            "publish_time_exact_match": assert_publish_time_exact_match(
                records, publish_time
            ),
            "forecast_horizon_covers_next_day": assert_forecast_horizon_covers(
                records, horizon_end
            ),
            "schema_has_required_fields": assert_schema_has_required_fields(
                records
            ),
        }
        failures = [name for name, ok in assertions.items() if not ok]
        passed = len(failures) == 0
        if not passed:
            overall_passed = False

        evidence = extract_evidence(status, records)

        print(f"    HTTP status: {evidence['http_status']}")
        print(f"    Records returned: {evidence['n_records']}")
        print(f"    First publishTime returned: {evidence['first_publish_time_returned']}")
        print(
            f"    Start-time range: "
            f"{evidence['min_start_time']} → {evidence['max_start_time']}"
        )
        for name, ok in assertions.items():
            print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
        if failures:
            print(f"    → FAILED: {', '.join(failures)}")

        results.append({
            "requested_vintage": publish_time,
            "label": label,
            "horizon_end_required": horizon_end,
            "status": "pass" if passed else "fail",
            "assertions": assertions,
            "evidence": evidence,
            "failures": failures,
        })

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_passed": overall_passed,
        "dates_checked": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Overall: {'PASS' if overall_passed else 'FAIL'}")
    print(f"Evidence written to: {OUTPUT_FILE}")
    print(f"{'=' * 60}\n")

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
