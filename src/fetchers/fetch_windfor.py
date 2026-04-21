"""
Fetch WINDFOR day-ahead wind forecasts for the full study period.
==================================================================

Fetches the 12:30 UTC vintage on day d-1 for every settlement date d
in the study period (2024-01-01 to 2025-12-31), as specified in the
pre-registration §Variables:

    DA_wind_forecast_evening_d: WINDFOR 12:30 UTC vintage on day d-1,
    retrieved via /forecast/generation/wind/history?publishTime=<d-1>T12:30Z.
    SP 34-44 values are extracted and averaged to produce the covariate.

Output
------
    data/raw/windfor/windfor_raw.parquet

    Columns:
        forecast_date       date  (the date being forecast, i.e. d)
        settlement_period   int   (1-48)
        publish_time        datetime (the actual vintage returned, ~11:30 or 12:30)
        generation          float (MW)

    One row per (forecast_date, settlement_period) using the 12:30 vintage.
    If the 12:30 vintage is unavailable for a given d-1, the nearest
    available vintage within ±90 minutes is used (per DEVIATIONS Entry 003).
    Days where no vintage is available within ±90 min are written to
    data/raw/windfor/windfor_missing_dates.csv.

Usage
-----
    python src/fetchers/fetch_windfor.py

    # Dry run (print dates, don't fetch)
    python src/fetchers/fetch_windfor.py --dry-run

    # Custom date range
    python src/fetchers/fetch_windfor.py --start 2024-06-01 --end 2024-06-30

    # Resume interrupted fetch (skips dates already in output)
    python src/fetchers/fetch_windfor.py --resume

Runtime: ~15-30 minutes for the full 2024-2025 period (730 API calls).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STUDY_START = date(2024, 1, 1)
STUDY_END   = date(2025, 12, 31)

TARGET_PUBLISH_HOUR   = 12
TARGET_PUBLISH_MINUTE = 30
TOLERANCE_MINUTES     = 90   # per DEVIATIONS Entry 003

OUTPUT_DIR  = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "windfor"
OUTPUT_FILE = OUTPUT_DIR / "windfor_raw.parquet"
MISSING_FILE = OUTPUT_DIR / "windfor_missing_dates.csv"

RATE_LIMIT_SLEEP = 0.5   # seconds between API calls


# ---------------------------------------------------------------------------
# Fetch one vintage
# ---------------------------------------------------------------------------

def fetch_one_day(
    d_minus_1: date,
    api,
) -> tuple[list[dict], str | None]:
    """
    Fetch WINDFOR for d-1 at the 12:30 UTC vintage.

    Returns (records, actual_publish_time_str).
    records is empty if nothing usable was returned.
    """
    from datetime import datetime, timezone, timedelta

    target_dt = datetime(
        d_minus_1.year, d_minus_1.month, d_minus_1.day,
        TARGET_PUBLISH_HOUR, TARGET_PUBLISH_MINUTE,
        tzinfo=timezone.utc,
    )
    tolerance = timedelta(minutes=TOLERANCE_MINUTES)

    try:
        resp = api.forecast_generation_wind_history_get(
            publish_time=target_dt,
            format="json",
        )
    except Exception as e:
        logger.warning(f"  {d_minus_1}: API error — {e}")
        return [], None

    if not (hasattr(resp, "data") and resp.data):
        logger.warning(f"  {d_minus_1}: empty response")
        return [], None

    records = [r.to_dict() for r in resp.data]

    # Check the returned publish_time is within tolerance
    first_pt = records[0].get("publish_time") or records[0].get("publishTime")
    if first_pt is None:
        logger.warning(f"  {d_minus_1}: no publish_time in response")
        return [], None

    if isinstance(first_pt, datetime):
        pt_dt = first_pt if first_pt.tzinfo else first_pt.replace(tzinfo=timezone.utc)
    else:
        from datetime import datetime as dt
        pt_str = str(first_pt).replace("Z", "+00:00")
        try:
            pt_dt = datetime.fromisoformat(pt_str)
        except ValueError:
            logger.warning(f"  {d_minus_1}: cannot parse publish_time {first_pt!r}")
            return [], None

    delta = pt_dt - target_dt
    if delta > tolerance or delta < -tolerance:
        logger.warning(
            f"  {d_minus_1}: publish_time {pt_dt} is {delta} from target "
            f"(outside ±{TOLERANCE_MINUTES} min) — skipping"
        )
        return [], None

    return records, pt_dt.isoformat()


# ---------------------------------------------------------------------------
# Parse records into rows
# ---------------------------------------------------------------------------

def parse_records(
    records: list[dict],
    forecast_date: date,
    actual_publish_time: str,
) -> list[dict]:
    """
    Convert raw API records for one day into flat rows.
    forecast_date is the date being forecast (d), not d-1.
    """
    from datetime import datetime, timezone

    rows = []
    for r in records:
        # start_time tells us which settlement period this is
        st = r.get("start_time") or r.get("startTime")
        gen = r.get("generation")

        if st is None or gen is None:
            continue

        if isinstance(st, datetime):
            st_dt = st if st.tzinfo else st.replace(tzinfo=timezone.utc)
        else:
            st_str = str(st).replace("Z", "+00:00")
            try:
                from datetime import datetime as dt
                st_dt = datetime.fromisoformat(st_str)
            except ValueError:
                continue

        # Only keep rows for forecast_date (the record may cover d+1, d+2, etc.)
        if st_dt.date() != forecast_date:
            continue

        # Derive settlement period from start_time
        # SP 1 starts at 00:00, SP 2 at 00:30, ..., SP 34 at 16:30
        midnight = st_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        minutes_since_midnight = int((st_dt - midnight).total_seconds() / 60)
        sp = minutes_since_midnight // 30 + 1

        rows.append({
            "forecast_date": forecast_date,
            "settlement_period": sp,
            "publish_time": actual_publish_time,
            "generation_mw": float(gen),
        })

    return rows


# ---------------------------------------------------------------------------
# Main fetch loop
# ---------------------------------------------------------------------------

def fetch_windfor(
    start: date = STUDY_START,
    end: date = STUDY_END,
    resume: bool = False,
    dry_run: bool = False,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing data if resuming
    if resume and OUTPUT_FILE.exists():
        existing = pd.read_parquet(OUTPUT_FILE)
        already_done = set(pd.to_datetime(existing["forecast_date"]).dt.date)
        logger.info(f"Resuming: {len(already_done)} dates already fetched")
    else:
        existing = pd.DataFrame()
        already_done = set()

    # Build date list: d iterates over forecast dates (d), we fetch d-1
    all_dates = []
    d = start
    while d <= end:
        if d not in already_done:
            all_dates.append(d)
        d += timedelta(days=1)

    logger.info(
        f"WINDFOR fetch: {len(all_dates)} dates to fetch "
        f"({start} → {end}, {len(already_done)} already done)"
    )

    if dry_run:
        logger.info("Dry run — not fetching.")
        for d in all_dates[:5]:
            logger.info(f"  Would fetch: d={d}, publishTime={d - timedelta(days=1)}T12:30Z")
        if len(all_dates) > 5:
            logger.info(f"  ... and {len(all_dates) - 5} more")
        return

    # Initialise elexonpy
    try:
        from elexonpy.api_client import ApiClient
        from elexonpy.api.generation_forecast_api import GenerationForecastApi
        client = ApiClient()
        api = GenerationForecastApi(client)
    except ImportError as e:
        logger.error(f"elexonpy not available: {e}")
        sys.exit(1)

    all_rows = []
    missing_dates = []

    for i, d in enumerate(all_dates):
        d_minus_1 = d - timedelta(days=1)

        if i % 50 == 0:
            logger.info(f"Progress: {i}/{len(all_dates)} — fetching {d} (publish {d_minus_1}T12:30Z)")

        records, actual_pt = fetch_one_day(d_minus_1, api)

        if not records:
            missing_dates.append({"forecast_date": d, "d_minus_1": d_minus_1, "reason": "no data"})
            logger.debug(f"  {d}: missing")
        else:
            rows = parse_records(records, d, actual_pt)
            if rows:
                all_rows.extend(rows)
                logger.debug(f"  {d}: {len(rows)} SP rows (publish_time={actual_pt})")
            else:
                missing_dates.append({
                    "forecast_date": d, "d_minus_1": d_minus_1,
                    "reason": "records returned but none matched forecast_date"
                })

        time.sleep(RATE_LIMIT_SLEEP)

    # Combine with existing and write
    new_df = pd.DataFrame(all_rows)
    if not new_df.empty:
        new_df["forecast_date"] = pd.to_datetime(new_df["forecast_date"])

    if not existing.empty and not new_df.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["forecast_date", "settlement_period"])
    elif not existing.empty:
        combined = existing
    else:
        combined = new_df

    if not combined.empty:
        combined = combined.sort_values(["forecast_date", "settlement_period"])
        combined.to_parquet(OUTPUT_FILE, index=False)
        logger.info(f"Saved {len(combined)} rows to {OUTPUT_FILE}")

    if missing_dates:
        missing_df = pd.DataFrame(missing_dates)
        missing_df.to_csv(MISSING_FILE, index=False)
        logger.warning(
            f"{len(missing_dates)} dates missing — see {MISSING_FILE}"
        )

    logger.info("WINDFOR fetch complete.")
    logger.info(f"  Dates fetched:  {len(all_rows) > 0 and len(new_df['forecast_date'].unique()) if not new_df.empty else 0}")
    logger.info(f"  Dates missing:  {len(missing_dates)}")
    logger.info(f"  Output:         {OUTPUT_FILE}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch WINDFOR 12:30 UTC vintages for the study period."
    )
    parser.add_argument("--start", type=date.fromisoformat, default=STUDY_START)
    parser.add_argument("--end",   type=date.fromisoformat, default=STUDY_END)
    parser.add_argument("--resume", action="store_true",
                        help="Skip dates already present in output file.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print dates without fetching.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
    )

    fetch_windfor(
        start=args.start,
        end=args.end,
        resume=args.resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
