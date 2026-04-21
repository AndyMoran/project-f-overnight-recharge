"""
Fetch TSDF (Transmission System Demand Forecast) for the study period.
=======================================================================

For each settlement date d in 2024-01-01 to 2025-12-31, fetches the
latest TSDF vintage with publish_time in [d-1T11:00Z, d-1T14:59Z],
as specified in the pre-registration §Variables (DEVIATIONS Entry 004).

Output
------
    data/raw/tsdf/tsdf_raw.parquet

    Columns:
        forecast_date       date
        settlement_period   int (1-48)
        publish_time        datetime (actual vintage used)
        demand_mw           float

Usage
-----
    python src/fetchers/fetch_tsdf.py
    python src/fetchers/fetch_tsdf.py --resume
    python src/fetchers/fetch_tsdf.py --start 2024-06-01 --end 2024-06-30
    python src/fetchers/fetch_tsdf.py --dry-run

Runtime: ~20-30 minutes (each day requires one API call covering the
4-hour publish window, returning ~10,000 records per call).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)

STUDY_START = date(2024, 1, 1)
STUDY_END   = date(2025, 12, 31)

# Vintage window: latest publish_time in [d-1T11:00Z, d-1T14:59Z]
WINDOW_FROM_HOUR = 11
WINDOW_TO_HOUR   = 14
WINDOW_TO_MINUTE = 59

OUTPUT_DIR  = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "tsdf"
OUTPUT_FILE = OUTPUT_DIR / "tsdf_raw.parquet"
MISSING_FILE = OUTPUT_DIR / "tsdf_missing_dates.csv"

RATE_LIMIT_SLEEP = 1.0   # TSDF returns large payloads; be gentle


def fetch_one_day(d_minus_1: date, api) -> tuple[pd.DataFrame, str | None]:
    """
    Fetch all TSDF records with publish_time in [d-1T11:00, d-1T14:59] UTC.
    Returns the records for forecast_date = d using the latest vintage,
    plus the actual publish_time string used.
    """
    from_str = f"{d_minus_1}T{WINDOW_FROM_HOUR:02d}:00:00"
    to_str   = f"{d_minus_1}T{WINDOW_TO_HOUR:02d}:{WINDOW_TO_MINUTE:02d}:00"

    try:
        resp = api.datasets_tsdf_get(
            publish_date_time_from=from_str,
            publish_date_time_to=to_str,
            format="json",
        )
    except Exception as e:
        logger.warning(f"  {d_minus_1}: API error — {e}")
        return pd.DataFrame(), None

    if not (hasattr(resp, "data") and resp.data):
        logger.warning(f"  {d_minus_1}: empty response")
        return pd.DataFrame(), None

    records = [r.to_dict() for r in resp.data]
    df = pd.DataFrame(records)

    if df.empty:
        return pd.DataFrame(), None

    # Normalise column names
    df.columns = [c.lower() for c in df.columns]

    # Find publish_time column
    pt_col = next((c for c in df.columns if "publish" in c), None)
    sd_col  = next((c for c in df.columns if "settlement_date" in c), None)
    sp_col  = next((c for c in df.columns if "settlement_period" in c), None)
    dem_col = next((c for c in df.columns if c in ("demand", "nd", "transmission_demand")), None)

    if not all([pt_col, sd_col, sp_col, dem_col]):
        logger.warning(f"  {d_minus_1}: missing columns — found {list(df.columns)}")
        return pd.DataFrame(), None

    # Use the latest publish_time vintage
    df[pt_col] = pd.to_datetime(df[pt_col], utc=True, errors="coerce")
    latest_pt = df[pt_col].max()
    df_latest = df[df[pt_col] == latest_pt].copy()

    # Filter to forecast_date = d (d_minus_1 + 1 day)
    forecast_date = d_minus_1 + timedelta(days=1)
    df_latest[sd_col] = pd.to_datetime(df_latest[sd_col], errors="coerce").dt.date
    df_day = df_latest[df_latest[sd_col] == forecast_date].copy()

    if df_day.empty:
        logger.debug(f"  {d_minus_1}: no records for forecast_date {forecast_date}")
        return pd.DataFrame(), None

    out = pd.DataFrame({
        "forecast_date":     df_day[sd_col],
        "settlement_period": pd.to_numeric(df_day[sp_col], errors="coerce"),
        "publish_time":      latest_pt.isoformat(),
        "demand_mw":         pd.to_numeric(df_day[dem_col], errors="coerce"),
    })
    return out.dropna(), latest_pt.isoformat()


def fetch_tsdf(
    start: date = STUDY_START,
    end: date = STUDY_END,
    resume: bool = False,
    dry_run: bool = False,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if resume and OUTPUT_FILE.exists():
        existing = pd.read_parquet(OUTPUT_FILE)
        already_done = set(pd.to_datetime(existing["forecast_date"]).dt.date)
        logger.info(f"Resuming: {len(already_done)} dates already fetched")
    else:
        existing = pd.DataFrame()
        already_done = set()

    all_dates = []
    d = start
    while d <= end:
        if d not in already_done:
            all_dates.append(d)
        d += timedelta(days=1)

    logger.info(f"TSDF fetch: {len(all_dates)} dates to fetch ({start} → {end})")

    if dry_run:
        for d in all_dates[:5]:
            logger.info(f"  Would fetch: d={d}, window={d - timedelta(days=1)}T11:00Z–14:59Z")
        return

    try:
        from elexonpy.api_client import ApiClient
        from elexonpy.api.datasets_api import DatasetsApi
        client = ApiClient()
        api = DatasetsApi(client)
    except ImportError as e:
        logger.error(f"elexonpy not available: {e}")
        sys.exit(1)

    all_frames = []
    missing_dates = []

    for i, d in enumerate(all_dates):
        if i % 50 == 0:
            logger.info(f"Progress: {i}/{len(all_dates)} — {d}")

        df_day, actual_pt = fetch_one_day(d - timedelta(days=1), api)

        if df_day.empty:
            missing_dates.append({"forecast_date": d, "reason": "no data in window"})
        else:
            all_frames.append(df_day)
            logger.debug(f"  {d}: {len(df_day)} SP rows (vintage {actual_pt})")

        time.sleep(RATE_LIMIT_SLEEP)

    new_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

    if not new_df.empty:
        new_df["forecast_date"] = pd.to_datetime(new_df["forecast_date"])

    combined = pd.concat(
        [f for f in [existing, new_df] if not f.empty], ignore_index=True
    ) if (not existing.empty or not new_df.empty) else pd.DataFrame()

    if not combined.empty:
        combined = combined.drop_duplicates(
            subset=["forecast_date", "settlement_period"]
        ).sort_values(["forecast_date", "settlement_period"])
        combined.to_parquet(OUTPUT_FILE, index=False)
        logger.info(f"Saved {len(combined)} rows → {OUTPUT_FILE}")

    if missing_dates:
        pd.DataFrame(missing_dates).to_csv(MISSING_FILE, index=False)
        logger.warning(f"{len(missing_dates)} missing dates → {MISSING_FILE}")

    logger.info("TSDF fetch complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch TSDF demand forecasts for the study period."
    )
    parser.add_argument("--start", type=date.fromisoformat, default=STUDY_START)
    parser.add_argument("--end",   type=date.fromisoformat, default=STUDY_END)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
    )
    fetch_tsdf(
        start=args.start, end=args.end,
        resume=args.resume, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
