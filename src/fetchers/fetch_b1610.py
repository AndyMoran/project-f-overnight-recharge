"""
B1610 fetcher: Actual Generation Output Per Generation Unit.
============================================================

Fetches half-hourly metered volume (MWh) from Elexon BMRS B1610, filtered
to the Project F BESS master list. Follows the resume-on-failure monthly-
chunk pattern established by the IV project's `fetch_wind_history_patch`.

Endpoint
--------
Elexon Insights `/datasets/B1610`, accessed via `elexonpy`. Each call
returns all BM units (or a specified subset) for a single
(settlement_date, settlement_period). Publication is 5 days after the
operational period via the II Settlement Run, refreshed by subsequent
runs.

Row schema returned by the API
------------------------------
Each record contains:
  - dataset                    = "B1610"
  - psr_type                   "Generation" / "Battery Storage" / ...
  - bm_unit                    Elexon BMU ID (e.g. "T_DRAXX-1")
  - national_grid_bm_unit_id   NG's alt ID
  - settlement_date            YYYY-MM-DD
  - settlement_period          1..48 (or 46/50 on DST days)
  - half_hour_end_time         UTC, end of the 30-min window — canonical
  - quantity                   metered MWh (signed)

Two notes on the schema worth flagging:

  * `half_hour_end_time` is the END of the 30-minute window. Project F's
    convention (shared with WINDFOR) uses `start_time` at the beginning
    of the window. This fetcher converts once at load time by
    subtracting 30 minutes, producing a `start_time` column that is
    tz-aware UTC and consistent with the rest of the pipeline. The
    original `half_hour_end_time` is preserved for audit.

  * `half_hour_end_time` is authoritative. We do NOT recompute times
    from (settlement_date, settlement_period) — that's the DST bug
    documented in the IV project's DEVIATIONS §4 and refused by Project
    F's `src/utils/time_conventions.py`. See that module for background.

Call volume and runtime
-----------------------
48 SPs × ~730 days = ~35,000 API calls to cover the full train + test
period. At 0.15s sleep between calls, network sleep alone is ~90
minutes; real wall clock is typically 2-3 hours depending on latency.
The fetcher is fully resumable — if it fails mid-run, rerunning skips
completed monthly chunks.

Preflight
---------
Before committing to the full run, call with `preflight_only=True` to
make a single sample API call and verify:
  (i)   The BMU list loads and has ≥ 1 member.
  (ii)  The API responds 200 with a non-empty payload for a recent date.
  (iii) At least one of our target BMUs appears in the response.
  (iv)  The row schema matches what this module expects.

This catches setup errors (wrong BMU list, API schema change, missing
credentials) in 30 seconds rather than 3 hours.

B1610 data revisions
--------------------
B1610 is refreshed by subsequent Settlement Runs. The same (date, SP)
queried at different times may return different values. For
reproducibility the fetcher saves a `fetched_at` UTC timestamp alongside
each row and NEVER re-fetches a completed chunk. If you need to refresh,
delete the specific chunk files and rerun; the combined parquet is
regenerated on every run from whichever chunks are present.

Publication delay
-----------------
Data is published 5 days after the operational period. Fetching a date
within 5 days of "now" will typically return zero rows or a partial
response. The fetcher warns if `end_date` is within `publication_delay_days`
of today (default 5) but does not refuse — you may want to deliberately
probe the publication window.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SETTLEMENT_PERIODS_PER_DAY = 48  # standard day; DST days handled by API
DEFAULT_RATE_LIMIT_SLEEP = 0.15  # seconds between API calls
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BASE_WAIT = 5  # seconds; wait = base * attempt


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PreflightResult:
    """
    Output of the preflight check.

    Attributes
    ----------
    ok : True iff every check passed.
    checks : dict of individual check-name → bool.
    sample_n_records : how many records the sample call returned.
    sample_n_matching_bmus : how many of our BESS BMUs appeared in the
        sample response.
    n_bmus_loaded : size of the BMU list.
    sample_record : first record from the sample call, as a dict.
    errors : list of human-readable error messages for any failing check.
    """
    ok: bool
    checks: dict
    sample_n_records: int
    sample_n_matching_bmus: int
    n_bmus_loaded: int
    sample_record: Optional[dict]
    errors: list


# ---------------------------------------------------------------------------
# BMU list loading
# ---------------------------------------------------------------------------

def _load_bmu_list(path: Path) -> list[str]:
    """
    Load the BESS BMU master list.

    Expected schema: a CSV with a column that is plausibly the BMU ID
    (`bmu_id`, `bm_unit`, `elexon_bmu_id`, or similar). The function uses
    the same defensive column-lookup pattern used in the IV project's
    loaders — find the first matching column name.

    Returns a list of BMU ID strings, deduplicated but order-preserving.

    Raises
    ------
    FileNotFoundError : if the CSV doesn't exist.
    ValueError : if no plausible BMU-ID column is found.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"BMU list not found at {path}. Project F requires the 219-BMU "
            f"master list to be committed before B1610 fetching. See "
            f"PROJECT_F_PRE_REGISTRATION_v3.md §Data sources."
        )

    df = pd.read_csv(path)
    candidate_cols = [
        "bmu_id", "bm_unit", "elexon_bmu_id", "elexon_bm_unit",
        "bmUnit", "bmuId", "BMU_ID",
    ]
    col = next((c for c in candidate_cols if c in df.columns), None)
    if col is None:
        raise ValueError(
            f"BMU list {path} has no plausible BMU-ID column. "
            f"Expected one of: {candidate_cols}. Got: {list(df.columns)}"
        )

    ids = df[col].dropna().astype(str).str.strip()
    ids = ids[ids != ""]
    # Preserve order, dedupe
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)

    if not out:
        raise ValueError(f"BMU list {path} yielded zero IDs after cleaning.")

    logger.info(f"Loaded {len(out)} BMU IDs from {path} (column: {col!r})")
    return out


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _make_api_client():
    """
    Construct an elexonpy DatasetsApi client. Imported here so the module
    doesn't require elexonpy at import time — only at fetch time.
    """
    try:
        from elexonpy.api_client import ApiClient
        from elexonpy.api.datasets_api import DatasetsApi
    except ImportError as e:
        raise ImportError(
            "elexonpy is required for B1610 fetching. "
            "Install via `pip install elexonpy>=0.0.12` or add to "
            "requirements.txt."
        ) from e

    return DatasetsApi(ApiClient())


def _fetch_one_sp(
    api,
    settlement_date: str,
    settlement_period: int,
    bmu_list: list[str],
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_base_wait: int = DEFAULT_RETRY_BASE_WAIT,
) -> list[dict]:
    """
    Fetch B1610 for one (settlement_date, settlement_period) with retry
    on transient failure.

    Returns a list of record dicts (possibly empty). Raises RuntimeError
    after exhausting retries.
    """
    last_err = None
    for attempt in range(retry_attempts):
        try:
            resp = api.datasets_b1610_get(
                settlement_date=settlement_date,
                settlement_period=settlement_period,
                bm_unit=bmu_list,
                format="json",
            )
            if hasattr(resp, "data") and resp.data:
                return [r.to_dict() for r in resp.data]
            return []
        except Exception as e:
            last_err = e
            if attempt < retry_attempts - 1:
                wait = retry_base_wait * (attempt + 1)
                logger.warning(
                    f"  B1610 {settlement_date} SP {settlement_period} "
                    f"attempt {attempt+1}: {e} — retry in {wait}s"
                )
                time.sleep(wait)

    raise RuntimeError(
        f"B1610 {settlement_date} SP {settlement_period}: all "
        f"{retry_attempts} attempts failed. Last error: {last_err}"
    )


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight(
    bmu_list_path: str | Path,
    sample_date: Optional[str] = None,
    sample_period: int = 34,  # SP 34 = 16:30 UTC in winter, a sensible "evening" SP
) -> PreflightResult:
    """
    Run all four preflight checks without committing to the full fetch.

    If sample_date is None, uses (today - 10 days) to land comfortably
    inside the 5-day publication delay window and well within typical
    retention.

    Parameters
    ----------
    bmu_list_path : path to the BESS BMU master list CSV.
    sample_date : YYYY-MM-DD; defaults to 10 days before today UTC.
    sample_period : which SP to query (default 34, evening window).

    Returns
    -------
    PreflightResult
    """
    errors: list[str] = []
    checks = {
        "bmu_list_loads": False,
        "api_call_succeeds": False,
        "at_least_one_matching_bmu": False,
        "schema_has_required_fields": False,
    }
    n_bmus = 0
    sample_n = 0
    sample_n_matching = 0
    sample_record: Optional[dict] = None

    # 1. Load BMU list.
    try:
        bmu_list = _load_bmu_list(Path(bmu_list_path))
        n_bmus = len(bmu_list)
        checks["bmu_list_loads"] = True
    except Exception as e:
        errors.append(f"bmu_list_loads: {e}")
        return PreflightResult(
            ok=False, checks=checks, sample_n_records=0,
            sample_n_matching_bmus=0, n_bmus_loaded=0,
            sample_record=None, errors=errors,
        )

    # 2. Resolve sample date.
    if sample_date is None:
        sample_date = (
            datetime.now(timezone.utc) - timedelta(days=10)
        ).strftime("%Y-%m-%d")

    # 3. One sample API call.
    try:
        api = _make_api_client()
        records = _fetch_one_sp(
            api, sample_date, sample_period, bmu_list,
            retry_attempts=1,  # preflight: no retries, fail fast
        )
        checks["api_call_succeeds"] = True
        sample_n = len(records)
    except Exception as e:
        errors.append(f"api_call_succeeds: {e}")
        return PreflightResult(
            ok=False, checks=checks, sample_n_records=0,
            sample_n_matching_bmus=0, n_bmus_loaded=n_bmus,
            sample_record=None, errors=errors,
        )

    if sample_n == 0:
        errors.append(
            f"api_call_succeeds: sample call to {sample_date} SP "
            f"{sample_period} returned zero records. Either the date is "
            f"within the 5-day publication delay, or all target BMUs had "
            f"zero output that SP."
        )
        # Not strictly a failure — zero records is a valid B1610 response.
        # But treat it as a preflight failure because we can't verify the
        # downstream checks without data.
        return PreflightResult(
            ok=False, checks=checks, sample_n_records=0,
            sample_n_matching_bmus=0, n_bmus_loaded=n_bmus,
            sample_record=None, errors=errors,
        )

    sample_record = records[0]

    # 4. At least one record corresponds to a target BMU (the bm_unit
    # filter should guarantee this, but verify).
    bmu_ids_in_response = {
        r.get("bm_unit") for r in records
    }
    bmu_ids_in_response.discard(None)
    sample_n_matching = len(bmu_ids_in_response & set(bmu_list))
    if sample_n_matching > 0:
        checks["at_least_one_matching_bmu"] = True
    else:
        errors.append(
            "at_least_one_matching_bmu: no records in sample match any "
            "BMU in our list. This is unexpected given the bm_unit filter."
        )

    # 5. Schema check.
    required_fields = {
        "bm_unit", "settlement_date", "settlement_period",
        "half_hour_end_time", "quantity",
    }
    missing = required_fields - set(sample_record.keys())
    if not missing:
        checks["schema_has_required_fields"] = True
    else:
        errors.append(
            f"schema_has_required_fields: sample record missing fields: "
            f"{missing}. Available: {list(sample_record.keys())}"
        )

    ok = all(checks.values())
    return PreflightResult(
        ok=ok,
        checks=checks,
        sample_n_records=sample_n,
        sample_n_matching_bmus=sample_n_matching,
        n_bmus_loaded=n_bmus,
        sample_record={k: str(v) for k, v in sample_record.items()},
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Monthly chunk fetching
# ---------------------------------------------------------------------------

def _fetch_month(
    api,
    month_start: pd.Timestamp,
    month_end_inclusive: pd.Timestamp,
    bmu_list: list[str],
    rate_limit_sleep: float,
    retry_attempts: int,
    retry_base_wait: int,
) -> pd.DataFrame:
    """
    Fetch B1610 for every SP in every day in [month_start, month_end_inclusive].

    Returns a DataFrame of all records for the month, with a `fetched_at`
    column added per-row. Empty DataFrame if no records produced.
    """
    records: list[dict] = []
    fetched_at = datetime.now(timezone.utc).isoformat()

    day = month_start
    while day <= month_end_inclusive:
        date_str = day.strftime("%Y-%m-%d")
        # Standard 48 SPs per day — Elexon handles DST 46/50 cases on
        # the server side. Requesting SP 49 on a normal day returns
        # zero rows, which is fine.
        for sp in range(1, SETTLEMENT_PERIODS_PER_DAY + 1):
            try:
                rows = _fetch_one_sp(
                    api, date_str, sp, bmu_list,
                    retry_attempts=retry_attempts,
                    retry_base_wait=retry_base_wait,
                )
            except RuntimeError as e:
                logger.error(f"  Giving up on {date_str} SP {sp}: {e}")
                rows = []

            if rows:
                for r in rows:
                    r["fetched_at"] = fetched_at
                records.extend(rows)

            time.sleep(rate_limit_sleep)

        day += timedelta(days=1)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main fetcher
# ---------------------------------------------------------------------------

def fetch_b1610(
    start_date: str,
    end_date: str,
    bmu_list_path: str | Path,
    chunks_dir: str | Path = "data/raw/elexon/b1610_chunks",
    combined_path: str | Path = "data/raw/elexon/b1610_all.parquet",
    preflight_only: bool = False,
    publication_delay_days: int = 5,
    rate_limit_sleep: float = DEFAULT_RATE_LIMIT_SLEEP,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_base_wait: int = DEFAULT_RETRY_BASE_WAIT,
) -> pd.DataFrame:
    """
    Fetch B1610 for all (date, SP) in [start_date, end_date], filtered to
    the BMUs in `bmu_list_path`.

    Monthly chunks are saved to `chunks_dir` for resume-on-failure. After
    all months are present, the chunks are concatenated, deduped, and
    saved to `combined_path`.

    Parameters
    ----------
    start_date, end_date : "YYYY-MM-DD", inclusive.
    bmu_list_path : path to the BESS BMU master list CSV.
    chunks_dir : directory for monthly parquet chunks.
    combined_path : final combined parquet output path.
    preflight_only : if True, run preflight only and return empty DF.
    publication_delay_days : warn if end_date is within this many days
        of today. Default 5 (B1610 publication delay).
    rate_limit_sleep, retry_attempts, retry_base_wait : knobs for
        network behaviour.

    Returns
    -------
    pd.DataFrame : combined B1610 with columns:
        bm_unit, national_grid_bm_unit_id, psr_type, settlement_date,
        settlement_period, half_hour_end_time (UTC, end-of-window),
        start_time (UTC, start-of-window — derived from half_hour_end_time),
        quantity (MWh), fetched_at (ISO timestamp of fetch).

    Raises
    ------
    FileNotFoundError : if bmu_list_path doesn't exist.
    RuntimeError : if preflight fails and preflight_only=False (refuse
        to commit to a multi-hour fetch with a broken setup).
    """
    chunks_dir = Path(chunks_dir)
    combined_path = Path(combined_path)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    combined_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Preflight --------------------------------------------------------
    logger.info("=" * 60)
    logger.info("B1610 fetcher — preflight")
    logger.info("=" * 60)
    pf = preflight(bmu_list_path)
    for name, ok in pf.checks.items():
        logger.info(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    for err in pf.errors:
        logger.warning(f"  {err}")

    if not pf.ok:
        if preflight_only:
            logger.info("Preflight returned non-OK. Returning without fetch.")
            return pd.DataFrame()
        raise RuntimeError(
            "B1610 preflight failed. Review errors above and fix before "
            "committing to the full fetch. See the module docstring for "
            "what each check verifies."
        )

    logger.info(
        f"  Preflight OK: {pf.n_bmus_loaded} BMUs, sample call returned "
        f"{pf.sample_n_records} records ({pf.sample_n_matching_bmus} "
        f"matched target BMUs)."
    )

    if preflight_only:
        logger.info("Preflight-only mode: exiting without full fetch.")
        return pd.DataFrame()

    # ---- Publication-delay warning ----------------------------------------
    end_ts = pd.Timestamp(end_date, tz="UTC")
    today = pd.Timestamp.now(tz="UTC").normalize()
    delay_warning_threshold = today - pd.Timedelta(days=publication_delay_days)
    if end_ts > delay_warning_threshold:
        logger.warning(
            f"end_date {end_date} is within {publication_delay_days} days "
            f"of today. B1610 is published 5 days after the operational "
            f"period, so the most recent days may return zero/partial "
            f"records."
        )

    # ---- Fetch by month ---------------------------------------------------
    bmu_list = _load_bmu_list(Path(bmu_list_path))
    api = _make_api_client()

    start_ts = pd.Timestamp(start_date, tz="UTC").normalize()
    end_ts = end_ts.normalize()

    total_months = (
        (end_ts.year - start_ts.year) * 12
        + (end_ts.month - start_ts.month) + 1
    )
    month_idx = 0
    current = start_ts

    logger.info("=" * 60)
    logger.info(f"B1610 fetch: {start_date} → {end_date} ({total_months} months)")
    logger.info("=" * 60)

    while current <= end_ts:
        month_idx += 1
        month_str = current.strftime("%Y-%m")
        chunk_file = chunks_dir / f"b1610_{month_str}.parquet"
        month_end_inclusive = min(
            (current + pd.offsets.MonthEnd(0)).normalize().tz_localize(None).tz_localize("UTC"),
            end_ts,
        )

        if chunk_file.exists():
            logger.info(
                f"[{month_idx}/{total_months}] B1610 {month_str}: cached, skipping"
            )
            current = (month_end_inclusive + pd.Timedelta(days=1)).normalize()
            continue

        logger.info(
            f"[{month_idx}/{total_months}] B1610 {month_str}: "
            f"fetching {month_end_inclusive.day} days..."
        )

        df_month = _fetch_month(
            api,
            current,
            month_end_inclusive,
            bmu_list,
            rate_limit_sleep=rate_limit_sleep,
            retry_attempts=retry_attempts,
            retry_base_wait=retry_base_wait,
        )

        if not df_month.empty:
            df_month.to_parquet(chunk_file, index=False)
            logger.info(
                f"  B1610 {month_str}: saved {len(df_month):,} rows → {chunk_file}"
            )
        else:
            logger.warning(f"  B1610 {month_str}: no records retrieved")

        current = (month_end_inclusive + pd.Timedelta(days=1)).normalize()

    # ---- Combine ----------------------------------------------------------
    chunks = sorted(chunks_dir.glob("b1610_*.parquet"))
    if not chunks:
        logger.error("B1610 fetch: no chunks produced")
        return pd.DataFrame()

    logger.info(f"Combining {len(chunks)} monthly chunks...")
    df = pd.concat([pd.read_parquet(f) for f in chunks], ignore_index=True)

    before = len(df)
    df = df.drop_duplicates(subset=["bm_unit", "settlement_date", "settlement_period"])
    logger.info(f"  B1610: {before:,} → {len(df):,} rows after dedup")

    # Type hygiene and canonical start_time.
    # half_hour_end_time is authoritative UTC (end-of-window).
    # start_time = half_hour_end_time - 30 min, per Project F convention.
    df["half_hour_end_time"] = pd.to_datetime(df["half_hour_end_time"], utc=True)
    df["start_time"] = df["half_hour_end_time"] - pd.Timedelta(minutes=30)
    df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date.astype(str)
    df["settlement_period"] = pd.to_numeric(
        df["settlement_period"], errors="coerce"
    ).astype("Int64")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)

    df.to_parquet(combined_path, index=False)
    logger.info(f"✓ B1610 combined: {len(df):,} rows → {combined_path}")

    # Diagnostic
    n_bmus = df["bm_unit"].nunique()
    date_range = f"{df['settlement_date'].min()} → {df['settlement_date'].max()}"
    logger.info(f"  Coverage: {n_bmus} unique BMUs, dates {date_range}")
    q_nonnull = df["quantity"].dropna()
    if len(q_nonnull) > 0:
        logger.info(
            f"  Quantity stats (MWh): mean={q_nonnull.mean():.2f}, "
            f"min={q_nonnull.min():.2f}, max={q_nonnull.max():.2f}, "
            f"pct_negative={(q_nonnull < 0).mean():.1%}"
        )

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Fetch B1610 metered volumes for Project F BESS fleet."
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--bmu-list",
        default="data/bess_master_list_219.csv",
        help="Path to BESS BMU master list CSV",
    )
    parser.add_argument(
        "--chunks-dir",
        default="data/raw/elexon/b1610_chunks",
        help="Directory for monthly chunk files",
    )
    parser.add_argument(
        "--combined-path",
        default="data/raw/elexon/b1610_all.parquet",
        help="Final combined parquet output path",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run preflight checks only, don't start the full fetch",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
    )

    fetch_b1610(
        start_date=args.start,
        end_date=args.end,
        bmu_list_path=args.bmu_list,
        chunks_dir=args.chunks_dir,
        combined_path=args.combined_path,
        preflight_only=args.preflight_only,
    )


if __name__ == "__main__":
    _main()
