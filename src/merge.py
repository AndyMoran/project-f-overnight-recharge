"""
Merge pipeline — assembles all fetched data into the study dataset.
====================================================================

Reads:
    data/raw/b1610/b1610_raw.parquet          (from fetch_b1610.py)
    data/raw/windfor/windfor_raw.parquet       (from fetch_windfor.py)
    data/raw/tsdf/tsdf_raw.parquet             (from fetch_tsdf.py)
    data/raw/prices/system_prices_raw.parquet  (from fetch_prices.py)
    data/raw/prices/da_prices_raw.parquet      (from fetch_prices.py)
    data/short_duration_bmu_list.csv           (from build_bmu_list.py)
    data/bmu_mwh_manual.csv                    (manual MWh ratings)

Writes:
    data/processed/project_f_merged.parquet   (SP-level, one row per date×SP)
    data/processed/bmu_day_detail.parquet     (BMU-day detail for HS-7)

Columns in project_f_merged.parquet
-------------------------------------
    date                                    settlement date (daily grain)
    settlement_period                       SP 1-48
    P_imb                                   system sell price (£/MWh)
    DA_overnight_price_d                    DA price avg SP 1-14 on date (£/MWh)
    DA_wind_forecast_evening_d              WINDFOR avg SP 34-44 (MW)
    DA_demand_forecast_evening_d            TSDF avg SP 34-44 (MW)
    DA_IC_schedule_overnight_d              placeholder 0.0 (IC data TBD)
    evening_depletion_d                     fleet net discharge SP 34-44 / fleet MWh
    BESS_SD_fleet_operational_bmu_sps_expected  int (for HS-1)
    BESS_SD_fleet_operational_bmu_sps_observed  int (for HS-1)
    day_of_week_d                           0=Mon...6=Sun
    month_d                                 1-12

Usage
-----
    python src/merge.py
    python src/merge.py --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR  = PROJECT_ROOT / "data" / "raw"
OUT_DIR  = PROJECT_ROOT / "data" / "processed"

STUDY_START = pd.Timestamp("2024-01-01")
STUDY_END   = pd.Timestamp("2025-12-31")

EVENING_SPS   = list(range(34, 45))   # SP 34-44
OVERNIGHT_SPS = list(range(1, 15))    # SP 1-14


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found at {path}. "
            f"Run the corresponding fetcher first."
        )
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path, comment="#")
    logger.info(f"  Loaded {label}: {len(df):,} rows")
    return df


def _date_col(df: pd.DataFrame, candidates: list[str]) -> str:
    return next((c for c in candidates if c in df.columns), None)


# ---------------------------------------------------------------------------
# Step 1: B1610 → evening depletion and BMU coverage
# ---------------------------------------------------------------------------

def build_bess_treatment(
    b1610: pd.DataFrame,
    bmu_short: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    From B1610 and the short-duration BMU list, compute:
      - evening_depletion_d: fleet net discharge SP34-44 / fleet MWh
      - daily BMU coverage counts for HS-1
      - bmu_day_detail for HS-7

    Returns (daily_df, bmu_day_df).
    """
    # Normalise B1610 columns
    b1610 = b1610.copy()
    b1610.columns = [c.lower() for c in b1610.columns]

    bmu_col  = next((c for c in b1610.columns if c in ("bm_unit", "bmunit", "elexon_bmu_id")), None)
    date_col = next((c for c in b1610.columns if "settlement_date" in c), None)
    sp_col   = next((c for c in b1610.columns if "settlement_period" in c), None)
    qty_col  = next((c for c in b1610.columns if c in ("quantity", "generation_mw", "quantity_mw")), None)

    if not all([bmu_col, date_col, sp_col, qty_col]):
        raise ValueError(f"B1610 missing required columns. Found: {list(b1610.columns)}")

    b1610[date_col] = pd.to_datetime(b1610[date_col])
    b1610[qty_col]  = pd.to_numeric(b1610[qty_col], errors="coerce")

    # BMU list normalisation
    bmu_short = bmu_short.copy()
    bmu_short.columns = [c.lower() for c in bmu_short.columns]
    bmu_id_col = next(
        (c for c in bmu_short.columns if c in ("elexon_bmu_id", "bm_unit", "bmu_id")), None
    )
    mwh_col = next((c for c in bmu_short.columns if "rated_mwh" in c or c == "mwh"), None)
    mw_col  = next((c for c in bmu_short.columns if "rated_mw" in c), None)

    if bmu_id_col is None:
        raise ValueError(f"BMU list missing ID column. Found: {list(bmu_short.columns)}")

    bmu_set = set(bmu_short[bmu_id_col].astype(str).str.strip())

    # Fleet total nameplate MWh (sum across short-duration fleet)
    if mwh_col:
        fleet_mwh = pd.to_numeric(bmu_short[mwh_col], errors="coerce").sum()
    else:
        # Fallback: use 2h × rated_mw
        fleet_mw = pd.to_numeric(bmu_short[mw_col], errors="coerce").sum() if mw_col else 1.0
        fleet_mwh = fleet_mw * 2.0
        logger.warning(f"  No MWh column — using 2h × rated_mw = {fleet_mwh:.0f} MWh")

    logger.info(f"  Fleet nameplate MWh: {fleet_mwh:.0f}")
    logger.info(f"  Short-duration BMUs: {len(bmu_set)}")

    # Filter B1610 to short-duration BMUs
    b1610_sd = b1610[b1610[bmu_col].astype(str).str.strip().isin(bmu_set)].copy()
    logger.info(f"  B1610 rows after SD filter: {len(b1610_sd):,}")

    # Evening window: SP 34-44
    b1610_eve = b1610_sd[b1610_sd[sp_col].isin(EVENING_SPS)].copy()

    # Per-day: fleet net discharge in the evening window
    # quantity > 0 = generation (discharge); < 0 = consumption (charge)
    daily_discharge = (
        b1610_eve.groupby(b1610_eve[date_col].dt.date)[qty_col]
        .sum()   # positive = net discharge
        .reset_index()
    )
    daily_discharge.columns = ["date", "fleet_evening_net_discharge_mwh"]
    daily_discharge["date"] = pd.to_datetime(daily_discharge["date"])

    # Normalise by fleet MWh → treatment metric
    # Each SP is 30 min = 0.5h; quantity in MW → MWh per SP = MW × 0.5
    n_sps = len(EVENING_SPS)
    daily_discharge["fleet_evening_net_discharge_mwh"] *= 0.5  # MW → MWh
    daily_discharge["evening_depletion_d"] = (
        daily_discharge["fleet_evening_net_discharge_mwh"] / fleet_mwh
    ).clip(lower=0)   # clip to 0 — negative (net charging evening) → 0 depletion

    # HS-1: coverage counts
    # Expected: len(bmu_set) × 11 SPs per date
    expected_per_day = len(bmu_set) * n_sps
    observed_counts = (
        b1610_eve.groupby(b1610_eve[date_col].dt.date)[[bmu_col]]
        .count()
        .reset_index()
    )
    observed_counts.columns = ["date", "BESS_SD_fleet_operational_bmu_sps_observed"]
    observed_counts["date"] = pd.to_datetime(observed_counts["date"])
    observed_counts["BESS_SD_fleet_operational_bmu_sps_expected"] = expected_per_day

    daily = daily_discharge.merge(observed_counts, on="date", how="outer")

    # BMU-day detail for HS-7
    bmu_day = (
        b1610_eve.groupby([b1610_eve[date_col].dt.date, b1610_eve[bmu_col]])[qty_col]
        .sum()
        .reset_index()
    )
    bmu_day.columns = ["date", "bmu_id", "evening_discharge_mwh"]
    bmu_day["evening_discharge_mwh"] *= 0.5   # MW → MWh

    # Add nameplate MWh per BMU for HS-7
    if mwh_col:
        mwh_lookup = (
            bmu_short[[bmu_id_col, mwh_col]]
            .rename(columns={bmu_id_col: "bmu_id", mwh_col: "nameplate_mwh"})
        )
        mwh_lookup["nameplate_mwh"] = pd.to_numeric(mwh_lookup["nameplate_mwh"], errors="coerce")
        bmu_day = bmu_day.merge(mwh_lookup, on="bmu_id", how="left")
    else:
        bmu_day["nameplate_mwh"] = fleet_mwh / len(bmu_set)

    bmu_day["date"] = pd.to_datetime(bmu_day["date"])

    return daily, bmu_day


# ---------------------------------------------------------------------------
# Step 2: WINDFOR → evening wind forecast covariate
# ---------------------------------------------------------------------------

def build_wind_covariate(windfor: pd.DataFrame) -> pd.DataFrame:
    windfor = windfor.copy()
    windfor.columns = [c.lower() for c in windfor.columns]

    date_col = next((c for c in windfor.columns if "forecast_date" in c or "date" in c), None)
    sp_col   = next((c for c in windfor.columns if "settlement_period" in c), None)
    gen_col  = next((c for c in windfor.columns if "generation" in c or "mw" in c), None)

    if not all([date_col, sp_col, gen_col]):
        raise ValueError(f"WINDFOR missing columns. Found: {list(windfor.columns)}")

    windfor[date_col] = pd.to_datetime(windfor[date_col])
    windfor[gen_col]  = pd.to_numeric(windfor[gen_col], errors="coerce")

    evening = windfor[windfor[sp_col].isin(EVENING_SPS)]
    daily = (
        evening.groupby(evening[date_col].dt.date)[gen_col]
        .mean()
        .reset_index()
    )
    daily.columns = ["date", "DA_wind_forecast_evening_d"]
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


# ---------------------------------------------------------------------------
# Step 3: TSDF → evening demand forecast covariate
# ---------------------------------------------------------------------------

def build_demand_covariate(tsdf: pd.DataFrame) -> pd.DataFrame:
    tsdf = tsdf.copy()
    tsdf.columns = [c.lower() for c in tsdf.columns]

    date_col = next((c for c in tsdf.columns if "forecast_date" in c or "date" in c), None)
    sp_col   = next((c for c in tsdf.columns if "settlement_period" in c), None)
    dem_col  = next((c for c in tsdf.columns if "demand" in c), None)

    if not all([date_col, sp_col, dem_col]):
        raise ValueError(f"TSDF missing columns. Found: {list(tsdf.columns)}")

    tsdf[date_col] = pd.to_datetime(tsdf[date_col])
    tsdf[dem_col]  = pd.to_numeric(tsdf[dem_col], errors="coerce")

    evening = tsdf[tsdf[sp_col].isin(EVENING_SPS)]
    daily = (
        evening.groupby(evening[date_col].dt.date)[dem_col]
        .mean()
        .reset_index()
    )
    daily.columns = ["date", "DA_demand_forecast_evening_d"]
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


# ---------------------------------------------------------------------------
# Step 4: prices → P_imb and DA overnight price
# ---------------------------------------------------------------------------

def build_price_covariates(
    system_prices: pd.DataFrame,
    da_prices: pd.DataFrame,
) -> pd.DataFrame:
    # System prices: keep SP-level for P_imb
    sp_df = system_prices.copy()
    sp_df.columns = [c.lower() for c in sp_df.columns]
    sd_col   = next((c for c in sp_df.columns if "settlement_date" in c), None)
    sp_col   = next((c for c in sp_df.columns if "settlement_period" in c), None)
    sell_col = next((c for c in sp_df.columns if "sell" in c or "imbalance" in c or "system" in c), None)

    if not all([sd_col, sp_col, sell_col]):
        raise ValueError(f"System prices missing columns. Found: {list(sp_df.columns)}")

    sp_df[sd_col]   = pd.to_datetime(sp_df[sd_col])
    sp_df[sell_col] = pd.to_numeric(sp_df[sell_col], errors="coerce")
    p_imb = sp_df[[sd_col, sp_col, sell_col]].rename(columns={
        sd_col: "date", sp_col: "settlement_period", sell_col: "P_imb"
    })

    # DA overnight price: average SP 1-14 on settlement_date for day-ahead price
    da_df = da_prices.copy()
    da_df.columns = [c.lower() for c in da_df.columns]

    # da_sd   = next((c for c in da_df.columns if "settlement_date" in c), None)
    # da_sp   = next((c for c in da_df.columns if "settlement_period" in c), None)
    # da_px   = next((c for c in da_df.columns if "price" in c or "da" in c), None)

    da_sd = next((c for c in da_df.columns if "settlement_date" in c), None)

    da_px = next(
        (
            c for c in da_df.columns
            if c not in [da_sd]
            and ("price" in c or "gbp" in c or "value" in c)
        ),
        None
    )

   # DA prices are daily, not SP-level
    if all([da_sd, da_px]):
        da_df[da_sd] = pd.to_datetime(da_df[da_sd])
        da_df[da_px] = pd.to_numeric(da_df[da_px], errors="coerce")

        da_overnight = da_df[[da_sd, da_px]].drop_duplicates(subset=[da_sd]).rename(columns={
            da_sd: "date",
            da_px: "DA_overnight_price_d"
        })
    else:
        logger.warning("DA prices: could not identify columns — DA_overnight_price_d will be NaN")
        da_overnight = pd.DataFrame(columns=["date", "DA_overnight_price_d"])



    return p_imb, da_overnight


# ---------------------------------------------------------------------------
# Main merge
# ---------------------------------------------------------------------------

def run_merge() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw data...")
    b1610        = _load(RAW_DIR / "b1610" / "b1610_raw.parquet", "B1610")
    windfor      = _load(RAW_DIR / "windfor" / "windfor_raw.parquet", "WINDFOR")
    tsdf         = _load(RAW_DIR / "tsdf" / "tsdf_raw.parquet", "TSDF")
    sys_prices   = _load(RAW_DIR / "prices" / "system_prices_raw.parquet", "System prices")
    da_prices    = _load(RAW_DIR / "prices" / "da_prices_raw.parquet", "DA prices")
    bmu_short    = _load(PROJECT_ROOT / "data" / "short_duration_bmu_list.csv", "Short-duration BMU list")

    logger.info("Building BESS treatment metric and coverage...")
    daily_bess, bmu_day = build_bess_treatment(b1610, bmu_short)

    # daily_bess already has a 'date' column and evening_depletion_d
    # Rename for consistency with other covariates
    daily_bess = daily_bess.rename(columns={
        "evening_depletion_d": "bess_treatment_d"
    })

    logger.info("Building WINDFOR covariate...")
    wind_cov = build_wind_covariate(windfor)

    logger.info("Building TSDF demand covariate...")
    demand_cov = build_demand_covariate(tsdf)

    logger.info("Building price covariates...")
    p_imb_sp, da_overnight = build_price_covariates(sys_prices, da_prices)

    # DA overnight price is a daily series — it's the avg SP1-14 DA price
    # on day d (i.e. the overnight window of the night following day d).
    # We need it aligned to date = d for the matching covariate.
    # Note: pre-reg uses "overnight hours of day d+1" — this means the
    # settlement periods SP1-14 on date d+1. So we shift by 1 day.
    if not da_overnight.empty:
        da_overnight_shifted = da_overnight.copy()

        # Ensure date column exists
        if "date" not in da_overnight_shifted.columns:
            logger.warning("DA_overnight missing 'date' column — creating from settlement_date")
            # Try common fallback names
            for candidate in ["settlement_date", "delivery_date"]:
                if candidate in da_overnight_shifted.columns:
                    da_overnight_shifted["date"] = pd.to_datetime(da_overnight_shifted[candidate])
                    break

        # Now safely shift
        # da_overnight_shifted["date"] = da_overnight_shifted["date"] - pd.Timedelta(days=1)
        if "date" in da_overnight_shifted.columns:
            da_overnight_shifted["date"] = da_overnight_shifted["date"] - pd.Timedelta(days=1)
        else:
            logger.warning("DA_overnight_shifted has no 'date' column — skipping shift")

    else:
        da_overnight_shifted = da_overnight



    logger.info("Assembling SP-level merged dataset...")

    # Start with P_imb (SP-level spine)
    merged = p_imb_sp.copy()
    merged["date"] = pd.to_datetime(merged["date"])

    # Promote SP-level table to daily table
    if "settlement_date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["settlement_date"])
    elif "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"])
    else:
        raise KeyError("No settlement_date or date column found in merged SP-level table.")

    # Restrict to study period
    merged = merged[
        (merged["date"] >= STUDY_START) &
        (merged["date"] <= STUDY_END)
    ].copy()
    # Ensure all daily covariates have a 'date' column
    for df in [daily_bess, wind_cov, demand_cov, da_overnight]:
        if "date" not in df.columns:
            for candidate in ["settlement_date", "settlementdate", "delivery_date"]:
                if candidate in df.columns:
                    df["date"] = pd.to_datetime(df[candidate])
                    break
            else:
                raise KeyError(f"Covariate missing a usable date column: {df.columns}")

    # Merge daily covariates
    for cov_df, label in [
        (daily_bess, "BESS treatment"),
        (wind_cov, "WINDFOR"),
        (demand_cov, "TSDF"),
        (da_overnight_shifted, "DA overnight price"),
    ]:
        if not cov_df.empty:
            cov_df["date"] = pd.to_datetime(cov_df["date"])
            before = len(merged)
            merged = merged.merge(cov_df, on="date", how="left")
            logger.info(f"  After {label} merge: {len(merged):,} rows (was {before:,})")


    # Add IC schedule placeholder (to be fetched later if needed)
    if "DA_IC_schedule_overnight_d" not in merged.columns:
        merged["DA_IC_schedule_overnight_d"] = 0.0
        logger.info("  DA_IC_schedule_overnight_d set to 0.0 (placeholder)")

    # Calendar features
    merged["day_of_week_d"] = merged["date"].dt.dayofweek
    merged["month_d"]       = merged["date"].dt.month

    # Also build overnight_price_d_plus_1: DA price on SP1-14 of date d+1
    # (the outcome variable) — this is the same as DA_overnight_price_d
    # shifted back: the overnight price for night of d is the SP1-14 avg
    # of date d (not d+1). Need to clarify: pre-reg says "overnight hours
    # of day d+1 (SP 1-14)", meaning the night that follows the trading day.
    # So overnight_price_d_plus_1 is the avg SP1-14 DA price on date d+1.
    # Build outcome variable
    if not da_overnight.empty:
        outcome_map = da_overnight.rename(
            columns={"DA_overnight_price_d": "overnight_price_d_plus_1"}
        )
        outcome_map["date"] = outcome_map["date"] - pd.Timedelta(days=1)
        merged = merged.merge(
            outcome_map[["date", "overnight_price_d_plus_1"]],
            on="date",
            how="left"
        )


    # Sort and deduplicate
    merged = merged.sort_values(["date", "settlement_period"]).reset_index(drop=True)

    logger.info(f"\nFinal merged dataset: {len(merged):,} rows")
    logger.info(f"  Date range: {merged['date'].min().date()} → {merged['date'].max().date()}")
    logger.info(f"  Columns: {list(merged.columns)}")
    logger.info(f"  NaN summary:")
    nan_pct = merged.isna().mean()
    for col, pct in nan_pct.items():
        if pct > 0:
            logger.info(f"    {col}: {pct:.1%} missing")

    # Write outputs
    merged.to_parquet(OUT_DIR / "project_f_merged.parquet", index=False)
    logger.info(f"\nSaved → {OUT_DIR / 'project_f_merged.parquet'}")

    bmu_day["date"] = pd.to_datetime(bmu_day["date"])
    bmu_day = bmu_day[
        (bmu_day["date"] >= STUDY_START) & (bmu_day["date"] <= STUDY_END)
    ]
    bmu_day.to_parquet(OUT_DIR / "bmu_day_detail.parquet", index=False)
    logger.info(f"Saved → {OUT_DIR / 'bmu_day_detail.parquet'}")
    logger.info("\nMerge complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Project F merge pipeline.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
    )
    run_merge()


if __name__ == "__main__":
    main()
