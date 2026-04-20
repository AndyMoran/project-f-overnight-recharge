"""
Build the Project F BESS master list.
======================================

Constructs `data/bess_master_list_219.csv` and
`data/short_duration_bmu_list.csv`.

How BESS units are identified in the Elexon reference data
-----------------------------------------------------------
After inspecting `bmu_reference.parquet`, the fuel-type structure is:

  fuelType = "OTHER"  →  74 units: confirmed BESS + a handful of solar
                          farms (T_CLVHS, T_LARKS, T_SUTBS) and virtual
                          aggregator units (V__ prefix, 0 MW).
  fuelType = NaN      →  2,395 units: all embedded behind-the-meter
                          assets. Most have no name that identifies them
                          as BESS. The ones that are BESS are identifiable
                          by bmUnitName keywords (bess, batter, stor).
  fuelType = "PS"     →  16 units: pumped-storage hydro. Excluded.
  Everything else     →  CCGT, WIND, NUCLEAR, etc. Excluded.

Identification rules applied (in order of reliability):
  1. fuelType == "OTHER" AND generationCapacity >= MIN_MW AND
     NOT a known non-BESS (solar farms by name, 0-MW virtual units)
  2. fuelType is NaN AND bmUnitName contains BESS keywords

After identification, the list is cross-referenced against BOALF to
confirm activity during the study period (2024-2025).

Why not just use the BOALF list?
----------------------------------
`boalf_bess.parquet` from the IV project contains ALL BMU acceptances,
not just BESS — the filename is misleading. It includes CCGTs, wind
farms, etc. It cannot be used as a BESS filter.

MWh ratings
-----------
Elexon does not publish MWh ratings. They must be looked up manually.
Run this script first to produce `data/bmu_exclusions.csv` (sorted by
rated_mw descending), then look up MWh for the largest units first and
add them to `data/bmu_mwh_manual.csv`. Re-run to apply the ≤2h filter.

Sources for MWh ratings:
  - NESO storage tracker: neso.energy/data-portal (search "battery storage")
  - Planning applications: search BMU name + local authority
  - Developer websites: Gore Street, Zenobe, Gresham House, Harmony Energy
  - Modo Energy (if subscription available)

Usage
-----
    python prelock_checks/build_bmu_list.py
    python prelock_checks/build_bmu_list.py --manual-mwh data/bmu_mwh_manual.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHORT_DURATION_RATIO_MAX = 2.0

IV_ROOT = Path(__file__).resolve().parent.parent.parent / "depletion-multiplier-iv"
DEFAULT_BMU_REF = IV_ROOT / "data" / "raw" / "elexon" / "bmu_reference.parquet"
DEFAULT_BOALF   = IV_ROOT / "data" / "raw" / "elexon" / "boalf_bess.parquet"

STUDY_START = "2024-01-01"
STUDY_END   = "2025-12-31"

# Minimum rated_mw to include. Excludes tiny virtual / aggregator units.
MIN_MW = 1.0

# Known non-BESS unit names within fuelType="OTHER". These are solar farms
# or gas peakers misclassified as OTHER. Excluded by substring match on
# bmUnitName (case-insensitive).
NON_BESS_NAME_SUBSTRINGS = [
    "solar",
    "solar farm",
    "lister storage",   # 1 MW gas storage, not BESS
]

# Keywords in bmUnitName that confirm a NaN-fuelType embedded unit is BESS.
BESS_NAME_KEYWORDS = [
    "battery", "batter", "bess", " stor", "storage",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def _is_non_bess_name(name: str) -> bool:
    n = str(name).lower()
    return any(sub in n for sub in NON_BESS_NAME_SUBSTRINGS)


def _is_bess_name(name: str) -> bool:
    n = str(name).lower()
    return any(kw in n for kw in BESS_NAME_KEYWORDS)


# ---------------------------------------------------------------------------
# Step 1: identify BESS from bmu_reference
# ---------------------------------------------------------------------------

def identify_bess_from_reference(bmu_ref_path: Path) -> pd.DataFrame:
    """
    Load bmu_reference.parquet and return confirmed BESS units with their
    rated_mw and identifying information.

    Returns DataFrame with columns:
        elexon_bmu_id, ng_bmu_id, bmu_name, rated_mw,
        lead_party, identification_method
    """
    if not bmu_ref_path.exists():
        raise FileNotFoundError(
            f"bmu_reference not found at:\n  {bmu_ref_path}\n"
            f"Pass the correct path with --bmu-ref"
        )

    ref = pd.read_parquet(bmu_ref_path)
    logger.info(f"  bmu_reference: {len(ref)} rows")

    # Column mapping
    elexon_col = _find_col(ref, ["elexonBmUnit", "elexon_bm_unit"])
    ng_col     = _find_col(ref, ["nationalGridBmUnit", "national_grid_bm_unit"])
    fuel_col   = _find_col(ref, ["fuelType", "fuel_type"])
    name_col   = _find_col(ref, ["bmUnitName", "bm_unit_name"])
    mw_col     = _find_col(ref, ["generationCapacity", "generation_capacity"])
    party_col  = _find_col(ref, ["leadPartyName", "lead_party_name"])

    if not elexon_col or not fuel_col:
        raise ValueError(
            f"Cannot find elexonBmUnit or fuelType columns.\n"
            f"Columns: {list(ref.columns)}"
        )

    # Log fuelType breakdown for transparency
    fuel_counts = ref[fuel_col].fillna("NaN").value_counts()
    logger.info(f"  fuelType breakdown: {dict(fuel_counts)}")

    rows = []

    # --- Group 1: fuelType == "OTHER" ------------------------------------
    # These are the registered BESS units. Filter out solar farms, virtual
    # units (V__ prefix), and sub-threshold MW units.
    other = ref[ref[fuel_col] == "OTHER"].copy()
    for _, row in other.iterrows():
        bmu = str(row[elexon_col]) if pd.notna(row[elexon_col]) else ""
        if not bmu or bmu == "nan":
            continue
        mw = float(row[mw_col]) if mw_col and pd.notna(row[mw_col]) else 0.0
        name = str(row[name_col]) if name_col and pd.notna(row[name_col]) else ""

        # Exclude virtual aggregator units (V__ prefix)
        if bmu.startswith("V__"):
            logger.debug(f"  Excluded (virtual): {bmu} — {name}")
            continue

        # Exclude known solar / non-BESS names
        if _is_non_bess_name(name):
            logger.debug(f"  Excluded (non-BESS name): {bmu} — {name}")
            continue

        # Exclude sub-threshold units
        if mw < MIN_MW:
            logger.debug(f"  Excluded (< {MIN_MW} MW): {bmu} — {mw} MW")
            continue

        rows.append({
            "elexon_bmu_id": bmu,
            "ng_bmu_id": str(row[ng_col]) if ng_col and pd.notna(row[ng_col]) else "",
            "bmu_name": name,
            "rated_mw": mw,
            "lead_party": str(row[party_col]) if party_col and pd.notna(row[party_col]) else "",
            "identification_method": "fuelType=OTHER",
        })

    n_other = len(rows)
    logger.info(f"  fuelType=OTHER after filtering: {n_other} BESS units")

    # --- Group 2: fuelType is NaN + name keyword -------------------------
    # Embedded BESS units are not classified in Elexon's reference.
    # Identify them by bmUnitName keyword match.
    nan_units = ref[ref[fuel_col].isna()].copy()
    n_name_match = 0
    existing_bmus = {r["elexon_bmu_id"] for r in rows}

    for _, row in nan_units.iterrows():
        bmu = str(row[elexon_col]) if pd.notna(row[elexon_col]) else ""
        if not bmu or bmu == "nan" or bmu in existing_bmus:
            continue
        name = str(row[name_col]) if name_col and pd.notna(row[name_col]) else ""
        mw = float(row[mw_col]) if mw_col and pd.notna(row[mw_col]) else 0.0

        if not _is_bess_name(name):
            continue
        if mw < MIN_MW:
            continue
        # Exclude virtual units
        if bmu.startswith("V__"):
            continue

        rows.append({
            "elexon_bmu_id": bmu,
            "ng_bmu_id": str(row[ng_col]) if ng_col and pd.notna(row[ng_col]) else "",
            "bmu_name": name,
            "rated_mw": mw,
            "lead_party": str(row[party_col]) if party_col and pd.notna(row[party_col]) else "",
            "identification_method": "name_keyword",
        })
        existing_bmus.add(bmu)
        n_name_match += 1

    logger.info(f"  NaN fuelType + name keyword: {n_name_match} additional BESS units")
    logger.info(f"  Total identified: {len(rows)}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2: cross-reference against BOALF study period
# ---------------------------------------------------------------------------

def filter_to_study_period(df: pd.DataFrame, boalf_path: Path) -> pd.DataFrame:
    """
    Keep only BMUs that appeared in BOALF during 2024-2025.

    Unlike previous versions, we do NOT use BOALF as the identification
    source — it's unfiltered. We use it only to confirm study-period
    activity for the units already identified from the reference.
    """
    if not boalf_path.exists():
        logger.warning(
            f"  BOALF not found at {boalf_path}. "
            f"Keeping all identified BESS BMUs without activity filter."
        )
        df["active_2024_25"] = True
        return df

    logger.info(f"  Loading BOALF to check study-period activity...")
    boalf = pd.read_parquet(boalf_path)

    bmu_col  = _find_col(boalf, ["bm_unit", "bmUnit", "elexonBmUnit"])
    date_col = _find_col(boalf, [
        c for c in boalf.columns if "settlement_date" in c.lower()
    ])

    if bmu_col is None:
        logger.warning("  No BMU column in BOALF. Keeping all.")
        df["active_2024_25"] = True
        return df

    if date_col:
        boalf[date_col] = pd.to_datetime(boalf[date_col], errors="coerce")
        study = boalf[
            (boalf[date_col] >= STUDY_START)
            & (boalf[date_col] <= STUDY_END)
        ]
    else:
        study = boalf

    active = set(study[bmu_col].dropna().astype(str).str.strip())
    df = df.copy()
    df["active_2024_25"] = df["elexon_bmu_id"].isin(active)

    n_active = int(df["active_2024_25"].sum())
    n_inactive = len(df) - n_active
    logger.info(
        f"  Active in BOALF 2024-2025: {n_active}/{len(df)} "
        f"({n_inactive} not seen in BM — may still have B1610 output)"
    )
    # Keep all — inactive-in-BOALF units may still have metered output
    # in B1610 (units that export to grid without bidding into the BM).
    return df


# ---------------------------------------------------------------------------
# Step 3: manual MWh overrides
# ---------------------------------------------------------------------------

def apply_manual_mwh(df: pd.DataFrame, manual_path: Path) -> pd.DataFrame:
    # Resolve relative paths against the project root (parent of prelock_checks/)
    # so the file is found regardless of which directory the script is run from.
    if not manual_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        manual_path = project_root / manual_path

    logger.info(f"  Looking for manual MWh file at: {manual_path} (exists: {manual_path.exists()})")

    if not manual_path.exists():
        logger.info(
            f"  No manual MWh file at {manual_path}. "
            f"Populate this file with bmu_id, rated_mwh, notes and re-run."
        )
        return df

    try:
        manual = pd.read_csv(manual_path, comment="#")
        manual.columns = [c.lower().strip() for c in manual.columns]
    except Exception as e:
        logger.warning(f"  Manual MWh file error: {e}")
        return df

    id_col  = _find_col(manual, ["bmu_id", "bm_unit", "elexon_bmu_id"])
    mwh_col = _find_col(manual, ["rated_mwh", "mwh"])

    if not id_col or not mwh_col:
        logger.warning(
            f"  Manual file needs 'bmu_id' and 'rated_mwh' columns. "
            f"Found: {list(manual.columns)}"
        )
        return df

    manual = manual.rename(columns={id_col: "_mid", mwh_col: "_mmwh"})
    df = df.merge(manual[["_mid", "_mmwh"]], left_on="elexon_bmu_id",
                  right_on="_mid", how="left")
    has_m = df["_mmwh"].notna()
    df.loc[has_m, "rated_mwh"] = df.loc[has_m, "_mmwh"]
    df.loc[has_m, "mwh_source"] = "MANUAL"
    df.drop(columns=["_mid", "_mmwh"], inplace=True)
    logger.info(f"  Manual MWh: {int(has_m.sum())} BMUs updated")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_bmu_list(
    bmu_ref_path: Path = DEFAULT_BMU_REF,
    boalf_path: Path = DEFAULT_BOALF,
    manual_mwh_path: Path = Path("data/bmu_mwh_manual.csv"),
    output_dir: Path = Path("data"),
) -> tuple[pd.DataFrame, pd.DataFrame]:

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Step 1: Identify BESS from bmu_reference")
    logger.info("=" * 60)
    df = identify_bess_from_reference(bmu_ref_path)
    if df.empty:
        logger.error("No BESS units identified.")
        sys.exit(1)

    df["rated_mwh"] = float("nan")
    df["mwh_source"] = "MISSING"

    logger.info("=" * 60)
    logger.info("Step 2: Cross-reference BOALF for study-period activity")
    logger.info("=" * 60)
    df = filter_to_study_period(df, boalf_path)

    logger.info("=" * 60)
    logger.info("Step 3: Apply manual MWh overrides")
    logger.info("=" * 60)
    df = apply_manual_mwh(df, manual_mwh_path)

    # Duration ratio
    df["rated_mwh"] = pd.to_numeric(df.get("rated_mwh"), errors="coerce")
    df["rated_mw"]  = pd.to_numeric(df.get("rated_mw"),  errors="coerce")
    df["duration_ratio_h"] = df["rated_mwh"] / df["rated_mw"]

    master_cols = [c for c in [
        "elexon_bmu_id", "ng_bmu_id", "bmu_name", "rated_mw",
        "rated_mwh", "duration_ratio_h", "mwh_source",
        "lead_party", "identification_method", "active_2024_25",
    ] if c in df.columns]

    df_master = df[master_cols].sort_values("rated_mw", ascending=False)
    df_master.to_csv(output_dir / "bess_master_list_219.csv", index=False)

    sd_mask = (
        df["duration_ratio_h"].notna()
        & (df["duration_ratio_h"] <= SHORT_DURATION_RATIO_MAX)
    )
    df_sd = df[sd_mask][master_cols].sort_values("rated_mw", ascending=False)
    df_sd.to_csv(output_dir / "short_duration_bmu_list.csv", index=False)

    # Exclusion list — sorted by rated_mw so largest units are prioritised
    excl = []
    for _, row in df[df["rated_mwh"].isna()].iterrows():
        excl.append({
            "elexon_bmu_id": row.get("elexon_bmu_id", ""),
            "bmu_name":      row.get("bmu_name", ""),
            "lead_party":    row.get("lead_party", ""),
            "rated_mw":      row.get("rated_mw", ""),
            "exclusion_reason": "rated_mwh MISSING — add to data/bmu_mwh_manual.csv",
        })
    for _, row in df[
        df["duration_ratio_h"].notna()
        & (df["duration_ratio_h"] > SHORT_DURATION_RATIO_MAX)
    ].iterrows():
        excl.append({
            "elexon_bmu_id": row.get("elexon_bmu_id", ""),
            "bmu_name":      row.get("bmu_name", ""),
            "lead_party":    row.get("lead_party", ""),
            "rated_mw":      row.get("rated_mw", ""),
            "exclusion_reason": f"duration {row.get('duration_ratio_h',0):.1f}h > 2.0h",
        })
    if excl:
        excl_df = pd.DataFrame(excl)
        excl_df["rated_mw"] = pd.to_numeric(excl_df["rated_mw"], errors="coerce")
        excl_df = excl_df.sort_values("rated_mw", ascending=False)
        excl_df.to_csv(output_dir / "bmu_exclusions.csv", index=False)

    # Summary
    n_master = len(df_master)
    n_sd     = len(df_sd)
    n_miss   = int(df["rated_mwh"].isna().sum())
    n_by_method = df["identification_method"].value_counts().to_dict()

    print(f"\n{'=' * 60}")
    print(f"  Master list:           {n_master:>4} BMUs → data/bess_master_list_219.csv")
    print(f"  Short-duration (≤2h):  {n_sd:>4} BMUs → data/short_duration_bmu_list.csv")
    print()
    print(f"  Identification breakdown:")
    for method, count in n_by_method.items():
        print(f"    {method:<30} {count}")
    print()
    print(f"  MWh source breakdown:")
    for src, cnt in df["mwh_source"].value_counts().items():
        print(f"    {src:<25} {cnt}")
    print()
    if n_miss:
        print(f"  ⚠  {n_miss} BMU(s) missing rated_mwh")
        print(f"     See data/bmu_exclusions.csv (sorted largest MW first)")
        print(f"     Add: bmu_id, rated_mwh, notes  →  data/bmu_mwh_manual.csv")
        print(f"     Then re-run. Short-duration count will increase.")
        print()
    ok_master = 150 <= n_master <= 300
    ok_sd = n_sd >= 50
    print(f"  {'✓' if ok_master else '⚠'} HS-2 master list:    {n_master} BMUs (need 150–300)")
    print(f"  {'✓' if ok_sd else '⚠'} HS-2 short-duration: {n_sd} BMUs (need ≥50)")
    print(f"{'=' * 60}\n")

    return df_master, df_sd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Project F BESS BMU master list."
    )
    parser.add_argument("--bmu-ref", type=Path, default=DEFAULT_BMU_REF,
                        help="Path to bmu_reference.parquet")
    parser.add_argument("--boalf-path", type=Path, default=DEFAULT_BOALF,
                        help="Path to boalf_bess.parquet (for activity filter)")
    parser.add_argument("--manual-mwh", type=Path,
                        default=Path("data/bmu_mwh_manual.csv"),
                        help="CSV of manual MWh ratings (bmu_id, rated_mwh, notes)")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
    )
    build_bmu_list(
        bmu_ref_path=args.bmu_ref,
        boalf_path=args.boalf_path,
        manual_mwh_path=args.manual_mwh,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
