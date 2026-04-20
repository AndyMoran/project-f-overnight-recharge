"""
Matched-pairs construction and balance diagnostics.
===================================================

This module implements the binary-specification matching pipeline
described in PROJECT_F_PRE_REGISTRATION_v3.md §Matching covariates and
§Matching algorithm. It also provides the balance diagnostics required
by HS-5 and HS-6 (standardised mean difference, Hotelling's T²).

Pipeline (reading top to bottom):

  1. assign_terciles        — per-month rank of evening_depletion_d into
                              top/middle/bottom terciles; middle excluded.
  2. build_matched_pairs    — for each HighDep day, find the nearest
                              LowDep day satisfying the pre-registered
                              constraints; return one row per matched pair.
  3. compute_smd            — standardised mean difference between HighDep
                              and LowDep groups on a given covariate.
  4. hotelling_t2           — joint balance test across all matching
                              covariates.

Pre-registration disambiguations
--------------------------------
Three points are left partially under-specified in the pre-reg. Choices
made here (documented in the function docstrings and collected here for
visibility):

  (A) Mahalanobis covariance matrix estimation pool: computed on the full
      training sample, not on HighDep+LowDep only or HighDep only. This
      is the convention in the matching literature and produces stable
      distance metrics regardless of tercile membership.

  (B) Decile boundaries for the ±1 decile matching constraint: computed
      once on the full training sample. Not re-computed per month.
      Stable boundaries mean the constraint is comparable across the
      year rather than a function of seasonal variation in the covariate.

  (C) Matching replacement policy: WITHOUT replacement. Each LowDep day
      can be matched to at most one HighDep day. This preserves the
      independence assumption underlying the block bootstrap. Unmatched
      HighDep days are dropped; unmatched LowDep days are unused. The
      estimand is therefore "effect on matchable HighDep days" per
      pre-reg §What this study does NOT claim.

These three choices mirror the under-specification lesson from the IV
project's DEVIATIONS.md §5–§8, where several pre-reg phrases required
post-hoc disambiguation. Documenting them up-front here, with inline
justification, avoids the same retrospective archaeology.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import mahalanobis

from .constants import (
    CONTINUOUS_MATCHING_COVARIATES,
    HS5_SMD_THRESHOLD,
    MATCHING_CALENDAR_PROXIMITY_DAYS,
    MATCHING_IC_DECILE_TOLERANCE,
    MATCHING_N_BINS,
    MATCHING_PRICE_DECILE_TOLERANCE,
)
from .errors import HardStopComputationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class MatchingResult:
    """
    Output of build_matched_pairs.

    Attributes
    ----------
    pairs : DataFrame with one row per matched pair. Columns:
        date_high, date_low (pd.Timestamp, tz-aware UTC)
        day_of_week_high, day_of_week_low (int 0-6, expected equal)
        <covariate>_high, <covariate>_low for each matching covariate
        overnight_price_high, overnight_price_low (£/MWh, SP1-14 mean on d+1)
        pair_distance (Mahalanobis distance between the two days' covariate
                       vectors in the scaled space)
    n_high_dep_total : int — total HighDep days in the input
    n_high_dep_matched : int — count of HighDep days with a valid match
    n_high_dep_dropped : int — total minus matched
    dropped_date_list : list[pd.Timestamp] — which HighDep dates were dropped
                                            (for Rosenbaum bounds diagnostic)
    """
    pairs: pd.DataFrame
    n_high_dep_total: int
    n_high_dep_matched: int
    n_high_dep_dropped: int
    dropped_date_list: list

    @property
    def dropped_proportion(self) -> float:
        if self.n_high_dep_total == 0:
            return 0.0
        return self.n_high_dep_dropped / self.n_high_dep_total


# ---------------------------------------------------------------------------
# Tercile assignment
# ---------------------------------------------------------------------------

def assign_terciles(
    df: pd.DataFrame,
    depletion_col: str = "evening_depletion_d",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Assign each day to HighDep / MiddleDep / LowDep within its calendar month.

    Per pre-reg §Binary and continuous treatment specifications:
        "Within each calendar month, rank evening_depletion_d and split
         into terciles. Top tercile = HighDep_d = 1; bottom tercile =
         LowDep_d = 1; middle tercile excluded."

    Parameters
    ----------
    df : day-level DataFrame with at minimum a `depletion_col` and `date_col`.
         Must have one row per date. `date_col` must be tz-aware UTC.
    depletion_col : name of the evening_depletion_d column
    date_col : name of the date column (tz-aware UTC)

    Returns
    -------
    pd.DataFrame : copy of input with two new columns:
        tercile : "high", "middle", or "low"
        is_high_dep : bool (True iff tercile == "high")
        is_low_dep : bool (True iff tercile == "low")

    Raises
    ------
    HardStopComputationError : if tercile assignment fails (e.g. all
        depletion values equal within a month).
    """
    if depletion_col not in df.columns or date_col not in df.columns:
        raise HardStopComputationError(
            check_id="MATCHING",
            reason=(
                f"assign_terciles: required columns {depletion_col!r} or "
                f"{date_col!r} missing. Available: {list(df.columns)}"
            ),
        )

    out = df.copy()
    # Build a month-grouping key that doesn't drop the timezone. We could
    # use .dt.to_period("M") but pandas strips tz during that conversion
    # and emits a UserWarning. A (year, month) tuple serialised to int
    # (year*12 + month) is tz-safe and hashes identically.
    out["_year_month"] = (
        out[date_col].dt.year * 12 + out[date_col].dt.month
    )

    # Rank within month, then cut into 3 equal-size bins.
    # qcut with duplicates="drop" collapses ties; if a month has too many
    # ties (e.g. all zero depletion), we get fewer than 3 bins and should
    # raise rather than silently collapse the categories.
    def _tercile_within_month(group: pd.Series) -> pd.Series:
        try:
            cats = pd.qcut(
                group, q=3, labels=["low", "middle", "high"], duplicates="drop"
            )
        except ValueError as e:
            raise HardStopComputationError(
                check_id="MATCHING",
                reason=(
                    f"assign_terciles: qcut failed on month "
                    f"{group.index[0] if len(group) else 'unknown'}: {e}. "
                    f"Likely cause: insufficient variation in {depletion_col} "
                    f"within this month."
                ),
            )
        # If duplicates="drop" removed categories, cats has fewer than 3
        # unique labels — the top tercile may not exist.
        n_labels = cats.dropna().nunique()
        if n_labels < 3:
            ym = out.loc[group.index[0], "_year_month"]
            year, month = divmod(int(ym), 12)
            if month == 0:
                year -= 1
                month = 12
            raise HardStopComputationError(
                check_id="MATCHING",
                reason=(
                    f"assign_terciles: month {year:04d}-{month:02d} "
                    f"produced only {n_labels} terciles (expected 3). "
                    f"Ties in {depletion_col} are collapsing the categories."
                ),
            )
        return cats

    # Group by month; apply per-group. We can't use transform here because
    # it would require a scalar return.
    out["tercile"] = (
        out.groupby("_year_month", group_keys=False)[depletion_col]
        .apply(_tercile_within_month)
    )

    out["is_high_dep"] = out["tercile"] == "high"
    out["is_low_dep"] = out["tercile"] == "low"
    out = out.drop(columns=["_year_month"])

    n_high = int(out["is_high_dep"].sum())
    n_mid = int((out["tercile"] == "middle").sum())
    n_low = int(out["is_low_dep"].sum())
    logger.info(
        f"Tercile assignment: high={n_high}, middle={n_mid}, low={n_low}, "
        f"total={len(out)}"
    )

    return out


# ---------------------------------------------------------------------------
# Decile computation
# ---------------------------------------------------------------------------

def compute_deciles(
    df: pd.DataFrame,
    covariate: str,
) -> pd.Series:
    """
    Compute decile membership (1-10) for a covariate using boundaries
    estimated on the full training sample.

    Per disambiguation (B) in the module docstring, decile boundaries are
    computed ONCE on the input `df`, not re-computed per month or per
    group. The returned series is integer-valued 1..10.

    Parameters
    ----------
    df : DataFrame with `covariate` column (non-null required rows only;
         caller is responsible for dropna).
    covariate : column name

    Returns
    -------
    pd.Series of int : decile membership, 1 = lowest, 10 = highest.
        Returned with the same index as df[covariate].

    Raises
    ------
    HardStopComputationError : if qcut fails (too many ties) or covariate
        column is missing.
    """
    if covariate not in df.columns:
        raise HardStopComputationError(
            check_id="MATCHING",
            reason=f"compute_deciles: column {covariate!r} not in DataFrame",
        )

    try:
        deciles = pd.qcut(
            df[covariate],
            q=MATCHING_N_BINS,
            labels=list(range(1, MATCHING_N_BINS + 1)),
            duplicates="drop",
        )
    except ValueError as e:
        raise HardStopComputationError(
            check_id="MATCHING",
            reason=(
                f"compute_deciles: qcut failed on {covariate!r}: {e}. "
                f"Likely cause: insufficient variation in training sample."
            ),
        )

    # If duplicates were dropped, fewer than 10 categories exist. This
    # would allow the ±1 tolerance to span more than ~20% of the sample,
    # defeating the purpose of the constraint. Raise.
    n_categories = deciles.dropna().nunique()
    if n_categories < MATCHING_N_BINS:
        raise HardStopComputationError(
            check_id="MATCHING",
            reason=(
                f"compute_deciles({covariate!r}): produced {n_categories} "
                f"categories, expected {MATCHING_N_BINS}. Likely cause: "
                f"ties in the covariate collapsed the decile boundaries."
            ),
        )

    return deciles.astype(int)


# ---------------------------------------------------------------------------
# Mahalanobis matching
# ---------------------------------------------------------------------------

def _mahalanobis_covariance(
    df: pd.DataFrame,
    covariates: tuple[str, ...],
) -> np.ndarray:
    """
    Estimate the covariance matrix used for the Mahalanobis distance.

    Per disambiguation (A) in the module docstring, we use the full
    training sample (all days, regardless of tercile membership). Drops
    rows with NaN in any covariate before computing.

    Returns the INVERSE covariance matrix (what mahalanobis expects).

    Raises
    ------
    HardStopComputationError : if the covariance matrix is singular.
    """
    x = df[list(covariates)].dropna().values
    if len(x) == 0:
        raise HardStopComputationError(
            check_id="MATCHING",
            reason="Mahalanobis covariance: no rows after dropping NaN.",
        )

    cov = np.cov(x, rowvar=False)
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError as e:
        raise HardStopComputationError(
            check_id="MATCHING",
            reason=(
                f"Mahalanobis covariance matrix is singular: {e}. "
                f"Likely cause: a matching covariate is (near-)perfectly "
                f"collinear with another."
            ),
            details={"covariates": list(covariates)},
        )
    return inv_cov


def build_matched_pairs(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    day_of_week_col: str = "day_of_week_d",
    overnight_price_col: str = "overnight_price_d_plus_1",
    covariates: tuple[str, ...] = CONTINUOUS_MATCHING_COVARIATES,
    price_decile_tolerance: int = MATCHING_PRICE_DECILE_TOLERANCE,
    ic_decile_tolerance: int = MATCHING_IC_DECILE_TOLERANCE,
    calendar_proximity_days: int = MATCHING_CALENDAR_PROXIMITY_DAYS,
) -> MatchingResult:
    """
    Build matched pairs per pre-reg §Matching covariates and §Matching
    algorithm.

    For each HighDep day, find the nearest LowDep day satisfying:
      - same day-of-week (exact)
      - within ±calendar_proximity_days calendar days
      - exact decile on DA_wind_forecast_evening_d
      - exact decile on DA_demand_forecast_evening_d
      - within ±price_decile_tolerance deciles on DA_overnight_price_d
      - within ±ic_decile_tolerance deciles on DA_IC_schedule_overnight_d

    Nearest defined by Mahalanobis distance over the four continuous
    covariates, with covariance estimated on the full input (disambiguation
    A). Matching is WITHOUT replacement (disambiguation C): a LowDep day
    matched to one HighDep day is not eligible for subsequent matches.

    Parameters
    ----------
    df : day-level DataFrame already processed by assign_terciles. Must have
        columns: date_col (tz-aware UTC), day_of_week_col, overnight_price_col,
        is_high_dep, is_low_dep, and all `covariates`.
    date_col, day_of_week_col, overnight_price_col : column names.
    covariates : tuple of continuous matching covariate names.
    price_decile_tolerance, ic_decile_tolerance : integer decile tolerances
        for the price and IC schedule constraints (wind and demand are
        always exact-decile per pre-reg).
    calendar_proximity_days : integer number of days for the ±day window.

    Returns
    -------
    MatchingResult

    Raises
    ------
    HardStopComputationError : on schema problems or singular covariance.
    """
    # ---- Schema checks ---------------------------------------------------
    required_cols = (
        [date_col, day_of_week_col, overnight_price_col, "is_high_dep", "is_low_dep"]
        + list(covariates)
    )
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HardStopComputationError(
            check_id="MATCHING",
            reason=f"build_matched_pairs: missing required columns: {missing}",
        )

    # ---- Drop days with any NaN in required fields -----------------------
    # A day with NaN in any matching covariate cannot be matched; drop it
    # before tercile-based selection to avoid silent exclusion later.
    before = len(df)
    df_clean = df.dropna(subset=required_cols).copy()
    if len(df_clean) < before:
        logger.info(
            f"build_matched_pairs: dropped {before - len(df_clean)} rows "
            f"with NaN in matching fields (of {before})."
        )

    # ---- Decile bins (disambiguation B) ----------------------------------
    # Compute once on the full cleaned input, before tercile split.
    df_clean["_wind_dec"] = compute_deciles(df_clean, covariates[0])
    df_clean["_demand_dec"] = compute_deciles(df_clean, covariates[1])
    df_clean["_price_dec"] = compute_deciles(df_clean, covariates[2])
    df_clean["_ic_dec"] = compute_deciles(df_clean, covariates[3])

    # ---- Mahalanobis inverse-covariance (disambiguation A) ---------------
    inv_cov = _mahalanobis_covariance(df_clean, covariates)

    # ---- Split into HighDep and LowDep frames ----------------------------
    high = df_clean[df_clean["is_high_dep"]].copy().reset_index(drop=True)
    low = df_clean[df_clean["is_low_dep"]].copy().reset_index(drop=True)

    if len(high) == 0 or len(low) == 0:
        raise HardStopComputationError(
            check_id="MATCHING",
            reason=(
                f"build_matched_pairs: empty tercile — "
                f"high={len(high)}, low={len(low)}. Did assign_terciles run?"
            ),
        )

    logger.info(
        f"build_matched_pairs: matching {len(high)} HighDep days to "
        f"{len(low)} LowDep candidates."
    )

    # ---- Greedy matching without replacement -----------------------------
    # Process HighDep days in date order (stable, reproducible). For each,
    # filter LowDep candidates by the hard constraints, then pick the one
    # with the smallest Mahalanobis distance. Remove the chosen LowDep day
    # from the pool.
    high = high.sort_values(date_col).reset_index(drop=True)

    # Pre-extract LowDep arrays for speed
    low_dates = low[date_col].values
    low_dow = low[day_of_week_col].values
    low_wind_dec = low["_wind_dec"].values
    low_demand_dec = low["_demand_dec"].values
    low_price_dec = low["_price_dec"].values
    low_ic_dec = low["_ic_dec"].values
    low_cov_matrix = low[list(covariates)].values

    low_available = np.ones(len(low), dtype=bool)

    pairs = []
    dropped_dates = []

    for _, h_row in high.iterrows():
        h_date = h_row[date_col]
        h_dow = h_row[day_of_week_col]
        h_vec = h_row[list(covariates)].values.astype(float)

        # Build eligibility mask over LowDep pool.
        # 1. Available (not already used).
        eligible = low_available.copy()
        # 2. Same day of week.
        eligible &= (low_dow == h_dow)
        # 3. Within ±N calendar days.
        date_diffs_days = (low_dates - h_date.to_numpy()).astype("timedelta64[D]").astype(int)
        eligible &= (np.abs(date_diffs_days) <= calendar_proximity_days)
        # 4. Exact wind decile.
        eligible &= (low_wind_dec == h_row["_wind_dec"])
        # 5. Exact demand decile.
        eligible &= (low_demand_dec == h_row["_demand_dec"])
        # 6. Price decile within tolerance.
        eligible &= (np.abs(low_price_dec - h_row["_price_dec"]) <= price_decile_tolerance)
        # 7. IC decile within tolerance.
        eligible &= (np.abs(low_ic_dec - h_row["_ic_dec"]) <= ic_decile_tolerance)

        if not eligible.any():
            dropped_dates.append(h_date)
            continue

        # Compute Mahalanobis distance for each eligible candidate.
        cand_idx = np.where(eligible)[0]
        distances = np.empty(len(cand_idx))
        for i, ci in enumerate(cand_idx):
            distances[i] = mahalanobis(h_vec, low_cov_matrix[ci], inv_cov)

        best_i = cand_idx[np.argmin(distances)]
        best_distance = distances[np.argmin(distances)]

        # Record the pair.
        pair_row = {
            "date_high": h_date,
            "date_low": low_dates[best_i],
            "day_of_week_high": h_dow,
            "day_of_week_low": low_dow[best_i],
            "overnight_price_high": h_row[overnight_price_col],
            "overnight_price_low": low.iloc[best_i][overnight_price_col],
            "pair_distance": float(best_distance),
        }
        # Add covariate values for audit/diagnostic use.
        for c in covariates:
            pair_row[f"{c}_high"] = h_row[c]
            pair_row[f"{c}_low"] = low.iloc[best_i][c]

        pairs.append(pair_row)
        low_available[best_i] = False

    pairs_df = pd.DataFrame(pairs)

    result = MatchingResult(
        pairs=pairs_df,
        n_high_dep_total=len(high),
        n_high_dep_matched=len(pairs_df),
        n_high_dep_dropped=len(dropped_dates),
        dropped_date_list=dropped_dates,
    )
    logger.info(
        f"build_matched_pairs: matched={result.n_high_dep_matched}/"
        f"{result.n_high_dep_total} HighDep days "
        f"({100 * (1 - result.dropped_proportion):.1f}% match rate)"
    )
    return result


# ---------------------------------------------------------------------------
# Balance diagnostics
# ---------------------------------------------------------------------------

def compute_smd(
    pairs_df: pd.DataFrame,
    covariate: str,
) -> float:
    """
    Standardised mean difference for a single covariate, between the
    HighDep and LowDep sides of the matched pairs.

    SMD = (mean_high - mean_low) / pooled_sd

    where pooled_sd = sqrt((var_high + var_low) / 2), using sample
    (unbiased) variance.

    Per pre-reg §Hard stops HS-5: |SMD| must be ≤ 0.10 for each matching
    covariate on the matched sample. A covariate with SMD > 0.10 after
    matching means the matching algorithm failed to balance that
    covariate.

    Parameters
    ----------
    pairs_df : DataFrame from MatchingResult.pairs. Must have columns
        `{covariate}_high` and `{covariate}_low`.
    covariate : base column name (not suffixed).

    Returns
    -------
    float : signed SMD. Positive means HighDep mean exceeds LowDep mean.

    Raises
    ------
    HardStopComputationError : if required columns are missing or the
        pooled standard deviation is zero.
    """
    col_high = f"{covariate}_high"
    col_low = f"{covariate}_low"
    if col_high not in pairs_df.columns or col_low not in pairs_df.columns:
        raise HardStopComputationError(
            check_id="HS-5",
            reason=(
                f"compute_smd: missing {col_high!r} or {col_low!r} in pairs_df. "
                f"Available: {list(pairs_df.columns)}"
            ),
        )

    x_high = pairs_df[col_high].dropna().values
    x_low = pairs_df[col_low].dropna().values

    if len(x_high) == 0 or len(x_low) == 0:
        raise HardStopComputationError(
            check_id="HS-5",
            reason=f"compute_smd({covariate!r}): empty group after NaN drop.",
        )

    mean_diff = x_high.mean() - x_low.mean()
    # ddof=1 for sample variance (unbiased).
    var_high = x_high.var(ddof=1) if len(x_high) > 1 else 0.0
    var_low = x_low.var(ddof=1) if len(x_low) > 1 else 0.0
    pooled_sd = np.sqrt((var_high + var_low) / 2.0)

    if pooled_sd == 0:
        raise HardStopComputationError(
            check_id="HS-5",
            reason=(
                f"compute_smd({covariate!r}): pooled SD is zero. "
                f"Both groups are constants."
            ),
        )

    return float(mean_diff / pooled_sd)


def hotelling_t2(
    pairs_df: pd.DataFrame,
    covariates: tuple[str, ...] = CONTINUOUS_MATCHING_COVARIATES,
) -> tuple[float, float]:
    """
    Two-sample Hotelling's T² test for joint balance between the HighDep
    and LowDep sides of matched pairs.

    Per pre-reg §Hard stops HS-5: joint balance must NOT be rejected at
    p < 0.10 (i.e. p-value ≥ 0.10 to pass).

    The test statistic is:
        T² = (n1*n2/(n1+n2)) * (x̄1 - x̄2)' * S_pooled^-1 * (x̄1 - x̄2)

    With F-distribution under H0:
        F = T² * (n1+n2-p-1) / (p*(n1+n2-2))   ~   F(p, n1+n2-p-1)

    Parameters
    ----------
    pairs_df : DataFrame from MatchingResult.pairs with `{c}_high` and
        `{c}_low` columns for each covariate in `covariates`.
    covariates : tuple of base covariate names.

    Returns
    -------
    (T², p_value) : both floats.

    Raises
    ------
    HardStopComputationError : on missing columns, singular pooled
        covariance matrix, or insufficient sample size.
    """
    high_cols = [f"{c}_high" for c in covariates]
    low_cols = [f"{c}_low" for c in covariates]
    missing = [c for c in high_cols + low_cols if c not in pairs_df.columns]
    if missing:
        raise HardStopComputationError(
            check_id="HS-5",
            reason=f"hotelling_t2: missing columns: {missing}",
        )

    x_high = pairs_df[high_cols].dropna().values
    x_low = pairs_df[low_cols].dropna().values

    n1, p1 = x_high.shape
    n2, p2 = x_low.shape
    if p1 != p2:
        raise HardStopComputationError(
            check_id="HS-5",
            reason=f"hotelling_t2: dim mismatch ({p1} vs {p2})",
        )
    p = p1
    if n1 + n2 - p - 1 <= 0:
        raise HardStopComputationError(
            check_id="HS-5",
            reason=(
                f"hotelling_t2: insufficient sample size "
                f"(n1+n2-p-1 = {n1 + n2 - p - 1} ≤ 0). "
                f"Need more pairs or fewer covariates."
            ),
        )

    mean_diff = x_high.mean(axis=0) - x_low.mean(axis=0)

    # Pooled covariance (equal-covariance assumption; standard for the
    # two-sample version of Hotelling's T²).
    cov_high = np.cov(x_high, rowvar=False, ddof=1)
    cov_low = np.cov(x_low, rowvar=False, ddof=1)
    s_pooled = ((n1 - 1) * cov_high + (n2 - 1) * cov_low) / (n1 + n2 - 2)

    try:
        s_pooled_inv = np.linalg.inv(s_pooled)
    except np.linalg.LinAlgError as e:
        raise HardStopComputationError(
            check_id="HS-5",
            reason=f"hotelling_t2: pooled covariance singular: {e}",
        )

    t2 = (n1 * n2 / (n1 + n2)) * mean_diff @ s_pooled_inv @ mean_diff

    # F-transformation
    f_stat = t2 * (n1 + n2 - p - 1) / (p * (n1 + n2 - 2))
    p_value = 1.0 - stats.f.cdf(f_stat, p, n1 + n2 - p - 1)

    return float(t2), float(p_value)


# ---------------------------------------------------------------------------
# Convenience summary
# ---------------------------------------------------------------------------

def summarise_balance(
    pairs_df: pd.DataFrame,
    covariates: tuple[str, ...] = CONTINUOUS_MATCHING_COVARIATES,
) -> dict:
    """
    Compute SMD for each covariate and the joint Hotelling T² test.
    Used directly by HS-5.

    Returns
    -------
    dict with keys:
        smd_per_covariate : {covariate: smd}
        max_abs_smd : float
        all_smd_ok : bool (all |SMD| ≤ HS5_SMD_THRESHOLD)
        hotelling_t2 : float
        hotelling_pvalue : float
    """
    smds = {c: compute_smd(pairs_df, c) for c in covariates}
    max_abs = max(abs(s) for s in smds.values())
    t2, pval = hotelling_t2(pairs_df, covariates)
    return {
        "smd_per_covariate": smds,
        "max_abs_smd": max_abs,
        "all_smd_ok": max_abs <= HS5_SMD_THRESHOLD,
        "hotelling_t2": t2,
        "hotelling_pvalue": pval,
    }
