"""
Inference engines for the primary and informational tests.
==========================================================

This module implements three inferential procedures named in the
pre-registration:

  1. bca_block_bootstrap   — BCa (bias-corrected accelerated) block
                             bootstrap; the single pre-registered
                             gate on primary hypothesis testing.
  2. within_dow_permutation — permutation placebo used by HS-3.
  3. tost_equivalence       — two one-sided tests equivalence procedure,
                             informational output for the null-affirmative
                             case.

The BCa bootstrap is the subtlest of the three. Percentile bootstrap is
what most people reach for first, but under heavy-tailed or skewed
sampling distributions it can produce a significant p-value while the
95% CI contains zero — a contradiction that undermines the gate. BCa
corrects for two sources of skew:

  * bias correction (z0): where the point estimate sits in the bootstrap
    distribution;
  * acceleration (a): how much the variance changes with the parameter,
    estimated by jackknife.

Both corrections are implemented here per the standard reference (Efron,
1987; Efron & Tibshirani, 1993, Chapter 14). The algorithms are checked
against `scipy.stats.bootstrap` behaviour in the tests, and the
block-resampling logic is unit-tested against a toy case where the
expected block structure is obvious by hand.

Block-bootstrap resampling unit
-------------------------------
Pre-reg §Matching algorithm specifies that a matched pair spanning two
calendar weeks is assigned to `d_high`'s week for bootstrap purposes.
Block resampling therefore operates on week-of-d_high. The LowDep side's
calendar week is ignored for the block structure — a deliberate leak
documented in the pre-reg.

Permutation strata
------------------
Pre-reg §Hard stops HS-3 (revised in v3) specifies within-day-of-week
permutation, NOT day-of-week × month. The v2 DoW×month version
preserved too much structure and narrowed the permutation null
artificially. The v3 within-DoW version widens the null by mixing
across seasons while preserving the weekly pattern that most matters
for energy markets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .constants import (
    BOOTSTRAP_SEED,
    HS3_N_PERMUTATIONS,
    HS3_PERMUTATION_SEED,
    N_BOOTSTRAP,
)
from .errors import HardStopComputationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BcaBootstrapResult:
    """
    Output of bca_block_bootstrap.

    Attributes
    ----------
    point_estimate : the observed Δ̄ on the full sample (no resampling).
    ci_lower, ci_upper : BCa 95% confidence interval bounds.
    one_sided_p_value : BCa-corrected p-value for the one-sided null that
        the true Δ̄ is ≥ 0 (the pre-registered test direction is "Δ̄ < 0
        represents suppression"; a small p-value rejects the null).
    z0 : bias-correction constant.
    a : acceleration constant.
    n_resamples : bootstrap replications actually completed.
    n_blocks : number of unique blocks (weeks) the pairs were grouped into.
    bootstrap_distribution : full array of bootstrap Δ̄_b values (length n_resamples).
    """
    point_estimate: float
    ci_lower: float
    ci_upper: float
    one_sided_p_value: float
    z0: float
    a: float
    n_resamples: int
    n_blocks: int
    bootstrap_distribution: np.ndarray = field(repr=False)


@dataclass
class PermutationResult:
    """
    Output of within_dow_permutation.

    Attributes
    ----------
    observed_delta : Δ̄ computed on the observed (unshuffled) data.
    permutation_deltas : array of Δ̄ from each permutation replicate.
    median_abs_perm_delta : median of |Δ̄_perm| — the HS-3 benchmark.
    observed_abs_delta : |observed_delta|, for HS-3 comparison.
    passes_hs3 : True iff |observed_delta| ≥ median(|Δ̄_perm|).
    n_permutations : replicates actually completed.
    """
    observed_delta: float
    permutation_deltas: np.ndarray = field(repr=False)
    median_abs_perm_delta: float
    observed_abs_delta: float
    passes_hs3: bool
    n_permutations: int


@dataclass
class TostResult:
    """
    Output of tost_equivalence.

    Attributes
    ----------
    point_estimate : observed Δ̄.
    lower_bound, upper_bound : equivalence bounds (typically ±£1/MWh).
    p_lower : one-sided p-value for H0: Δ̄ ≤ lower_bound.
    p_upper : one-sided p-value for H0: Δ̄ ≥ upper_bound.
    tost_p_value : max(p_lower, p_upper) — the TOST p-value.
    equivalent_at_alpha : True iff tost_p_value < alpha.
    alpha : significance level used for the equivalence decision.
    """
    point_estimate: float
    lower_bound: float
    upper_bound: float
    p_lower: float
    p_upper: float
    tost_p_value: float
    equivalent_at_alpha: bool
    alpha: float


# ---------------------------------------------------------------------------
# BCa block bootstrap
# ---------------------------------------------------------------------------

def _assign_blocks(pairs_df: pd.DataFrame, date_high_col: str = "date_high") -> pd.Series:
    """
    Assign each pair to its block ID: the ISO-week of d_high, as a
    (iso_year, iso_week) tuple encoded to an integer
    (iso_year * 100 + iso_week). Stable across year boundaries.

    Returns a Series of int, same index as `pairs_df`.
    """
    if date_high_col not in pairs_df.columns:
        raise HardStopComputationError(
            check_id="BOOTSTRAP",
            reason=f"_assign_blocks: column {date_high_col!r} missing.",
        )
    iso = pairs_df[date_high_col].dt.isocalendar()
    return (iso["year"].astype(int) * 100 + iso["week"].astype(int)).rename("_block_id")


def _bootstrap_by_week(
    pair_deltas: np.ndarray,
    block_ids: np.ndarray,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw `n_resamples` bootstrap means by resampling blocks (weeks) with
    replacement, preserving all pairs within a resampled block.

    Internally, we build once:
      * unique block IDs (array of length B)
      * for each block, a numpy array of pair-delta values

    Then each resample draws B blocks with replacement (with replacement
    means each bootstrap sample has the same number of blocks as the
    original, not necessarily the same number of pairs — block sizes
    vary), concatenates their deltas, and takes the mean.
    """
    unique_blocks, inverse = np.unique(block_ids, return_inverse=True)
    n_blocks = len(unique_blocks)

    # Pre-split deltas by block for O(1) lookup per draw.
    block_to_deltas = [
        pair_deltas[inverse == i] for i in range(n_blocks)
    ]

    resample_means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        chosen_idx = rng.integers(0, n_blocks, size=n_blocks)
        # Concatenate deltas from chosen blocks.
        concatenated = np.concatenate([block_to_deltas[j] for j in chosen_idx])
        resample_means[i] = concatenated.mean()

    return resample_means


def _bca_z0(observed: float, resamples: np.ndarray) -> float:
    """
    Bias-correction constant z0 = Φ⁻¹(fraction of resamples ≤ observed).
    """
    frac = (resamples < observed).mean()
    # Handle the degenerate case where all resamples are on one side —
    # Φ⁻¹(0) or Φ⁻¹(1) is ±∞, which propagates into NaN CIs. We clip
    # the fraction into (0, 1) by a tiny amount to get a large but
    # finite z0 and let the CI reflect the degeneracy through its width.
    frac = np.clip(frac, 1e-10, 1 - 1e-10)
    return float(stats.norm.ppf(frac))


def _bca_acceleration(
    pair_deltas: np.ndarray,
    block_ids: np.ndarray,
) -> float:
    """
    Acceleration constant `a` via the jackknife, over BLOCKS (not pairs).

    For block-bootstrap BCa, the jackknife units must match the bootstrap
    units — blocks, not individual observations. Otherwise the acceleration
    mis-estimates the variance's rate of change with respect to the
    parameter.

    Formula (Efron & Tibshirani 1993, eq 14.15):

        a = Σ(θ̄ - θ_{(i)})^3 / (6 * [Σ(θ̄ - θ_{(i)})^2]^{3/2})

    where θ_{(i)} is the leave-one-block-out estimate.
    """
    unique_blocks, inverse = np.unique(block_ids, return_inverse=True)
    n_blocks = len(unique_blocks)

    if n_blocks < 2:
        # With <2 blocks, acceleration is undefined. Return 0 and let
        # the BCa degenerate back to the basic bias-corrected bootstrap.
        logger.warning(
            "_bca_acceleration: <2 blocks, returning a=0 (bias-corrected only)."
        )
        return 0.0

    jackknife_means = np.empty(n_blocks, dtype=float)
    for i in range(n_blocks):
        keep = inverse != i
        if not keep.any():
            raise HardStopComputationError(
                check_id="BOOTSTRAP",
                reason=(
                    f"Acceleration jackknife: block {i} contains all pairs. "
                    f"Cannot leave-one-block-out."
                ),
            )
        jackknife_means[i] = pair_deltas[keep].mean()

    grand_mean = jackknife_means.mean()
    residuals = grand_mean - jackknife_means
    num = (residuals ** 3).sum()
    den = 6.0 * (np.sum(residuals ** 2) ** 1.5)

    if den == 0:
        # All jackknife means equal → variance doesn't change with theta.
        # a = 0 is the correct value.
        return 0.0
    return float(num / den)


def _bca_quantile(
    alpha: float,
    z0: float,
    a: float,
) -> float:
    """
    Compute the BCa-adjusted quantile α₁ = Φ(z0 + (z0 + zα) / (1 - a(z0 + zα))).

    Used to map the nominal α to a BCa-corrected quantile in the bootstrap
    distribution.
    """
    z_alpha = stats.norm.ppf(alpha)
    adjusted = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
    return float(stats.norm.cdf(adjusted))


def bca_block_bootstrap(
    pairs_df: pd.DataFrame,
    *,
    price_high_col: str = "overnight_price_high",
    price_low_col: str = "overnight_price_low",
    date_high_col: str = "date_high",
    n_resamples: int = N_BOOTSTRAP,
    ci_level: float = 0.95,
    seed: int = BOOTSTRAP_SEED,
) -> BcaBootstrapResult:
    """
    BCa block-bootstrap on the matched pairs, resampling by ISO-week of
    d_high.

    Per pre-reg §Test statistic and inference, the output BCa one-sided
    p-value is the single pre-registered gate on the primary hypothesis.
    The BCa CI is reported descriptively — it does not independently
    determine passing.

    Parameters
    ----------
    pairs_df : DataFrame from matching.build_matched_pairs.pairs. Must have
        price_high_col, price_low_col, date_high_col columns.
    price_high_col, price_low_col : names of the two overnight-price columns.
    date_high_col : name of the d_high column (used for week assignment).
    n_resamples : bootstrap replications.
    ci_level : nominal CI coverage (default 0.95 → 95% CI).
    seed : RNG seed for reproducibility.

    Returns
    -------
    BcaBootstrapResult

    Raises
    ------
    HardStopComputationError : on schema problems, insufficient blocks,
        or degenerate bootstrap distribution.
    """
    required = [price_high_col, price_low_col, date_high_col]
    missing = [c for c in required if c not in pairs_df.columns]
    if missing:
        raise HardStopComputationError(
            check_id="BOOTSTRAP",
            reason=f"bca_block_bootstrap: missing columns: {missing}",
        )

    df_clean = pairs_df.dropna(subset=required).copy()
    if len(df_clean) == 0:
        raise HardStopComputationError(
            check_id="BOOTSTRAP",
            reason="bca_block_bootstrap: no rows after dropping NaN.",
        )

    pair_deltas = (
        df_clean[price_high_col].values - df_clean[price_low_col].values
    ).astype(float)
    observed = float(pair_deltas.mean())

    block_ids = _assign_blocks(df_clean, date_high_col=date_high_col).values
    unique_blocks = np.unique(block_ids)
    n_blocks = len(unique_blocks)

    if n_blocks < 2:
        raise HardStopComputationError(
            check_id="BOOTSTRAP",
            reason=(
                f"bca_block_bootstrap: only {n_blocks} unique block(s); "
                f"need ≥ 2 for meaningful block resampling."
            ),
        )

    rng = np.random.default_rng(seed)

    logger.info(
        f"bca_block_bootstrap: n_pairs={len(df_clean)}, n_blocks={n_blocks}, "
        f"n_resamples={n_resamples}, observed Δ̄={observed:.4f}"
    )

    # ---- Resample ---------------------------------------------------------
    bootstrap_means = _bootstrap_by_week(
        pair_deltas, block_ids, n_resamples, rng
    )

    # ---- Bias correction and acceleration --------------------------------
    z0 = _bca_z0(observed, bootstrap_means)
    a = _bca_acceleration(pair_deltas, block_ids)

    # ---- BCa CI ----------------------------------------------------------
    alpha = (1 - ci_level) / 2
    alpha_lower = _bca_quantile(alpha, z0, a)
    alpha_upper = _bca_quantile(1 - alpha, z0, a)
    ci_lower = float(np.quantile(bootstrap_means, alpha_lower))
    ci_upper = float(np.quantile(bootstrap_means, alpha_upper))

    # ---- One-sided BCa p-value for H0: Δ̄ ≥ 0 ----------------------------
    # Goal: find the largest one-sided significance level alpha_p such that
    # the BCa-corrected upper bound for the interval (-∞, θ̂_{alpha_p}) = 0.
    # The direct way: solve for the alpha_p at which the BCa upper quantile
    # is exactly 0, i.e. where the uncorrected bootstrap fraction below 0
    # produces a BCa-corrected upper bound at 0.
    #
    # Equivalently, per Efron & Tibshirani (1993, Section 14.3), the BCa
    # p-value is obtained by inverting the BCa adjustment:
    #
    #   Let p0 = fraction of bootstrap resamples ≤ 0.
    #   zp = Φ⁻¹(p0)
    #   BCa p-value = Φ(2*z0 - zp + a_adjusted_term)  ... this is the
    #   bias-corrected-only (BC) form. The full BCa form requires
    #   iterating the (z0 + zα) / (1 - a(z0 + zα)) relationship.
    #
    # We use a numerical approach: find the alpha for which the BCa-adjusted
    # quantile equals 0 in the bootstrap distribution, then that alpha IS
    # the one-sided p-value for H0: Δ̄ ≥ 0.
    p_one_sided = _bca_one_sided_pvalue(bootstrap_means, z0=z0, a=a, null_value=0.0)

    return BcaBootstrapResult(
        point_estimate=observed,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        one_sided_p_value=p_one_sided,
        z0=z0,
        a=a,
        n_resamples=n_resamples,
        n_blocks=n_blocks,
        bootstrap_distribution=bootstrap_means,
    )


def _bca_one_sided_pvalue(
    resamples: np.ndarray,
    *,
    z0: float,
    a: float,
    null_value: float,
) -> float:
    """
    One-sided BCa p-value for H0: θ ≥ null_value vs H1: θ < null_value.

    Interpretation:
      - small p-value ⇒ observed is substantially below null_value ⇒
        reject H0 in favour of H1.
      - p ≈ 0.5 ⇒ observed ≈ null_value ⇒ inconclusive.
      - p near 1 ⇒ observed is substantially above null_value ⇒ cannot
        reject H0.

    Construction (Efron & Tibshirani 1993, §14.3): a BCa one-sided CI for
    θ at nominal level (1-α) has upper bound q̂(1-α), where q̂ is the
    BCa-adjusted quantile of the bootstrap distribution. The p-value for
    the one-sided test H0: θ ≥ null_value is the smallest α such that
    null_value ≤ q̂(1-α). Because q̂ is monotonic in α, we binary-search
    over α ∈ (0,1) for the α where q̂(1-α) = null_value.

    Edge cases: if null_value is above every resample, H0 cannot be
    rejected at any α → p-value is 1. If null_value is below every
    resample, H0 is rejected at every α → p-value is 0.
    """
    # Edge cases at the support boundaries.
    # For H0: θ ≥ null_value vs H1: θ < null_value:
    #   - if null_value > max(resamples): the bootstrap distribution lies
    #     entirely below null → strong evidence against H0 → p = 0.
    #   - if null_value < min(resamples): the bootstrap distribution lies
    #     entirely above null → no evidence against H0 → p = 1.
    if null_value >= resamples.max():
        return 0.0
    if null_value <= resamples.min():
        return 1.0

    def q_upper_at_alpha(alpha: float) -> float:
        """BCa upper-bound quantile at nominal two-sided level alpha, i.e.
        the (1 - alpha) BCa quantile."""
        bca_q = _bca_quantile(1.0 - alpha, z0, a)
        bca_q = float(np.clip(bca_q, 1e-10, 1 - 1e-10))
        return float(np.quantile(resamples, bca_q))

    # Binary search for alpha ∈ (0, 1) such that q_upper_at_alpha(alpha) = null_value.
    # q_upper_at_alpha is DECREASING in alpha (larger alpha → narrower interval →
    # lower upper bound). We want the smallest alpha at which q_upper ≤ null_value.
    lo, hi = 1e-10, 1.0 - 1e-10

    # Sanity: at lo, q_upper should be very high (near resamples.max());
    # at hi, q_upper should be very low (near resamples.min()).
    q_at_lo = q_upper_at_alpha(lo)
    q_at_hi = q_upper_at_alpha(hi)
    if q_at_lo < null_value:
        # Even the widest interval's upper bound is below null → null is
        # well above the bootstrap distribution → can't reject → p = 1.
        return 1.0
    if q_at_hi > null_value:
        # Even the narrowest interval's upper bound is above null → null
        # is well below the bootstrap distribution → always reject → p = 0.
        return 0.0

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        q_mid = q_upper_at_alpha(mid)
        if q_mid > null_value:
            # Upper bound too high → need larger alpha (narrower interval)
            lo = mid
        else:
            # Upper bound too low → need smaller alpha
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Within-DoW permutation placebo
# ---------------------------------------------------------------------------

def within_dow_permutation(
    day_frame: pd.DataFrame,
    *,
    build_matched_pairs_fn,
    compute_delta_fn=None,
    depletion_col: str = "evening_depletion_d",
    day_of_week_col: str = "day_of_week_d",
    n_permutations: int = HS3_N_PERMUTATIONS,
    seed: int = HS3_PERMUTATION_SEED,
) -> PermutationResult:
    """
    HS-3 permutation placebo.

    Per pre-reg §Hard stops HS-3: permute `evening_depletion_d` within
    day-of-week strata (NOT day-of-week × month — see module docstring),
    rerun the full matched-pair pipeline for each permutation, compute
    Δ̄_perm. Trigger if |observed Δ̄| < median |Δ̄_perm|.

    This is expensive: `n_permutations` full matching rebuilds. For the
    pre-registered n=1000, expect minutes-not-seconds runtime. If you're
    iterating on the pipeline during development, temporarily reduce
    n_permutations via the HS3_N_PERMUTATIONS constant (but don't commit
    the change).

    Parameters
    ----------
    day_frame : day-level frame after assign_terciles has NOT yet run.
        Must contain depletion_col and day_of_week_col, plus all columns
        required by build_matched_pairs_fn.
    build_matched_pairs_fn : callable(shuffled_day_frame) → MatchingResult.
        Injected so this module doesn't import matching — caller wires it.
    compute_delta_fn : callable(MatchingResult) → float. Defaults to
        computing the mean price difference if None.
    depletion_col : name of the depletion column to permute.
    day_of_week_col : name of the strata column.
    n_permutations : number of permutation replicates.
    seed : RNG seed for reproducibility.

    Returns
    -------
    PermutationResult
    """
    if depletion_col not in day_frame.columns:
        raise HardStopComputationError(
            check_id="HS-3",
            reason=f"within_dow_permutation: column {depletion_col!r} missing.",
        )
    if day_of_week_col not in day_frame.columns:
        raise HardStopComputationError(
            check_id="HS-3",
            reason=f"within_dow_permutation: column {day_of_week_col!r} missing.",
        )

    if compute_delta_fn is None:
        def compute_delta_fn(matching_result):
            if matching_result.pairs.empty:
                return np.nan
            deltas = (
                matching_result.pairs["overnight_price_high"]
                - matching_result.pairs["overnight_price_low"]
            )
            return float(deltas.mean())

    rng = np.random.default_rng(seed)

    # ---- Observed Δ̄ ------------------------------------------------------
    observed_result = build_matched_pairs_fn(day_frame)
    observed_delta = compute_delta_fn(observed_result)

    # ---- Permutations ----------------------------------------------------
    # For efficiency, pre-compute the per-group indices once. For each
    # replicate, shuffle depletion values within each DoW stratum.
    dow_values = day_frame[day_of_week_col].values
    dow_unique = np.unique(dow_values)
    dow_to_indices = {d: np.where(dow_values == d)[0] for d in dow_unique}

    permutation_deltas = np.full(n_permutations, np.nan, dtype=float)
    original_depletion = day_frame[depletion_col].values.copy()

    for i in range(n_permutations):
        shuffled = original_depletion.copy()
        for d in dow_unique:
            idx = dow_to_indices[d]
            shuffled_subset = shuffled[idx].copy()
            rng.shuffle(shuffled_subset)
            shuffled[idx] = shuffled_subset

        permuted_frame = day_frame.copy()
        permuted_frame[depletion_col] = shuffled

        try:
            permuted_result = build_matched_pairs_fn(permuted_frame)
            permutation_deltas[i] = compute_delta_fn(permuted_result)
        except HardStopComputationError as e:
            # A permutation may fail (e.g. insufficient variation after
            # shuffle). We record NaN and continue; the final median is
            # taken over the valid values.
            logger.warning(
                f"Permutation {i} failed: {e}. Recording NaN."
            )

    valid_mask = ~np.isnan(permutation_deltas)
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        raise HardStopComputationError(
            check_id="HS-3",
            reason=(
                "within_dow_permutation: all permutations failed. "
                "Cannot compute benchmark distribution."
            ),
        )
    if n_valid < n_permutations * 0.9:
        logger.warning(
            f"within_dow_permutation: only {n_valid} / {n_permutations} "
            f"permutations succeeded (≥ 90% expected)."
        )

    median_abs_perm = float(np.median(np.abs(permutation_deltas[valid_mask])))
    observed_abs = float(abs(observed_delta)) if not np.isnan(observed_delta) else 0.0

    return PermutationResult(
        observed_delta=observed_delta,
        permutation_deltas=permutation_deltas,
        median_abs_perm_delta=median_abs_perm,
        observed_abs_delta=observed_abs,
        passes_hs3=observed_abs >= median_abs_perm,
        n_permutations=n_valid,
    )


# ---------------------------------------------------------------------------
# TOST equivalence
# ---------------------------------------------------------------------------

def tost_equivalence(
    pair_deltas: np.ndarray | pd.Series,
    *,
    equivalence_bound: float,
    alpha: float = 0.05,
) -> TostResult:
    """
    Two one-sided tests (TOST) equivalence procedure on matched-pair deltas.

    Per pre-reg §Test statistic and inference §Equivalence test, this is
    an informational output for the null-affirmative case. A significant
    TOST result (p < alpha) supports the pre-registered statement "no
    economically meaningful suppression effect detected" at the
    specified bound.

    The procedure runs two one-sided t-tests:
      H0_lower: Δ̄ ≤ -equivalence_bound   (rejected if Δ̄ substantially > -bound)
      H0_upper: Δ̄ ≥ +equivalence_bound   (rejected if Δ̄ substantially < +bound)

    TOST p-value is max(p_lower, p_upper). Rejection of both one-sided
    nulls at level alpha means the mean lies in the equivalence interval
    at the (1 - 2*alpha) confidence level.

    Parameters
    ----------
    pair_deltas : array or Series of per-pair price differences
        (overnight_price_high - overnight_price_low).
    equivalence_bound : positive scalar (£/MWh). The equivalence interval
        is [-equivalence_bound, +equivalence_bound].
    alpha : one-sided significance level per TOST half-test (default 0.05
        → 90% confidence-interval equivalent).

    Returns
    -------
    TostResult
    """
    if equivalence_bound <= 0:
        raise HardStopComputationError(
            check_id="TOST",
            reason=f"equivalence_bound must be positive (got {equivalence_bound}).",
        )

    x = np.asarray(pair_deltas, dtype=float)
    x = x[~np.isnan(x)]

    n = len(x)
    if n < 2:
        raise HardStopComputationError(
            check_id="TOST",
            reason=f"tost_equivalence: need ≥ 2 valid deltas, got {n}.",
        )

    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd == 0:
        raise HardStopComputationError(
            check_id="TOST",
            reason="tost_equivalence: zero variance in pair deltas.",
        )
    se = sd / np.sqrt(n)

    # Lower test: H0: mean ≤ -bound vs H1: mean > -bound
    t_lower = (mean - (-equivalence_bound)) / se
    p_lower = 1.0 - stats.t.cdf(t_lower, df=n - 1)

    # Upper test: H0: mean ≥ +bound vs H1: mean < +bound
    t_upper = (mean - equivalence_bound) / se
    p_upper = stats.t.cdf(t_upper, df=n - 1)

    tost_p = max(p_lower, p_upper)

    return TostResult(
        point_estimate=mean,
        lower_bound=-equivalence_bound,
        upper_bound=+equivalence_bound,
        p_lower=float(p_lower),
        p_upper=float(p_upper),
        tost_p_value=float(tost_p),
        equivalent_at_alpha=bool(tost_p < alpha),
        alpha=float(alpha),
    )
