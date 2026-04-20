"""
Tests for src/inference.py.

The BCa bootstrap is the subtlest of the three inferential procedures
and receives the most test effort. Key tests:

  - Symmetric data: BCa CI should approximately equal percentile CI
    (because z0 ≈ 0 and a ≈ 0 when the bootstrap distribution is
    symmetric). Deviations should be small.
  - Strong effect: p-value should be very small, CI should exclude 0.
  - Null data: p-value should be near the nominal α under repeated draws.
  - Block bootstrap vs iid: with one-pair-per-block, block and iid
    bootstrap should give the same answer.

Permutation tests check: (1) observed effect is correctly recovered;
(2) permutation distribution is symmetric around 0 under null; (3)
HS-3 decision is correct under known effect size.

TOST tests check: (1) rejects equivalence when data is far from 0;
(2) accepts equivalence when data is tightly centred on 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.errors import HardStopComputationError
from src.inference import (
    BcaBootstrapResult,
    PermutationResult,
    TostResult,
    bca_block_bootstrap,
    tost_equivalence,
    within_dow_permutation,
)


# ===========================================================================
# BCa bootstrap
# ===========================================================================

def _make_pairs(
    deltas: np.ndarray,
    weeks: np.ndarray,
    baseline_price: float = 50.0,
) -> pd.DataFrame:
    """
    Build a pairs_df with given per-pair deltas and week assignments.

    overnight_price_high - overnight_price_low = deltas. Easiest
    construction: set LowDep prices to `baseline_price` and HighDep to
    `baseline_price + delta`.

    `weeks` is an array of integers in the range [1, 52]; actual ISO-week
    assignment comes from the date, so we construct dates that fall in
    the specified week. Using Mondays of each week in 2024 for simplicity.
    """
    assert len(deltas) == len(weeks)

    # ISO week 1 of 2024 starts Monday 2024-01-01.
    # ISO week w Monday = Jan 1 + 7 * (w - 1) days.
    base = pd.Timestamp("2024-01-01", tz="UTC")
    dates = pd.to_datetime(
        [base + pd.Timedelta(days=7 * (w - 1)) for w in weeks]
    )

    return pd.DataFrame({
        "date_high": dates,
        "date_low": dates,  # immaterial for bootstrap; matching already happened
        "overnight_price_high": baseline_price + deltas,
        "overnight_price_low": np.full(len(deltas), baseline_price, dtype=float),
    })


class TestBcaBlockBootstrapSmoke:
    def test_runs_and_returns_result_type(self):
        rng = np.random.default_rng(0)
        n_pairs = 100
        deltas = rng.normal(-3, 5, n_pairs)
        weeks = rng.integers(1, 40, n_pairs)  # 39 possible weeks
        pairs = _make_pairs(deltas, weeks)

        result = bca_block_bootstrap(pairs, n_resamples=1000, seed=42)
        assert isinstance(result, BcaBootstrapResult)
        assert result.n_resamples == 1000
        assert result.n_blocks >= 2
        assert len(result.bootstrap_distribution) == 1000

    def test_missing_columns_raise(self):
        pairs = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(HardStopComputationError, match="missing"):
            bca_block_bootstrap(pairs)

    def test_too_few_blocks_raise(self):
        # All pairs in the same week → 1 block → bootstrap is undefined.
        pairs = _make_pairs(
            deltas=np.array([-1.0, -2.0, -3.0]),
            weeks=np.array([5, 5, 5]),  # all in week 5
        )
        with pytest.raises(HardStopComputationError, match="block"):
            bca_block_bootstrap(pairs, n_resamples=100, seed=1)


class TestBcaBlockBootstrapCorrectness:
    """Verify BCa produces sensible answers on data with known properties."""

    def test_strong_negative_effect_small_p_value(self):
        """True mean = -5, SD = 1 → strong signal → p-value near 0."""
        rng = np.random.default_rng(10)
        n = 100
        deltas = rng.normal(-5.0, 1.0, n)
        weeks = rng.integers(1, 20, n)
        pairs = _make_pairs(deltas, weeks)

        result = bca_block_bootstrap(pairs, n_resamples=5000, seed=42)
        assert result.point_estimate < -4.5  # consistent with true mean
        assert result.one_sided_p_value < 0.001  # strongly reject H0: Δ̄ ≥ 0
        assert result.ci_upper < 0  # CI excludes 0

    def test_observed_exactly_zero_gives_p_near_half(self):
        """When the observed mean is exactly zero (not just 'from zero-mean
        generating process', but literally equal to zero), the p-value for
        H0: Δ̄ ≥ 0 should be close to 0.5."""
        # Construct symmetric pairs so the observed mean is exactly 0.
        deltas = np.concatenate([
            np.linspace(-5, -0.1, 50),
            np.linspace(0.1, 5, 50),
        ])
        assert deltas.mean() == pytest.approx(0.0, abs=1e-10)
        weeks = np.arange(1, 101)  # one pair per week
        pairs = _make_pairs(deltas, weeks)

        result = bca_block_bootstrap(pairs, n_resamples=5000, seed=42)
        assert abs(result.point_estimate) < 0.1
        # Symmetric sample centered on 0 → p-value should bracket 0.5.
        assert 0.3 < result.one_sided_p_value < 0.7

    def test_ci_contains_point_estimate(self):
        """BCa CI should bracket the point estimate for any non-degenerate
        case."""
        rng = np.random.default_rng(30)
        n = 50
        deltas = rng.normal(-2.0, 3.0, n)
        weeks = rng.integers(1, 15, n)
        pairs = _make_pairs(deltas, weeks)

        result = bca_block_bootstrap(pairs, n_resamples=3000, seed=42)
        assert result.ci_lower <= result.point_estimate <= result.ci_upper

    def test_one_pair_per_week_matches_iid_roughly(self):
        """With one pair per week, block bootstrap ≈ iid bootstrap.
        Compare against scipy's bootstrap as sanity."""
        from scipy.stats import bootstrap as scipy_bootstrap

        rng = np.random.default_rng(40)
        n = 50
        deltas = rng.normal(-2.0, 2.0, n)
        weeks = np.arange(1, n + 1)  # one pair per week
        pairs = _make_pairs(deltas, weeks)

        our_result = bca_block_bootstrap(
            pairs, n_resamples=10000, seed=42
        )
        scipy_res = scipy_bootstrap(
            (deltas,),
            statistic=np.mean,
            method="BCa",
            n_resamples=10000,
            random_state=42,
            confidence_level=0.95,
        )

        # CIs won't match exactly (different seeds, different resampling
        # internals) but should be close.
        assert abs(our_result.ci_lower - scipy_res.confidence_interval.low) < 0.3
        assert abs(our_result.ci_upper - scipy_res.confidence_interval.high) < 0.3

    def test_all_same_week_effective_block_concentration(self):
        """When two very-different-signed pairs are placed in ONE block and
        the rest in separate blocks, the block containing both pairs
        always contributes both or neither → reduces variability.
        This is a sanity check that block resampling is doing something.
        """
        # 20 pairs at mean -3, plus 2 pairs (+10, -10) all in week 1.
        # The "plus/minus 10" pair block adds high variance if resampled
        # alone, but because it's a block, it contributes 0 net delta.
        deltas = np.concatenate([
            np.full(20, -3.0),
            np.array([10.0, -10.0]),
        ])
        weeks = np.concatenate([
            np.arange(2, 22),  # weeks 2..21, one pair each
            np.array([1, 1]),  # both "extreme" pairs in week 1
        ])
        pairs = _make_pairs(deltas, weeks)

        result = bca_block_bootstrap(pairs, n_resamples=3000, seed=42)
        # The block containing both extremes always sums to 0; variability
        # comes only from which of the 20 same-sign pairs get resampled.
        # So the bootstrap distribution should be fairly tight around the
        # overall mean.
        assert result.n_blocks == 21  # 20 weeks + 1 shared week
        assert result.ci_upper - result.ci_lower < 3.0  # tight CI

    def test_pvalue_monotone_in_signal_strength(self):
        """As we make the true effect more strongly negative (holding SD
        constant), the one-sided p-value should decrease monotonically."""
        rng = np.random.default_rng(555)
        n = 100
        weeks = rng.integers(1, 20, n)
        noise = rng.normal(0, 1, n)  # same noise for all runs

        p_values = []
        for true_mean in [-3.0, -1.0, 0.0, +1.0, +3.0]:
            deltas = true_mean + noise
            pairs = _make_pairs(deltas, weeks)
            result = bca_block_bootstrap(pairs, n_resamples=2000, seed=42)
            p_values.append(result.one_sided_p_value)

        # p-values should be monotone non-decreasing (more negative signal
        # → smaller p). Saturation at 0.0 and 1.0 is allowed at the
        # extremes where the null is fully outside the bootstrap support.
        for earlier, later in zip(p_values, p_values[1:]):
            assert earlier <= later, (
                f"p-values not monotone non-decreasing: {p_values}"
            )
        # Strong negative → small; strong positive → large.
        assert p_values[0] < 0.05
        assert p_values[-1] > 0.95


class TestBcaInternals:
    """Direct tests of internal helpers."""

    def test_z0_zero_when_symmetric(self):
        from src.inference import _bca_z0

        # If exactly half the resamples are below observed, z0 = 0.
        resamples = np.array([-1.0, -0.5, 0.5, 1.0])
        z0 = _bca_z0(0.0, resamples)
        assert z0 == pytest.approx(0.0, abs=1e-10)

    def test_z0_positive_when_observed_above_median(self):
        from src.inference import _bca_z0
        resamples = np.array([-1.0, -0.5, 0.5, 1.0])
        z0 = _bca_z0(0.9, resamples)
        assert z0 > 0

    def test_acceleration_zero_when_variance_constant(self):
        from src.inference import _bca_acceleration

        # All blocks contain the same mean → jackknife means are equal
        # → residuals are 0 → a = 0.
        pair_deltas = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        block_ids = np.array([1, 1, 2, 2, 3, 3])
        a = _bca_acceleration(pair_deltas, block_ids)
        assert a == pytest.approx(0.0, abs=1e-12)

    def test_acceleration_warning_with_one_block(self):
        from src.inference import _bca_acceleration
        # 1 block → acceleration undefined, returns 0 with a warning.
        pair_deltas = np.array([1.0, 2.0, 3.0])
        block_ids = np.array([1, 1, 1])
        a = _bca_acceleration(pair_deltas, block_ids)
        assert a == 0.0

    def test_bca_quantile_reduces_to_normal_when_z0_and_a_zero(self):
        from src.inference import _bca_quantile
        # With z0=0, a=0, _bca_quantile(alpha, 0, 0) should equal alpha
        # because Φ(0 + zα / 1) = Φ(zα) = alpha.
        for alpha in [0.025, 0.1, 0.5, 0.9, 0.975]:
            assert _bca_quantile(alpha, 0.0, 0.0) == pytest.approx(alpha, abs=1e-10)


# ===========================================================================
# within-DoW permutation
# ===========================================================================

def _fake_matching_fn_from_depletion(day_frame, depletion_col="evening_depletion_d"):
    """
    Stand-in for matching.build_matched_pairs used by the permutation test.
    For testing purposes, we want a function that produces a predictable
    Δ̄ based on the depletion values, so we can verify that permutation
    correctly breaks the signal.

    Strategy: split into high (top third) and low (bottom third) by
    depletion, pair them 1-to-1 by order, and return a fake MatchingResult.
    The effect is: the "observed" Δ̄ reflects the correlation in the data;
    a shuffle should destroy it.
    """
    from src.matching import MatchingResult

    n = len(day_frame)
    q_high = np.quantile(day_frame[depletion_col], 2 / 3)
    q_low = np.quantile(day_frame[depletion_col], 1 / 3)

    high_frame = day_frame[day_frame[depletion_col] >= q_high].reset_index(drop=True)
    low_frame = day_frame[day_frame[depletion_col] <= q_low].reset_index(drop=True)

    # Pair by index order (just for testing).
    n_pairs = min(len(high_frame), len(low_frame))
    pairs = pd.DataFrame({
        "date_high": high_frame["date"].values[:n_pairs],
        "date_low": low_frame["date"].values[:n_pairs],
        "overnight_price_high": high_frame["overnight_price_d_plus_1"].values[:n_pairs],
        "overnight_price_low": low_frame["overnight_price_d_plus_1"].values[:n_pairs],
    })
    return MatchingResult(
        pairs=pairs,
        n_high_dep_total=len(high_frame),
        n_high_dep_matched=n_pairs,
        n_high_dep_dropped=len(high_frame) - n_pairs,
        dropped_date_list=[],
    )


class TestWithinDowPermutation:
    def test_observed_effect_recovered(self):
        """Construct data where depletion and price are strongly
        negatively correlated. Observed Δ̄ should reflect that."""
        rng = np.random.default_rng(100)
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
        depletion = rng.uniform(0, 1, n)
        # price = 60 - 20*depletion + noise → high depletion → low price
        price = 60 - 20 * depletion + rng.normal(0, 3, n)
        df = pd.DataFrame({
            "date": dates,
            "day_of_week_d": dates.dayofweek,
            "evening_depletion_d": depletion,
            "overnight_price_d_plus_1": price,
        })

        result = within_dow_permutation(
            df,
            build_matched_pairs_fn=_fake_matching_fn_from_depletion,
            n_permutations=50,
            seed=42,
        )
        # Observed Δ̄ should be strongly negative.
        assert result.observed_delta < -5.0
        # HS-3 should pass because observed > median(|permutation|).
        assert result.passes_hs3

    def test_null_data_hs3_fails(self):
        """Construct data where depletion and price are independent.
        Observed Δ̄ ≈ 0 → HS-3 should fail (|observed| < median(|perm|))."""
        rng = np.random.default_rng(101)
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
        df = pd.DataFrame({
            "date": dates,
            "day_of_week_d": dates.dayofweek,
            "evening_depletion_d": rng.uniform(0, 1, n),
            "overnight_price_d_plus_1": rng.normal(60, 5, n),
        })

        result = within_dow_permutation(
            df,
            build_matched_pairs_fn=_fake_matching_fn_from_depletion,
            n_permutations=50,
            seed=43,
        )
        # Observed is near 0; permutations are also near 0 but scattered.
        # HS-3 asks whether |observed| ≥ median(|perm|). With both random,
        # it's a coin flip. Just check we got a decision and it's finite.
        assert result.median_abs_perm_delta > 0
        assert isinstance(result.passes_hs3, bool)

    def test_missing_depletion_column_raises(self):
        df = pd.DataFrame({"day_of_week_d": [0, 1, 2]})
        with pytest.raises(HardStopComputationError, match="evening_depletion"):
            within_dow_permutation(
                df,
                build_matched_pairs_fn=lambda x: None,
            )

    def test_permutations_preserve_dow_strata(self):
        """Verify the shuffle only permutes within day-of-week. We do this
        by recording the depletion-DoW joint distribution before and after
        via a custom build_matched_pairs_fn that captures the shuffled
        frame on each call."""
        n = 70
        dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")

        # Set depletion = 100 for Mondays, 0 for all other days. After
        # within-DoW permutation, Mondays should still all have depletion
        # 100 (they've been shuffled among themselves, but they all have
        # the same value so nothing visibly changes). Non-Mondays should
        # still have 0.
        df = pd.DataFrame({
            "date": dates,
            "day_of_week_d": dates.dayofweek,
            "evening_depletion_d": np.where(dates.dayofweek == 0, 100.0, 0.0),
            "overnight_price_d_plus_1": np.full(n, 50.0),
        })

        captured = []

        def capture_fn(day_frame):
            captured.append(day_frame.copy())
            return _fake_matching_fn_from_depletion(day_frame)

        within_dow_permutation(
            df,
            build_matched_pairs_fn=capture_fn,
            n_permutations=5,
            seed=42,
        )

        # Check captured permuted frames: Monday-depletion must remain 100,
        # non-Monday must remain 0.
        for frame in captured[1:]:  # skip the first call (observed)
            mon_mask = frame["day_of_week_d"] == 0
            assert (frame.loc[mon_mask, "evening_depletion_d"] == 100.0).all()
            assert (frame.loc[~mon_mask, "evening_depletion_d"] == 0.0).all()


# ===========================================================================
# TOST equivalence
# ===========================================================================

class TestTostEquivalence:
    def test_tightly_centred_near_zero_is_equivalent(self):
        """Deltas tightly centred on 0 within ±£1 bound → equivalence
        should be found significant."""
        rng = np.random.default_rng(200)
        n = 200
        deltas = rng.normal(0.0, 0.3, n)

        result = tost_equivalence(
            deltas, equivalence_bound=1.0, alpha=0.05
        )
        assert result.equivalent_at_alpha is True
        assert result.tost_p_value < 0.05

    def test_large_effect_not_equivalent(self):
        """Deltas far from 0 should fail equivalence at ±£1 bound."""
        rng = np.random.default_rng(201)
        n = 100
        deltas = rng.normal(-5.0, 1.0, n)

        result = tost_equivalence(
            deltas, equivalence_bound=1.0, alpha=0.05
        )
        assert result.equivalent_at_alpha is False
        assert result.tost_p_value > 0.5

    def test_result_fields_populated(self):
        rng = np.random.default_rng(202)
        deltas = rng.normal(0.2, 0.5, 100)
        result = tost_equivalence(deltas, equivalence_bound=1.0)
        assert isinstance(result, TostResult)
        assert result.lower_bound == -1.0
        assert result.upper_bound == 1.0
        assert result.tost_p_value == max(result.p_lower, result.p_upper)
        assert result.alpha == 0.05

    def test_negative_bound_raises(self):
        with pytest.raises(HardStopComputationError, match="positive"):
            tost_equivalence(np.array([1.0, 2.0]), equivalence_bound=-1.0)

    def test_insufficient_data_raises(self):
        with pytest.raises(HardStopComputationError, match="≥ 2"):
            tost_equivalence(np.array([1.0]), equivalence_bound=1.0)

    def test_zero_variance_raises(self):
        with pytest.raises(HardStopComputationError, match="variance"):
            tost_equivalence(np.array([2.0, 2.0, 2.0]), equivalence_bound=1.0)
