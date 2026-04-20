"""
Tests for src/matching.py.

Philosophy: every statistical function is tested against at least one
synthetic fixture with HAND-COMPUTED expected values. "It didn't crash"
is not a sufficient test for a helper that's going to determine whether
a pre-registered hypothesis passes or fails.

The fixtures are small (typically 4-8 rows) and deliberately simple so
that the expected values can be verified by hand from the formulas in
the module docstrings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import constants
from src.errors import HardStopComputationError
from src.matching import (
    MatchingResult,
    assign_terciles,
    build_matched_pairs,
    compute_deciles,
    compute_smd,
    hotelling_t2,
    summarise_balance,
)


# ===========================================================================
# FIXTURE HELPERS
# ===========================================================================

def _make_day_frame(
    n_days: int = 90,
    start_date: str = "2024-01-01",
    seed: int = 0,
) -> pd.DataFrame:
    """
    Build a synthetic day-level frame with all columns required for the
    matching pipeline. Depletion is drawn uniformly [0, 1]; covariates
    are independent normals.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_days, freq="D", tz="UTC")
    return pd.DataFrame({
        "date": dates,
        "day_of_week_d": dates.dayofweek,
        "evening_depletion_d": rng.uniform(0, 1, n_days),
        "DA_wind_forecast_evening_d": rng.normal(5, 2, n_days),
        "DA_demand_forecast_evening_d": rng.normal(30, 5, n_days),
        "DA_overnight_price_d": rng.normal(60, 15, n_days),
        "DA_IC_schedule_overnight_d": rng.normal(0, 2, n_days),
        "overnight_price_d_plus_1": rng.normal(60, 15, n_days),
    })


# ===========================================================================
# assign_terciles
# ===========================================================================

class TestAssignTerciles:
    def test_basic_split_has_three_groups(self):
        df = _make_day_frame(n_days=30, seed=1)  # all January — one month
        out = assign_terciles(df)
        assert set(out["tercile"].unique()) == {"low", "middle", "high"}

    def test_per_month_sizes_balanced(self):
        """qcut with q=3 should produce roughly-equal tercile sizes
        per month, within 1 from rounding."""
        df = _make_day_frame(n_days=31, seed=2)  # 31 days = roughly 10-11-10
        out = assign_terciles(df)
        counts = out["tercile"].value_counts()
        assert counts.min() >= 10
        assert counts.max() <= 11

    def test_middle_excluded_from_treatment_flags(self):
        df = _make_day_frame(n_days=30, seed=3)
        out = assign_terciles(df)
        middle_mask = out["tercile"] == "middle"
        assert not out.loc[middle_mask, "is_high_dep"].any()
        assert not out.loc[middle_mask, "is_low_dep"].any()

    def test_tercile_computed_per_month(self):
        """Depletion in January is 0-1; in February it's 100-200.
        Without per-month ranking, all Feb days would be HighDep.
        With per-month ranking, each month's top third is HighDep."""
        rng = np.random.default_rng(10)
        jan = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=15, freq="D", tz="UTC"),
            "evening_depletion_d": rng.uniform(0, 1, 15),
        })
        feb = pd.DataFrame({
            "date": pd.date_range("2024-02-01", periods=15, freq="D", tz="UTC"),
            "evening_depletion_d": rng.uniform(100, 200, 15),
        })
        df = pd.concat([jan, feb], ignore_index=True)
        out = assign_terciles(df)

        # HighDep count in January ≈ 5, in February ≈ 5 (each month has
        # its own top-third).
        jan_high = out[(out["date"].dt.month == 1) & out["is_high_dep"]]
        feb_high = out[(out["date"].dt.month == 2) & out["is_high_dep"]]
        assert 4 <= len(jan_high) <= 6
        assert 4 <= len(feb_high) <= 6

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        with pytest.raises(HardStopComputationError, match="missing"):
            assign_terciles(df)

    def test_all_tied_depletion_raises(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "evening_depletion_d": [0.5] * 10,  # all identical
        })
        with pytest.raises(HardStopComputationError, match="tercile"):
            assign_terciles(df)


# ===========================================================================
# compute_deciles
# ===========================================================================

class TestComputeDeciles:
    def test_returns_integers_1_to_10(self):
        df = pd.DataFrame({"x": np.arange(100, dtype=float)})
        deciles = compute_deciles(df, "x")
        assert deciles.min() == 1
        assert deciles.max() == 10
        assert set(deciles.unique()) == set(range(1, 11))

    def test_hand_computed_deciles_uniform(self):
        """For x = 0..99, deciles should split into 10 groups of 10.
        The smallest 10 values (0-9) are decile 1; the largest 10 (90-99)
        are decile 10."""
        df = pd.DataFrame({"x": np.arange(100, dtype=float)})
        deciles = compute_deciles(df, "x")
        # Check endpoints by hand.
        assert deciles.iloc[0] == 1    # x=0
        assert deciles.iloc[9] == 1    # x=9
        assert deciles.iloc[10] == 2   # x=10
        assert deciles.iloc[89] == 9   # x=89
        assert deciles.iloc[90] == 10  # x=90
        assert deciles.iloc[99] == 10  # x=99

    def test_missing_column_raises(self):
        df = pd.DataFrame({"y": [1, 2, 3]})
        with pytest.raises(HardStopComputationError):
            compute_deciles(df, "x")

    def test_heavy_ties_raises(self):
        """If the covariate has massive ties, qcut collapses bins and
        we should refuse to return a partial result."""
        df = pd.DataFrame({"x": [0.0] * 80 + list(range(20))})
        with pytest.raises(HardStopComputationError):
            compute_deciles(df, "x")


# ===========================================================================
# compute_smd
# ===========================================================================

class TestComputeSmd:
    def test_zero_when_identical(self):
        """SMD = 0 when both groups have identical values."""
        pairs = pd.DataFrame({
            "x_high": [1.0, 2.0, 3.0, 4.0],
            "x_low": [1.0, 2.0, 3.0, 4.0],
        })
        assert compute_smd(pairs, "x") == 0.0

    def test_positive_when_high_larger(self):
        pairs = pd.DataFrame({
            "x_high": [2.0, 3.0, 4.0, 5.0],  # mean 3.5
            "x_low": [1.0, 2.0, 3.0, 4.0],   # mean 2.5
        })
        # mean_diff = 1.0. Both groups have sample var = 1.667, pooled SD
        # = sqrt(1.667) ≈ 1.291. SMD ≈ 0.775.
        smd = compute_smd(pairs, "x")
        assert smd > 0
        expected = 1.0 / np.sqrt((1.6666667 + 1.6666667) / 2)
        assert smd == pytest.approx(expected, abs=1e-4)

    def test_hand_computed_smd(self):
        """Fully hand-computed example: two pairs, tight variance.
        HighDep values: [10, 12]; LowDep values: [8, 10].
        mean_high = 11, mean_low = 9 → diff = 2.
        var_high = 2 (ddof=1: sum((10-11)^2 + (12-11)^2) / (2-1) = 2).
        var_low = 2.
        pooled_sd = sqrt((2 + 2) / 2) = sqrt(2).
        SMD = 2 / sqrt(2) = sqrt(2) ≈ 1.4142.
        """
        pairs = pd.DataFrame({"x_high": [10.0, 12.0], "x_low": [8.0, 10.0]})
        assert compute_smd(pairs, "x") == pytest.approx(np.sqrt(2), abs=1e-10)

    def test_missing_columns_raises(self):
        pairs = pd.DataFrame({"y_high": [1], "y_low": [2]})
        with pytest.raises(HardStopComputationError):
            compute_smd(pairs, "x")

    def test_zero_pooled_sd_raises(self):
        pairs = pd.DataFrame({
            "x_high": [5.0, 5.0, 5.0],
            "x_low": [3.0, 3.0, 3.0],
        })
        with pytest.raises(HardStopComputationError, match="pooled SD"):
            compute_smd(pairs, "x")


# ===========================================================================
# hotelling_t2
# ===========================================================================

class TestHotellingT2:
    def test_balanced_groups_p_near_1(self):
        """Identical groups → T² = 0, p = 1."""
        rng = np.random.default_rng(42)
        n = 50
        # Two identical samples.
        x = rng.normal(0, 1, (n, 2))
        pairs = pd.DataFrame({
            "a_high": x[:, 0], "a_low": x[:, 0],
            "b_high": x[:, 1], "b_low": x[:, 1],
        })
        t2, p = hotelling_t2(pairs, covariates=("a", "b"))
        assert t2 == pytest.approx(0.0, abs=1e-10)
        assert p == pytest.approx(1.0, abs=1e-10)

    def test_separated_groups_p_small(self):
        """Clearly separated means → T² large, p small."""
        rng = np.random.default_rng(43)
        n = 50
        pairs = pd.DataFrame({
            "a_high": rng.normal(5, 1, n),
            "a_low": rng.normal(0, 1, n),
            "b_high": rng.normal(5, 1, n),
            "b_low": rng.normal(0, 1, n),
        })
        t2, p = hotelling_t2(pairs, covariates=("a", "b"))
        assert t2 > 50   # very separated
        assert p < 1e-10

    def test_missing_columns_raises(self):
        pairs = pd.DataFrame({"a_high": [1, 2]})
        with pytest.raises(HardStopComputationError):
            hotelling_t2(pairs, covariates=("a",))

    def test_insufficient_sample_raises(self):
        """Need n1+n2-p-1 > 0; if we ask for 2 covariates with only 1 pair,
        n1+n2-p-1 = 1+1-2-1 = -1, which should raise."""
        pairs = pd.DataFrame({
            "a_high": [1.0], "a_low": [2.0],
            "b_high": [3.0], "b_low": [4.0],
        })
        with pytest.raises(HardStopComputationError, match="sample size"):
            hotelling_t2(pairs, covariates=("a", "b"))


# ===========================================================================
# build_matched_pairs — integration tests
# ===========================================================================

class TestBuildMatchedPairs:
    def test_end_to_end_runs(self):
        """Full pipeline: 90 days, random data, should produce some pairs
        and not raise."""
        df = _make_day_frame(n_days=90, seed=100)
        df = assign_terciles(df)
        result = build_matched_pairs(df)
        assert isinstance(result, MatchingResult)
        assert result.n_high_dep_total > 0
        assert 0 <= result.n_high_dep_matched <= result.n_high_dep_total
        assert result.n_high_dep_matched + result.n_high_dep_dropped == result.n_high_dep_total

    def test_pair_columns_present(self):
        df = _make_day_frame(n_days=90, seed=101)
        df = assign_terciles(df)
        result = build_matched_pairs(df)
        if result.n_high_dep_matched > 0:
            expected = {
                "date_high", "date_low",
                "day_of_week_high", "day_of_week_low",
                "overnight_price_high", "overnight_price_low",
                "pair_distance",
            }
            for c in constants.CONTINUOUS_MATCHING_COVARIATES:
                expected.add(f"{c}_high")
                expected.add(f"{c}_low")
            assert expected.issubset(set(result.pairs.columns))

    def test_same_day_of_week(self):
        """Every matched pair must have equal day_of_week on both sides."""
        df = _make_day_frame(n_days=120, seed=102)
        df = assign_terciles(df)
        result = build_matched_pairs(df)
        if result.n_high_dep_matched > 0:
            assert (
                result.pairs["day_of_week_high"] == result.pairs["day_of_week_low"]
            ).all()

    def test_calendar_proximity_respected(self):
        """No matched pair may span more than ±14 days."""
        df = _make_day_frame(n_days=120, seed=103)
        df = assign_terciles(df)
        result = build_matched_pairs(df, calendar_proximity_days=14)
        if result.n_high_dep_matched > 0:
            day_diffs = (
                result.pairs["date_high"] - result.pairs["date_low"]
            ).dt.days.abs()
            assert (day_diffs <= 14).all()

    def test_without_replacement(self):
        """No LowDep date appears more than once across the pairs."""
        df = _make_day_frame(n_days=150, seed=104)
        df = assign_terciles(df)
        result = build_matched_pairs(df)
        if result.n_high_dep_matched > 1:
            low_dates = result.pairs["date_low"]
            assert low_dates.is_unique, (
                f"LowDep dates reused: {low_dates.value_counts().head()}"
            )

    def test_dropped_dates_not_in_pairs(self):
        """Dropped HighDep dates should not appear in the pairs frame."""
        df = _make_day_frame(n_days=150, seed=105)
        df = assign_terciles(df)
        result = build_matched_pairs(df)
        if result.n_high_dep_dropped > 0 and result.n_high_dep_matched > 0:
            matched = set(result.pairs["date_high"])
            dropped = set(result.dropped_date_list)
            assert matched.isdisjoint(dropped)

    def test_dropped_proportion_matches_counts(self):
        df = _make_day_frame(n_days=90, seed=106)
        df = assign_terciles(df)
        result = build_matched_pairs(df)
        assert result.dropped_proportion == pytest.approx(
            result.n_high_dep_dropped / result.n_high_dep_total
        )

    def test_empty_tercile_raises(self):
        """If assign_terciles hasn't run, is_high_dep is missing."""
        df = _make_day_frame(n_days=30, seed=107)
        with pytest.raises(HardStopComputationError, match="missing required"):
            build_matched_pairs(df)


# ===========================================================================
# summarise_balance
# ===========================================================================

class TestSummariseBalance:
    def test_well_balanced_data_passes_smd(self):
        """Construct pairs where HighDep and LowDep are identical — all
        SMDs must be 0 and all_smd_ok should be True."""
        n = 20
        cov_names = constants.CONTINUOUS_MATCHING_COVARIATES
        rng = np.random.default_rng(200)
        data = {}
        for c in cov_names:
            vals = rng.normal(0, 1, n)
            data[f"{c}_high"] = vals
            data[f"{c}_low"] = vals.copy()
        pairs = pd.DataFrame(data)

        summary = summarise_balance(pairs)
        for c in cov_names:
            assert summary["smd_per_covariate"][c] == pytest.approx(0.0, abs=1e-10)
        assert summary["max_abs_smd"] == pytest.approx(0.0, abs=1e-10)
        assert summary["all_smd_ok"] is True
        assert summary["hotelling_pvalue"] == pytest.approx(1.0, abs=1e-10)

    def test_unbalanced_data_triggers_smd_threshold(self):
        """Shift one covariate by 1 SD between groups — SMD on that one
        should be ≈ 1, breaching the 0.10 threshold."""
        n = 50
        cov_names = constants.CONTINUOUS_MATCHING_COVARIATES
        rng = np.random.default_rng(201)
        data = {}
        for i, c in enumerate(cov_names):
            base = rng.normal(0, 1, n)
            data[f"{c}_low"] = base
            if i == 0:
                # Shift first covariate by 1 SD → SMD ~ 1
                data[f"{c}_high"] = base + 1.0
            else:
                data[f"{c}_high"] = base + rng.normal(0, 0.01, n)
        pairs = pd.DataFrame(data)

        summary = summarise_balance(pairs)
        # First covariate should have large SMD, others small.
        first_smd = summary["smd_per_covariate"][cov_names[0]]
        assert abs(first_smd) > 0.5
        assert summary["all_smd_ok"] is False
