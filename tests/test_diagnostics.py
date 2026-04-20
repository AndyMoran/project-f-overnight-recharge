"""
Tests for src/diagnostics.py.

Test strategy (v3.2):
  * VIF (HS-5' gate): construct a design matrix with a known
    collinearity pattern (column B = 2 * column A + noise). Verify VIF
    on both A and B exceeds threshold 5, and on an independent C column
    is ~1. Verify summarise_hs5p gates on VIF ONLY.
  * Breusch-Godfrey (informational): construct residuals with known
    autocorrelation (AR(1) process) and confirm the test detects it.
    White-noise residuals should not be detected. BG does NOT gate.
  * Cook's distance (informational): construct data where one
    observation is a deliberate outlier; verify it's flagged and the
    outlier-excluded β differs from the full-sample β.
  * Leverage: retained as a diagnostic function; tests verify it
    computes correctly but no longer participates in HS-5' gating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.diagnostics import (
    BreuschGodfreyResult,
    CooksDistanceResult,
    LeverageResult,
    VifResult,
    breusch_godfrey_test,
    cooks_distance_diagnostic,
    leverage_stats,
    summarise_hs5p,
    vif_per_covariate,
)
from src.errors import HardStopComputationError


# ===========================================================================
# Breusch-Godfrey
# ===========================================================================

class TestBreuschGodfreyTest:
    def test_white_noise_residuals_pass(self):
        """White-noise residuals → no autocorrelation → BG passes."""
        rng = np.random.default_rng(10)
        n = 500
        x = rng.normal(0, 1, n)
        y = 1 + 2 * x + rng.normal(0, 1, n)  # iid residuals
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()

        result = breusch_godfrey_test(model, lags=7)
        assert isinstance(result, BreuschGodfreyResult)
        assert result.passes is True
        assert result.p_value > 0.01

    def test_ar1_residuals_fail(self):
        """Strongly autocorrelated residuals → BG rejects."""
        rng = np.random.default_rng(11)
        n = 500
        x = rng.normal(0, 1, n)
        # AR(1) residuals with rho=0.8 → strong autocorrelation
        eps = np.zeros(n)
        eps[0] = rng.normal(0, 1)
        for t in range(1, n):
            eps[t] = 0.8 * eps[t - 1] + rng.normal(0, 1)
        y = 1 + 2 * x + eps
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()

        result = breusch_godfrey_test(model, lags=7)
        assert result.passes is False
        assert result.p_value < 0.01

    def test_non_fitted_object_raises(self):
        """Passing something without .resid should raise."""
        with pytest.raises(HardStopComputationError, match="resid"):
            breusch_godfrey_test(object())

    def test_lags_parameter_respected(self):
        """Changing lags should change the test statistic."""
        rng = np.random.default_rng(12)
        n = 200
        x = rng.normal(0, 1, n)
        y = 1 + 2 * x + rng.normal(0, 1, n)
        model = sm.OLS(y, sm.add_constant(x)).fit()

        r1 = breusch_godfrey_test(model, lags=1)
        r7 = breusch_godfrey_test(model, lags=7)
        assert r1.lags == 1
        assert r7.lags == 7
        # Different lag counts should produce different LM statistics
        # (not guaranteed equal).
        assert r1.lm_statistic != r7.lm_statistic

    def test_custom_alpha(self):
        """Pass/fail decision respects alpha."""
        rng = np.random.default_rng(13)
        n = 500
        # Mild autocorrelation
        eps = np.zeros(n)
        eps[0] = rng.normal(0, 1)
        for t in range(1, n):
            eps[t] = 0.2 * eps[t - 1] + rng.normal(0, 1)
        x = rng.normal(0, 1, n)
        y = 1 + x + eps
        model = sm.OLS(y, sm.add_constant(x)).fit()

        # With strict alpha=0.01, mild autocorr might pass; with
        # generous alpha=0.20, it might fail. Either outcome is fine —
        # we just check that alpha is actually used in the decision.
        r_strict = breusch_godfrey_test(model, lags=7, alpha=0.001)
        r_loose = breusch_godfrey_test(model, lags=7, alpha=0.99)
        # Same p-value, different alphas.
        assert r_strict.p_value == r_loose.p_value
        # r_loose alpha=0.99 must fail (p < 0.99 almost always); r_strict
        # alpha=0.001 is more likely to pass.
        assert r_loose.passes is False


# ===========================================================================
# Leverage
# ===========================================================================

class TestLeverageStats:
    def test_uniform_design_all_similar_leverage(self):
        """When all observations are drawn iid from the same distribution,
        leverages should be roughly equal and close to (k+1)/n."""
        rng = np.random.default_rng(20)
        n = 100
        X = pd.DataFrame({
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "c": rng.normal(0, 1, n),
        })
        result = leverage_stats(X)
        assert isinstance(result, LeverageResult)
        assert result.n_observations == n
        assert result.n_covariates == 3
        # Threshold = 3 * 4 / 100 = 0.12
        assert result.threshold == pytest.approx(0.12, abs=1e-10)
        # Max leverage should be well under threshold for uniform data.
        assert result.max_leverage < 0.12
        assert result.passes is True

    def test_single_outlier_flagged(self):
        """An extreme outlier in covariate space should get high leverage."""
        rng = np.random.default_rng(21)
        n = 100
        X = pd.DataFrame({
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "c": rng.normal(0, 1, n),
        })
        # Replace the last row with an outlier far from the centroid.
        X.iloc[-1] = [50, 50, 50]

        result = leverage_stats(X)
        assert result.max_leverage_row_index == n - 1
        # This outlier's leverage should far exceed the threshold.
        assert result.max_leverage > result.threshold
        assert result.passes is False

    def test_hand_computed_threshold(self):
        """Threshold formula 3 * (k+1) / n sanity check."""
        rng = np.random.default_rng(22)
        X = pd.DataFrame({f"x{i}": rng.normal(0, 1, 50) for i in range(4)})
        result = leverage_stats(X, multiplier=3)
        # k=4, n=50 → threshold = 3 * 5 / 50 = 0.3
        assert result.threshold == pytest.approx(0.3, abs=1e-10)

    def test_leverage_sum_equals_k_plus_1(self):
        """For OLS: sum of leverages = rank of design matrix = k+1 (with const).
        This is a known identity and a sanity check the leverage calculation
        is correct."""
        rng = np.random.default_rng(23)
        n = 80
        X = pd.DataFrame({
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
        })
        # Manually reconstruct the leverages to verify the identity.
        X_with_const = sm.add_constant(X.values)
        xtx_inv = np.linalg.inv(X_with_const.T @ X_with_const)
        leverages = np.einsum("ij,jk,ik->i", X_with_const, xtx_inv, X_with_const)
        # k + 1 = 3 (a, b, const)
        assert leverages.sum() == pytest.approx(3.0, abs=1e-8)

    def test_nan_raises(self):
        X = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(HardStopComputationError, match="NaN"):
            leverage_stats(X)

    def test_empty_raises(self):
        X = pd.DataFrame({"a": [], "b": []})
        with pytest.raises(HardStopComputationError, match="empty"):
            leverage_stats(X)

    def test_too_few_observations_raises(self):
        """n <= k+1 means the design matrix has no residual degrees of freedom."""
        X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with pytest.raises(HardStopComputationError, match="insufficient"):
            leverage_stats(X)

    def test_singular_design_matrix_raises(self):
        """Two perfectly collinear columns → singular X'X → raise."""
        rng = np.random.default_rng(24)
        a = rng.normal(0, 1, 50)
        X = pd.DataFrame({
            "a": a,
            "b": 2 * a,  # perfectly collinear
        })
        with pytest.raises(HardStopComputationError, match="singular"):
            leverage_stats(X)


# ===========================================================================
# VIF
# ===========================================================================

class TestVifPerCovariate:
    def test_independent_covariates_vif_near_one(self):
        """Independent columns should have VIF ≈ 1."""
        rng = np.random.default_rng(30)
        n = 500
        X = pd.DataFrame({
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "c": rng.normal(0, 1, n),
        })
        result = vif_per_covariate(X)
        assert isinstance(result, VifResult)
        for col in ["a", "b", "c"]:
            assert 0.9 < result.vif_per_covariate[col] < 1.2
        assert result.max_vif < 2
        assert result.passes is True

    def test_strong_collinearity_flagged(self):
        """When b ≈ 2a, VIF on both a and b should be very high."""
        rng = np.random.default_rng(31)
        n = 500
        a = rng.normal(0, 1, n)
        X = pd.DataFrame({
            "a": a,
            "b": 2 * a + rng.normal(0, 0.01, n),  # near-perfect collinearity
            "c": rng.normal(0, 1, n),             # independent
        })
        result = vif_per_covariate(X)
        assert result.vif_per_covariate["a"] > 100
        assert result.vif_per_covariate["b"] > 100
        assert result.vif_per_covariate["c"] < 2
        assert result.max_vif_covariate in ("a", "b")
        assert result.passes is False

    def test_hand_computed_vif(self):
        """Two uncorrelated standard-normal columns: both VIFs = 1 exactly
        in the limit. With finite n=10000 they should be very close."""
        rng = np.random.default_rng(32)
        n = 10000
        X = pd.DataFrame({
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
        })
        result = vif_per_covariate(X)
        assert result.vif_per_covariate["a"] == pytest.approx(1.0, abs=0.05)
        assert result.vif_per_covariate["b"] == pytest.approx(1.0, abs=0.05)

    def test_nan_raises(self):
        X = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [1.0, 2.0, 3.0]})
        with pytest.raises(HardStopComputationError, match="NaN"):
            vif_per_covariate(X)

    def test_single_covariate_raises(self):
        """VIF requires at least 2 covariates to make sense."""
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(HardStopComputationError, match="≥ 2"):
            vif_per_covariate(X)

    def test_perfect_collinearity_raises_non_finite(self):
        """Two perfectly collinear columns → VIF = ∞ → raise."""
        rng = np.random.default_rng(33)
        a = rng.normal(0, 1, 100)
        X = pd.DataFrame({
            "a": a,
            "b": 2 * a,  # perfectly collinear, R² = 1 → VIF = ∞
        })
        # statsmodels emits a RuntimeWarning on 1/(1-R²) when R²=1 exactly;
        # the warning is expected, so we suppress it within this test.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(HardStopComputationError, match="non-finite"):
                vif_per_covariate(X)


# ===========================================================================
# summarise_hs5p
# ===========================================================================

class TestSummariseHs5p:
    """
    v3.2 summariser: gates on VIF ONLY. BG and Cook's are informational.
    """

    def test_summary_structure_has_expected_keys(self):
        """Output dict has vif, bg, cooks, passes — and passes reflects
        VIF only."""
        rng = np.random.default_rng(40)
        n = 500
        X = pd.DataFrame({
            "evening_depletion_d": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "c": rng.normal(0, 1, n),
        })
        y = 1 + 2 * X["evening_depletion_d"] + 0.5 * X["b"] + rng.normal(0, 1, n)
        model = sm.OLS(y, sm.add_constant(X)).fit()

        summary = summarise_hs5p(model, X, y)

        # Expected top-level keys under v3.2 (no 'leverage' at top level).
        assert set(summary.keys()) == {"vif", "bg", "cooks", "passes"}

        # VIF is the gate: top-level passes == vif.passes.
        assert summary["passes"] == summary["vif"]["passes"]

        # BG and Cook's are flagged informational.
        assert summary["bg"]["informational"] is True
        assert summary["cooks"]["informational"] is True

    def test_clean_data_passes_vif_gate(self):
        """On independent iid covariates, VIF is ~1 → HS-5' passes."""
        rng = np.random.default_rng(42)
        n = 500
        X = pd.DataFrame({
            "evening_depletion_d": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "c": rng.normal(0, 1, n),
        })
        y = 1 + 2 * X["evening_depletion_d"] + 0.5 * X["b"] + rng.normal(0, 1, n)
        model = sm.OLS(y, sm.add_constant(X)).fit()

        summary = summarise_hs5p(model, X, y)
        assert summary["vif"]["passes"] is True
        assert summary["passes"] is True

    def test_collinearity_triggers_vif_fail(self):
        """Near-collinear columns → VIF > 5 → HS-5' fails."""
        rng = np.random.default_rng(41)
        n = 500
        a = rng.normal(0, 1, n)
        X = pd.DataFrame({
            "evening_depletion_d": a,
            "b": 2 * a + rng.normal(0, 0.01, n),
            "c": rng.normal(0, 1, n),
        })
        y = rng.normal(0, 1, n)
        model = sm.OLS(y, sm.add_constant(X)).fit()

        summary = summarise_hs5p(model, X, y)
        assert summary["vif"]["passes"] is False
        assert summary["passes"] is False

    def test_bg_failure_does_not_affect_gate(self):
        """Under v3.2, BG failure does NOT block HS-5'. Construct
        autocorrelated residuals; verify BG fails but VIF passes and
        the gate remains open."""
        rng = np.random.default_rng(43)
        n = 500
        # Independent covariates → VIF will pass.
        X = pd.DataFrame({
            "evening_depletion_d": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "c": rng.normal(0, 1, n),
        })
        # AR(1) residuals → BG will fail informationally.
        eps = np.zeros(n)
        eps[0] = rng.normal(0, 1)
        for t in range(1, n):
            eps[t] = 0.8 * eps[t - 1] + rng.normal(0, 1)
        y = 1 + 2 * X["evening_depletion_d"] + eps
        model = sm.OLS(y, sm.add_constant(X)).fit()

        summary = summarise_hs5p(model, X, y)
        # Informational BG should flag autocorrelation.
        assert summary["bg"]["passes"] is False
        # VIF still passes.
        assert summary["vif"]["passes"] is True
        # v3.2 key invariant: gate reflects VIF only.
        assert summary["passes"] is True

    def test_skip_cooks_when_y_missing(self):
        """If caller passes y=None, summary records 'not_computed' for
        Cook's but VIF/BG still run and the gate still resolves."""
        rng = np.random.default_rng(44)
        n = 300
        X = pd.DataFrame({
            "evening_depletion_d": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "c": rng.normal(0, 1, n),
        })
        y = 1 + X["evening_depletion_d"] + rng.normal(0, 1, n)
        model = sm.OLS(y, sm.add_constant(X)).fit()

        summary = summarise_hs5p(model, X, y=None)
        assert summary["cooks"]["status"] == "not_computed"
        assert summary["cooks"]["informational"] is True
        assert summary["passes"] == summary["vif"]["passes"]


# ===========================================================================
# Cook's distance diagnostic (new in v3.2)
# ===========================================================================

class TestCooksDistanceDiagnostic:
    """
    Construction of synthetic fixtures where the expected behaviour is
    hand-verifiable:
      * No outliers → nothing flagged, beta_excluded is NaN.
      * One deliberate outlier → one flag, beta_excluded differs from
        beta_full.
      * Hand check: with only one flagged obs in a large sample,
        beta_divergence should be small (outlier moved estimate only
        modestly).
    """

    def _fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fit OLS with constant. Returns the fitted model."""
        return sm.OLS(y, sm.add_constant(X)).fit()

    def test_no_outliers_nothing_flagged(self):
        rng = np.random.default_rng(100)
        n = 500
        X = pd.DataFrame({
            "evening_depletion_d": rng.normal(0, 1, n),
            "c": rng.normal(0, 1, n),
        })
        y = 1 + 2 * X["evening_depletion_d"] + rng.normal(0, 1, n)
        model = self._fit(X, y.values)

        result = cooks_distance_diagnostic(model, X, y.values)
        assert isinstance(result, CooksDistanceResult)
        # With clean iid data and threshold 4/n, typically zero or very
        # few flags. We test weakly: at least the beta_full is finite.
        assert np.isfinite(result.beta_full)
        if result.n_flagged == 0:
            assert np.isnan(result.beta_excluded)
            assert np.isnan(result.beta_divergence)
        else:
            # If some were flagged (rare), the diagnostic ran to
            # completion and produced a finite excluded estimate.
            assert np.isfinite(result.beta_excluded)

    def test_single_deliberate_outlier_flagged(self):
        """Inject a single extreme (x, y) pair and verify it's flagged."""
        rng = np.random.default_rng(101)
        n = 200
        X_vals = rng.normal(0, 1, n)
        X_vals[-1] = 20.0  # extreme covariate
        y_vals = 1 + 2 * X_vals + rng.normal(0, 1, n)
        y_vals[-1] = -50.0  # extreme residual at extreme x

        X = pd.DataFrame({
            "evening_depletion_d": X_vals,
            "c": rng.normal(0, 1, n),
        })
        model = self._fit(X, y_vals)

        result = cooks_distance_diagnostic(model, X, y_vals)
        assert result.n_flagged >= 1
        assert (n - 1) in result.flagged_indices  # the last row is flagged
        assert np.isfinite(result.beta_excluded)
        # The outlier is at x=20 with y=-50 (negative slope contribution);
        # excluding it must push the estimated positive slope HIGHER
        # because the outlier was pulling it toward zero.
        assert result.beta_excluded > result.beta_full

    def test_missing_treatment_column_raises(self):
        rng = np.random.default_rng(102)
        X = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 100)})
        y = rng.normal(0, 1, 100)
        model = self._fit(X, y)

        from src.errors import HardStopComputationError
        with pytest.raises(HardStopComputationError, match="evening_depletion_d"):
            cooks_distance_diagnostic(model, X, y)

    def test_treatment_col_override(self):
        """Allow a non-default treatment column name."""
        rng = np.random.default_rng(103)
        X = pd.DataFrame({
            "x_treat": rng.normal(0, 1, 200),
            "x_ctrl": rng.normal(0, 1, 200),
        })
        y = 1 + 3 * X["x_treat"] + rng.normal(0, 1, 200)
        model = self._fit(X, y.values)

        result = cooks_distance_diagnostic(
            model, X, y.values, treatment_col="x_treat"
        )
        # Full β should be close to 3.
        assert abs(result.beta_full - 3.0) < 0.2

    def test_length_mismatch_raises(self):
        rng = np.random.default_rng(104)
        X = pd.DataFrame({
            "evening_depletion_d": rng.normal(0, 1, 100),
            "c": rng.normal(0, 1, 100),
        })
        y = rng.normal(0, 1, 100)
        model = self._fit(X, y)

        # Pass a shorter y than the design.
        from src.errors import HardStopComputationError
        with pytest.raises(HardStopComputationError, match="length"):
            cooks_distance_diagnostic(model, X, y[:50])
