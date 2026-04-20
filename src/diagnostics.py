"""
Diagnostics for the continuous OLS specification.
=================================================

The continuous specification is (pre-reg §Secondary test):

  overnight_price_{d+1} = α + β * evening_depletion_d
                          + γ * (DA matching covariates, continuous)
                          + month_FE + day-of-week_FE + ε_d

This module provides four diagnostic helpers:

  1. vif_per_covariate       — HS-5' hard-stop gate (v3.2).
  2. breusch_godfrey_test    — informational residual autocorrelation test.
  3. cooks_distance_diagnostic — informational outlier-robustness check.
  4. leverage_stats          — diagnostic only; retained for inspection use.

HS-5' scope (v3.2, per DEVIATIONS.md Entry 001)
-----------------------------------------------
Only VIF gates HS-5'. If max VIF on the five continuous covariates
(evening_depletion_d + four DA matching covariates) exceeds 5, X'X
is near-singular and the continuous β̂ is mathematically unreliable.

Breusch-Godfrey and Cook's distance are computed and reported but do
NOT gate Notebook 01. Residual autocorrelation is expected in
sequential grid data and is absorbed by the pre-registered Newey-West
HAC standard errors. Cook's distance identifies influential points
for an outlier-excluded robustness estimate of β; substantial
divergence between full-sample and outlier-excluded β informs
interpretation but does not stop the study.

Leverage is retained as a diagnostic function for ad-hoc inspection
but is not wired into `summarise_hs5p` — it was removed from the
gate in v3.2 after simulation showed the `3×(k+1)/n` threshold fails
on clean iid data at Project F sample sizes (see DEVIATIONS.md 001).

Scope of "continuous covariates" for VIF, leverage, Cook's distance
-------------------------------------------------------------------
All three operate on the design matrix of continuous covariates,
NOT the full design matrix with month and day-of-week fixed effects.
Fixed effects are binary indicators that can produce high apparent
leverage on rare categories (e.g. a single February observation) —
that is not the concern HS-5' is trying to catch. HS-5' targets
collinearity among the continuous variables specifically.

Breusch-Godfrey operates on the fitted-model residuals and so takes
the fitted model directly, with FEs included, because the test is
about the residual time-series structure, not the design geometry.

Pre-registration disambiguation
-------------------------------

  (D) VIF, leverage, and Cook's distance operate on the five-column
      design matrix: evening_depletion_d, DA_wind_forecast_evening_d,
      DA_demand_forecast_evening_d, DA_overnight_price_d,
      DA_IC_schedule_overnight_d. A constant is added internally.
      Fixed-effect columns are excluded.

  (E) Breusch-Godfrey is run at lag = HAC_BANDWIDTH_DAYS (7 from
      constants.py), matching the HAC bandwidth.

  (F) Cook's distance flags observations with D_i > 4/n, re-estimates
      β on `evening_depletion_d` excluding ALL flagged observations
      simultaneously (not leave-one-out), and reports both full-sample
      and outlier-excluded β.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .constants import (
    BG_INFORMATIONAL_ALPHA,
    COOKS_DISTANCE_MULTIPLIER,
    CONTINUOUS_MATCHING_COVARIATES,
    HAC_BANDWIDTH_DAYS,
    HS5P_LEVERAGE_MULTIPLIER,
    HS5P_VIF_THRESHOLD,
)
from .errors import HardStopComputationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BreuschGodfreyResult:
    """Output of breusch_godfrey_test."""
    lm_statistic: float
    p_value: float
    lags: int
    passes: bool


@dataclass
class LeverageResult:
    """Output of leverage_stats."""
    max_leverage: float
    max_leverage_row_index: int
    threshold: float
    n_observations: int
    n_covariates: int
    passes: bool


@dataclass
class VifResult:
    """Output of vif_per_covariate."""
    vif_per_covariate: dict[str, float]
    max_vif: float
    max_vif_covariate: str
    threshold: float
    passes: bool


@dataclass
class CooksDistanceResult:
    """
    Output of cooks_distance_diagnostic.

    Attributes
    ----------
    beta_full : full-sample coefficient on the treatment variable
        (evening_depletion_d).
    beta_excluded : coefficient after excluding all flagged observations.
        NaN if no observations were flagged (in which case beta_excluded
        equals beta_full; the field is set to NaN so the caller can
        distinguish "nothing to exclude" from "excluded and got same
        answer").
    n_flagged : number of observations with D_i > threshold.
    flagged_indices : list of integer row indices (into the input
        continuous_design_matrix) that exceeded the threshold.
    threshold : Cook's distance cutoff = multiplier / n.
    max_cooks_d : maximum observed Cook's distance.
    beta_divergence : relative change |beta_excluded - beta_full| /
        |beta_full|. NaN if beta_excluded is NaN (nothing flagged) or
        beta_full is zero.
    """
    beta_full: float
    beta_excluded: float
    n_flagged: int
    flagged_indices: list
    threshold: float
    max_cooks_d: float
    beta_divergence: float


# ---------------------------------------------------------------------------
# Breusch-Godfrey
# ---------------------------------------------------------------------------

def breusch_godfrey_test(
    fitted_ols_model,
    *,
    lags: int = HAC_BANDWIDTH_DAYS,
    alpha: float = BG_INFORMATIONAL_ALPHA,
) -> BreuschGodfreyResult:
    """
    Breusch-Godfrey LM test for residual autocorrelation up to `lags`.

    INFORMATIONAL ONLY (v3.2). The test H0 is "no serial correlation at
    lags 1..lags". A low p-value rejects H0 and indicates mild residual
    autocorrelation — expected in sequential grid data and absorbed by
    the pre-registered Newey-West HAC standard errors. BG does NOT gate
    HS-5' in v3.2; see DEVIATIONS.md Entry 001.

    The returned `passes` field is True iff `p >= alpha` (i.e. the test
    does not reject "no autocorrelation"). A False value indicates mild
    autocorrelation is present and should be noted in interpretation,
    but inference proceeds on HAC-corrected standard errors regardless.

    Parameters
    ----------
    fitted_ols_model : a fitted statsmodels OLSResults object. Must
        expose `.resid` and be compatible with
        statsmodels.stats.diagnostic.acorr_breusch_godfrey.
    lags : number of lags to test. Default HAC_BANDWIDTH_DAYS (7).
    alpha : informational significance level. Default
        BG_INFORMATIONAL_ALPHA (0.05).

    Returns
    -------
    BreuschGodfreyResult

    Raises
    ------
    HardStopComputationError : if the fitted model lacks residuals, or
        the test itself raises.
    """
    if not hasattr(fitted_ols_model, "resid"):
        raise HardStopComputationError(
            check_id="HS-5'",
            reason=(
                "breusch_godfrey_test: passed object has no `.resid` "
                "attribute. Expected a fitted statsmodels OLSResults object."
            ),
        )

    try:
        lm_stat, lm_pvalue, _f_stat, _f_pvalue = acorr_breusch_godfrey(
            fitted_ols_model, nlags=lags
        )
    except Exception as e:
        raise HardStopComputationError(
            check_id="HS-5'",
            reason=f"Breusch-Godfrey test failed: {e}",
            details={"lags": lags},
        )

    return BreuschGodfreyResult(
        lm_statistic=float(lm_stat),
        p_value=float(lm_pvalue),
        lags=int(lags),
        passes=bool(lm_pvalue >= alpha),
    )


# ---------------------------------------------------------------------------
# Leverage
# ---------------------------------------------------------------------------

def leverage_stats(
    design_matrix: pd.DataFrame,
    *,
    multiplier: float = HS5P_LEVERAGE_MULTIPLIER,
) -> LeverageResult:
    """
    Compute the maximum leverage (hat-matrix diagonal) on the design
    matrix of continuous covariates.

    Leverage threshold per pre-reg HS-5':
        h_ii > multiplier * (k + 1) / n  ⇒  FLAG

    where:
        k = number of continuous covariates in `design_matrix`
            (the added constant is the "+1" in the formula)
        n = number of observations
        multiplier = 3 by default

    Pre-reg §Hard stops HS-5' passes iff max(h_ii) ≤ threshold.

    Parameters
    ----------
    design_matrix : DataFrame of continuous covariates only (NO fixed
        effects, NO constant column — a constant is added internally).
        One row per observation, one column per covariate. Must have
        no NaN values; caller is responsible for dropna.
    multiplier : leverage threshold multiplier. Default 3 per pre-reg.

    Returns
    -------
    LeverageResult

    Raises
    ------
    HardStopComputationError : if the matrix is empty, has NaN values,
        or produces a singular normal-equations matrix.
    """
    if design_matrix.isna().any().any():
        raise HardStopComputationError(
            check_id="HS-5'",
            reason=(
                "leverage_stats: design matrix contains NaN. "
                "Caller must dropna first so n and k reflect actual "
                "observations used in the regression."
            ),
        )

    n, k = design_matrix.shape
    if n == 0:
        raise HardStopComputationError(
            check_id="HS-5'",
            reason="leverage_stats: empty design matrix.",
        )
    if k == 0:
        raise HardStopComputationError(
            check_id="HS-5'",
            reason="leverage_stats: design matrix has no columns.",
        )
    if n <= k + 1:
        raise HardStopComputationError(
            check_id="HS-5'",
            reason=(
                f"leverage_stats: insufficient observations "
                f"(n={n}, k={k}). Need n > k+1."
            ),
        )

    # Build X with constant column; compute hat diagonals.
    X = sm.add_constant(design_matrix.values, has_constant="add")

    # Detect rank deficiency explicitly. numpy.linalg.inv does NOT raise
    # on a singular matrix — it returns an inverse polluted with huge
    # numerical errors. Use matrix_rank with a tolerance for the check.
    rank = np.linalg.matrix_rank(X)
    expected_rank = X.shape[1]  # n_covariates + 1 (constant)
    if rank < expected_rank:
        raise HardStopComputationError(
            check_id="HS-5'",
            reason=(
                f"leverage_stats: design matrix is singular "
                f"(rank {rank} < expected {expected_rank}). Likely cause: "
                f"perfect collinearity between continuous covariates."
            ),
            details={"covariates": list(design_matrix.columns)},
        )

    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError as e:
        raise HardStopComputationError(
            check_id="HS-5'",
            reason=(
                f"leverage_stats: (X'X) is singular. Likely cause: "
                f"perfect collinearity between continuous covariates. "
                f"Error: {e}"
            ),
            details={"covariates": list(design_matrix.columns)},
        )

    # Vectorised hat-diagonal: h_ii = x_i' (X'X)^-1 x_i
    leverages = np.einsum("ij,jk,ik->i", X, xtx_inv, X)

    max_lev = float(leverages.max())
    max_idx = int(np.argmax(leverages))
    threshold = multiplier * (k + 1) / n

    return LeverageResult(
        max_leverage=max_lev,
        max_leverage_row_index=max_idx,
        threshold=float(threshold),
        n_observations=int(n),
        n_covariates=int(k),
        passes=bool(max_lev <= threshold),
    )


# ---------------------------------------------------------------------------
# VIF
# ---------------------------------------------------------------------------

def vif_per_covariate(
    design_matrix: pd.DataFrame,
    *,
    threshold: float = HS5P_VIF_THRESHOLD,
) -> VifResult:
    """
    Variance inflation factor for each continuous covariate.

    VIF for covariate j is:
        VIF_j = 1 / (1 - R²_j)

    where R²_j is the R² of regressing covariate j on all other
    continuous covariates and a constant. VIF = 1 indicates no
    collinearity; VIF > 10 is the conventional threshold for "problem."

    Pre-reg §Hard stops HS-5' passes iff max VIF ≤ threshold
    (default 10).

    Parameters
    ----------
    design_matrix : DataFrame of continuous covariates only (no FEs,
        no constant). One row per observation. Must have no NaN.
    threshold : VIF cutoff. Default 10.

    Returns
    -------
    VifResult

    Raises
    ------
    HardStopComputationError : if the matrix is invalid or a VIF
        computation fails for one of the covariates.
    """
    if design_matrix.isna().any().any():
        raise HardStopComputationError(
            check_id="HS-5'",
            reason="vif_per_covariate: design matrix contains NaN.",
        )

    n, k = design_matrix.shape
    if n == 0:
        raise HardStopComputationError(
            check_id="HS-5'",
            reason="vif_per_covariate: empty design matrix.",
        )
    if k < 2:
        raise HardStopComputationError(
            check_id="HS-5'",
            reason=(
                f"vif_per_covariate: need ≥ 2 covariates to compute VIF "
                f"(got {k}). With only one covariate, VIF is trivially 1."
            ),
        )

    # statsmodels' variance_inflation_factor expects a numpy matrix with
    # a constant column added, and an index of which column to test.
    X_with_const = sm.add_constant(design_matrix.values, has_constant="add")

    vifs = {}
    for i, col in enumerate(design_matrix.columns):
        # Column index in X_with_const is i+1 because add_constant put the
        # constant at column 0.
        try:
            vif = variance_inflation_factor(X_with_const, i + 1)
        except Exception as e:
            raise HardStopComputationError(
                check_id="HS-5'",
                reason=f"VIF computation failed for {col!r}: {e}",
                details={"covariate": col},
            )
        if not np.isfinite(vif):
            # Infinity or NaN VIF indicates perfect collinearity.
            raise HardStopComputationError(
                check_id="HS-5'",
                reason=(
                    f"VIF for {col!r} is non-finite ({vif}). Likely cause: "
                    f"perfect collinearity with another covariate."
                ),
                details={"covariate": col, "vif": str(vif)},
            )
        vifs[col] = float(vif)

    max_vif = max(vifs.values())
    max_cov = max(vifs, key=vifs.get)

    return VifResult(
        vif_per_covariate=vifs,
        max_vif=float(max_vif),
        max_vif_covariate=max_cov,
        threshold=float(threshold),
        passes=bool(max_vif <= threshold),
    )


# ---------------------------------------------------------------------------
# Cook's distance (informational outlier robustness)
# ---------------------------------------------------------------------------

def cooks_distance_diagnostic(
    fitted_ols_model,
    continuous_design_matrix: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    treatment_col: str = "evening_depletion_d",
    multiplier: float = COOKS_DISTANCE_MULTIPLIER,
) -> CooksDistanceResult:
    """
    Flag high-influence observations via Cook's distance and report a
    full-sample vs outlier-excluded β̂ on the treatment variable.

    INFORMATIONAL ONLY (v3.2). Per pre-reg §Secondary test, this is an
    outlier-robustness sensitivity check, not a gate. Substantial
    divergence between `beta_full` and `beta_excluded` indicates the
    continuous point estimate is driven by individual high-influence
    days rather than a systematic dose-response, diminishing confidence
    in the continuous specification.

    Cook's distance is computed by statsmodels' OLSInfluence helper on
    the fitted model. Observations with D_i > multiplier/n are flagged.
    Re-estimation excludes ALL flagged observations simultaneously (per
    pre-reg disambiguation F), fits a fresh OLS on the reduced sample,
    and reports the new β̂ on `treatment_col`.

    Parameters
    ----------
    fitted_ols_model : fitted statsmodels OLSResults. Must expose
        `params` and be compatible with OLSInfluence.
    continuous_design_matrix : DataFrame of continuous covariates ONLY
        (evening_depletion_d + four DA matching covariates).
        Same rows and order as the fitted model's training data.
        No constant, no fixed effects.
    y : the outcome series used to fit the model. Must align
        row-for-row with continuous_design_matrix.
    treatment_col : the column in continuous_design_matrix whose
        coefficient is the β of interest. Default "evening_depletion_d".
    multiplier : Cook's distance threshold numerator (D_i > multiplier/n
        triggers flag). Default COOKS_DISTANCE_MULTIPLIER (4).

    Returns
    -------
    CooksDistanceResult

    Raises
    ------
    HardStopComputationError : if treatment_col is absent; if fitted
        model lacks the expected attributes; if row-count mismatch
        between design matrix and fitted residuals.
    """
    from statsmodels.stats.outliers_influence import OLSInfluence

    if treatment_col not in continuous_design_matrix.columns:
        raise HardStopComputationError(
            check_id="HS-5'",
            reason=(
                f"cooks_distance_diagnostic: treatment column "
                f"{treatment_col!r} not in continuous_design_matrix. "
                f"Available: {list(continuous_design_matrix.columns)}"
            ),
        )

    if not hasattr(fitted_ols_model, "params"):
        raise HardStopComputationError(
            check_id="HS-5'",
            reason=(
                "cooks_distance_diagnostic: fitted_ols_model has no "
                "`.params` attribute. Expected a fitted OLSResults object."
            ),
        )

    n = len(continuous_design_matrix)
    if not hasattr(fitted_ols_model, "resid") or len(fitted_ols_model.resid) != n:
        raise HardStopComputationError(
            check_id="HS-5'",
            reason=(
                f"cooks_distance_diagnostic: fitted model has "
                f"{len(getattr(fitted_ols_model, 'resid', []))} residuals "
                f"but design matrix has {n} rows. Row counts must match."
            ),
        )

    y_arr = np.asarray(y, dtype=float)
    if len(y_arr) != n:
        raise HardStopComputationError(
            check_id="HS-5'",
            reason=(
                f"cooks_distance_diagnostic: y has length {len(y_arr)} "
                f"but design matrix has {n} rows."
            ),
        )

    # Full-sample β̂ on the treatment column, pulled from the fitted model's
    # params. The fitted model includes all covariates and possibly fixed
    # effects; we index by name rather than position.
    try:
        beta_full = float(fitted_ols_model.params[treatment_col])
    except (KeyError, IndexError):
        raise HardStopComputationError(
            check_id="HS-5'",
            reason=(
                f"cooks_distance_diagnostic: fitted model has no "
                f"coefficient named {treatment_col!r}. Model params: "
                f"{list(fitted_ols_model.params.index) if hasattr(fitted_ols_model.params, 'index') else 'unknown'}"
            ),
        )

    # Compute Cook's distance using statsmodels.
    try:
        influence = OLSInfluence(fitted_ols_model)
        cooks_d, _ = influence.cooks_distance
    except Exception as e:
        raise HardStopComputationError(
            check_id="HS-5'",
            reason=f"Cook's distance computation failed: {e}",
        )

    threshold = float(multiplier / n)
    max_d = float(np.nanmax(cooks_d))
    flagged = np.where(cooks_d > threshold)[0]
    n_flagged = int(len(flagged))

    # Default outputs if nothing is flagged.
    beta_excluded: float = float("nan")
    beta_divergence: float = float("nan")

    if n_flagged > 0:
        # Re-fit on the continuous design only (no FEs), since the
        # interpretation target is β on treatment_col and the pre-reg
        # specifies "re-estimate the regression excluding flagged
        # observations". We fit on the same column set used for the
        # coefficient of interest; if the caller wants to include FEs
        # in the re-estimation they should pass a design matrix that
        # already contains them. The simplest, most reproducible
        # re-estimation for the sensitivity diagnostic is a continuous-
        # only OLS on the reduced sample.
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[flagged] = False
        X_reduced = continuous_design_matrix.loc[keep_mask].values
        y_reduced = y_arr[keep_mask]

        if len(y_reduced) < X_reduced.shape[1] + 2:
            raise HardStopComputationError(
                check_id="HS-5'",
                reason=(
                    f"cooks_distance_diagnostic: excluding {n_flagged} "
                    f"flagged observations leaves {len(y_reduced)} rows, "
                    f"insufficient for OLS with {X_reduced.shape[1]} "
                    f"covariates + constant."
                ),
            )

        X_with_const = sm.add_constant(X_reduced, has_constant="add")
        try:
            reduced_model = sm.OLS(y_reduced, X_with_const).fit()
        except Exception as e:
            raise HardStopComputationError(
                check_id="HS-5'",
                reason=f"Reduced-sample OLS failed: {e}",
            )

        # Find treatment coefficient by column position: add_constant
        # prepends the constant, so treatment_col's position is
        # original_col_index + 1.
        treat_pos = list(continuous_design_matrix.columns).index(treatment_col) + 1
        beta_excluded = float(reduced_model.params[treat_pos])

        if beta_full != 0:
            beta_divergence = float(
                abs(beta_excluded - beta_full) / abs(beta_full)
            )

    return CooksDistanceResult(
        beta_full=beta_full,
        beta_excluded=beta_excluded,
        n_flagged=n_flagged,
        flagged_indices=flagged.tolist(),
        threshold=threshold,
        max_cooks_d=max_d,
        beta_divergence=beta_divergence,
    )


# ---------------------------------------------------------------------------
# Convenience summary for HS-5'
# ---------------------------------------------------------------------------

def summarise_hs5p(
    fitted_ols_model,
    continuous_design_matrix: pd.DataFrame,
    y: pd.Series | np.ndarray | None = None,
    *,
    treatment_col: str = "evening_depletion_d",
) -> dict:
    """
    Run the VIF gate plus the informational continuous-spec diagnostics,
    returning a dict suitable for direct inclusion in the Notebook 00
    results JSON.

    PASS/FAIL COMPOSITION (v3.2, per DEVIATIONS.md Entry 001):
    - `passes` reflects VIF only. That is the sole HS-5' gate.
    - `bg` is reported informationally (autocorrelation expected in
      sequential grid data; HAC SEs handle it).
    - `cooks` is reported informationally (outlier robustness).

    Parameters
    ----------
    fitted_ols_model : fitted statsmodels OLSResults (for BG and Cook's)
    continuous_design_matrix : DataFrame of continuous covariates only
        (evening_depletion_d + four DA matching covariates). No
        constant, no fixed effects.
    y : outcome series used to fit the model. Required for Cook's
        distance. If None, Cook's diagnostic is skipped and the summary
        records "not_computed".
    treatment_col : column in continuous_design_matrix whose coefficient
        is β̂ of interest. Default "evening_depletion_d".

    Returns
    -------
    dict with keys: vif, bg, cooks, passes (bool — VIF-only).
    """
    vif = vif_per_covariate(continuous_design_matrix)
    bg = breusch_godfrey_test(fitted_ols_model)

    out = {
        "vif": {
            "vif_per_covariate": vif.vif_per_covariate,
            "max_vif": vif.max_vif,
            "max_vif_covariate": vif.max_vif_covariate,
            "threshold": vif.threshold,
            "passes": vif.passes,
        },
        "bg": {
            "lm_statistic": bg.lm_statistic,
            "p_value": bg.p_value,
            "lags": bg.lags,
            "passes": bg.passes,
            "informational": True,
        },
        "passes": vif.passes,  # VIF is the sole gate
    }

    if y is not None:
        cooks = cooks_distance_diagnostic(
            fitted_ols_model,
            continuous_design_matrix,
            y,
            treatment_col=treatment_col,
        )
        out["cooks"] = {
            "beta_full": cooks.beta_full,
            "beta_excluded": cooks.beta_excluded,
            "n_flagged": cooks.n_flagged,
            "flagged_indices": cooks.flagged_indices,
            "threshold": cooks.threshold,
            "max_cooks_d": cooks.max_cooks_d,
            "beta_divergence": cooks.beta_divergence,
            "informational": True,
        }
    else:
        out["cooks"] = {"status": "not_computed", "informational": True}

    return out
