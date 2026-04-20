"""
Locked pre-registration constants.
==================================

Single source of truth for every numeric parameter fixed by
PROJECT_F_PRE_REGISTRATION_v3.md (v3.1). Every other module in src/ and
every notebook imports from here — no other module is permitted to
hard-code these values.

Why this file exists
--------------------
The pre-registration locks specific numeric thresholds. If those numbers
appear in multiple places in the code, a copy-paste error could produce
silent drift between the pre-reg and the implementation. Centralising
them here means:

  1. The pre-reg and the code are compared in exactly one file.
  2. A git diff on this file is a one-line signal that a pre-reg value
     has (accidentally or deliberately) changed.
  3. The constants are immutable at import time (tuples, frozensets,
     int/float primitives) — downstream code cannot mutate them.

Do not modify after lock. Any change is a deviation and must be
documented in DEVIATIONS.md BEFORE the relevant notebook is re-run.
Pre-reg reference is given inline for every constant.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Pre-registration identity
# ---------------------------------------------------------------------------
# Bumped on any pre-reg edit. Notebooks assert this matches the lock tag.
PRE_REG_VERSION = "v3.2"
PRE_REG_FILENAME = "PROJECT_F_PRE_REGISTRATION_v3.md"


# ---------------------------------------------------------------------------
# Study period (pre-reg §Study period and splits)
# ---------------------------------------------------------------------------
TRAIN_START = pd.Timestamp("2024-01-01", tz="UTC")
TRAIN_END = pd.Timestamp("2024-12-31", tz="UTC")
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
TEST_END = pd.Timestamp("2025-12-31", tz="UTC")


# ---------------------------------------------------------------------------
# Settlement period windows (pre-reg §Variables)
# ---------------------------------------------------------------------------
# Evening window: SP 34–44 inclusive (16:30–22:00 UTC).
# Overnight window: SP 1–14 inclusive (00:00–07:00 UTC).
# Tuples (not ranges) so they are hashable and length-checked.
EVENING_SPS: tuple[int, ...] = tuple(range(34, 45))   # 34, 35, ..., 44
OVERNIGHT_SPS: tuple[int, ...] = tuple(range(1, 15))  # 1, 2, ..., 14


# ---------------------------------------------------------------------------
# Matching covariates (pre-reg §Matching covariates)
# ---------------------------------------------------------------------------
# The four continuous covariates used in the Mahalanobis distance. Order is
# fixed — downstream code iterates this tuple and expects a stable order.
CONTINUOUS_MATCHING_COVARIATES: tuple[str, ...] = (
    "DA_wind_forecast_evening_d",
    "DA_demand_forecast_evening_d",
    "DA_overnight_price_d",
    "DA_IC_schedule_overnight_d",
)

# Matching constraints (pre-reg §Matching covariates).
MATCHING_CALENDAR_PROXIMITY_DAYS = 14  # ±14 calendar days
MATCHING_WIND_DECILE_TOLERANCE = 0     # exact decile
MATCHING_DEMAND_DECILE_TOLERANCE = 0   # exact decile
MATCHING_PRICE_DECILE_TOLERANCE = 1    # ±1 decile
MATCHING_IC_DECILE_TOLERANCE = 1       # ±1 decile
MATCHING_N_BINS = 10                   # deciles


# ---------------------------------------------------------------------------
# Interconnector list (pre-reg §Matching covariates)
# ---------------------------------------------------------------------------
# Seven GB interconnectors, 2024–2025. Order is fixed so that aggregated
# sums are reproducible row-by-row in intermediate tables.
GB_INTERCONNECTORS: tuple[str, ...] = (
    "IFA",
    "IFA2",
    "BritNed",
    "NSL",
    "ElecLink",
    "Viking Link",
    "Nemo",
)


# ---------------------------------------------------------------------------
# WINDFOR vintage selection (pre-reg §Matching covariates, v3.1 addendum)
# ---------------------------------------------------------------------------
# The 12:30 UTC vintage on day d-1 is the primary matching covariate source.
# The 10:30 UTC vintage is the pre-registered robustness source.
WINDFOR_VINTAGE_PRIMARY_HOUR = 12
WINDFOR_VINTAGE_PRIMARY_MINUTE = 30
WINDFOR_VINTAGE_ROBUSTNESS_HOUR = 10
WINDFOR_VINTAGE_ROBUSTNESS_MINUTE = 30

# All eight WINDFOR publication slots, for fetcher schedule and diagnostics.
WINDFOR_ALL_PUBLISH_SLOTS: tuple[tuple[int, int], ...] = (
    (3, 30), (5, 30), (8, 30), (10, 30),
    (12, 30), (16, 30), (19, 30), (23, 30),
)

# Predetermination cutoff: matching covariates must be fixed before 15:00 UTC
# on day d-1. Asserted by the fetcher for every covariate.
DA_COVARIATE_CUTOFF_HOUR = 15
DA_COVARIATE_CUTOFF_MINUTE = 0


# ---------------------------------------------------------------------------
# Hard-stop thresholds (pre-reg §Hard stops)
# ---------------------------------------------------------------------------
# HS-1: B1610 coverage
HS1_MISSING_THRESHOLD = 0.05  # >5% missing → stop

# HS-2: BESS identification integrity
HS2_MASTER_LIST_MIN = 150
HS2_MASTER_LIST_MAX = 300
HS2_SHORT_DURATION_LIST_MIN = 50
HS2_VISUAL_REVIEW_NON_BESS_MAX = 2  # >2 non-BESS in random 20 → stop

# HS-3: Within-DoW permutation placebo
HS3_N_PERMUTATIONS = 1000
HS3_PERMUTATION_SEED = 42  # Fixed seed for reproducibility

# HS-4: Sign of effect — no numeric threshold; triggers on Δ̄ > 0.

# HS-5: Match quality (binary)
HS5_MIN_MATCHED_PAIRS = 75
HS5_SMD_THRESHOLD = 0.10      # per-covariate SMD
HS5_HOTELLING_ALPHA = 0.10    # joint balance; reject if p < 0.10

# HS-5': Continuous-spec multicollinearity (v3.2 — VIF-only gate)
# See DEVIATIONS.md Entry 001 for the reasoning behind the v3.1 → v3.2 change.
HS5P_VIF_THRESHOLD = 5        # max VIF per continuous covariate → hard stop

# Informational diagnostics on the continuous spec (NOT gating per v3.2).
# Computed and reported alongside β̂ for interpretive context; failure
# does not stop Notebook 01.
BG_INFORMATIONAL_ALPHA = 0.05   # Breusch-Godfrey significance level for reporting
COOKS_DISTANCE_MULTIPLIER = 4   # Cook's D threshold: D_i > multiplier/n → flag
# Leverage helper remains in diagnostics.py with its own default multiplier
# for diagnostic use, but no longer gates HS-5'.
HS5P_LEVERAGE_MULTIPLIER = 3  # diagnostic-only: BKW flag threshold

# HS-6: Evening-window IC confounder
HS6_SMD_THRESHOLD = 0.20      # |SMD| on realised evening IC flow

# HS-7: B1610 physical plausibility
HS7_DISCHARGE_NAMEPLATE_RATIO_CAP = 1.5  # flag BMU-days above this ratio
HS7_VIOLATION_RATE_THRESHOLD = 0.05      # >5% flagged → stop


# ---------------------------------------------------------------------------
# Inference (pre-reg §Test statistic and inference)
# ---------------------------------------------------------------------------
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 123      # Fixed seed for reproducibility
HAC_BANDWIDTH_DAYS = 7    # Newey-West for continuous secondary spec


# ---------------------------------------------------------------------------
# Success criteria (pre-reg §Success criteria)
# ---------------------------------------------------------------------------
TIER1_P_VALUE = 0.05
TIER1_ABSOLUTE_THRESHOLD_GBP_PER_MWH = -3.0
TIER1_RELATIVE_THRESHOLD = -0.05  # fraction of mean overnight price

TIER2_P_VALUE = 0.10  # relaxed for smaller test sample

# Equivalence test (informational only)
TOST_EQUIVALENCE_BOUND_GBP_PER_MWH = 1.0

# Selection diagnostic trigger for Rosenbaum bounds
ROSENBAUM_UNMATCHED_PROPORTION_TRIGGER = 0.20  # >20% unmatched → run bounds


# ---------------------------------------------------------------------------
# Expected treatment distribution (pre-reg §Expected treatment distribution)
# ---------------------------------------------------------------------------
EXPECTED_DEPLETION_MEDIAN = 0.20
EXPECTED_DEPLETION_IQR_LOW = 0.05
EXPECTED_DEPLETION_IQR_HIGH = 0.50
EXPECTED_DEPLETION_MAX = 1.5  # bounded by HS-7


# ---------------------------------------------------------------------------
# Short-duration criterion (pre-reg §Short-duration BMU list handling)
# ---------------------------------------------------------------------------
SHORT_DURATION_RATIO_MAX_HOURS = 2.0  # rated_MWh / rated_MW ≤ 2.0
