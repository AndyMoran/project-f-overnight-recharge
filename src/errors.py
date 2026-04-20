"""
Project F exception types.
==========================

Motivation
----------
The pre-registration says "all hard stops must pass for Notebook 01 to run".
In practice a hard-stop cell can fail for two very different reasons:

  1. COMPUTATIONAL FAILURE: the check couldn't be evaluated because a column
     was missing, a matrix was singular, all observations were NaN, etc.
     This is a bug in the pipeline, not a statistical rejection. The study
     is not over; the pipeline needs fixing and the notebook needs a
     re-run with a DEVIATIONS.md entry documenting the fix.

  2. STATISTICAL FAILURE: the check ran correctly and the test statistic
     crossed the pre-registered threshold. This IS a pre-registered study
     outcome. Per pre-reg §Success criteria, the study reports the null
     (or the HS-4 anti-hypothesis result) and does not re-run.

Both cases block Notebook 01 — correctly. But the distinction matters for
post-hoc decision-making (deviate and re-run, or publish the null), and it
matters for the results JSON so that a reviewer can tell afterwards which
happened.

These exception types let a hard-stop cell raise specifically on a
computational failure. The notebook catches them, writes a distinct
"***ERROR***" status to the results JSON, and proceeds to the guard cell
which blocks Notebook 01 either way.

STATISTICAL failures do NOT raise — they record `{"passed": False, "status":
"*** HARD STOP ***"}` in the normal way. Only computational failures raise.

Inheritance
-----------
All project exceptions inherit from ProjectFError so a top-level handler
can catch all project-specific failures without catching general Python
exceptions that should propagate.
"""

from __future__ import annotations


class ProjectFError(Exception):
    """Base class for all Project F exceptions."""


# ---------------------------------------------------------------------------
# Computational failures (raised from within hard-stop cells)
# ---------------------------------------------------------------------------

class HardStopComputationError(ProjectFError):
    """
    A hard-stop check could not be computed.

    This is distinct from a statistical hard-stop trigger. Examples:
      - a required column is missing from the merged DataFrame
      - all observations are NaN after filtering
      - a covariance matrix is singular and cannot be inverted
      - Hotelling's T² denominator is zero

    The notebook catches this, records the check as status="***ERROR***"
    in the results JSON, and proceeds to the guard cell. The guard blocks
    Notebook 01 because all_passed is False. The analyst then decides
    whether the error reflects a pipeline bug (→ fix, re-run, DEVIATIONS.md
    entry) or a genuine design problem (→ study ends at this check).

    Attributes
    ----------
    check_id : which HS raised (e.g. "HS-5")
    reason : short human-readable reason (e.g. "Hotelling T² singular
        covariance")
    details : optional dict with diagnostic context
    """

    def __init__(
        self,
        check_id: str,
        reason: str,
        details: dict | None = None,
    ):
        self.check_id = check_id
        self.reason = reason
        self.details = details or {}
        super().__init__(f"{check_id} computation failed: {reason}")


# ---------------------------------------------------------------------------
# Fetcher failures (raised from src/fetchers/)
# ---------------------------------------------------------------------------

class VintageAssertionError(ProjectFError):
    """
    The publish timestamp of a fetched record does not match the requested
    vintage.

    Raised by WINDFOR and demand-forecast fetchers when the Elexon/NESO API
    returns a record whose publish_time does not equal the pre-registered
    vintage. This is typically because NESO delayed a scheduled run.

    Per pre-reg §Matching covariates, the fetcher must assert on the publish
    timestamp in the response and abort if assertion fails. The affected
    date is flagged as a fetch exception and excluded from analysis — NOT
    substituted with a different vintage.

    Attributes
    ----------
    requested_vintage : the pd.Timestamp that was requested
    received_vintage : what the response actually contained
    source : fetcher name (e.g. "WINDFOR", "NESO_demand")
    """

    def __init__(
        self,
        requested_vintage,
        received_vintage,
        source: str,
    ):
        self.requested_vintage = requested_vintage
        self.received_vintage = received_vintage
        self.source = source
        super().__init__(
            f"{source}: requested vintage {requested_vintage} but received "
            f"{received_vintage}"
        )


class PreLockCheckFailure(ProjectFError):
    """
    A pre-lock empirical check failed.

    Raised during the WINDFOR vintage-retention check or the NESO
    publication-time check (pre-reg §Review and adversarial check).

    If either fails for any date in the train or test period, the pre-reg
    cannot lock with the relevant covariate specified as written. The
    analyst must either find an alternative source or accept that the
    study cannot be executed as designed.

    This is NEVER caught or converted to a ***ERROR*** status. A pre-lock
    check failure blocks the lock entirely.
    """


# ---------------------------------------------------------------------------
# Data-integrity failures (raised from loaders / mergers)
# ---------------------------------------------------------------------------

class TimeConventionError(ProjectFError):
    """
    A DataFrame lacks the `start_time` column required for UTC-safe joins,
    or the column contains naive datetimes where tz-aware UTC is required.

    Raised by src/utils/time_conventions.py::require_start_time() and
    related helpers. The underlying risk is the DST offset bug documented
    in the IV project's DEVIATIONS.md §4 — see docstring of the time
    conventions module for full background.
    """


class SchemaError(ProjectFError):
    """
    A DataFrame is missing a required column or has an unexpected dtype.

    Raised by loaders when Elexon/NESO API responses don't match the
    expected schema. Typically means the API was updated upstream.
    """
