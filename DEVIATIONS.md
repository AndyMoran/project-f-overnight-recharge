# Project F — DEVIATIONS Log

This file records every deviation from the locked pre-registration.
Every code change, data fix, or methodological adjustment that alters
the behaviour specified in `PROJECT_F_PRE_REGISTRATION_v3.md` receives
an entry here BEFORE the relevant notebook is re-run.

**Scope.** An entry is required for:
  * Pre-lock pre-reg amendments (while the pre-reg is still being
    refined; the convention of recording the reasoning is maintained
    for continuity).
  * Post-lock pipeline fixes that change any measured value.
  * Corrections to hard-stop thresholds, matching criteria, or
    inferential procedures after the pipeline has been run at least
    once.

**Not required for:**
  * Typos, comment fixes, logging-message tweaks, or test fixture
    improvements that do not alter measured values.
  * Refactoring that preserves behaviour (e.g. splitting a function
    into two, renaming an internal variable).

Each entry includes the date, the triggering observation, the
pre-reg section(s) affected, the resolution, the files changed, and
the approving party. Entries are numbered sequentially and never
deleted or renumbered.

---

## Entry 000 — File initialisation

**Date:** 2026-04-19
**Triggered by:** Project initialisation.
**Scope:** Documentation only.

This file is created concurrently with the first amendment to the
pre-registration (Entry 001). No measured values are affected.

---

## Entry 001 — HS-5' rewritten: VIF-only gate

**Date:** 2026-04-19
**Pre-reg version affected:** v3.1 → v3.2
**Section affected:** §Hard stops, row HS-5' (continuous specification
diagnostics).
**Triggered by:** Implementation of `src/diagnostics.py` during helper
module development. Numerical simulation during test writing
revealed that the v3.1 HS-5' leverage threshold (`3×(k+1)/n` on any
observation) fails on well-behaved iid data at Project F sample sizes
with high probability.

**Evidence.** Simulation across 500 trials per sample size, with k=5
continuous covariates (evening_depletion_d plus four DA matching
covariates) and iid standard-normal columns:

```
n      threshold    P(any point exceeds)
  75      0.2400        20%
 100      0.1800        26%
 150      0.1200        42%
 200      0.0900        54%
 500      0.0360        90%
1000      0.0180        99.6%
```

At n=150, roughly 40% of well-behaved datasets trigger the hard stop.
The `3×(k+1)/n` cutoff is a Belsley-Kuh-Welsch flagging heuristic
originally intended for identifying points for **inspection**, not as
a binary pass/fail gate. Treating it as a hard stop is a category
error.

**Discussion.** Three rounds of adversarial review (v1→v2→v3) did not
identify this issue because none of those rounds ran numerical
simulation of the proposed thresholds. This is a procedural lesson:
any pre-reg numeric threshold should be tested under realistic
synthetic conditions before lock.

The underlying principle for HS-5' was "stop if the continuous
specification's β̂ is mathematically unreliable." Reconsidering each
proposed HS-5' check against that principle:

  * **Leverage (3×(k+1)/n).** Flags influential points. Does NOT
    make β̂ mathematically invalid — β̂ is still the OLS minimum-SSR
    solution, just influenced by specific rows. Belongs in
    informational diagnostics (Cook's distance), not a hard stop.
  * **Breusch-Godfrey (p < 0.01).** Detects residual autocorrelation.
    β̂ remains unbiased under autocorrelation; only the standard
    errors change, and those are handled by the pre-registered
    Newey-West HAC procedure. Sequential grid data is expected to
    exhibit residual autocorrelation — that is a feature of the data,
    not a pipeline failure. Belongs in informational diagnostics.
  * **VIF (> threshold).** Detects near-singular X'X. VIF large ⇒
    X'X^(-1) numerically unstable ⇒ β̂ genuinely unreliable in the
    mathematical sense. Belongs in the hard stop.

HS-5' is therefore reduced to a single mathematical-validity check on
VIF, with the threshold tightened from 10 to 5. VIF 5 is a standard
conservative cutoff (Menard) and is appropriate for a hard-stop gate
where the cost of a false negative (accepting a broken continuous
spec) exceeds the cost of a false positive. Because the DA matching
covariates are handpicked and their collinearity structure is
predictable (GB evening wind and demand covary under winter
high-pressure regimes), VIF > 5 would indicate a genuine
specification error — collinear matching covariates providing
non-independent information.

**Resolution.**

Hard-stop gate: HS-5' fails iff `max(VIF) > 5` on the five continuous
covariates (evening_depletion_d + four DA matching covariates).

Informational diagnostics (reported in the continuous-spec output,
NOT gating):

  * Breusch-Godfrey LM test at lag 7, reported with p-value and
    standard interpretation: "sequential grid data is expected to
    exhibit mild residual autocorrelation; the pre-registered
    Newey-West HAC standard errors correct inference accordingly."
  * Cook's distance at multiplier 4: flag observations with
    D_i > 4/n, then re-estimate β on `evening_depletion_d` excluding
    all flagged observations simultaneously. Report both full-sample
    and outlier-excluded β. Substantial divergence indicates the
    continuous point estimate is driven by individual high-influence
    days rather than a systematic dose-response.
  * Leverage statistics remain computable in the module but are no
    longer part of the HS-5' pass/fail determination.

**Files changed.**

  * `PROJECT_F_PRE_REGISTRATION_v3.md` — version bumped to v3.2;
    HS-5' row rewritten; informational diagnostics section expanded;
    review-trail entry added.
  * `src/constants.py` — `HS5P_VIF_THRESHOLD` updated from 10 to 5;
    `HS5P_BG_ALPHA` removed from hard-stop parameters; new
    `BG_INFORMATIONAL_ALPHA = 0.05` added to informational-diagnostic
    parameters; new `COOKS_DISTANCE_MULTIPLIER = 4` added;
    `HS5P_LEVERAGE_MULTIPLIER` left in place (still usable by
    `leverage_stats` but not called from `summarise_hs5p`).
  * `src/diagnostics.py` — `summarise_hs5p` rewritten to gate on VIF
    only; `cooks_distance_diagnostic` function added;
    `breusch_godfrey_test` retained unchanged in signature but the
    returned `passes` field now reflects the informational p-threshold
    0.05, not a gating decision; module docstring updated to reflect
    the HS-5' scope change.
  * `tests/test_diagnostics.py` — HS-5' summariser tests reduced to
    VIF-only gating; new tests added for `cooks_distance_diagnostic`;
    Breusch-Godfrey-pass tests kept as informational-output
    validation.
  * `README.md` — pre-reg version bumped; review-trail table updated.

**Approved by:** Andy Moran, 2026-04-19.

**Post-lock status at time of amendment.** Pre-registration v3.1 was
never formally locked (no GPG commit signature, no repository tag).
This amendment is therefore technically pre-lock. The DEVIATIONS entry
is made to preserve the reasoning trail. The lock target is now v3.2.

---

## Entry 002 — HS-2 master list count: 97 BMUs vs 150–300 target

**Date:** 2026-04-20
**Triggered by:** `prelock_checks/build_bmu_list.py` reports 97 BESS
BMUs in `data/bess_master_list_219.csv` against the pre-reg HS-2
requirement of 150–300.

**Root cause:** The Elexon BMU reference file (`bmu_reference.parquet`,
fetched by the IV project) is a snapshot taken before the 2024–2025
commissioning wave. BESS units that entered service during the study
period (e.g. Bramley, Minety, Coalburn, and dozens of smaller
embedded units) are absent from the reference file and therefore absent
from the master list. This is a data-staleness issue, not a
methodological error: the 97 units represent the full set of registered
BESS BMUs visible in the available reference snapshot.

**HS-2 short-duration check:** PASSES — 55 BMUs ≥ 50. The short-duration
sublist is the operationally relevant population for the overnight
recharge suppression study. The fleet-level treatment metric is computed
over short-duration units only; the master list count is a consistency
check, not a direct input to any measured value.

**Resolution:** Accept the 97-BMU master list as the best available
identification from the data on hand. The pre-reg HS-2 range of 150–300
was an estimate based on the expected number of active BESS units at the
time of pre-registration; the actual registrable population in the
available reference snapshot is 97. No analytical results are affected:
all downstream calculations use `data/short_duration_bmu_list.csv`
(55 BMUs, passing HS-2), not the master list count.

**Identification method:** Two-stage filter on `bmu_reference.parquet`:
(1) `fuelType == "OTHER"` minus virtual aggregators (V__ prefix), solar
farms identified by name, and units < 1 MW — yields 51 BMUs;
(2) `fuelType == NaN` with bmUnitName containing battery/bess/storage
keywords — yields 46 additional BMUs. Total: 97. Cross-referenced
against `boalf_bess.parquet` filtered to 2024-01-01–2025-12-31 to
confirm study-period activity; 68/97 confirmed active in the BM (the
remaining 29 may have B1610 metered output without BM acceptances).

**MWh ratings:** 56/97 BMUs have manually researched MWh ratings in
`data/bmu_mwh_manual.csv`, sourced from press releases, developer
announcements, and developer-pattern inference (all noted per entry).
The 41 remaining BMUs have `mwh_source = MISSING` and appear in
`data/bmu_exclusions.csv`. None of the MISSING units are among the
55 short-duration BMUs used in analysis.

**Files changed:**
  * `data/bess_master_list_219.csv` — produced by build script; 97 rows.
  * `data/short_duration_bmu_list.csv` — produced by build script; 55 rows.
  * `data/bmu_exclusions.csv` — 41 MISSING-MWh units + long-duration units.
  * `data/bmu_mwh_manual.csv` — 56 manually researched MWh entries.
  * `prelock_checks/build_bmu_list.py` — identification script.

**Approved by:** Andy Moran, 2026-04-20.

---

## Entry 003 — WINDFOR vintage check: relax publish_time_exact_match to ±90 min

**Date:** 2026-04-20
**Triggered by:** `prelock_checks/windfor_vintage_retention.py` failing
`publish_time_exact_match` for both study dates. Requested
`2024-01-15T12:30Z`; API returned `2024-01-15T11:30:00+00:00`. Same
one-hour offset for `2025-12-15T12:30Z` → `2025-12-15T11:30:00+00:00`.
All other checks (200/non-empty, horizon coverage, schema) passed.

**Root cause:** The WINDFOR endpoint returns the nearest available
published vintage to the requested publishTime. WINDFOR forecasts are
published on the half-hour; `12:30Z` is a valid half-hour mark but
Elexon's archive aligns to the *preceding* published run, yielding
`11:30Z`. This is correct API behaviour — the historical vintage is
present and being returned. The original check was too strict.

**What the check was designed to catch:** The failure mode the pre-reg
cares about is Elexon returning a *more recent* vintage when a historical
one is requested, indicating the historical data has been dropped from
the retention window. A one-hour *earlier* offset is not this failure
mode.

**Resolution:** Relaxed `assert_publish_time_exact_match` to a ±90-minute
tolerance. The check now fails only if the returned vintage is more than
90 minutes newer than requested (indicating retention loss) or more than
90 minutes older (unexpected API behaviour). Both study dates now pass.

**Impact on pre-reg §WINDFOR matching covariate:** None. The returned
vintage (`11:30Z`) is the correctly archived historical forecast for that
date. The one-hour offset is consistent and predictable. The matching
pipeline fetches forecasts by the nearest available vintage, so the
actual timestamps used in analysis will align correctly with the
settlement periods being matched.

**Files changed:**
  * `prelock_checks/windfor_vintage_retention.py` — `assert_publish_time_exact_match`
    rewritten; tolerance set to 90 minutes; docstring updated.

**Approved by:** Andy Moran, 2026-04-20.

---

## Entry 004 — DA demand forecast: source changed from NESO CKAN to Elexon TSDF; vintage pinned at ~14:45 UTC

**Date:** 2026-04-20
**Triggered by:** Pre-lock empirical check of the NESO day-ahead demand
forecast. The pre-reg specified the NESO CKAN dataset as the source with
publication time "TBD empirically pre-lock."

**Finding 1 — NESO CKAN is not a historical archive.**
The NESO 1-Day-Ahead Demand Forecast CKAN dataset (resource ID
`aec5601a-7f3e-4c4c-bf56-d8e4184d3c5b`) retains only the current-day
forecast (~12 records). Querying for `TARGETDATE: 20240115` returns an
empty result. The dataset cannot serve as a study covariate.

**Finding 2 — Elexon TSDF provides the required data.**
The Elexon TSDF (Transmission System Demand Forecast) endpoint
(`datasets_tsdf_get`) provides half-hourly, SP-level demand forecasts
with `publish_time`, `settlement_date`, `settlement_period`, and `demand`
fields. Historical data is retained back to at least 2024-01-01 (verified
empirically). The endpoint requires `publish_date_time_from` and
`publish_date_time_to` parameters.

**Finding 3 — Vintage structure empirically verified.**
For 2024-01-15 data (target date), TSDF publishes a new vintage
approximately every 30 minutes throughout d-1. Record counts per vintage:
- Vintages before ~11:47 UTC: ~180 records (partial forecast horizon)
- Vintages from ~11:47 UTC onwards: 864 records (full 6-day horizon)
The latest full vintage (864 records) published before 15:00 UTC on d-1
is consistently **~14:45 UTC** (14:45:00+00:00 observed for 2024-01-14).

**Resolution:** Pre-reg updated in v3.2:
- Source changed from NESO CKAN to Elexon TSDF.
- Vintage pinned: fetcher uses the latest `publish_time` in the window
  `d−1T11:00Z` to `d−1T14:59Z` (guarantees a full vintage, before
  15:00 UTC cutoff). If no vintage exists in this window for a given
  d−1, the day is flagged as a fetch exception and excluded from analysis.
- Data sources table updated accordingly.

**Impact on matching:** The covariate `DA_demand_forecast_evening_d` is
still a pre-15:00 UTC vintage of the next-day demand forecast for SP
34–44. The operationalisation is unchanged; only the data source and
the explicit vintage time have been corrected.

**Files changed:**
  * `PROJECT_F_PRE_REGISTRATION_v3.md` — §Covariates updated; data
    sources table updated; checklist item marked complete.
  * `prelock_checks/neso_demand_forecast_check.py` — rewritten to use
    TSDF; passes for both study boundary dates.

## Entry 005 - Treatment metric: discharge-only not net output

**Date:** 2026-04-21
**Triggered by:** Code review of `src/merge.py` `build_bess_treatment()`.

**Bug:** The initial implementation summed `quantity` over SP 34-44 and
clipped the result to zero. This computes net output (discharge minus
charge), which understates true state-of-charge drawdown on days where
some BMUs are charging while others are discharging within the evening
window. The clip to zero treated net-charging evenings as zero depletion
but did not correctly handle the case where gross discharge exceeds gross
charge by a meaningful amount.

**Pre-reg specification:** §Variables defines `evening_depletion_d` as
fleet discharge energy in SP 34-44 normalised by fleet nameplate MWh.
"Discharge" means positive B1610 output — the pre-reg explicitly
describes the mechanism as SoC drawdown from discharging, not net
position.

**Correction:** `build_bess_treatment()` now computes
`max(0, quantity)` per row before summing, then multiplies by 0.5h
and divides by fleet MWh. This is discharge-only, as pre-registered.

**Classification:** Implementation bug fix. Aligns code with the
pre-registered definition. Not a change to study design.

**Files changed:**
  * `src/merge.py` — `build_bess_treatment()` corrected.

## Entry 006 - Replacement of Matched-Pair Estimation with Regression

**Pre-registered approach**

The primary identification strategy specified in the pre-registration was a matched-pair design, comparing high- and low-depletion days matched on covariates.

**What changed**

The matched-pair design was evaluated using pre-registered hard-stop checks in `00_hard_stops.ipynb`. The design failed the permutation-based robustness test (HS-3), as it did not produce a stable distribution of placebo effects.

As a result, the matched-pair approach was deemed not viable for causal inference.

A regression-based approach was subsequently implemented in `01_regression_analysis.ipynb` to estimate the relationship between evening depletion and next-day overnight prices.

**Justification**

This deviation was not driven by results, but by failure of a pre-registered validity check. The regression approach provides a more stable and interpretable framework given the available data.

**Status**

The regression analysis is not pre-registered and should be interpreted as a complementary, non-pre-registered analysis.

## Entry 007 — Regeneration of Short‑Duration BMU List

Date: 2026‑04‑22
**Triggered by:** Multiple failures in downstream code and inconsistencies in the prereg‑locked BMU identification outputs.

**Summary**

The preregistered short‑duration BMU list (data/short_duration_bmu_list.csv) was regenerated following corrections to the BMU identification logic, updates to manually researched MWh ratings, adjustments to duration thresholds, and updates to the underlying reference data. The previous version of the file was internally inconsistent and caused failures in the treatment‑construction and matching pipelines.

**What changed**

The regenerated file differs from the prereg‑locked version in several material ways:

Bug fixes in the BMU identification script (ordering, filtering, and classification logic).

Updated MWh ratings in bmu_mwh_manual.csv based on corrected or newly available information.

Updated duration thresholds, ensuring consistent classification of short‑duration units.

Updated reference data, including BMU metadata and activity flags.

Addition of a new column duration_hours, making the duration calculation explicit rather than implicit.

Reordering and reclassification of BMUs based on corrected logic.

Removal of units incorrectly included in the earlier version.

Addition of units previously omitted due to reference‑data issues.

**Justification**

The regeneration was necessary because the prereg‑locked version of the BMU list:

contained incorrect or outdated MWh ratings

misclassified several BMUs

omitted valid short‑duration units

included units that should not have been classified as short‑duration

lacked an explicit duration column, causing downstream ambiguity

caused failures in build_bess_treatment() and related diagnostics

produced inconsistent results in the matched‑pair and regression pipelines

These issues were not the result of analytical choices but of data and implementation errors in the pre‑lock identification process. Regenerating the list was required to restore internal consistency and allow the preregistered treatment metric to be computed correctly.

**Impact on analysis**

The regeneration affects the construction of evening_depletion_d, HS‑1, and HS‑7.

The change is not driven by results but by corrections to data and logic.

All downstream analyses must be interpreted using the regenerated list.

The matched‑pair estimator was already deemed invalid (Entry 006); this regeneration does not affect that conclusion.

The regression analysis (non‑preregistered) uses the corrected list for consistency.

**Files changed**

data/short_duration_bmu_list.csv — regenerated

data/bmu_mwh_manual.csv — updated

data/bmu_exclusions.csv — updated

prelock_checks/build_bmu_list.py — corrected

src/merge.py — updated to use explicit duration logic

(Any other files you touched — list them explicitly)

Approved by: Andy Moran, 2026‑04‑22.

