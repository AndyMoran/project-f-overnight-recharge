# Pre-Registration Record — Overnight Recharge Suppression in GB Short-Duration BESS

**Project codename:** F (overnight-recharge), short-duration variant
**Draft revision:** v3.2, 2026-04-19
**Target lock date:** [fill in before data access]
**Status:** DRAFT — not yet locked. Do not access B1610 or analyse merged data until this file is committed.

**Revision note:** v3 incorporates two rounds of adversarial review (see `REVIEW_NOTES.md` for the full critiques and point-by-point responses). Key changes from v2:

1. **Scope narrowed to short-duration (≤2h) BESS.** The fleet is restricted to a locked `SHORT_DURATION_BMU_LIST` (assets with `rated_MWh / rated_MW ≤ 2.0`). This aligns the treatment metric with the physical recharge-pressure mechanism the hypothesis describes. Long-duration assets, which need not mechanically recharge overnight, are explicitly out of scope.
2. **HS-6 rewritten.** The v2 check on realised overnight interconnector flow was a collider (descendant of the outcome), which would fire on true positives. Replaced with a check on realised *evening-window* IC flow, which is pre-treatment for the overnight outcome.
3. **DA wind forecast vintage pinned.** Uses the WINDFOR 12:30 UTC vintage on day `d−1` via the `/forecast/generation/wind/history` endpoint, which takes `publishTime` as a query parameter and returns the specific vintage (not the latest-available replay). Fetcher asserts on the publish timestamp in the response.
4. **Interconnector list corrected.** Removed the `ElecLink`/`Eleclink` duplicate. Added Nemo Link. Correct list for 2024–2025: IFA, IFA2, BritNed, NSL, ElecLink, Viking Link, Nemo.
5. **Inference switched to BCa bootstrap.** The one-sided BCa p-value is the single inferential gate; BCa CI is reported descriptively. Percentile bootstrap was subject to CI/p-value contradiction under heavy-tailed price differentials.
6. **Economic threshold tightened.** Changed from `max(−£3/MWh, −0.05 × mean)` to `min(−£3/MWh, −0.05 × mean)` so the stricter threshold binds in all price regimes.
7. **IDaP residual added as secondary outcome.** Imbalance minus day-ahead overnight price differences out market-wide gas, demand and wind level effects. Provides a robustness check against omitted level confounders (especially gas) without matching on a noisy proxy.
8. **Gas price: principled omission, not a data gap.** GB NBP day-ahead is not available in public data with a publication time preceding 15:00 UTC day-ahead cutoff; continental/US proxies degrade match quality. Omission is documented as a deliberate methodological choice, with the IDaP residual serving as the empirical safeguard.
9. **HS-3 permutation strata widened.** Permute within day-of-week only (not day-of-week × month), so the permutation null is not artificially narrow.
10. **Cross-week pair handling specified.** Matched pair is assigned to `d_high`'s week for bootstrap resampling; matching window tightened from ±2 weeks to ±14 calendar days; leak documented if >10% of pairs are cross-week.
11. **Estimand made explicit.** Target is "average effect on matchable HighDep short-duration days," not the population average. Proportion of HighDep days dropped due to no match is reported; Rosenbaum bounds run if this proportion exceeds 20%.
12. **Expected treatment distribution pre-registered** (§Expected treatment distribution).
13. **B1610 physical-plausibility check added** (HS-7).
14. **Equivalence test (TOST) added as informational output** for the null-affirmative case. Not a gate.
15. **Continuous specification diagnostic checks added** (HS-5').
16. **Train-pass / Test-fail language pre-specified** (§Success criteria).

**v3.1 addendum (vintage correction):** During v3 review the wind-forecast vintage was provisionally set to 11:00 UTC on day `d−1`, pending verification that such a vintage exists on the Elexon API. Verification found that (a) Elexon's `/forecast/generation/wind-and-solar/day-ahead` (B1440) endpoint has a regulatory publication deadline of "no later than 17:00 UTC day-before" with no guaranteed earlier vintage; (b) the underlying WINDFOR dataset exposes eight published vintages per day (03:30, 05:30, 08:30, 10:30, 12:30, 16:30, 19:30, 23:30) via the `/forecast/generation/wind/history` endpoint, which takes `publishTime` as a query parameter and returns that specific vintage; (c) vintage-filtered historical queries are therefore supported. v3.1 consequently switches from B1440 to WINDFOR, and from the 11:00 UTC target to the **12:30 UTC vintage on day `d−1`** — the latest vintage cleanly published before the 15:00 UTC cutoff specified in v3's matching logic, which gives a fresher forecast while remaining predetermined with respect to the evening window. A pre-lock empirical check (§Review and adversarial check) is added to confirm Elexon has retained 12:30 UTC vintages back to 2024-01-01. The NESO day-ahead demand forecast vintage, previously asserted by analogy as 11:00 UTC, is downgraded to "a pre-15:00 UTC vintage on day `d−1`, exact publication time to be empirically verified pre-lock" pending the same check.

**v3.2 amendment (HS-5' rewrite):** Implementation of `src/diagnostics.py` and numerical simulation of the HS-5' thresholds revealed that the v3.1 specification would fail on well-behaved iid data at Project F sample sizes with probabilities from 20% (n=75) to >50% (n=200). The `3×(k+1)/n` leverage rule is a Belsley-Kuh-Welsch flagging heuristic for inspection, not a pass/fail gate; treating it as one is a category error. The Breusch-Godfrey autocorrelation check is also mis-cast as a gate: serial correlation in sequential grid data is expected by construction, leaves β̂ unbiased, and is already handled by the pre-registered Newey-West HAC standard errors. HS-5' is therefore reduced to its one mathematically rigorous check: **VIF > 5 on any continuous covariate**, which is the only one of the three that detects a condition (near-singular X′X) under which β̂ is genuinely unreliable. Breusch-Godfrey and Cook's distance are retained as informational diagnostics in the continuous-spec output (not gates). Full reasoning and simulation evidence: `DEVIATIONS.md` Entry 001.

---

## Purpose

Test whether GB short-duration BESS fleet depletion during the evening demand peak is followed by measurable suppression of overnight imbalance prices, as depleted short-duration assets are mechanically forced to recharge overnight.

This is a **descriptive-causal study under a selection-on-observables assumption**, not an instrumental variables design. The previous IV programme (depletion-multiplier-iv) was terminated for documented reasons; this project does not use the same identification strategy.

**Scope note:** This study is explicitly restricted to BESS with duration (`rated_MWh / rated_MW`) ≤ 2.0 hours. For these assets, discharging through the full evening window approaches full state-of-charge drawdown by physical necessity, making "overnight recharge pressure" a mechanical constraint rather than a soft economic preference. Long-duration assets (≥3h) are not analysed; results cannot be extrapolated to the aggregate GB BESS fleet.

## Hypothesis

**Primary hypothesis (H1):** Imbalance prices in the overnight window (SP 1–14 on day `d+1`) are lower on average following evenings (SP 34–44 on day `d`) with high short-duration BESS fleet depletion than following evenings with low depletion, conditional on matched pre-delivery expected conditions.

**Null hypothesis (H0):** No conditional difference in overnight prices between high-depletion and low-depletion preceding evenings after matching on predetermined variables.

## Study period and splits

| Set | Period | Short-duration BESS capacity (approx.) | Purpose |
|-----|--------|---------------|---------|
| Train | 2024-01-01 – 2024-12-31 | ~2.0 – 2.5 GW | Matching specification, estimate main effect |
| Test | 2025-01-01 – 2025-12-31 | ~2.5 – 3.0 GW | Out-of-sample confirmation |

Capacity figures are approximate pending final short-duration list confirmation. No data prior to 2024-01-01 will be used.

## Variables

### Outcome — primary

**`P_imb`**: GB single imbalance price, defined as `system_sell_price` from Elexon system prices parquet. Under BSC Modification P305 (single pricing regime, effective November 2015), `system_sell_price` equals `system_buy_price` in every settlement period.

**`overnight_price_{d+1}`**: mean `P_imb` across SP 1–14 on day `d+1` (00:00–07:00 UTC). Mean rather than median: sensitive to tail effects that matter economically for a BESS operator's dispatch decision.

### Outcome — secondary (IDaP residual)

**`overnight_IDaP_{d+1}`**: `overnight_price_{d+1} − DA_overnight_price_d`, where `DA_overnight_price_d` is the day-ahead auction clearing price for overnight hours (defined in §Matching covariates).

Rationale: the day-ahead clearing price embeds the market's best pre-delivery estimate of overnight gas, demand and wind conditions. Subtracting it from the realised imbalance price differences out those level effects. If the primary result (raw imbalance price) and the IDaP residual result agree in sign and approximate magnitude, gas-price confounding is empirically ruled out as a dominant driver. Disagreement triggers diagnostic investigation (but does not alter the pre-registered pass/fail conclusion on the primary test).

### Treatment (evening depletion, short-duration fleet)

**`evening_depletion_d`**: defined on each calendar day `d`, computed over the evening window (SP 34–44, 16:30–22:00 UTC) over the short-duration BESS fleet only:

```
evening_depletion_d = Σ_{s ∈ SP 34..44} max(0, -BESS_SD_fleet_net_output_s) × 0.5h
                      ÷ BESS_SD_fleet_energy_capacity_d
```

Where:
- `BESS_SD_fleet_net_output_s`: sum of B1610 `quantity` at SP `s` across BMUs in the locked `SHORT_DURATION_BMU_LIST`, restricted to BMUs operational on day `d`. Negative values = net charging. Positive values = net discharging. We take `max(0, −x)` to extract discharge only.
- The factor 0.5h converts per-SP MW to MWh.
- **`BESS_SD_fleet_energy_capacity_d`**: total nameplate *energy* capacity (MWh) of BMUs in `SHORT_DURATION_BMU_LIST` with `commissioning_date ≤ d` and (`decommissioning_date > d` or `decommissioning_date IS NULL`). Daily series.

**Units and interpretation:** `evening_depletion_d` is dimensionless — it is the fraction of fleet energy capacity (MWh) discharged during the evening window. For a fleet composed entirely of ≤2h assets, a value of 1.0 corresponds to full state-of-charge drawdown. Under perfect dispatch `evening_depletion_d ∈ [0, 1]`; in practice mid-evening recharging or within-SP cycling can push it slightly above 1 (bounded above by the physical-plausibility cap in HS-7).

Normalising by MWh (not MW) is the fix for the duration-heterogeneity omitted variable raised in v2 review. Combined with the ≤2h scope restriction, the metric now directly approximates state-of-charge drawdown on the evening window.

### Expected treatment distribution

Pre-registered expectations, drawn from 2023 preview data (explicitly used for sanity-check only, not for estimation) and Modo Energy published dispatch statistics:

- **Median:** ~0.20 (20% of fleet energy discharged in evening)
- **IQR:** approximately [0.05, 0.50]
- **Max:** ~1.5 (bounded above by HS-7 at 1.5)
- **Skew:** right-skewed

If realised distribution deviates materially from these expectations (e.g., bimodal, heavily left-skewed, median above 0.6), this is documented in `DEVIATIONS.md` before Notebook 01. The tercile-based split is retained regardless — deviations are noted, not acted on, to avoid post-hoc rationalisation.

### Binary and continuous treatment specifications

Both specifications are pre-registered.

**Primary (binary):** within each calendar month, rank `evening_depletion_d` and split into terciles. Top tercile = `HighDep_d = 1`; bottom tercile = `LowDep_d = 1`; middle tercile excluded.

**Secondary (continuous):** regress `overnight_price_{d+1}` on `evening_depletion_d` directly, treating it as a continuous covariate. Uses the full sample. Tests for monotonic dose-response.

Both specifications also run with `overnight_IDaP_{d+1}` as the outcome (secondary-outcome × binary; secondary-outcome × continuous).

The binary-primary-outcome test is the single pre-registered gate (see §Success criteria). All other combinations are informational.

### Matching covariates (all predetermined before evening window begins)

All matching covariates are fixed by **15:00 UTC on day `d−1`**, two hours before the evening window begins (17:00 UTC day `d`). This preserves the causal chain from depletion to overnight price.

- **`day_of_week_d`**: exact match (Monday–Sunday)
- **`calendar_proximity`**: LowDep candidate date within ±14 calendar days of HighDep date
- **`DA_wind_forecast_evening_d`**: day-ahead wind forecast for SP 34–44 on day `d`. **Source and vintage:** the WINDFOR forecast published at **12:30 UTC on day `d−1`**, retrieved via the Elexon `/forecast/generation/wind/history` endpoint with `publishTime=<d−1>T12:30Z`. This endpoint takes `publishTime` as a query parameter and returns the specific vintage published at that time (not a latest-available replay), which is what makes vintage-pinned historical matching possible. The fetcher asserts that the `publishTime` field in the response equals the requested 12:30 UTC nominal; assertion failure aborts the fetch. If NESO has delayed a particular day's 12:30 run and only a later vintage is available for that `d−1`, the day is flagged as a fetch exception and excluded from analysis rather than substituted. Binned into deciles; exact decile match required. **WINDFOR is wind-only — solar is not included.** See §Solar omission note below.
- **`DA_demand_forecast_evening_d`**: day-ahead demand forecast for SP 34–44 on day `d`, from the Elexon TSDF (Transmission System Demand Forecast) dataset. **Source change from v3.1:** The NESO day-ahead demand forecast CKAN dataset (`aec5601a`) retains only current-day data (12 records as of verification); it is not a historical archive. The Elexon TSDF endpoint (`datasets_tsdf_get`) provides half-hourly SP-level demand forecasts with `publish_time`, `settlement_date`, and `settlement_period` fields, and retains full history back to at least 2024-01-01. **Vintage pinned empirically pre-lock:** TSDF publishes a new vintage approximately every 30 minutes. The record count jumps from 180 to 864 at the ~11:47 UTC vintage, indicating a qualitatively fuller forecast from that point. The latest full vintage (864 records) published before 15:00 UTC on d−1 is consistently **~14:45 UTC** (empirically verified for 2024-01-14). The fetcher uses the latest vintage with `publish_time` between `d−1T11:00Z` and `d−1T14:59Z`, taking the maximum available. If no such vintage exists for a given d−1, the day is flagged as a fetch exception and excluded rather than substituted. Binned into deciles; exact decile match required.
- **`DA_overnight_price_d`**: day-ahead auction clearing price averaged over the overnight hours of day `d+1` (SP 1–14), published at ~15:00 UTC on day `d−1` via N2EX/EPEX. Binned into deciles; match within ±1 decile.
- **`DA_IC_schedule_overnight_{d+1}`**: day-ahead scheduled interconnector net flow for the overnight hours, from ENTSO-E scheduled commercial exchanges. Summed across the seven GB interconnectors: **IFA, IFA2, BritNed, NSL, ElecLink, Viking Link, Nemo Link**. Binned into deciles; match within ±1 decile.

**On solar forecast (bounded omission):** The WINDFOR dataset used for `DA_wind_forecast_evening_d` is wind-only; it does not include a day-ahead solar component. The B1440 day-ahead wind-and-solar product includes both but has a 17:00 UTC publication deadline, which post-dates the 15:00 UTC cutoff required by the predetermination rule (§Matching covariates). Using B1440 for solar alone would therefore violate the cutoff. For the evening window (16:30–22:00 UTC) solar output is near-zero from October through February and modest in summer months, so the confounding risk from omitting day-ahead solar is bounded: it can only bias the summer subset of the match. The omission is accepted on this basis. A sensitivity check — running the primary test on the November–February subset only, where solar is mechanically near-zero — is reported as informational output. If the effect estimate in the Nov–Feb subset differs materially from the full-sample estimate in a direction consistent with solar confounding, this is documented in `DEVIATIONS.md` but does not alter the pre-registered pass/fail conclusion.

**On gas price (principled omission):** GB day-ahead NBP is the primary marginal fuel price setter for overnight CCGT. However, no public, daily-frequency NBP source exists with a publication time preceding the 15:00 UTC day-ahead cutoff. The available public proxies (TTF, Henry Hub + basis) exhibit structural divergence from GB gas dynamics at the daily resolution required for matching. Including an imprecise proxy introduces measurement error into the Mahalanobis distance metric, which degrades match quality on the remaining valid covariates without reliably controlling for gas-price confounding. Therefore, NBP is intentionally omitted from the matching set. Gas-price confounding is addressed empirically through the IDaP residual secondary outcome (§Outcome — secondary), which differences out the level component of gas-driven price variation that the day-ahead clearing price already embeds.

### Matching algorithm

A HighDep day is matched to the nearest LowDep day satisfying all constraints. Nearest defined by Mahalanobis distance on the continuous covariate values. Unmatched HighDep days are dropped.

**Estimand:** The target is the **average effect on matchable HighDep short-duration days**, not a population-average effect. This is reported explicitly with every effect estimate.

**Selection diagnostic:** The proportion of HighDep days dropped due to no LowDep match within constraints is reported. If > 20% of HighDep days are dropped, a Rosenbaum bounds sensitivity analysis is run and reported as informational output (not a pass/fail gate).

**Cross-week pairs:** A matched pair may span two calendar weeks (HighDep in week `w`, LowDep in week `w−2` through `w+2`). For block-bootstrap resampling, each pair is assigned to the calendar week of `d_high`. If > 10% of matched pairs are cross-week, this is noted in results as a known limitation of the bootstrap's block structure.

### Short-duration BMU list handling

**`SHORT_DURATION_BMU_LIST`** is a subset of the 219-BMU master BESS list. Membership criterion: `rated_MWh / rated_MW ≤ 2.0`. MWh ratings are sourced in order of precedence:

1. Elexon BMU reference where populated
2. NESO BESS register / REPD
3. Modo Energy public BESS list (cross-reference)
4. Manual verification for the top 20 largest BMUs by MW

BMUs for which no MWh rating can be determined are **excluded from the short-duration list** (conservative: absence of evidence of short-duration treated as evidence of absence of short-duration membership). The number of excluded BMUs and their aggregate MW capacity is reported.

`SHORT_DURATION_BMU_LIST` is locked at pre-registration commit, alongside the 219-BMU master list.

**Commission/decommission handling** (unchanged from v2):
- Dates sourced from Elexon BMU reference, NESO register / REPD, B1610 first-appearance, or manual verification for top 20.
- A BMU contributes to fleet output and capacity if and only if `commissioning_date ≤ d` and (`decommissioning_date > d` or `decommissioning_date IS NULL`).
- New BMUs enter the fleet from commissioning forward; pre-commission SPs are absent, not zero-imputed.
- Post-decommission SPs are excluded from the fleet.

**Missing B1610 records:** where a BMU is operational on day `d` but has no B1610 record for SP `s`, treat as `NaN` and exclude from the sum for that SP only. If more than 5% of operational-BMU-SPs have missing B1610, HS-1 triggers.

## Test statistic and inference

### Primary test (binary specification, primary outcome)

For each matched pair `(d_high, d_low)`:

```
Δ_pair = overnight_price_{d_high + 1} − overnight_price_{d_low + 1}
```

Effect estimate: `Δ̄ = mean(Δ_pair)`.

**Inference by BCa (bias-corrected accelerated) block bootstrap**, 10,000 replications:
- Group matched pairs by the calendar week of `d_high`.
- Resample weeks with replacement, preserving all pairs within a resampled week as a block.
- Compute `Δ̄_b` for each bootstrap sample.
- **BCa one-sided p-value:** bias-corrected accelerated p-value for `H0: Δ̄ ≥ 0`. This is the sole inferential gate.
- **BCa 95% CI:** bias-corrected accelerated 2.5th–97.5th percentile interval. Reported descriptively, does not independently determine passing.

**Rationale for BCa over percentile bootstrap:** percentile bootstrap can produce a significant p-value and a CI that includes zero simultaneously when the bootstrap distribution is skewed, which is plausible for heavy-tailed price differentials. BCa corrects for bias and skewness and eliminates this inconsistency.

**Rationale for block bootstrap:** paired t-test standard errors assume independence across pairs. Consecutive-day pairs share weather systems and weekly demand patterns, producing serial correlation. Calendar-week blocks preserve this dependence structure under resampling.

### Secondary test (continuous specification, primary outcome)

OLS regression with HAC standard errors (Newey-West, bandwidth = 7 days):

```
overnight_price_{d+1} = α + β × evening_depletion_d
                      + γ × (DA matching covariates, continuous)
                      + month FE + day-of-week FE + ε_d
```

Coefficient of interest: `β` (£/MWh per unit of evening_depletion). One-sided null: `β ≥ 0`.

#### Informational diagnostics on the continuous specification

The following diagnostics are reported alongside `β̂` for interpretive context. They are **informational only** — they do not gate Notebook 01 execution. Gating on the continuous specification is via HS-5' (VIF, §Hard stops) only.

- **Breusch-Godfrey LM test for residual autocorrelation**, lag 7 (equal to the HAC bandwidth). Reported as LM statistic and p-value. Sequential GB grid data is expected to exhibit mild residual autocorrelation; a BG p-value below 0.05 is anticipated and does not invalidate the point estimate `β̂`. Inference relies on the pre-registered Newey-West HAC standard errors, which are robust to autocorrelation up to the specified bandwidth.
- **Outlier robustness via Cook's distance.** Compute Cook's distance for all observations in the continuous OLS fit. Flag any observation with `D_i > 4/n`. Re-estimate the regression excluding all flagged observations simultaneously and report both the full-sample and outlier-excluded `β̂` on `evening_depletion_d`. Substantial divergence between the two (e.g. sign change, or magnitude difference greater than 50% of the full-sample estimate) indicates that the continuous point estimate is driven by individual high-influence days rather than a systematic dose-response, and accordingly diminishes confidence in the continuous specification. This is a descriptive interpretation, not a gate.

### Secondary outcome variants

Both the binary-primary and continuous-primary specifications are re-run with `overnight_IDaP_{d+1}` substituted for `overnight_price_{d+1}`. These are informational. Agreement with the primary outcome in sign and approximate magnitude provides empirical evidence against dominant gas/demand/wind level confounding. Disagreement triggers diagnostic investigation in `DEVIATIONS.md` but does not alter the pre-registered pass/fail conclusion on the primary test.

### Equivalence test (informational)

For the null-affirmative case, a two-one-sided-tests (TOST) equivalence procedure bounding `|Δ̄| < £1/MWh` is reported as an informational output. A significant TOST result supports a pre-registered statement of "no economically meaningful suppression effect detected." This is informational and does not constitute a gate.

### Multiple testing correction

The **primary binary BCa block-bootstrap test on the primary outcome is the single pre-registered gate**. All other tests (continuous primary-outcome; binary and continuous IDaP-outcome; equivalence TOST) are informational.

No Bonferroni correction is applied because only one test is the gate.

## Hard stops (Notebook 00)

All hard stops below must pass before Notebook 01 runs. Any triggered stop ends the stage.

| ID | Check | Trigger condition | Action |
|----|-------|-------------------|--------|
| HS-1 | B1610 coverage | Missing B1610 for > 5% of operational-BMU-SPs in train period | Stop — data quality insufficient |
| HS-2 | BESS identification integrity | Master list contains < 150 or > 300 BMUs, or random sample of 20 BMUs visually reviewed shows > 2 non-BESS entries; OR `SHORT_DURATION_BMU_LIST` contains < 50 BMUs | Stop — identification list needs refinement |
| HS-3 | Within-DoW permutation placebo | Permute `evening_depletion_d` within day-of-week strata (NOT day-of-week × month), 1000 reps, re-run primary test. **Trigger:** observed `\|Δ̄\|` < median permutation `\|Δ̄\|` | Stop — design has insufficient sensitivity |
| HS-4 | Sign of effect | Observed `Δ̄ > 0` in train period (prices higher following high-depletion evenings, contradicting physical mechanism) | Stop — evidence against hypothesis; report null |
| HS-5 | Match quality (binary) | < 75 matched pairs in train period; OR standardised mean difference (SMD) > 0.10 on any individual matching covariate; OR Hotelling's T² joint balance test rejects balance at p < 0.10 | Stop — insufficient matching power or poor balance |
| HS-5' | Continuous spec multicollinearity | VIF > 5 for any covariate in the continuous OLS specification (evening_depletion_d + four DA matching covariates) | Stop — coefficient estimates are mathematically unstable (primary binary test unaffected) |
| HS-6 | Evening-window IC confounder | Realised interconnector net flow **during the evening window SP 34–44 on day `d`** (pre-treatment for the overnight outcome) differs between HighDep and LowDep matched groups at SMD > 0.20 | Stop — pre-treatment IC flow differs materially despite DA schedule matching |
| HS-7 | B1610 physical plausibility | For any BMU-day in the short-duration fleet, cumulative discharge over SP 34–44 exceeds **1.5 × nameplate MWh** (50% tolerance for measurement error). Violating BMU-days are flagged and dropped from the analysis. **Trigger:** > 5% of short-duration BMU-days in train period violate | Stop — B1610 accuracy insufficient for short-duration analysis |

**HS-6 v3 note:** v2's HS-6 checked realised *overnight* IC flow, which is a descendant of the overnight price and would fire on true positives (lower GB prices pull imports). The v3 check uses evening-window IC flow on day `d`, which is pre-treatment for the overnight outcome and therefore a valid confounder check.

All HS results logged to `output/notebook_00_results.json` before Notebook 01.

## Success criteria

### Tier 1 — Primary gate (must pass)

- BCa block-bootstrap one-sided p-value < 0.05 in training period (primary outcome, binary specification)
- Effect size `Δ̄ < min(−£3/MWh, −0.05 × mean_overnight_price_train)`
  - Using `min` (stricter) rather than `max` (v2): at £40 overnight this is −£3 (requiring 7.5% suppression); at £65 this is −£3.25 (requiring 5.0% suppression). The stricter threshold binds in all price regimes.
  - Justification: round-trip losses (12–15%, ~£2–3/MWh), degradation (£3–7/MWh depending on chemistry; £7 assumed conservatively but LFP deployments are closer to £3–4), and bid-offer spread (~£2/MWh) imply a suppression of less than ~£3/MWh has no operational significance. Note that the assumed £7/MWh degradation is conservative for the growing LFP share; tightening would raise the gate economically but is not re-litigated in v3.

### Tier 2 — Out-of-sample confirmation (must pass for publication of positive result)

- Same specification applied to 2025 test period
- Sign of `Δ̄` matches training period
- BCa block-bootstrap one-sided p-value < 0.10 (relaxed for smaller test sample)

### Outcome specifications by pass/fail combination

- **Tier 1 passes, Tier 2 passes:** report as "detected, replicated"; publish full result.
- **Tier 1 passes, Tier 2 fails:** report as **"training-period effect that did not replicate out-of-sample; likely overfitting or regime shift; no tradeable signal claimed."** No onward economic or strategy interpretation.
- **Tier 1 fails (p ≥ 0.05 or effect size does not clear threshold):** report as "no suppression effect detected at pre-registered threshold." Tier 2 not run. If the TOST equivalence test is also significant at |Δ̄| < £1, additionally report as "consistent with no economically meaningful effect."
- **HS-4 triggered in training (Δ̄ > 0):** report as "evidence against hypothesis direction; physical mechanism not supported in training data."

### Tier 3 — Economic feasibility (informational, only if Tier 1 and Tier 2 pass)

- Magnitude expressed as fraction of overnight price mean and as £/MWh
- Rough P&L for a short-duration BESS operator recharging on HighDep evenings vs neutral strategy, explicit cost assumptions: £0.50/MWh trading costs, 1 tick slippage, 87% round-trip efficiency, £7/MWh degradation (conservative; LFP-specific P&L reported alongside at £4/MWh degradation)

## What this study does NOT claim

1. **Not randomised.** Confounders beyond the matched DA covariates may bias estimates. Matching on day-ahead values controls for information available before the evening window; unobserved intra-day news (grid events, trades) could correlate with both depletion and overnight prices and is not captured.

2. **Selection-on-observables assumption.** The causal claim rests on "no unobserved confounder predicts both evening depletion and overnight price after matching." This is an assumption, not a proof. Unlike IV, it cannot be tested directly.

3. **Not structural.** No attempt to estimate structural parameters of BESS behaviour or market clearing. This is a reduced-form descriptive test.

4. **GB short-duration BESS only, 2024–2025.** **This study does not claim to measure the recharge behaviour of long-duration (≥3h) BESS**, which may retain state-of-charge after evening discharge and are not mechanically constrained to recharge overnight. Applying these results to the aggregate GB BESS fleet would be an error. Results may also not transfer to other jurisdictions or future fleet sizes/compositions.

5. **Gas price explicitly not matched.** GB day-ahead NBP is not included in the matching set (see §Matching covariates for rationale). The IDaP residual secondary outcome provides an empirical check against level-based gas confounding but does not rule out interaction or nonlinear gas effects.

6. **Estimand is effect on matchable HighDep days.** Not a population-average effect. If the proportion of unmatchable HighDep days is high, the signal estimated here may not exist on days where a trader would want to trade it.

7. **No instrument.** The previous project (depletion-multiplier-iv) attempted an IV identification and failed at the exclusion restriction. No valid instrument for evening depletion has been identified in available GB public data.

## Data sources

| Variable | Source | Frequency | Publication time |
|---|---|---|---|
| `P_imb` | Elexon `/bmrs/api/v1/balancing/settlement/system-prices` | 30 min | Ex-post |
| `BESS_SD_fleet_net_output` | Elexon B1610 via `/bmrs/api/v1/datasets/B1610`, filtered to `SHORT_DURATION_BMU_LIST` | 30 min | Ex-post |
| `BESS_SD_fleet_energy_capacity` | BMU reference + NESO register + Modo Energy public list (MWh ratings) | Daily | — |
| `DA_wind_forecast_evening` | Elexon `/forecast/generation/wind/history` (WINDFOR), **12:30 UTC vintage on day `d−1`** queried via `publishTime` parameter; response `publishTime` asserted by fetcher | 30 min | 12:30 UTC day-before |
| `DA_demand_forecast_evening` | Elexon TSDF dataset (`datasets_tsdf_get`), latest vintage with `publish_time` between `d−1T11:00Z` and `d−1T14:59Z` (empirically ~14:45 UTC); NESO CKAN demand forecast not retained historically | 30 min | ~14:45 UTC day-before (latest pre-15:00 UTC full vintage) |
| `DA_overnight_price` | N2EX/EPEX UK hourly day-ahead clearing | 1 hr | ~15:00 UTC day-before |
| `DA_IC_schedule_overnight` | ENTSO-E scheduled commercial exchanges, 7 GB ICs (IFA, IFA2, BritNed, NSL, ElecLink, Viking Link, Nemo) | 1 hr | Day-before |
| `realised_IC_evening` (for HS-6) | Elexon physical notifications / ENTSO-E actual flows, SP 34–44 | 30 min | Ex-post |

The DA overnight price, DA interconnector schedule, and realised evening IC flow are data sources new to this project relative to depletion-multiplier-iv. All must be fetched and confirmed retrievable for train + test periods before this pre-registration is locked.

## Scientific integrity rules

- No data accessed or analysed until this pre-registration is committed to main.
- Any deviation documented in `DEVIATIONS.md` before running the relevant notebook.
- All results published within 30 days of completion, regardless of outcome. Audit trail via git history on a public repo.
- No financial position taken based on the results during the study period.
- This pre-registration is immutable once locked.
- The 219-BMU master BESS list, the `SHORT_DURATION_BMU_LIST`, and their commission/decommission and MWh-rating fields are locked as of the commit hash.

## Prior literature

_To be populated before lock. Minimum two relevant references:_
- Staffell, Pfenninger et al. on GB wind and BESS economics
- Modo Energy industry analysis of BESS dispatch patterns and short-duration vs long-duration dispatch behaviour
- Relevant depletion/flexibility literature from Journal of Energy Markets or Energy Economics
- Prior project: depletion-multiplier-iv (archived)

## Review and adversarial check

Before locking:
- [ ] Adversarial review of v3.2 by a peer or third independent LLM session
- [ ] All TODOs in this document resolved
- [ ] **WINDFOR vintage-retention empirical check:** query `/forecast/generation/wind/history?publishTime=2024-01-15T12:30Z` and confirm (a) a 200 response with non-empty content, (b) the returned `publishTime` field equals `2024-01-15T12:30Z` (not a more recent vintage), (c) the forecast horizon covers at least through end of 2024-01-16. Repeat for one date near the end of the test period (e.g. `2025-12-15T12:30Z`). If either date fails, vintage-filtered history is not reliably available for the full study period and the pre-reg cannot lock with this matching covariate specified as written.
- [x] **NESO demand forecast vintage empirical check:** TSDF (Elexon) confirmed as the correct source — NESO CKAN demand forecast is current-only, not archived. TSDF publishes ~every 30 min; full vintage (864 records) available from ~11:47 UTC; latest pre-15:00 UTC full vintage is ~14:45 UTC on d−1. Fetcher will use latest vintage in window d−1T11:00Z–d−1T14:59Z. Pre-reg updated: source changed from NESO CKAN to Elexon TSDF, vintage pinned at ~14:45 UTC. See DEVIATIONS Entry 004.
- [ ] DA fetchers tested end-to-end for the full train + test period; any dates where the required vintage is missing flagged as fetch exceptions.
- [ ] 219-BMU master list and `SHORT_DURATION_BMU_LIST` committed alongside this document, with MWh ratings populated or explicit exclusions documented
- [ ] REVIEW_NOTES.md committed documenting all three rounds of critique (v1→v2, v2→v3, v3→v3.1) and the responses
- [ ] DEVIATIONS.md Entry 001 reviewed and approved (v3.1 → v3.2 amendment)

---

## Sign-off

- **Author:** Andy Moran
- **Lock date:** [to be filled]
- **Git commit hash at lock:** [to be filled]
- **Repository:** [to be filled]
