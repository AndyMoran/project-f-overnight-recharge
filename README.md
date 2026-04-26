# Overnight Recharge Suppression in GB Short-Duration BESS

## Status — Pre-registration locked, analysis completed

Pre-registration v3.2 was locked at commit `e454425f5f0a18901795ecbc46a3e6b4eca530b4` (tag: `prereg-lock-v3.2`) prior to any data access.

The pre-registered analysis has since been executed.

* The primary matched-pairs design failed a hard-stop validity check (HS-3), preventing causal inference.
* Subsequent regression and trading analysis were conducted under explicitly **non-pre-registered (post-lock)** status.

👉 **Post-lock analysis, results, and investment memo are available in the [`postlock-analysis`](../../tree/postlock-analysis) branch.**

This branch separation ensures that:

* the pre-registered specification remains immutable
* exploratory analysis is clearly distinguished from causal inference

---

## Original Pre-Lock State (Preserved)

At the time of lock:

No B1610 data had been accessed.
No matching had been run.
No results existed.

The primary artefact in this repository remains the pre-registration, not the result.

## Status — Pre-registration locked, analysis completed

Pre-registration v3.2 locked at commit `[e454425f5f0a18901795ecbc46a3e6b4eca530b4]`, tag `prereg-lock-v3.2`, on `[Mon Apr 20 20:51:53 2026 +0100]`. No B1610 data has been accessed. No matching has been run. No results exist.

The primary artefact in this repo **is the pre-registration**, not the result. The point is that the design, the hard stops, and the pass/fail criteria were frozen before the data was touched. Any result produced later is interpretable precisely because the specification is immutable and publicly timestamped.

## The Question

GB BESS fleet capacity has grown from ~3.5 GW in early 2024 to ~4.7 GW by end of 2025. A large share of this is short-duration (≤2h). These assets, by physical construction, cannot retain state-of-charge across an evening of sustained discharge — they must recharge overnight.

If a substantial fraction of the short-duration fleet discharges through the evening peak and then mechanically recharges during the overnight window, the resulting charging demand should suppress overnight imbalance prices on a day-after-day basis.

The question is whether this effect is detectable in GB 2024–2025 data under a rigorous matched-pairs design, and whether it is large enough to matter economically for a BESS operator optimising overnight recharge timing.

## The Test

| Element | Specification |
|---|---|
| Treatment | `evening_depletion_d` — fraction of short-duration fleet MWh discharged over SP 34–44 on day `d` |
| Outcome | Mean imbalance price over SP 1–14 on day `d+1` |
| Design | Matched pairs on day-ahead predetermined covariates |
| Matching covariates | Day-of-week, ±14-day calendar proximity, DA wind forecast (WINDFOR 12:30 UTC vintage), DA demand forecast, DA overnight clearing price, DA interconnector schedule |
| Inference | BCa block-bootstrap by calendar week, 10,000 reps |
| Primary gate | One-sided BCa p < 0.05 AND `Δ̄ < min(−£3/MWh, −0.05 × mean overnight price)` |
| Replication gate | Same sign and p < 0.10 in 2025 test period |

The binary matched-pairs test on raw imbalance price is the **single pre-registered gate**. A continuous-treatment specification, an IDaP-residual secondary outcome, a TOST equivalence test, and a Nov–Feb solar-sensitivity subset are informational outputs. Seven hard stops (HS-1 through HS-7) gate the analysis from running at all.

Full specification in [`PROJECT_F_PRE_REGISTRATION_v3.md`](PROJECT_F_PRE_REGISTRATION_v3.md).

## Why ≤2h Only

The treatment variable is "fraction of fleet energy discharged." For the physical recharge-pressure mechanism to apply, the asset has to actually be depleted. A 4-hour BESS discharging 100 MW for 2 hours scores the same on a MW-normalised metric as a 1-hour BESS doing the same — but the first has 2 hours of energy left and is under no mechanical recharge constraint, while the second is empty.

Matching on MW-capacity conflates these. Restricting the fleet to `rated_MWh / rated_MW ≤ 2.0` and normalising by MWh makes the treatment metric an approximate state-of-charge drawdown. The hypothesis then applies to an asset class where it is physically unavoidable, not to one where it is an economic tendency.

The cost is external validity — the study does not claim anything about long-duration BESS. This is disclosed prominently in the pre-reg and should be disclosed again in any publication of the result.

## What This Study Does Not Claim

- **Not aggregate-fleet.** Results do not generalise to ≥3h BESS. The recharge constraint is a physical property of short-duration assets.
- **Not a population-average effect.** The estimand is "effect on matchable HighDep days." If HighDep days cluster in sparse regions of the covariate space, the effect estimated here may not exist on the days a trader would want to act on it. Rosenbaum bounds are run if >20% of HighDep days fail to match.
- **Not gas-matched.** GB day-ahead NBP is omitted from the matching set by design — the available public proxies (TTF, Henry Hub + basis) would add measurement error to the Mahalanobis distance without reliably controlling for the confounder. The IDaP residual secondary outcome handles level-based gas confounding empirically; nonlinear or interaction effects are not controlled.
- **Null is a valid outcome.** Publication is committed within 30 days of analysis completion regardless of result.

## The Review Trail

The pre-registration went through four rounds of adversarial review (v1 → v2 → v3 → v3.1 → v3.2). Every round caught at least one error that would have invalidated or weakened the study:

| Round | Error caught | Consequence if missed |
|---|---|---|
| v1 → v2 | Matching on endogenous evening price | Estimates biased in unknown direction |
| v1 → v2 | Paired t-test assumes pair independence | Standard errors understated |
| v2 → v3 | HS-6 checked a collider (overnight IC flow) | Hard stop would fire on true positives |
| v2 → v3 | Duplicated `ElecLink` in IC list | Decile binning corrupted |
| v2 → v3 | Treatment MW-normalised, duration-heterogeneous | Signal diluted by long-duration assets |
| v3 → v3.1 | Wind forecast vintage (11:00 UTC) not actually published by Elexon | Fetcher would fail or silently grab wrong vintage |
| v3.1 → v3.2 | HS-5' leverage threshold `3×(k+1)/n` fails 20–54% of the time on clean iid data at Project F sample sizes | Study would hard-stop before running primary test, a false-positive hard stop per trial |

Full critique and responses in [`REVIEW_NOTES.md`](REVIEW_NOTES.md). The document is stronger for every round. It is also longer than it would be if the errors had gone through — which is a cost the reader pays honestly.

## Hard Stops

Seven checks gate Notebook 01 from running. Any triggered stop ends the study at that point.

| ID | Check | Trigger |
|---|---|---|
| HS-1 | B1610 coverage | >5% of operational-BMU-SPs missing in train period |
| HS-2 | BESS identification integrity | Master list <150 or >300 BMUs, or visual review finds >2 non-BESS in random 20, or short-duration list <50 |
| HS-3 | Within-DoW permutation placebo | Observed `\|Δ̄\|` < median permutation `\|Δ̄\|` |
| HS-4 | Sign of effect | Observed `Δ̄ > 0` in train (contradicts physical mechanism) |
| HS-5 | Match quality | <75 pairs, SMD >0.10 on any covariate, or Hotelling's T² rejects at p<0.10 |
| HS-5' | Continuous-spec multicollinearity | VIF > 5 on any continuous covariate (β̂ mathematically unstable) |
| HS-6 | Evening-window IC confounder | Realised evening-window IC flow SMD >0.20 between groups |
| HS-7 | B1610 physical plausibility | >5% of BMU-days have evening discharge exceeding 1.5× nameplate MWh |

HS-6 in particular was rewritten between v2 and v3. The v2 version checked realised *overnight* IC flow, which is a descendant of the overnight price — lower GB prices pull imports, so the stop would fire on exactly the days the hypothesis was correct. The v3 check uses evening-window IC flow, which is pre-treatment.

## Data Sources

| Variable | Source | Cutoff | Status |
|---|---|---|---|
| `P_imb` | Elexon `/balancing/settlement/system-prices` | Ex-post | Available |
| `BESS_SD_fleet_net_output` | Elexon B1610 (short-duration subset) | Ex-post | Not accessed pre-lock |
| `BESS_SD_fleet_energy_capacity` | Elexon BMU ref + NESO register + Modo list | Daily | List locked |
| `DA_wind_forecast_evening` | Elexon `/forecast/generation/wind/history` (WINDFOR), 12:30 UTC vintage on `d−1` via `publishTime` parameter | 12:30 UTC day-before | Endpoint verified |
| `DA_demand_forecast_evening` | NESO day-ahead demand forecast, pre-15:00 UTC vintage | Pre-15:00 UTC day-before | Publication time TBD empirically pre-lock |
| `DA_overnight_price` | N2EX/EPEX UK hourly day-ahead clearing | ~15:00 UTC day-before | Available |
| `DA_IC_schedule_overnight` | ENTSO-E scheduled commercial exchanges, 7 ICs (IFA, IFA2, BritNed, NSL, ElecLink, Viking Link, Nemo) | Day-before | Available |
| `realised_IC_evening` (HS-6) | Elexon physical notifications / ENTSO-E actual flows, SP 34–44 on `d` | Ex-post | Available |

**Gas price is not matched.** Principled omission, not a data gap. See pre-reg §Matching covariates and §What this study does NOT claim.

**Solar is not matched.** WINDFOR is wind-only. B1440 includes solar but has a 17:00 UTC deadline that post-dates the 15:00 UTC cutoff. The Nov–Feb subset sensitivity analysis bounds the confounding risk.

## Running the Analysis (Post-Lock)

Do not run out of order.

1. **Pre-lock empirical checks** — WINDFOR vintage-retention and NESO demand publication-time verification. Outputs committed to `prelock_checks/` as part of the lock commit.
2. **Fetch raw data** via `src/fetchers/`. Each fetcher asserts on vintage timestamps. A day with a missing required vintage is flagged as a fetch exception, not substituted.
3. **Run `notebooks/00_hard_stops.ipynb`.** Executes all seven HS checks. If any triggers, the study ends. `output/notebook_00_results.json` is written regardless.
4. **Run `notebooks/01_primary_analysis.ipynb`** only if all hard stops passed. Produces the binary-primary test (the gate), plus all informational outputs (continuous, IDaP residual, TOST, Rosenbaum bounds if needed, Nov–Feb subset).

## Repo Structure

```
project-f-overnight-recharge/
│
├── README.md                              This file
├── PROJECT_F_PRE_REGISTRATION_v3.md       The locked pre-registration (v3.2)
├── REVIEW_NOTES.md                        Three rounds of adversarial critique + responses
├── DEVIATIONS.md                          Entry 001 documents v3.1 → v3.2 amendment (HS-5' rewrite)
│
├── data/
│   ├── bess_master_list_219.csv           Locked 219-BMU master list
│   ├── short_duration_bmu_list.csv        Locked ≤2h subset with MWh ratings
│   └── README.md                          Sourcing, lock date, exclusions
│
├── prelock_checks/
│   ├── windfor_vintage_retention.json     Empirical verification output
│   └── neso_demand_publication_time.json  Empirical verification output
│
├── src/
│   ├── identify_bess.py                   Code that produced the 219 list
│   └── fetchers/                          DA fetchers with vintage-timestamp assertions
│
├── notebooks/
│   ├── 00_hard_stops.ipynb                Gates Notebook 01
│   └── 01_primary_analysis.ipynb          Runs only if Notebook 00 passes
│
├── output/                                Populated post-run
├── .gitignore
└── LICENSE
```

## Verifying the Lock

```bash
git clone [repo-url]
cd project-f-overnight-recharge
git show prereg-lock-v3.2
git log --show-signature prereg-lock-v3.2
```

The lock commit is GPG-signed by `[key fingerprint]`. GitHub displays a verified badge on the commit page.

## Prior Work

Pre-registered gate structure and null-testing discipline carried forward from **Ahead of the Curve (2026)**¹ and **Across the Water (2026)**². Both gates failed. Both failures were informative — the first identified an arbitraged regulatory signal, the second located the exact mechanism (IFA2 implicit coupling) by which a candidate signal is absorbed.

This project applies the same discipline to a new physical mechanism. The gate may pass. It may not. Either result will be published.

¹ github.com/AndyMoran/ahead-of-the-curve
² github.com/AndyMoran/across-the-water

---

Built 2026. Part of a quantitative research portfolio focused on causal identification, pre-registered hypothesis testing, and honest constraint disclosure under real-world data limitations.

andrewgmoran@gmail.com

## Repository Structure (Research Separation)

- `main` → Pre-registered design and locked specification  
- `postlock-analysis` → Regression, trading analysis, and investment memo  

This separation preserves the integrity of causal claims while allowing exploratory analysis.
