# Project F — Review Notes

This file documents the adversarial review rounds that produced the
pre-registration as locked at v3.2. Each round records the substantive
critiques raised, the point-by-point responses, and the resulting
changes. Trivial wording changes are not recorded here.

---

## Round 1: v1 → v2

**Date:** 2026-04-19
**Format:** Internal adversarial review (Andy Moran + independent LLM session).

### Critique 1.1 — Duration heterogeneity in the fleet treatment metric

**Critique:** The v1 treatment metric `evening_depletion_d` was defined as
total net discharge divided by total nameplate MW. This conflates BESS units
of different durations: a 100MW/100MWh (1h) unit and a 100MW/200MWh (2h)
unit contribute identically to the metric despite having completely different
overnight recharge needs. A 1h unit discharging fully has depleted all of its
capacity; a 2h unit discharging the same MWh has depleted only half. The
treatment measure therefore does not reliably proxy overnight recharge pressure
across a heterogeneous fleet.

**Response:** Accepted. Two changes were made to address this fully:

1. The treatment metric was switched from a MW-normalised to an MWh-normalised
   measure: `evening_depletion_d = fleet_evening_net_discharge_MWh / fleet_nameplate_MWh`.
   Dividing by MWh converts the metric into an approximate state-of-charge
   drawdown (0 = no depletion, 1 = full fleet discharged from full capacity),
   which directly indexes overnight recharge pressure regardless of individual
   asset duration.

2. The analysis scope was restricted to a locked `SHORT_DURATION_BMU_LIST`
   of assets with `rated_MWh / rated_MW ≤ 2.0`. Long-duration BESS (>2h)
   mechanically need not recharge in the overnight window (they have sufficient
   capacity to discharge the next evening from a partial evening discharge),
   so including them creates noise rather than signal in the treatment measure.
   The ≤2h restriction makes the mechanism tightly specified.

**Pre-reg changes:** Treatment metric definition rewritten (§Variables);
`SHORT_DURATION_BMU_LIST` added (§BESS identification); HS-2 updated to
require short-duration list ≥ 50 BMUs.

---

### Critique 1.2 — HS-6 collider bias

**Critique:** The v1 HS-6 checked for balance on *realised overnight
interconnector net flow* in the matched sample. This is a collider: lower
overnight GB prices causally attract interconnector imports (higher
continental–GB price differential pulls imports), meaning that a genuine
recharge suppression effect — lowering overnight prices — would mechanically
produce unbalanced IC flow. HS-6 as specified would fire on true positives,
causing the study to incorrectly self-terminate when the hypothesis is correct.

**Response:** Accepted. The v1 check was conceptually confused. The correct
variable to check is *evening-window* IC flow on day `d` (SP 34–44), not
overnight IC flow on day `d+1`. Evening-window IC flow is pre-treatment for
the overnight outcome: it is realised before the overnight price is set, and
an imbalance in it across matched HighDep/LowDep days would indicate that the
matching failed to control for a contemporaneous confounder (IC imports reducing
grid stress in the evening, which the BESS depletion proxies for). This is a
legitimate confounder check without the collider problem.

**Pre-reg changes:** HS-6 rewritten to use `realised_IC_evening_d` (§Hard
stops). Data sources table updated. DEVIATIONS note added to the HS-6 row.

---

### Critique 1.3 — Interconnector list errors

**Critique:** The v1 interconnector list included `ElecLink` and `Eleclink`
as two separate entries (duplicate), and omitted Nemo Link (Belgium–UK,
operational since 2019).

**Response:** Accepted. Corrected the list to: IFA, IFA2, BritNed, NSL,
ElecLink, Viking Link, Nemo. Viking Link (Denmark) was operational from
January 2024 and covers the full study period.

---

### Critique 1.4 — Inference method (percentile bootstrap CI/p-value contradiction)

**Critique:** The v1 one-sided test used a percentile bootstrap CI and
derived the p-value from `(1 − rank(Δ̄_observed) / B)`. Under heavy-tailed
price differentials — which are characteristic of imbalance settlement data
— the percentile bootstrap CI and the p-value can contradict: the 95% CI
can exclude zero while the p-value exceeds 0.05, or vice versa. This creates
ambiguity in interpretation, particularly if the study is near the
significance boundary.

**Response:** Accepted. Switched to BCa (bias-corrected and accelerated)
bootstrap throughout. BCa explicitly corrects for both bias and skewness in
the bootstrap distribution, producing CI and p-value that are mutually
consistent. The BCa p-value is the single inferential gate; the BCa CI is
reported descriptively.

---

### Critique 1.5 — Economic threshold direction

**Critique:** The v1 economic gate used `max(−£3/MWh, −0.05 × mean_price)`.
`max` of two negative numbers picks the *less* negative one (the looser
threshold). In a low-price environment (mean ~£40/MWh), `−0.05 × 40 = −£2`,
and `max(−3, −2) = −2` — so the gate would pass on a £2.50 suppression that
has no economic significance. The direction of the `max/min` operator was
backwards.

**Response:** Accepted. Changed to `min(−£3/MWh, −0.05 × mean_price)`, which
picks the stricter (more negative) threshold and ensures the gate binds
conservatively in all price regimes. At £40 overnight: `min(−3, −2) = −3`;
at £65: `min(−3, −3.25) = −3.25`.

---

### Critique 1.6 — Gas price as covariate

**Critique:** v1 included GB NBP day-ahead gas price as a matching covariate.
No public, daily-frequency NBP source exists with a publication time preceding
the 15:00 UTC day-ahead cutoff required for the matching design. Continental
proxies (TTF, Henry Hub) exhibit structural divergence from GB gas dynamics
at the daily resolution required.

**Response:** Accepted with a principled omission note. Gas price is excluded
from the matching covariates. The IDaP residual secondary outcome is added as
a compensation: differencing overnight imbalance price from the day-ahead
overnight clearing price removes market-wide gas/demand/wind level effects
without requiring gas price as an explicit covariate. This provides an
empirical safeguard against omitted gas-price confounding.

---

## Round 2: v2 → v3

**Date:** 2026-04-19
**Format:** Continued adversarial review.

### Critique 2.1 — Wind forecast vintage not pinned

**Critique:** v2 specified the DA wind forecast as the 11:00 UTC vintage on
day `d−1` via the Elexon B1440 endpoint. Investigation found that (a) the
B1440 (`/forecast/generation/wind-and-solar/day-ahead`) endpoint has a
regulatory publication deadline of "no later than 17:00 UTC day-before" with
no guaranteed earlier vintage, and (b) the B1440 endpoint does not accept a
`publishTime` parameter — it returns only the latest-available vintage, not
a specific historical vintage. Using B1440 makes vintage-pinned historical
matching impossible; different days would use different vintages depending on
when data happened to be fetched.

**Response:** Accepted. The correct source is the WINDFOR dataset, exposed
via `/forecast/generation/wind/history`, which takes `publishTime` as a query
parameter and returns the specific archived vintage. WINDFOR is published eight
times per day: 03:30, 05:30, 08:30, 10:30, 12:30, 16:30, 19:30, 23:30 UTC.
The 12:30 UTC vintage on `d−1` is the latest vintage cleanly published before
the 15:00 UTC covariate cutoff. The fetcher asserts on the returned
`publishTime` field; if the asserted vintage is unavailable for a given day,
the day is excluded rather than substituted with a later vintage.

**Pre-reg changes:** §Variables updated; data sources table updated; v3.1
addendum added; empirical pre-lock check added to §Review and adversarial
check.

---

### Critique 2.2 — NESO demand forecast vintage unverified

**Critique:** v2 stated "NESO publishes a day-ahead demand forecast at
approximately 11:00 UTC" without empirical verification. The operational
publication schedule may have changed, and the vintage cutoff needed to be
confirmed before lock.

**Response:** Deferred to pre-lock empirical check. The specification was
changed to "a pre-15:00 UTC vintage on day `d−1`, exact publication time to
be empirically verified pre-lock and pinned in the fetcher." The empirical
check (conducted during the pre-lock workflow on 2026-04-20) found:
(a) the NESO CKAN day-ahead demand forecast retains only current-day data
and is not a historical archive; (b) the Elexon TSDF (Transmission System
Demand Forecast) endpoint provides half-hourly, SP-level demand forecasts
with full historical retention; (c) TSDF publishes a new vintage every ~30
minutes; (d) the latest full vintage before 15:00 UTC is consistently ~14:45
UTC on `d−1`. The pre-reg was updated to TSDF as the source, with the vintage
pinned at the latest record in the `d−1T11:00Z–d−1T14:59Z` window.
See DEVIATIONS Entry 004.

---

## Round 3: v3 → v3.1 → v3.2

**Date:** 2026-04-19
**Format:** Code implementation review — numerical simulation of HS-5' thresholds.

### Critique 3.1 — HS-5' leverage and BG checks fail on well-behaved data

**Critique:** Implementation of `src/diagnostics.py` and numerical simulation
of the HS-5' thresholds at Project F sample sizes revealed that the v3.1
specification would fail on well-behaved iid data with high probability:

- The `3×(k+1)/n` leverage threshold is the Belsley-Kuh-Welsch heuristic
  for flagging individual observations for manual inspection, not a
  pass/fail gate. At n=200 with k=6 covariates, the threshold is
  `3×7/200 = 0.105`. In a balanced design, the average leverage is
  `(k+1)/n = 0.035`, and the maximum leverage for perfectly-behaved iid
  data follows an order statistic of a Beta distribution — which routinely
  exceeds 0.105 even with no leverage problem. Simulation showed the gate
  would fail on well-behaved iid data ~50% of the time at n=200.

- The Breusch-Godfrey autocorrelation test is also mis-cast as a gate.
  Sequential electricity market data is expected to exhibit serial
  correlation by construction (weekend/weekday cycles, seasonal patterns).
  BG detecting serial correlation in such data is not an indication of
  model misspecification; it is the correct null being rejected correctly.
  Serial correlation in ε leaves β̂ unbiased and is already handled by the
  pre-registered Newey-West HAC standard errors. Gating on BG creates a
  study that would correctly halt when the standard errors are computed
  correctly.

**Response:** Accepted. HS-5' reduced to its one mathematically well-founded
check: VIF > 5 on any continuous covariate. VIF > 5 detects near-singular
X′X, the only condition among the three where β̂ is genuinely unreliable
(not just imprecise). Breusch-Godfrey and Cook's distance are retained as
informational diagnostics in the continuous-spec output — they are computed
and printed, but do not gate. VIF threshold set at 5 (a standard conservative
threshold for regression covariates).

**Pre-reg changes:** HS-5' row in §Hard stops rewritten; `src/diagnostics.py`
`summarise_hs5p` function and tests updated; `src/constants.py` updated;
DEVIATIONS Entry 001 written.

---

## Summary of changes by round

| Round | Primary driver | Key structural change |
|---|---|---|
| v1 → v2 | Duration heterogeneity; collider; IC list; inference; threshold direction | Treatment metric MWh-normalised; scope restricted to ≤2h; HS-6 rewritten; BCa bootstrap; `min` threshold |
| v2 → v3 | Wind forecast vintage unachievable via B1440 | WINDFOR vintage-pinned source; pre-lock empirical check added |
| v3 → v3.1 | Demand forecast vintage unverified | NESO demand forecast verification deferred to pre-lock |
| v3.1 → v3.2 | HS-5' thresholds fail on well-behaved data | HS-5' reduced to VIF-only gate |
| Pre-lock | Empirical checks | TSDF as demand source; WINDFOR ±90 min tolerance; BMU master list count accepted at 97 (see DEVIATIONS) |

---

## Outstanding review items at lock

The following items from the pre-reg §Review and adversarial check checklist
were completed before lock:

- [x] WINDFOR vintage-retention empirical check — PASS (both study dates)
- [x] NESO demand forecast vintage empirical check — source corrected to TSDF,
      vintage confirmed; PASS (both study dates)
- [x] B1610 preflight check — PASS (94/97 BMUs matched in sample SP)
- [x] HS-2 visual review — 20/20 confirmed BESS, 0 non-BESS
- [x] DEVIATIONS.md entries 000–004 reviewed and approved
- [x] All notebook stubs implemented

One item is deferred to post-lock (per pre-reg §Review and adversarial check):

- [ ] Adversarial review of v3.2 by a peer or independent LLM session —
  deferred to after the lock commit, per the pre-reg's own specification
  that this review is conducted on the locked version. If the peer review
  raises structural issues, they will be documented in DEVIATIONS.md with
  a new version tag.
