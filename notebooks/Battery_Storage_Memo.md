# Overnight Recharge Suppression in GB BESS

**Result:** Economic null (statistically fragile, non-monetisable effect)

A pre-registered study of whether evening depletion of short-duration (≤2h) GB battery storage suppresses next-day overnight imbalance prices.

Python 3.11 | Energy markets | 2026

---

## Executive Summary

This study tests whether large-scale evening discharge of short-duration batteries creates a measurable and monetisable “recharging footprint” in overnight electricity prices.

A pre-registered matched-pair design failed a hard-stop validity check (HS-3), preventing causal inference. A secondary regression analysis was conducted under explicit non-pre-registered status.

The baseline estimate implies a ~£4.7/MWh reduction in overnight prices for a 10pp increase in depletion. While this exceeds the pre-registered economic threshold (~£3.25/MWh), the result is statistically fragile (p ≈ 0.05), unstable across specifications, and not economically reliable.

A key result is a sign reversal between raw and controlled estimates:

* Unconditional: +£4.4/MWh
* Controlled: −£4.7/MWh

This reflects omitted variable bias: high depletion occurs during low-wind, high-demand conditions, which independently raise prices. Controlling for fundamentals reveals a weak negative mechanical effect.

Trading strategies based on the signal produce strongly negative Sharpe ratios.

**Conclusion:** The mechanism exists but is not observable at scale in a way that supports profitable trading.

---

## Research Question

Short-duration batteries must recharge after evening discharge. If many do so simultaneously, this should suppress overnight prices.

This study tests whether that effect is:

* observable in GB market data (2024–2025)
* large enough to monetise

---

## Method Overview

**Pre-registration**

* Specification locked prior to data access (`PROJECT_F_PRE_REGISTRATION_v3.md`)
* Seven hard stops gate validity

**Primary design**

* Matched pairs on day-ahead covariates
* Binary treatment: high vs low depletion

**Outcome**

* Overnight imbalance price (SP 1–14, day d+1)

**Hard-stop outcome**

* HS-3 (permutation test) failed → matched-pair estimator invalidated

**Secondary analysis**

* Regression with controls (wind, demand, calendar)
* Explicitly non-pre-registered

---

## What Was Found

* A borderline statistically significant relationship exists
* The effect size is unstable and sensitive to specification
* Conditional signals add negligible predictive power
* Backtests produce negative Sharpe ratios

---

## Why the Signal Fails

Three structural features of the GB market suppress the effect:

* **Day-ahead optimisation** — recharge is largely pre-scheduled
* **Short duration** — recharge demand is brief, not sustained
* **Ancillary incentives** — assets remain charged for reserve markets

The mechanism is real but diluted by market structure.

---

## Market Implications (Non Pre-Registered)

As BESS capacity expands:

* ancillary revenues are likely to compress
* more assets may shift toward energy arbitrage

However:

* increased participation will compress spreads
* standalone arbitrage strategies remain structurally challenged

The null result is informative: it suggests that visible system dynamics are already efficiently priced.

---

## Rigor and Design Discipline

* Pre-registration with locked specification
* Seven hard-stop validity checks
* Explicit invalidation of failed causal design
* Clear separation of pre-registered vs exploratory analysis
* Full deviations log (`DEVIATIONS.md`)

This project prioritises **identification over outcome**.

---

## Scope and Limitations

* Applies only to ≤2h BESS
* Not a population-average estimate
* Gas not explicitly matched (by design)
* Regression results are non-pre-registered

A null result is treated as a valid outcome.

---

## Reproducibility

Run in order:

1. `00_hard_stops.ipynb`
2. If all pass, `01_regression.ipynb`

Full specification: `PROJECT_F_PRE_REGISTRATION_v3.md`

---

## What This Project Shows

* Pre-registration prevents overfitting
* Structural intuition does not guarantee tradability
* Market prices absorb observable system effects
* Negative results can be economically informative

---

## Repository Structure

See:

* `PROJECT_F_PRE_REGISTRATION_v3.md` — full specification
* `DEVIATIONS.md` — audit trail
* `REVIEW_NOTES.md` — adversarial critique
* `notebooks/` — analysis pipeline

---

Built 2026
[andrewgmoran@gmail.com](mailto:andrewgmoran@gmail.com)
