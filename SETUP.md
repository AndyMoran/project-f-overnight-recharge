# Project F — Setup & First Run

These are the exact commands to take this repo from downloaded-and-unzipped to "131 tests passing."

## 1. Create a virtual environment

From inside the unzipped project folder:

```bash
python3 -m venv venv
source venv/bin/activate         # Linux / macOS
# or: .\venv\Scripts\activate    # Windows
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

Takes a few minutes. `statsmodels` and `elexonpy` are the two largest installs.

## 3. Run the test suite

```bash
pytest
```

Expected output: `131 passed in ~7s`. If you see this, every helper module is wired correctly and the project is ready for empirical work.

If you see import errors, something went wrong in the copy. Check:

- `ls src/` shows: `__init__.py constants.py diagnostics.py errors.py inference.py matching.py utils/`
- `ls src/utils/` shows: `__init__.py time_conventions.py`
- `ls tests/` shows: `__init__.py test_diagnostics.py test_inference.py test_matching.py test_time_conventions.py`
- `conftest.py` exists at the project root

## 4. Copy data from the IV project (if reusing)

The IV project's `data/` folder contains Elexon system prices, wind forecasts, FUELHH, and the BMU reference data that Project F also needs. Copy it in:

```bash
# From project root, assuming the IV project lives at ~/depletion-multiplier-iv/
cp -r ~/depletion-multiplier-iv/data/raw ./data/raw
cp -r ~/depletion-multiplier-iv/data/processed ./data/processed
```

Important caveats:

- Project F needs **B1610** half-hourly BESS metered volume, which the IV project does NOT fetch. That's a separate fetcher job before Notebook 00 can run — see §B1610 fetcher below.
- Project F needs the **219-BMU master list** and the **short-duration subset** in `data/bess_master_list_219.csv` and `data/short_duration_bmu_list.csv`. These need to be committed before lock per the pre-reg checklist — see §Build the BMU master list below.

## 5. Build the BMU master list

Before B1610 can be fetched, you need the list of BESS BMU IDs to filter on. Run:

```bash
python prelock_checks/build_bmu_list.py
```

This pulls Battery Storage BMUs from the Elexon BMU reference API, attempts to resolve MWh ratings from the NESO Embedded Capacity Register, and writes:

- `data/bess_master_list_219.csv` — all BESS BMUs operational during 2024-2025
- `data/short_duration_bmu_list.csv` — the ≤2h subset (rated_mwh / rated_mw ≤ 2.0)
- `data/bmu_exclusions.csv` — BMUs excluded and why

It also prints a pre-reg HS-2 check: master list must have 150-300 BMUs and the short-duration list must have ≥50.

**If some BMUs have no MWh rating** (likely — Elexon doesn't publish MWh, and NESO ECR coverage is incomplete), they appear in `data/bmu_exclusions.csv` with `exclusion_reason = "MWh rating missing"`. Look these up manually (Modo Energy, planning applications, manufacturer specs) and add them to `data/bmu_mwh_manual.csv`:

```csv
bmu_id,rated_mwh,notes
E_DAGENHAM-1,50.0,"Modo Energy, April 2026"
```

Then re-run the build script with the override:

```bash
python prelock_checks/build_bmu_list.py --manual-mwh data/bmu_mwh_manual.csv
```

Iterate until `bmu_exclusions.csv` contains only units you have actively decided to exclude (e.g. units where MWh genuinely cannot be found), and both pre-reg HS-2 size checks pass.

## 6. Run the pre-lock WINDFOR vintage-retention check

```bash
python prelock_checks/windfor_vintage_retention.py
```

This hits the Elexon API twice, asserts four properties per date, writes evidence to `prelock_checks/windfor_vintage_retention.json`, and prints a summary. Expected output: `Overall: PASS`. If it fails, the pre-reg cannot lock as written — follow the diagnostic branches described in the script's docstring.

## 7. Fetch B1610 half-hourly metered volumes

B1610 is the per-BMU half-hourly metered generation dataset from Elexon. The IV project's pipeline used BOALF (accepted volumes) instead; Project F needs actual metered output, so this is a new fetch.

**Preflight first (30 seconds):**

```bash
python prelock_checks/b1610_preflight.py
```

Runs one sample API call, verifies the BMU list loads, at least one of your target BMUs appears in the response, and the schema matches expectations. Writes evidence to `prelock_checks/b1610_preflight.json`. If this fails, do NOT run the full fetch — investigate and fix first.

**Full fetch (~2-3 hours wall clock):**

```bash
python -m src.fetchers.fetch_b1610 \
    --start 2024-01-01 \
    --end 2025-12-31 \
    --bmu-list data/bess_master_list_219.csv
```

Saves monthly chunks to `data/raw/elexon/b1610_chunks/b1610_YYYY-MM.parquet`, then combines them into `data/raw/elexon/b1610_all.parquet`. Resumable — if the fetch is interrupted, rerunning skips completed months.

Two things to know:

- **Publication delay.** B1610 is published 5 days after the operational period via the II Settlement Run. Fetching dates within 5 days of today returns zero or partial rows. The fetcher warns but doesn't refuse.
- **Data revisions.** B1610 is refreshed by subsequent Settlement Runs. Each row records its `fetched_at` timestamp. For reproducibility, the fetcher never re-fetches a completed chunk; if you need a refresh, delete the specific chunk file and rerun.

## 8. The remaining pre-lock work

Per `PROJECT_F_PRE_REGISTRATION_v3.md` §Review and adversarial check, before the lock commit:

- [ ] WINDFOR vintage-retention check passes (step 5 above).
- [ ] NESO demand-forecast publication-time verification.
- [ ] 219-BMU master list + short-duration subset committed with MWh ratings.
- [ ] `REVIEW_NOTES.md` finalised.
- [ ] Fill in the `notebooks/00_hard_stops.ipynb` skeleton (each HS cell calls into `src.matching`, `src.inference`, or `src.diagnostics`).
- [ ] `DEVIATIONS.md` Entry 001 reviewed and approved (v3.1 → v3.2 amendment).
- [ ] Adversarial review of v3.2 by a peer or independent LLM session.

Then the lock itself:

```bash
git add --all
git commit -S -m "Lock pre-registration v3.2"     # -S for GPG signing
git tag -s prereg-lock-v3.2 -m "Pre-registration v3.2 locked"
git push origin main --tags
```

Fill `[hash]`, `[date]`, `[key fingerprint]`, `[repo-url]` in `README.md` with the lock details after.

## Repository layout at this point

```
project-f-overnight-recharge/
├── README.md                              Public description
├── PROJECT_F_PRE_REGISTRATION_v3.md       The locked pre-reg (v3.2)
├── DEVIATIONS.md                          Amendment log (Entry 000, 001)
├── SETUP.md                               This file
├── requirements.txt
├── conftest.py                            pytest path config
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── constants.py                       All locked numeric parameters
│   ├── errors.py                          Project-F exception types
│   ├── matching.py                        Matched-pairs pipeline + SMD + T²
│   ├── inference.py                       BCa bootstrap, permutation, TOST
│   ├── diagnostics.py                     HS-5' VIF gate, BG + Cook's informational
│   ├── utils/
│   │   ├── __init__.py
│   │   └── time_conventions.py            DST-safe time helpers
│   └── fetchers/
│       ├── __init__.py
│       └── fetch_b1610.py                 Resumable monthly-chunk B1610 fetcher
│
├── tests/                                 131 passing tests
│   ├── __init__.py
│   ├── test_time_conventions.py
│   ├── test_matching.py
│   ├── test_inference.py
│   ├── test_diagnostics.py
│   └── test_fetch_b1610.py
│
├── prelock_checks/
│   ├── windfor_vintage_retention.py       Empirical API check (WINDFOR vintages)
│   ├── b1610_preflight.py                 Single-call B1610 sanity check
│   └── build_bmu_list.py                  Builds BMU master list from Elexon + NESO
│
├── notebooks/
│   └── 00_hard_stops.ipynb                Skeleton — stubs not yet filled
│
├── data/
│   ├── bmu_mwh_manual.csv                 Template for manual MWh overrides
│   └── ...                                (rest populated from IV project + build script)
└── output/                                (empty; populated post-run)
```

## Common problems

- **`ModuleNotFoundError: No module named 'src.errors'`**: you're running from the wrong directory or `conftest.py` is missing. `cd` to the project root first.

- **`ModuleNotFoundError: No module named 'elexonpy'`**: activate the venv (`source venv/bin/activate`) and re-install (`pip install -r requirements.txt`).

- **Tests hang for >30 seconds**: one of the inference tests runs 10,000 bootstrap replications; this can be slow on older CPUs but shouldn't hang indefinitely. If it truly hangs, interrupt and run `pytest -v` to see which test.

- **`UserWarning: Converting to PeriodArray/Index representation will drop timezone information`**: this was fixed in an earlier patch; if you see it the `matching.py` in place is from before that patch. Check that the file's tercile assignment uses `year * 12 + month` arithmetic rather than `.dt.to_period("M")`.

## Where to ask for help

If something in the setup fails in a way the troubleshooting section above doesn't cover, paste the exact command and full traceback. A screenshot of `ls -la src/` and `ls -la tests/` alongside the error is especially useful for import issues.
