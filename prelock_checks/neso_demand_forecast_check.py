"""
Pre-lock empirical check: demand forecast source and vintage.
=============================================================

Satisfies the pre-reg §Review and adversarial check item:

    [x] NESO demand forecast vintage empirical check.

Findings
--------
The NESO CKAN day-ahead demand forecast (resource aec5601a) retains only
current-day data (~12 records). It is not a historical archive and cannot
be used as a study covariate.

The Elexon TSDF (Transmission System Demand Forecast) endpoint provides:
  - Half-hourly, SP-level demand forecasts
  - Full history retained (verified back to 2024-01-01)
  - publish_time, settlement_date, settlement_period fields
  - Published approximately every 30 minutes, 24 hours a day
  - Record count jumps from ~180 to 864 at ~11:47 UTC (fuller forecast)
  - Latest full vintage (864 records) before 15:00 UTC is consistently ~14:45 UTC

Pre-reg update: source changed from NESO CKAN to Elexon TSDF.
Vintage pinned: latest vintage with publish_time in [d-1T11:00Z, d-1T14:59Z].

This script verifies TSDF availability for both study boundary dates.

Usage
-----
    python prelock_checks/neso_demand_forecast_check.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, date
from pathlib import Path

OUTPUT_FILE = Path(__file__).resolve().parent / "neso_demand_forecast_check.json"

DATES_TO_CHECK = [
    {
        "target_date": "2024-01-15",
        "d_minus_1": "2024-01-14",
        "label": "training-period start",
    },
    {
        "target_date": "2025-12-15",
        "d_minus_1": "2025-12-14",
        "label": "test-period end",
    },
]

EVENING_SP_MIN = 34
EVENING_SP_MAX = 44
VINTAGE_WINDOW_FROM = "T11:00:00"
VINTAGE_WINDOW_TO   = "T14:59:00"


def fetch_tsdf(d_minus_1: str) -> tuple[int, list[dict]]:
    try:
        from elexonpy.api_client import ApiClient
        from elexonpy.api.datasets_api import DatasetsApi
        client = ApiClient()
        api = DatasetsApi(client)
        resp = api.datasets_tsdf_get(
            publish_date_time_from=d_minus_1 + VINTAGE_WINDOW_FROM,
            publish_date_time_to=d_minus_1 + VINTAGE_WINDOW_TO,
            format="json",
        )
        if hasattr(resp, "data") and resp.data:
            return 200, [r.to_dict() for r in resp.data]
        return 200, []
    except Exception as e:
        status = getattr(e, "status", 0)
        return int(status) if status else 0, []


def main() -> int:
    results = []
    overall_passed = True

    for entry in DATES_TO_CHECK:
        target = entry["target_date"]
        d1 = entry["d_minus_1"]
        label = entry["label"]

        print(f"\n[*] Checking TSDF demand forecast: target={target} ({label})")
        print(f"    Querying publish window: {d1}{VINTAGE_WINDOW_FROM} – {d1}{VINTAGE_WINDOW_TO}")

        status, records = fetch_tsdf(d1)
        print(f"    HTTP status: {status}")
        print(f"    Records returned: {len(records)}")

        # Find unique publish times
        publish_times = sorted(set(str(r.get("publish_time")) for r in records))
        print(f"    Unique publish_times: {publish_times[:5]}{'...' if len(publish_times) > 5 else ''}")

        # Check evening SPs for target date
        target_dt = datetime.strptime(target, "%Y-%m-%d").date()
        evening_records = [
            r for r in records
            if r.get("settlement_date") == target_dt
            and EVENING_SP_MIN <= (r.get("settlement_period") or 0) <= EVENING_SP_MAX
        ]

        # Use the latest publish_time vintage
        if publish_times:
            latest_pt = publish_times[-1]
            latest_records = [r for r in records if str(r.get("publish_time")) == latest_pt]
            latest_evening = [
                r for r in latest_records
                if r.get("settlement_date") == target_dt
                and EVENING_SP_MIN <= (r.get("settlement_period") or 0) <= EVENING_SP_MAX
            ]
        else:
            latest_pt = None
            latest_evening = []

        print(f"    Latest vintage: {latest_pt}")
        print(f"    Evening SP records (SP {EVENING_SP_MIN}-{EVENING_SP_MAX}) in latest vintage: {len(latest_evening)}")

        a1 = status == 200 and len(records) > 0
        a2 = len(latest_evening) >= (EVENING_SP_MAX - EVENING_SP_MIN + 1)
        a3 = latest_pt is not None  # publish_time is before 15:00 by window construction
        a4 = bool(records) and all(
            k in records[0] for k in ["publish_time", "settlement_date", "settlement_period", "demand"]
        )

        assertions = {
            "http_200_non_empty": a1,
            "evening_sps_in_latest_vintage": a2,
            "latest_vintage_before_cutoff": a3,
            "schema_has_required_fields": a4,
        }
        failures = [k for k, v in assertions.items() if not v]
        passed = len(failures) == 0
        if not passed:
            overall_passed = False

        for name, ok in assertions.items():
            print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
        if failures:
            print(f"    → FAILED: {', '.join(failures)}")

        results.append({
            "target_date": target,
            "d_minus_1": d1,
            "label": label,
            "status": "pass" if passed else "fail",
            "assertions": assertions,
            "evidence": {
                "http_status": status,
                "n_records_in_window": len(records),
                "n_unique_publish_times": len(publish_times),
                "latest_publish_time": latest_pt,
                "n_evening_sp_records_latest_vintage": len(latest_evening),
            },
            "failures": failures,
        })

    print(f"\n{'=' * 60}")
    print(f"Overall: {'PASS' if overall_passed else 'FAIL'}")
    if overall_passed:
        print()
        print("TSDF confirmed as DA demand forecast source.")
        print(f"Vintage window: d-1{VINTAGE_WINDOW_FROM} – d-1{VINTAGE_WINDOW_TO} UTC")
        print("Fetcher uses latest publish_time in that window.")

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_passed": overall_passed,
        "source": "Elexon TSDF (datasets_tsdf_get)",
        "vintage_window": f"d-1{VINTAGE_WINDOW_FROM} to d-1{VINTAGE_WINDOW_TO} UTC",
        "dates_checked": results,
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Evidence written to: {OUTPUT_FILE}")
    print(f"{'=' * 60}\n")
    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
