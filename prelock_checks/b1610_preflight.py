"""
Pre-lock sanity check: B1610 preflight.
=======================================

A thin wrapper around `src.fetchers.fetch_b1610.preflight()` that runs
a single sample B1610 API call and writes JSON evidence. Use this
BEFORE committing to the full multi-hour B1610 fetch.

What this script does
---------------------
Makes one API call to Elexon `/datasets/B1610` for a recent date (10
days before today by default), filtered to the Project F BESS master
list. Verifies:

  1. The BMU list CSV loads and has ≥ 1 member.
  2. The API responds with non-empty data.
  3. At least one of our target BMUs appears in the response.
  4. The row schema matches the module's expectations.

Runtime: ~30 seconds (one API call + overhead).

Output
------
Writes `prelock_checks/b1610_preflight.json` with all four check
results, the sample record, and the number of matched BMUs.

Usage
-----
    python prelock_checks/b1610_preflight.py

    # Optional: specify an alternate BMU list path or sample date.
    python prelock_checks/b1610_preflight.py \\
        --bmu-list data/bess_master_list_219.csv \\
        --sample-date 2026-01-15 \\
        --sample-period 34

Exit status
-----------
    0 if all four checks pass.
    1 if any check fails.

If this exits nonzero, do NOT start the full fetch — investigate the
reported failure, fix, and re-run the preflight until it passes. The
full fetch will also run this preflight internally; running it
standalone is for fast iteration during setup.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make project root importable regardless of where the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fetchers.fetch_b1610 import preflight


def main() -> int:
    parser = argparse.ArgumentParser(
        description="B1610 API preflight check for Project F."
    )
    parser.add_argument(
        "--bmu-list",
        default="data/bess_master_list_219.csv",
        help="Path to the BESS BMU master list CSV.",
    )
    parser.add_argument(
        "--sample-date",
        default=None,
        help="YYYY-MM-DD; defaults to 10 days before today UTC.",
    )
    parser.add_argument(
        "--sample-period",
        type=int,
        default=34,
        help="Settlement period to query (default: 34, evening window).",
    )
    parser.add_argument(
        "--output",
        default="prelock_checks/b1610_preflight.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    print(f"\n[*] B1610 preflight")
    print(f"    BMU list: {args.bmu_list}")
    print(f"    Sample date/SP: {args.sample_date or '(auto, 10d ago)'} / "
          f"SP {args.sample_period}")
    print()

    result = preflight(
        bmu_list_path=args.bmu_list,
        sample_date=args.sample_date,
        sample_period=args.sample_period,
    )

    # Human-readable summary
    for name, ok in result.checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")

    print()
    print(f"    BMUs loaded: {result.n_bmus_loaded}")
    print(f"    Sample records returned: {result.sample_n_records}")
    print(f"    Matching target BMUs in sample: {result.sample_n_matching_bmus}")

    if result.errors:
        print()
        print("    Errors:")
        for err in result.errors:
            print(f"      - {err}")

    # Write JSON evidence
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ok": result.ok,
        "checks": result.checks,
        "n_bmus_loaded": result.n_bmus_loaded,
        "sample_n_records": result.sample_n_records,
        "sample_n_matching_bmus": result.sample_n_matching_bmus,
        "sample_record": result.sample_record,
        "errors": result.errors,
        "configuration": {
            "bmu_list_path": args.bmu_list,
            "sample_date": args.sample_date,
            "sample_period": args.sample_period,
        },
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Overall: {'PASS' if result.ok else 'FAIL'}")
    print(f"Evidence written to: {output_path}")
    print(f"{'=' * 60}\n")

    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
