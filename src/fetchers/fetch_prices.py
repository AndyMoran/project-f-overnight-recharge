from __future__ import annotations
import argparse, logging, sys, time
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
logger = logging.getLogger(__name__)

STUDY_START = date(2024, 1, 1)
STUDY_END   = date(2025, 12, 31)
OUTPUT_DIR  = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "prices"
RATE_LIMIT_SLEEP = 0.3
ELEXON_BASE = "https://data.elexon.co.uk/bmrs/api/v1"

def _save(frames, existing, out_file, dedup_cols):
    new_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    to_concat = [f for f in [existing, new_df] if f is not None and len(f) > 0]
    if not to_concat:
        return
    combined = pd.concat(to_concat, ignore_index=True)
    if not combined.empty:
        combined[dedup_cols[0]] = pd.to_datetime(combined[dedup_cols[0]])
        combined = combined.drop_duplicates(subset=dedup_cols).sort_values(dedup_cols)
        combined.to_parquet(out_file, index=False)
        logger.info(f"  Saved {len(combined)} rows -> {out_file}")

def fetch_system_prices(start, end, resume, dry_run):
    out_file    = OUTPUT_DIR / "system_prices_raw.parquet"
    missing_file = OUTPUT_DIR / "system_prices_missing.csv"
    existing = pd.DataFrame()
    already_done = set()
    if resume and out_file.exists():
        existing = pd.read_parquet(out_file)
        already_done = set(pd.to_datetime(existing["settlement_date"]).dt.date)
        logger.info(f"  Resuming: {len(already_done)} dates done")
    dates = [start + timedelta(days=i) for i in range((end-start).days+1)
             if (start + timedelta(days=i)) not in already_done]
    logger.info(f"  System prices: {len(dates)} dates to fetch")
    if dry_run:
        logger.info(f"  Dry run — first 3: {dates[:3]}")
        return
    frames, missing = [], []
    session = requests.Session()
    for i, d in enumerate(dates):
        if i % 100 == 0:
            logger.info(f"  System prices: {i}/{len(dates)} — {d}")
        url = f"{ELEXON_BASE}/balancing/settlement/system-prices/{d.isoformat()}"
        try:
            resp = session.get(url, params={"format": "json"}, timeout=30)
            if resp.status_code != 200:
                missing.append({"date": d, "reason": f"http_{resp.status_code}"})
                time.sleep(RATE_LIMIT_SLEEP); continue
            data = resp.json().get("data") or []
            if not data:
                missing.append({"date": d, "reason": "empty"}); time.sleep(RATE_LIMIT_SLEEP); continue
            df = pd.DataFrame(data)
            df.columns = [c.lower() for c in df.columns]
            sd  = next((c for c in df.columns if "settlementdate" in c or "settlement_date" in c), None)
            sp  = next((c for c in df.columns if "settlementperiod" in c or "settlement_period" in c), None)
            px  = next((c for c in df.columns if c in (
                "systemsellprice","system_sell_price","sellprice","sell_price",
                "imbalanceprice","imbalance_price","bsaddefaultsellprice")), None)
            bx  = next((c for c in df.columns if c in (
                "systembuyprice","system_buy_price","buyprice","buy_price")), None)
            if sd and sp and px:
                out = pd.DataFrame({"settlement_date": df[sd],
                                    "settlement_period": pd.to_numeric(df[sp], errors="coerce"),
                                    "system_sell_price": pd.to_numeric(df[px], errors="coerce")})
                if bx:
                    out["system_buy_price"] = pd.to_numeric(df[bx], errors="coerce")
                frames.append(out.dropna(subset=["settlement_date","settlement_period"]))
            else:
                logger.debug(f"    {d}: cols {list(df.columns)}")
                missing.append({"date": d, "reason": "unexpected columns"})
        except Exception as e:
            logger.warning(f"  {d}: {e}")
            missing.append({"date": d, "reason": str(e)[:80]})
        time.sleep(RATE_LIMIT_SLEEP)
    _save(frames, existing, out_file, ["settlement_date","settlement_period"])
    if missing:
        pd.DataFrame(missing).to_csv(missing_file, index=False)
        logger.warning(f"  {len(missing)} missing -> {missing_file}")

def fetch_da_prices(start, end, resume, dry_run):
    out_file    = OUTPUT_DIR / "da_prices_raw.parquet"
    missing_file = OUTPUT_DIR / "da_prices_missing.csv"
    existing = pd.DataFrame()
    already_done = set()
    if resume and out_file.exists():
        existing = pd.read_parquet(out_file)
        already_done = set(pd.to_datetime(existing["settlement_date"]).dt.date)
        logger.info(f"  Resuming: {len(already_done)} dates done")
    dates = [start + timedelta(days=i) for i in range((end-start).days+1)
             if (start + timedelta(days=i)) not in already_done]
    logger.info(f"  DA prices: {len(dates)} dates to fetch")
    if dry_run:
        logger.info(f"  Dry run — first 3: {dates[:3]}")
        return
    try:
        from elexonpy.api_client import ApiClient
        from elexonpy.api.datasets_api import DatasetsApi
        api = DatasetsApi(ApiClient())
    except Exception as e:
        logger.error(f"elexonpy unavailable for DA prices: {e}"); return
    frames, missing = [], []
    for i, d in enumerate(dates):
        if i % 100 == 0:
            logger.info(f"  DA prices: {i}/{len(dates)} — {d}")
        try:
            resp = api.datasets_mid_get(_from=str(d), to=str(d), format="json")
            if hasattr(resp, "data") and resp.data:
                records = [r.to_dict() for r in resp.data]
                df = pd.DataFrame(records)
                df.columns = [c.lower() for c in df.columns]
                sd = next((c for c in df.columns if "settlement_date" in c), None)
                sp = next((c for c in df.columns if "settlement_period" in c), None)
                px = next((c for c in df.columns if "price" in c or "index" in c), None)
                if sd and sp and px:
                    out = pd.DataFrame({"settlement_date": df[sd],
                                        "settlement_period": pd.to_numeric(df[sp], errors="coerce"),
                                        "da_price": pd.to_numeric(df[px], errors="coerce")})
                    frames.append(out.dropna(subset=["settlement_date","settlement_period"]))
                else:
                    missing.append({"date": d, "reason": f"cols: {list(df.columns)}"})
            else:
                missing.append({"date": d, "reason": "empty"})
        except Exception as e:
            logger.warning(f"  {d}: {e}")
            missing.append({"date": d, "reason": str(e)[:80]})
        time.sleep(RATE_LIMIT_SLEEP)
    _save(frames, existing, out_file, ["settlement_date","settlement_period"])
    if missing:
        pd.DataFrame(missing).to_csv(missing_file, index=False)
        logger.warning(f"  {len(missing)} missing -> {missing_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, default=STUDY_START)
    parser.add_argument("--end",   type=date.fromisoformat, default=STUDY_END)
    parser.add_argument("--resume",   action="store_true")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--series",   choices=["system","da","both"], default="both")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s — %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.series in ("system","both"):
        logger.info("Fetching system prices (P_imb)...")
        fetch_system_prices(args.start, args.end, args.resume, args.dry_run)
    if args.series in ("da","both"):
        logger.info("Fetching day-ahead prices...")
        fetch_da_prices(args.start, args.end, args.resume, args.dry_run)
    logger.info("Price fetch complete.")

if __name__ == "__main__":
    main()