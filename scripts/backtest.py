#!/usr/bin/env python3
"""Backtest helper: run WIO queries over a CSV and compute Brier/RMSE.
CSV: question,location_raw,valid_from,valid_to,observed_rain_mm
"""
import argparse, csv, json, httpx, asyncio
from pathlib import Path

async def run_one(client, url, q, loc):
    r = await client.post(url, json={"question": q, "location":{"raw": loc}, "lang":"en"})
    r.raise_for_status()
    return r.json()

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wio-endpoint", default="http://localhost:8001/wio/query")
    ap.add_argument("--csv", type=str, required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    import numpy as np
    briers=[]; rmses=[]
    async with httpx.AsyncClient(timeout=30) as client:
        for row in rows:
            q=row["question"]; loc=row.get("location_raw","Nagpur")
            data = await run_one(client, args.wio_endpoint, q, loc)
            wio=data["wio"]
            prob = (wio["weather"].get("rain") or {}).get("probability")
            pred_mm = (wio["weather"].get("rain") or {}).get("value_mm")
            obs = float(row.get("observed_rain_mm",0))
            obs_bin = 1 if obs>0.5 else 0
            if prob is not None:
                briers.append((prob-obs_bin)**2)
            if pred_mm is not None:
                rmses.append((pred_mm-obs)**2)
            print(f"{loc} prob={prob} pred={pred_mm} obs={obs}")
    if briers:
        print(f"Brier {np.mean(briers):.4f}  n={len(briers)}")
    if rmses:
        print(f"RMSE {np.sqrt(np.mean(rmses)):.3f} mm  n={len(rmses)}")

if __name__ == "__main__":
    asyncio.run(main())
