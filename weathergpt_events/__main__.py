"""Command line: python -m weathergpt_events [--source DIR_OR_HF_REPO] {status,catalogue,forecast} ...

    python -m weathergpt_events --source Arko007/weathergpt-events forecast --lat 28.61 --lon 77.21 --targets rain_curve,tmax_range --horizon 3
    python -m weathergpt_events --source ./artifacts catalogue --json

HF_TOKEN (environment) is used for private repos.  `forecast` and `status` fetch the newest GFS/GEFS run (about 1.5 minutes the first
time, then cached in --cache); `catalogue` needs no network after the artifacts are local.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from weathergpt_events.loader import load_engine
from weathergpt_events.service import ForecastService, ToolCallError, summarise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m weathergpt_events", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=os.environ.get("WEATHERGPT_EVENTS_SOURCE", "Arko007/weathergpt-events"),
                        help="artifact folder or Hugging Face repo id (default: env WEATHERGPT_EVENTS_SOURCE or Arko007/weathergpt-events)")
    parser.add_argument("--cache", default="run_cache", help="folder for the downloaded GFS/GEFS run (default ./run_cache)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="which models pass the gate, and the newest run")
    sub.add_parser("catalogue", help="the tools an orchestrator can call (JSON schema included with --json)")
    f = sub.add_parser("forecast", help="answers for one place")
    f.add_argument("--lat", type=float, required=True)
    f.add_argument("--lon", type=float, required=True)
    f.add_argument("--targets", default="", help="comma-separated model names (default: all served)")
    f.add_argument("--horizon", type=int, default=10, help="forecast days, 1-10")
    f.add_argument("--include-unvalidated", action="store_true", help="debug: show numbers even where the model has no held-out skill")
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = load_engine(args.source, token=os.environ.get("HF_TOKEN"), cache_dir=None)
    service = ForecastService.live(engine, args.cache)
    try:
        if args.command == "catalogue":
            tools_ = service.catalogue()
            if args.json:
                print(json.dumps(tools_, indent=1))
            else:
                for t in tools_:
                    state = "served " if t["available"] else "refused"
                    print(f"{state} {t['name']:22s} {t['kind']:9s} horizon {t['horizon_days']} d | {t['answers'][:90]}")
            return 0
        if args.command == "status":
            status = service.status()
            if args.json:
                print(json.dumps(status, indent=1))
            else:
                print(f"newest fresh run: {status['run']}  | served {len(status['served'])} of {len(status['models'])}")
                for m in status["models"]:
                    print(f"  {'SERVED ' if m['loaded'] else 'REFUSED'} {m['target']:22s} {'' if m['loaded'] else m['reason'][:100]}")
            return 0
        targets = [t for t in args.targets.split(",") if t] or None
        result = service.forecast(args.lat, args.lon, targets=targets, horizon_days=args.horizon, include_unvalidated=args.include_unvalidated)
        print(json.dumps(result, indent=1) if args.json else summarise(result))
        return 0 if result.get("available") else 2
    except ToolCallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
