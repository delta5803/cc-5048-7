"""Command line entry point.

python -m stereocheck generate [DIR]
python -m stereocheck eval      DIR
python -m stereocheck report    DIR
python -m stereocheck serve     DIR [--port PORT]
"""
from __future__ import annotations

import argparse
import json

from . import footage
from .evaluation import evaluate
from .report import build_report
from .server import serve
from .session import Session


def main() -> None:
    p = argparse.ArgumentParser(prog="stereocheck")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="render the synthetic stereo clip")
    g.add_argument("dir", nargs="?", default="output/footage")

    e = sub.add_parser("eval", help="measure localisation error and false alarms")
    e.add_argument("dir")
    e.add_argument("--grid-step", type=int, default=48)

    r = sub.add_parser("report", help="rebuild report from the approved transform")
    r.add_argument("dir")

    s = sub.add_parser("serve", help="start the browser inspection station")
    s.add_argument("dir")
    s.add_argument("--port", type=int, default=8765)

    args = p.parse_args()
    if args.cmd == "generate":
        info = footage.generate(args.dir)
        print(json.dumps({k: v for k, v in info.items() if k != "frames"}, indent=2))
    elif args.cmd == "eval":
        result = evaluate(args.dir, args.grid_step)
        print(json.dumps({
            "localisation": result["localisation"],
            "alarms": result["alarms"],
            "global_model": result["global_model"],
            "per_shot_model": result["per_shot_model"],
            "vertical_alarms_global": result["vertical_alarms_global"],
            "vertical_alarms_per_shot": result["vertical_alarms_per_shot"],
        }, indent=2))
    elif args.cmd == "report":
        out = build_report(Session.load(args.dir))
        print(json.dumps({"status": out["status"], "aggregate": out.get("aggregate")}, indent=2))
    elif args.cmd == "serve":
        serve(args.dir, args.port)


if __name__ == "__main__":
    main()
