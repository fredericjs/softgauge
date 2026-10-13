"""Command line: ``softgauge info`` · ``softgauge export``."""
from __future__ import annotations

import argparse
import collections
import sys

from .spec import load_spec


def main(argv=None):
    ap = argparse.ArgumentParser(prog="softgauge", description="Analytic soft gauges for surface texture software.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="gauges, cases and tiers of the bundled spec")
    ex = sub.add_parser("export", help="write gauges as files (+ a JSON sidecar each)")
    ex.add_argument("out", help="output directory")
    ex.add_argument("-f", "--format", default="sdf", choices=["sdf", "sdf-ascii"])
    ex.add_argument("-g", "--gauge", action="append", help="gauge id (repeatable; default: all)")
    ex.add_argument("-t", "--tier", action="append", choices=["core", "operators", "extension"],
                    help="only gauges used by cases of this tier (repeatable)")
    args = ap.parse_args(argv)
    spec = load_spec()

    if args.cmd == "info":
        print(f"spec {spec['version']}: {len(spec['gauges'])} gauges, {len(spec['cases'])} cases")
        print("tiers:  ", dict(collections.Counter(c["tier"] for c in spec["cases"])))
        print("domains:", dict(collections.Counter(c["domain"] for c in spec["cases"])))
        return 0

    from .export import export_all

    gauges = args.gauge
    if args.tier:
        used = {c["gauge"] for c in spec["cases"] if c["tier"] in args.tier}
        gauges = [g for g in (gauges or spec["gauges"]) if g in used]
    paths = export_all(args.out, args.format, spec, gauges)
    print(f"wrote {len(paths)} gauges ({args.format}) + sidecars to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
