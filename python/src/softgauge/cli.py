"""Command line: ``softgauge info`` · ``softgauge report`` · ``softgauge export``."""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from .spec import load_spec


def _report(args, spec):
    from .report import report

    files = [f for p in map(Path, args.results) for f in (sorted(p.glob("*.json")) if p.is_dir() else [p])]
    order = [k for k in (args.order or "").split(",") if k]
    hide = {k for k in (args.hide or "").split(",") if k}

    def rank(key):
        return order.index(key) if key in order else len(order)

    runs = [{"key": f.stem, **json.loads(f.read_text(encoding="utf-8"))} for f in files]
    runs = sorted((r for r in runs if r["key"] not in hide), key=lambda r: (rank(r["key"]), r["key"].casefold(), r["key"]))
    if not runs:
        print("no result files", file=sys.stderr)
        return 1
    r = report(spec, runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(r["json"], ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    (out / "results.csv").write_text(r["csv"], encoding="utf-8", newline="")
    (out / "results.tex").write_text(r["tex"], encoding="utf-8", newline="")
    for i in r["implementations"]:
        t = i["tally"]
        print(f"{i['name']} {i.get('version') or ''}: {t['pass']} pass, {t['fail']} fail, {t['error']} error, "
              f"{t['unsupported']} unsupported — gauges {'verified' if i['gaugesOk'] else 'MISMATCH'}")
    print(f"wrote results.json, results.csv, results.tex to {out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="softgauge", description="Analytic soft gauges for surface texture software.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="gauges, cases and tiers of the bundled spec")
    rp = sub.add_parser("report", help="merge result files into results.json, results.csv and results.tex")
    rp.add_argument("results", nargs="+", help="result files or directories of them (one per implementation; key = file name)")
    rp.add_argument("--out", default=".", help="output directory")
    rp.add_argument("--order", help="column order: comma-separated keys")
    rp.add_argument("--hide", help="keys to leave out: comma-separated")
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
    if args.cmd == "report":
        return _report(args, spec)

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
