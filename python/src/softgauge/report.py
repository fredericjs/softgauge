"""Merge the spec with any number of result files (mirrored in js/src/report.js) into

* ``json``  one document for a web page (cases × implementations, verdicts, tallies)
* ``csv``   one row per case × implementation
* ``tex``   a LaTeX longtable (compile with LuaLaTeX/XeLaTeX, UTF-8)

The CSV and LaTeX text is identical to what the JavaScript package writes.
"""
from __future__ import annotations

import re

from .judge import judge
from .numfmt import exponential, num, precision, utc_now

__all__ = ["case_label", "measure_label", "op_label", "report"]

# ── human-readable label of what a case measures ──────────────────────────────
_OPN = {
    "level": "level", "levelTLS": "level (total least squares)", "detrend": "detrend", "levelPoints": "level to points",
    "removeSphere": "remove sphere", "removeCylinder": "remove cylinder", "filter": "filter", "center": "center", "zero": "zero",
    "invert": "invert", "threshold": "threshold", "removeOutliers": "remove outliers", "despike": "despike", "fill": "fill",
    "crop": "crop", "zoom": "zoom", "rotate": "rotate", "maskRect": "mask rectangle", "maskCircle": "mask circle",
    "maskThreshold": "mask by height", "maskInvert": "invert mask", "maskToNaN": "mask → non-measured",
}

_STATS = {"mean": "mean height", "min": "min height", "max": "max height", "nanCount": "non-measured points",
          "maskedCount": "masked points", "validMean": "mean of unmasked", "rows": "rows", "cols": "columns",
          "stepX": "Δx", "stepY": "Δy"}


def op_label(o):
    b = _OPN.get(o["op"], o["op"])
    if o["op"] == "detrend":
        return f"{b} (degree {num(o['degree'])})"
    if o["op"] == "filter":
        mode = f"{o['mode']} " if o.get("mode") and o["mode"] != "reflect" else ""
        band = f"/{num(o['cutoff2'])}" if o.get("cutoff2") else ""
        return f"{mode}{o['type']} λc = {num(o['cutoff'])}{band} µm"
    if o["op"] in ("removeOutliers", "despike", "fill"):
        return f"{b} ({o['method']})"
    if o["op"] == "rotate":
        return f"{b} {num(o['angle'])}°"
    if o["op"] == "zoom":
        return f"{b} ×{num(o['factor'])}"
    return b


def measure_label(m):
    k = m["kind"]
    if k == "param":
        return f"{m['name']}({', '.join(num(a) for a in m['args'])})" if m.get("args") else m["name"]
    if k == "stat":
        return _STATS.get(m["stat"], m["stat"])
    return {
        "fieldError": "max |Δz|", "sineAmplitude": "amplitude transmission", "nanMismatch": "misclassified points",
        "planeGradient": "|∇z|" if m.get("component") == "mag" else "gradient direction", "formRadius": "fitted radius",
        "radialPsd": "Σ PSD·Δf² (radial PSD)", "fourierPeak": "peak frequency (2D DFT)",
        "acfMap": f"ACF(τ = {num(m['tau'][0])}, {num(m['tau'][1])} µm)" if k == "acfMap" else "",
        "abbottCurveError": "max |Δh| of the displayed curve",
        "regionVolume": "regions" if m.get("what") == "nRegions" else "region volume",
        "fitCircle": "fitted radius" if m.get("what") == "r" else "centre error",
        "fitLine": "fitted direction" if m.get("what") == "angle" else "line offset",
        "sampleHeight": "z(x, y)", "profileLine": "max |Δz| along the line",
        "cavities": "cavities found" if m.get("what") == "count" else f"cavity {m.get('what')}",
    }.get(k, k)


def case_label(c):
    return " → ".join([*(op_label(o) for o in c["ops"]), measure_label(c["measure"])])


def _csv_cell(v):
    if v is None:
        return ""
    s = num(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)
    return '"' + s.replace('"', '""') + '"' if re.search(r'[",\n]', s) else s


_TEX_ESC = {"\\": r"\textbackslash{}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _tex(s):
    return re.sub(r"[\\~^&%$#_{}]", lambda m: _TEX_ESC.get(m.group(0), "\\" + m.group(0)), "" if s is None else str(s))


def _tex_num(v):
    if v is None:
        return "--"
    if abs(v) >= 1e4 or (abs(v) < 1e-3 and v != 0):
        return re.sub(r"e([+-]\d+)", lambda m: f"\\times10^{{{int(m.group(1))}}}", exponential(v, 4))
    return num(float(precision(v, 7)))


def report(spec, runs):
    """Merge ``runs`` (result files of spec/README.md, Results, each with a short ``key``; the order given is the
    column order) into ``{"json", "csv", "tex", "implementations"}``."""
    groups, cases = [], []
    for c in spec["cases"]:
        if c["group"] not in groups:
            groups.append(c["group"])
        results = {}
        for r in runs:
            res = r["cases"].get(c["id"])
            j = judge(c, res)
            x = {"verdict": j["verdict"], "value": None if not res or res.get("nan") else res.get("value"),
                 "nan": bool(res and res.get("nan")), "error": j["error"], "limit": j["limit"], "ms": res.get("ms") if res else None}
            if res and res.get("error"):
                x["message"] = res["error"]
            results[r["key"]] = x
        j0 = judge(c, None)
        entry = {"id": c["id"], "group": c["group"], "gauge": c["gauge"], "tier": c["tier"], "domain": c["domain"],
                 "label": case_label(c), "unit": c["unit"], "standards": c["standards"], "method": c.get("method"),
                 "cite": c.get("cite"), "reference": c["reference"], "discrete": c.get("discrete"), "judgedRef": j0["ref"],
                 "samplingError": j0["samplingError"], "tol": c["tol"], "basis": c["basis"], "note": c.get("note"),
                 "expectNaN": bool(c.get("expectNaN")), "series": c.get("series"), "results": results}
        optional = ("method", "cite", "discrete", "note", "series")  # absent in the spec → absent here, as in JS
        cases.append({k: v for k, v in entry.items() if v is not None or k not in optional})
    implementations = []
    for r in runs:
        tally = {"pass": 0, "fail": 0, "unsupported": 0, "error": 0}
        for c in cases:
            tally[c["results"][r["key"]]["verdict"]] += 1
        implementations.append({"key": r["key"], **r["implementation"], "date": r.get("date"), "specVersion": r.get("specVersion"),
                                "tally": tally, "gaugesOk": all(g["ok"] for g in (r.get("gauges") or {}).values())})
    gauges = {gid: {**{k: v for k, v in g.items() if k != "fingerprint"}, "cases": sum(c["gauge"] == gid for c in cases)}
              for gid, g in spec["gauges"].items()}
    doc = {"name": spec["name"], "title": spec.get("title"), "specVersion": spec["version"], "conventions": spec["conventions"],
           "references": spec.get("references"), "generated": utc_now(),
           "implementations": implementations, "groups": groups, "gauges": gauges, "cases": cases}

    # ── CSV ─────────────────────────────────────────────────────────────────────
    rows = [",".join(["case", "group", "tier", "domain", "gauge", "quantity", "unit", "standards", "reference", "reference_formula",
                      "reference_text", "sampled_reference", "tol_abs", "tol_rel", "implementation", "value", "error", "limit", "verdict"])]
    for c in cases:
        for r in runs:
            x = c["results"][r["key"]]
            ref = c["reference"]
            rows.append(",".join(_csv_cell(v) for v in [
                c["id"], c["group"], c["tier"], c["domain"], c["gauge"], c["label"], c["unit"], "; ".join(c["standards"]), ref["value"],
                ref.get("formula"), ref.get("text"), c["discrete"]["value"] if "discrete" in c else "", c["tol"].get("abs", ""),
                c["tol"].get("rel", ""), r["key"], "NaN" if x["nan"] else x["value"], x["error"], x["limit"], x["verdict"]]))

    # ── LaTeX ───────────────────────────────────────────────────────────────────
    def vcell(x):
        if x["verdict"] == "unsupported":
            return r"\textit{n/a}"
        if x["verdict"] == "error":
            return r"\textbf{error}"
        v = r"\mathrm{NaN}" if x["nan"] else _tex_num(x["value"])
        return f"${v}$" + (r"\,\textbf{✗}" if x["verdict"] == "fail" else r"\,✓")

    names = "; ".join(f"{i['name']} {i.get('version') or ''}".strip() for i in implementations)
    T = f"% Generated by softgauge report — do not edit by hand.\n% Spec {spec['version']}; {names}.\n"
    T += (f"\\begin{{longtable}}{{@{{}}p{{2.6cm}}p{{3.6cm}}r{'r' * len(runs)}@{{}}}}\n\\toprule\nGauge & Quantity & Reference"
          + "".join(f" & {_tex(i['name'])}" for i in implementations) + "\\\\\n\\midrule\\endhead\n")
    for gname in groups:
        T += f"\\multicolumn{{{3 + len(runs)}}}{{@{{}}l}}{{\\textbf{{{_tex(gname)}}}}}\\\\\n"
        for c in (k for k in cases if k["group"] == gname):
            unit = f" [{_tex(c['unit'])}]" if c["unit"] else ""
            ref = r"\text{undef.}" if c["expectNaN"] else _tex_num(c["judgedRef"])
            T += f"{_tex(gauges[c['gauge']]['title'])} & {_tex(c['label'])}{unit} & ${ref}$" + "".join(f" & {vcell(c['results'][r['key']])}" for r in runs) + "\\\\\n"
    T += "\\bottomrule\n\\end{longtable}\n"

    return {"json": doc, "csv": "\n".join(rows) + "\n", "tex": T, "implementations": implementations}
