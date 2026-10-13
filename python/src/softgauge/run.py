"""Run the spec against any implementation through a binding (mirrored in js/src/run.js).

softgauge knows nothing about the software under test. A binding is any object with

``load(z, gauge) -> handle``
    ``z`` is a fresh ``(ny, nx)`` float array (``(1, nx)`` for a profile gauge,
    ``gauge["profile"]`` true), heights in µm, steps ``gauge["dx"]``/``gauge["dy"]``.
``apply(handle, op, gauge) -> handle``
    One operation of ``case["ops"]`` (SPEC.md, Operations). Must not modify ``handle``:
    chains are cached.
``parameter(handle, name, args, gauge) -> float``
    Measure kind 'param'.
``measure(handle, m, gauge) -> float`` *(optional)*
    Any measure; return ``NotImplemented`` to fall back to the array measures.
``to_array(handle) -> ArrayResult | dict | tuple`` *(optional)*
    The output heights, for the array measures.

Raise :class:`~softgauge.Unsupported` (or leave a method out) for anything the software
cannot do: the case is reported 'unsupported', never 'fail'.
"""
from __future__ import annotations

import datetime
import json
import math
import time

from .gauges import build_heights, verify_gauge
from .judge import judge
from .measures import ARRAY_MEASURES, Unsupported, array_measure
from .spec import load_spec, select_cases

__all__ = ["Runner", "run"]


def _encode(v):
    """JSON has no NaN/Infinity: NaN is flagged, ±inf (e.g. an unfilled hole) is clamped to ±1e308."""
    if v is None:
        return {"status": "ok", "value": None, "nan": True}
    v = float(v)
    if math.isnan(v):
        return {"status": "ok", "value": None, "nan": True}
    if math.isinf(v):
        return {"status": "ok", "value": math.copysign(1e308, v)}
    return {"status": "ok", "value": v}


class Runner:
    """Runs cases one at a time (e.g. one pytest test per case), caching gauges and chains."""

    def __init__(self, binding, spec=None):
        self.binding = binding
        self.spec = load_spec() if spec is None else spec
        self.gauges = {}
        self._heights = {}
        self._chains = {}

    def heights(self, gauge_id):
        if gauge_id not in self._heights:
            g = self.spec["gauges"][gauge_id]
            z = build_heights(g)
            self.gauges[gauge_id] = verify_gauge(g, z)
            self._heights[gauge_id] = z
        return self._heights[gauge_id]

    def _evaluate(self, h, m, g):
        b = self.binding
        if m["kind"] == "param":
            if not hasattr(b, "parameter"):
                raise Unsupported("parameters")
            return b.parameter(h, m["name"], list(m.get("args", [])), g)
        if hasattr(b, "measure"):
            v = b.measure(h, m, g)
            if v is not NotImplemented:
                return v
        if m["kind"] in ARRAY_MEASURES and hasattr(b, "to_array"):
            return array_measure(m, b.to_array(h), g)
        raise Unsupported(f"measure {m['kind']}")

    def case(self, c):
        """Result of one case: ``{"status": "ok"|"unsupported"|"error", "value", "nan"?, "error"?, "ms"}``."""
        t0 = time.perf_counter()
        try:
            g = self.spec["gauges"][c["gauge"]]
            z = self.heights(c["gauge"])
            if not self.gauges[c["gauge"]]["ok"]:
                raise RuntimeError("gauge fingerprint mismatch")
            key = c["gauge"] + "|" + json.dumps(c["ops"], sort_keys=True)
            if key not in self._chains:
                h = self.binding.load(z.copy(), g)
                for o in c["ops"]:
                    if not hasattr(self.binding, "apply"):
                        raise Unsupported(f"operation {o['op']}")
                    h = self.binding.apply(h, o, g)
                self._chains[key] = h
            res = _encode(self._evaluate(self._chains[key], c["measure"], g))
        except Unsupported as e:
            res = {"status": "unsupported", "error": str(e)}
        except Exception as e:  # an implementation error is a result, not a crash
            res = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        res["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return res


def run(binding, spec=None, *, implementation=None, cases=None, on_case=None, **filters):
    """Run ``cases`` (default: the spec's cases filtered by ``tier``/``domain``/``group``/``only``)
    through ``binding`` and return the result file of SPEC.md, Results.

    ``on_case(case, result, verdict)`` is called after each case.
    """
    runner = Runner(binding, spec)
    selected = select_cases(runner.spec, **filters) if cases is None else cases
    out = {}
    for c in selected:
        res = runner.case(c)
        out[c["id"]] = res
        if on_case:
            on_case(c, res, judge(c, res)["verdict"])
    return {
        "implementation": implementation or {"name": "unnamed"},
        "specVersion": runner.spec["version"],
        "gauges": runner.gauges,
        "cases": out,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
