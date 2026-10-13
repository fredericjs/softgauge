"""The single pass/fail rule of the suite (mirrored in js/src/judge.js)."""
from __future__ import annotations

__all__ = ["judge", "judged_reference"]


def judged_reference(c):
    """Reference a case is judged against: the sampled-gauge value when given, else the continuous one."""
    d = c.get("discrete")
    return d["value"] if d and d.get("value") is not None else c["reference"]["value"]


def judge(c, r):
    """Judge one result ``r`` (``{"status", "value", "nan"?}``) of case ``c``.

    Returns ``{"verdict": "pass"|"fail"|"unsupported"|"error", "ref", "error", "limit",
    "samplingError"}`` with error = value − ref, limit = allowed |error| and
    samplingError = discrete − continuous reference.
    """
    ref = judged_reference(c)
    out = {"ref": ref, "error": None, "limit": None, "samplingError": None}
    d = c.get("discrete")
    if d and d.get("value") is not None and c["reference"].get("value") is not None:
        out["samplingError"] = d["value"] - c["reference"]["value"]
    if not r or r.get("status") == "unsupported":
        return {**out, "verdict": "unsupported"}
    if r.get("status") != "ok":
        return {**out, "verdict": "error"}
    value = r.get("value")
    is_nan = bool(r.get("nan")) or value is None or value != value
    if c.get("expectNaN"):
        return {**out, "verdict": "pass" if is_nan else "fail"}
    if is_nan:
        return {**out, "verdict": "fail"}
    tol = c.get("tol") or {}
    out["limit"] = (tol.get("abs") or 0) + (tol.get("rel") or 0) * abs(ref)
    out["error"] = value - ref
    return {**out, "verdict": "pass" if abs(out["error"]) <= out["limit"] else "fail"}
