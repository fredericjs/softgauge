"""Measures softgauge evaluates itself, from the height array a binding returns after the
operations (mirrored in js/src/measures.js).

A binding only has to hand back its result as a plain array, and every implementation is
then measured by the same code: fieldError, sineAmplitude, nanMismatch, planeGradient and
stat. Parameters ('param') and the extension measures stay with the binding (SPEC.md,
Measures).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .gauges import build_heights

__all__ = ["ARRAY_MEASURES", "ArrayResult", "Unsupported", "array_measure"]

TAU = 2 * math.pi

#: Measure kinds evaluated by :func:`array_measure`.
ARRAY_MEASURES = frozenset({"stat", "fieldError", "sineAmplitude", "nanMismatch", "planeGradient"})


class Unsupported(Exception):
    """Raised by a binding (or softgauge) for anything the software cannot do: the case is
    reported 'unsupported', never 'fail'."""


@dataclass
class ArrayResult:
    """A binding's output: heights ``z`` (rows × cols, µm, NaN = non-measured), the
    sampling steps in µm and an optional boolean mask (True = masked)."""

    z: np.ndarray
    dx: float
    dy: float
    mask: np.ndarray | None = None

    @classmethod
    def coerce(cls, a):
        if isinstance(a, cls):
            out = a
        elif isinstance(a, dict):
            out = cls(a["z"], a["dx"], a["dy"], a.get("mask"))
        else:
            out = cls(*a)
        z = np.asarray(out.z, dtype=float)
        z = z[None, :] if z.ndim == 1 else z
        mask = None if out.mask is None else np.asarray(out.mask, dtype=bool).reshape(z.shape)
        return cls(z, float(out.dx), float(out.dy), mask)


def _lsq(a, basis, use):
    rows, cols = a.z.shape
    r, c = np.mgrid[0:rows, 0:cols]
    sel = np.isfinite(a.z) & use(r, c)
    A = np.column_stack(basis(c[sel] * a.dx, r[sel] * a.dy))
    coef, *_ = np.linalg.lstsq(A, a.z[sel], rcond=None)
    return coef


def _stat(a, name):
    z = a.z
    rows, cols = z.shape
    if name == "rows":
        return rows
    if name == "cols":
        return cols
    if name == "stepX":
        return a.dx
    if name == "stepY":
        return a.dy
    if name == "nanCount":
        return int(np.isnan(z).sum())
    if name == "maskedCount":
        return 0 if a.mask is None else int(a.mask.sum())
    if name in ("mean", "min", "max", "validMean"):
        sel = np.isfinite(z)
        if name == "validMean" and a.mask is not None:
            sel &= ~a.mask
        v = z[sel]
        if not v.size:
            return math.nan
        return float(v.min() if name == "min" else v.max() if name == "max" else v.mean())
    raise Unsupported(f"stat {name}")


def array_measure(m, a, gauge=None):
    """Evaluate measure ``m`` (``case["measure"]``, kind in :data:`ARRAY_MEASURES`) on a
    binding's output ``a`` (:class:`ArrayResult`, a dict with z/dx/dy/mask, or a tuple).

    Returns ±inf where the spec asks for it (requireFinite on a NaN).
    """
    a = ArrayResult.coerce(a)
    z = a.z
    rows, cols = z.shape
    kind = m["kind"]
    if kind == "stat":
        return _stat(a, m["stat"])
    if kind == "fieldError":
        ez = build_heights({"nx": cols, "ny": rows, "dx": a.dx, "dy": a.dy, "parts": m["expect"]}).astype(float)
        sel = np.ones(z.shape, dtype=bool)
        mg, mr = m.get("margin", 0), m.get("marginRows", 0)
        if mg:
            sel[:, :mg] = False
            sel[:, cols - mg:] = False
        if mr:
            sel[:mr, :] = False
            sel[rows - mr:, :] = False
        if m.get("skipSpikes"):
            for rr, cc, _ in (gauge or {}).get("spikes", []):
                sel[rr, cc] = False
        if m.get("requireFinite") and np.isnan(z[sel]).any():
            return math.inf
        sel &= np.isfinite(z)
        return float(np.max(np.abs(z[sel] - ez[sel]))) if sel.any() else 0.0
    if kind == "sineAmplitude":
        k, mg = TAU / m["lambda"], m.get("margin", 0)
        if m.get("axis") == "y":
            coef = _lsq(a, lambda x, y: [np.ones_like(y), np.sin(k * y), np.cos(k * y)], lambda r, c: (r >= mg) & (r < rows - mg))
        else:
            coef = _lsq(a, lambda x, y: [np.ones_like(x), np.sin(k * x), np.cos(k * x)], lambda r, c: (c >= mg) & (c < cols - mg))
        return float(math.hypot(coef[1], coef[2]))
    if kind == "nanMismatch":
        want = np.zeros(z.shape, dtype=bool)
        for rr, cc in m["points"]:
            want[rr, cc] = True
        return int((np.isnan(z) != want).sum())
    if kind == "planeGradient":
        mg = m.get("margin", 0)
        coef = _lsq(a, lambda x, y: [np.ones_like(x), x, y], lambda r, c: (r >= mg) & (r < rows - mg) & (c >= mg) & (c < cols - mg))
        gx, gy_up = coef[1], -coef[2]  # y measured upwards (rows run downwards)
        return float(math.hypot(gx, gy_up)) if m["component"] == "mag" else float(math.degrees(math.atan2(gy_up, gx)))
    raise Unsupported(f"measure {kind}")
