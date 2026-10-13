"""Gauge builder: recipe → height map (numpy twin of js/src/gauges.js).

Both builders are checked against the fingerprints frozen into spec.json, so every
implementation under test sees the same surface, whichever language its binding uses.

Grid convention: row-major, ``ny`` rows × ``nx`` columns, pixel (r, c) at x = c·dx,
y = r·dy (µm; row 0 is the first row of the array); components defined in the
"y measured from the bottom" box convention use yb = (ny − 1 − r)·dy.
"""
from __future__ import annotations

import math

import numpy as np

TAU = 2 * math.pi

__all__ = ["build_heights", "fingerprint", "verify_gauge"]


def tri(u):
    return 4 * np.abs(u - np.floor(u) - 0.5) - 1


def ramp_index(p, g, r, c, k):
    order = p.get('order', 'raster')
    if order == 'raster':
        return k
    if order == 'serpentine':
        return np.where(r % 2 == 1, r * g['nx'] + (g['nx'] - 1 - c), k)
    if order == 'scramble':
        return np.mod(p['mult'] * k, g['nx'] * g['ny'])
    raise ValueError(f"unknown ramp order {order}")


def component(p, g, r, c, k):
    x, y = c * g['dx'], r * g['dy']
    yb = (g['ny'] - 1 - r) * g['dy']
    t = p['type']
    if t == 'const':
        return np.full(x.shape, float(p['value']))
    if t == 'plane':
        return p['a'] * x + p['b'] * y + p.get('c', 0)
    if t == 'poly':
        u, v = x - p['xc'], y - p['yc']
        s = np.zeros(x.shape)
        for i, j, coef in p['terms']:
            s = s + coef * u ** i * v ** j
        return s
    if t == 'sine':
        th = p.get('theta', 0) * math.pi / 180
        return p['A'] * np.sin(TAU * (x * math.cos(th) + y * math.sin(th)) / p['lambda'] + p.get('phase', 0))
    if t == 'eggcrate':
        return p['A'] * np.cos(TAU * (x - p.get('x0', 0)) / p['lambdaX']) * np.cos(TAU * (y - p.get('y0', 0)) / p['lambdaY'])
    if t == 'triangle':
        z = p['A'] * tri(x / p['lambda'])
        if p.get('axes') == 'xy':
            z = z + p['A'] * tri(y / p['lambda'])
        return z
    if t == 'cos':
        return p['A'] * np.cos(TAU * (x - p.get('x0', 0)) / p['lambda'])
    if t == 'columns':
        return np.where(c < p['split'], float(p['left']), float(p['right']))
    if t == 'disc':
        rho = np.hypot(x - p['cx'], (y if p.get('frame') == 'top' else yb) - p['cy'])
        if not p.get('w'):
            return np.where(rho <= p['r'], float(p['h']), 0.0)
        return 0.5 * p['h'] * (1 - np.tanh((rho - p['r']) / p['w']))
    if t == 'bearing':
        N = g['nx'] * g['ny']
        m = 100 * k / (N - 1)
        kn = np.asarray(p['knots'], dtype=float)
        return np.interp(m, kn[:, 0], kn[:, 1])
    if t == 'bearingCos':
        N = g['nx'] * g['ny']
        return p['A'] * np.cos(math.pi * k / (N - 1))
    if t == 'bearingPoly':
        u = ramp_index(p, g, r, c, k) / (g['nx'] * g['ny'] - 1)
        h = np.zeros(u.shape)
        for a in reversed(p['coefs']):
            h = h * u + a
        return h
    if t == 'dither':
        return p['amp'] * np.sin(TAU * (0.7071067811865476 * c + 0.3183098861837907 * r)) * np.cos(TAU * (0.4142135623730951 * r - 0.2718281828459045 * c))
    if t == 'sphere':
        return p['zc'] + np.sqrt(p['R'] ** 2 - ((x - p['xc']) ** 2 + (y - p['yc']) ** 2))
    if t == 'cylinder':
        ph = p['phi'] * math.pi / 180
        d = -(x - p['px']) * math.sin(ph) + (y - p['py']) * math.cos(ph)
        return p['zc'] + np.sqrt(p['R'] ** 2 - d * d)
    if t == 'gauss':
        return p['h'] * np.exp(-((x - p['xc']) ** 2 + (y - p['yc']) ** 2) / (2 * p['s'] ** 2))
    if t == 'bump':
        rho = np.hypot(x - p['xc'], y - p['yc'])
        return np.where(rho < p['r0'], p['h'] * np.cos(math.pi * rho / (2 * p['r0'])) ** 2, 0.0)
    if t == 'edge':
        ph = p['phi'] * math.pi / 180
        d = -(x - p['x0']) * math.sin(ph) + (y - p['y0']) * math.cos(ph)
        return 0.5 * p['h'] * (1 + np.tanh(d / p['w']))
    if t == 'pockets':
        z = np.zeros(x.shape)
        for q in p['pockets']:
            inside = (np.abs(x - q['cx']) < q['w'] / 2) & (np.abs(y - q['cy']) < q['h'] / 2) & (z == 0)
            z = np.where(inside, -float(q['depth']), z)
        return z
    raise ValueError(f"unknown gauge component {t}")


def build_heights(g):
    r, c = np.mgrid[0:g['ny'], 0:g['nx']].astype(float)
    k = r * g['nx'] + c
    z = np.zeros((g['ny'], g['nx']))
    for p in g['parts']:
        z = z + component(p, g, r, c, k)
    for rr, cc, dz in g.get('spikes', []):
        z[rr, cc] += dz
    for r0, r1, c0, c1 in g.get('holes', []):
        z[r0:r1 + 1, c0:c1 + 1] = np.nan
    return z.astype(np.float32) if g.get('dtype') == 'float32' else z


def _probes(g):
    nx, ny = g["nx"], g["ny"]
    return [(0, 0), (ny - 1, nx - 1), (ny >> 1, nx >> 1), (ny >> 2, (3 * nx) >> 2), ((3 * ny) >> 2, nx >> 3)]


def fingerprint(g, z=None):
    """Count of measured points, their mean and RMS about zero, min, max, and z at fixed probe pixels."""
    z = build_heights(g) if z is None else np.asarray(z).reshape(g["ny"], g["nx"])
    v = z[np.isfinite(z)].astype(float)
    probes = []
    for r, c in _probes(g):
        x = float(z[r, c])
        probes.append([r, c, None if math.isnan(x) else x])
    return {
        "n": int(v.size), "mean": float(v.mean()), "rms": math.sqrt(float(np.mean(v * v))),
        "min": float(v.min()), "max": float(v.max()), "probes": probes,
    }


def verify_gauge(g, z=None):
    """Check heights against the fingerprint frozen into the spec → ``{"ok": bool, "maxRelDiff": float}``.

    Differences are scaled by the gauge RMS: math libraries differ in the last bit, and a
    zero-mean gauge's mean is ~1e−17.
    """
    fp, ref = fingerprint(g, z), g["fingerprint"]
    scale = ref["rms"] + 1e-300

    def rel(a, b):
        return 0.0 if a == b else abs(a - b) / (scale + abs(b))

    d = max(rel(fp[k], ref[k]) for k in ("mean", "rms", "min", "max"))
    for (_, _, got), (_, _, want) in zip(fp["probes"], ref["probes"]):
        if got != want:
            d = max(d, math.inf if got is None or want is None else rel(got, want))
    tol = 1e-6 if g.get("dtype") == "float32" else 1e-11
    return {"ok": fp["n"] == ref["n"] and d <= tol, "maxRelDiff": d}
