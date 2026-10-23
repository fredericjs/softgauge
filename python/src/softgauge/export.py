"""Export gauges as files, for software that cannot run a binding (closed-source packages,
instrument software): load the file, compute the parameters, compare with the sidecar.

SDF (ISO 25178-71) is written by `sdfio <https://pypi.org/project/sdfio/>`_, an
independent implementation (``pip install softgauge[sdf]``); softgauge does not carry a
file writer of its own. Heights are stored as binary64 (or 17 significant digits in
ASCII), so the file holds the gauge to the last bit up to the µm → m conversion, and
non-measured points use the format's own marker (``BAD`` in ASCII, the sentinel value in
binary).

Next to every file a ``<gauge>.json`` sidecar holds the fingerprint (point count, mean,
RMS, min, max and heights at probe pixels) — check it first to prove that your software
read the gauge correctly — and every case defined on the gauge with its reference and
tolerance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .gauges import build_heights, verify_gauge
from .judge import judged_reference
from .spec import load_spec

__all__ = ["FORMATS", "export_all", "export_gauge", "read_sdf", "sidecar"]

UM = 1e-6  # µm → m

#: Supported formats: name → file suffix.
FORMATS = {"sdf": ".sdf", "sdf-ascii": ".sdf"}


def _sdfio():
    try:
        import sdfio
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("SDF export needs sdfio: pip install softgauge[sdf]") from e
    return sdfio


def sidecar(gauge_id, spec=None, file_name=None):
    """The sidecar document of one gauge."""
    spec = load_spec() if spec is None else spec
    g = spec["gauges"][gauge_id]
    cases = [
        {k: c.get(k) for k in ("id", "tier", "group", "cite", "ops", "measure", "unit", "standards", "method", "reference", "discrete", "expectNaN", "tol", "basis", "note") if c.get(k) is not None}
        | {"judgedReference": judged_reference(c)}
        for c in spec["cases"] if c["gauge"] == gauge_id
    ]
    return {
        "softgauge": spec["version"],
        "file": file_name,
        "gauge": {k: g[k] for k in ("id", "title", "formula", "description", "cite", "nx", "ny", "dx", "dy") if k in g}
        | {"profile": bool(g.get("profile")), "units": "lengths and heights in µm (the SDF file itself stores metres)"},
        "conventions": spec["conventions"],
        "fingerprint": g["fingerprint"],
        "cases": cases,
        "references": {k: spec["references"][k] for k in sorted({k for x in [g, *cases] for k in x.get("cite", [])})},
    }


def export_gauge(gauge_id, out_dir, fmt="sdf", spec=None):
    """Write one gauge (+ its sidecar) to ``out_dir``; returns the file path."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; one of {', '.join(FORMATS)}")
    spec = load_spec() if spec is None else spec
    sdfio = _sdfio()
    g = spec["gauges"][gauge_id]
    z = build_heights(g).astype(float)
    if not verify_gauge(g, z)["ok"]:
        raise RuntimeError(f"gauge {gauge_id} does not reproduce its fingerprint")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{gauge_id}{FORMATS[fmt]}"
    sdfio.write(path, z * UM, x_scale=g["dx"] * UM, y_scale=g["dy"] * UM, z_scale=UM,
                format=sdfio.FileFormat.ASCII if fmt == "sdf-ascii" else sdfio.FileFormat.BINARY)
    back = read_sdf(path)
    if back.shape != z.shape or not verify_gauge(g, back)["ok"]:
        raise RuntimeError(f"{path} does not read back as gauge {gauge_id}")
    (out / f"{gauge_id}.json").write_text(json.dumps(sidecar(gauge_id, spec, path.name), indent=1, allow_nan=False) + "\n", encoding="utf-8")
    return path


def export_all(out_dir, fmt="sdf", spec=None, gauges=None):
    """Write every gauge (or the ids in ``gauges``); returns the file paths."""
    spec = load_spec() if spec is None else spec
    return [export_gauge(gid, out_dir, fmt, spec) for gid in (gauges or spec["gauges"])]


def read_sdf(path):
    """Heights of an SDF file in µm, as a (rows, cols) array (NaN = non-measured)."""
    return np.asarray(_sdfio().read(path).data, dtype=float) / UM
