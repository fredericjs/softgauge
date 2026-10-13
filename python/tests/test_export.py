import json

import numpy as np
import pytest

from softgauge import build_heights, load_spec, verify_gauge
from softgauge.cli import main
from softgauge.export import export_gauge, read_sdf

sdfio = pytest.importorskip("sdfio")
SPEC = load_spec()


@pytest.mark.parametrize("fmt", ["sdf", "sdf-ascii"])
@pytest.mark.parametrize("gid", list(SPEC["gauges"]))
def test_round_trip(tmp_path, gid, fmt):
    """sdfio writes the gauge, sdfio reads it back, and the heights reproduce the fingerprint."""
    g = SPEC["gauges"][gid]
    path = export_gauge(gid, tmp_path, fmt)
    z = read_sdf(path)
    assert z.shape == (g["ny"], g["nx"])
    assert verify_gauge(g, z)["ok"]
    want = build_heights(g).astype(float)
    assert np.array_equal(np.isnan(z), np.isnan(want))
    ok = np.isfinite(want)
    assert np.max(np.abs(z[ok] - want[ok]), initial=0) <= 1e-12 * max(1.0, np.max(np.abs(want[ok])))
    sdf = sdfio.read(path)
    assert sdf.header.x_scale == pytest.approx(g["dx"] * 1e-6, rel=1e-15)
    assert sdf.header.y_scale == pytest.approx(g["dy"] * 1e-6, rel=1e-15)
    side = json.loads((tmp_path / f"{gid}.json").read_text(encoding="utf-8"))
    assert side["fingerprint"] == g["fingerprint"]
    assert {c["id"] for c in side["cases"]} == {c["id"] for c in SPEC["cases"] if c["gauge"] == gid}


def test_holes_are_non_measured(tmp_path):
    gid = next(k for k, g in SPEC["gauges"].items() if g.get("holes"))
    for fmt in ("sdf", "sdf-ascii"):
        z = read_sdf(export_gauge(gid, tmp_path / fmt, fmt))
        assert np.isnan(z).sum() == np.isnan(build_heights(SPEC["gauges"][gid])).sum() > 0


def test_cli_export_by_tier(tmp_path):
    assert main(["export", str(tmp_path), "--tier", "core"]) == 0
    used = {c["gauge"] for c in SPEC["cases"] if c["tier"] == "core"}
    assert {p.stem for p in tmp_path.glob("*.sdf")} == used
