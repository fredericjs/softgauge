import json
import math
from pathlib import Path

import numpy as np
import pytest

import softgauge
from softgauge import array_measure, build_heights, judge, load_spec, select_cases, verify_gauge
from softgauge.testing import cases, check

from naive_binding import NaiveBinding

SPEC = load_spec()
REPO_SPEC = Path(__file__).resolve().parents[2] / "spec" / "spec.json"


@pytest.mark.skipif(not REPO_SPEC.exists(), reason="outside the repository")
def test_bundled_spec_is_the_repository_spec():
    assert SPEC == json.loads(REPO_SPEC.read_text(encoding="utf-8"))


@pytest.mark.parametrize("gid", list(SPEC["gauges"]))
def test_builder_reproduces_the_js_fingerprint(gid):
    """The numpy builder against the fingerprints computed by the JS builder."""
    res = verify_gauge(SPEC["gauges"][gid])
    assert res["ok"], res


def test_judge():
    c = {"reference": {"value": 1}, "tol": {"abs": 0.1, "rel": 0.1}}
    assert judge(c, {"status": "ok", "value": 1.19})["verdict"] == "pass"
    assert judge(c, {"status": "ok", "value": 1.21})["verdict"] == "fail"
    assert judge(c, {"status": "ok", "value": None, "nan": True})["verdict"] == "fail"
    assert judge(c, {"status": "unsupported"})["verdict"] == "unsupported"
    assert judge(c, None)["verdict"] == "unsupported"
    assert judge(c, {"status": "error"})["verdict"] == "error"
    assert judge({**c, "discrete": {"value": 2}}, {"status": "ok", "value": 2.25})["verdict"] == "pass"
    assert judge({**c, "expectNaN": True}, {"status": "ok", "value": None, "nan": True})["verdict"] == "pass"


def test_array_measures_on_the_unmodified_gauge():
    g = SPEC["gauges"]["plane"]
    a = (build_heights(g), g["dx"], g["dy"])
    assert array_measure({"kind": "fieldError", "expect": g["parts"]}, a, g) == 0
    assert array_measure({"kind": "stat", "stat": "cols"}, a) == g["nx"]
    p = g["parts"][0]
    assert abs(array_measure({"kind": "planeGradient", "component": "mag"}, a) - math.hypot(p["a"], p["b"])) < 1e-12
    with pytest.raises(softgauge.Unsupported):
        array_measure({"kind": "cavities"}, a)


def test_select_cases():
    core = select_cases(tier="core", domain="profile")
    assert core and all(c["tier"] == "core" and c["domain"] == "profile" for c in core)
    assert len(select_cases(tier=["core", "operators", "extension"])) == len(SPEC["cases"])


def test_naive_binding_passes_every_case_it_supports():
    res = softgauge.run(NaiveBinding(), implementation={"name": "naive"})
    verdicts = {c["id"]: judge(c, res["cases"][c["id"]])["verdict"] for c in SPEC["cases"]}
    bad = {k: res["cases"][k] for k, v in verdicts.items() if v in ("fail", "error")}
    assert not bad
    assert sum(v == "pass" for v in verdicts.values()) >= 30
    assert all(g["ok"] for g in res["gauges"].values())
    json.dumps(res, allow_nan=False)


RUNNER = softgauge.Runner(NaiveBinding())


@pytest.mark.parametrize("case", cases(only="height."))
def test_pytest_helper(case):
    check(RUNNER, case)
