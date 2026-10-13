"""pytest helpers: one test per case.

.. code-block:: python

    # tests/test_softgauge.py
    import pytest
    import softgauge
    from softgauge.testing import cases, check

    from mypackage_binding import MyBinding

    runner = softgauge.Runner(MyBinding())
    KNOWN_DEFECTS = {"hybrid.egg.Sdr"}          # judged as xfail

    @pytest.mark.parametrize("case", cases(tier=["core", "operators"]))
    def test_softgauge(case):
        check(runner, case, xfail=KNOWN_DEFECTS)

'unsupported' cases are skipped, failures report value, reference, error and tolerance.
pytest is imported lazily, so softgauge itself does not depend on it.
"""
from __future__ import annotations

from .judge import judge
from .spec import load_spec, select_cases

__all__ = ["cases", "check"]


def cases(spec=None, **filters):
    """``pytest.param`` per selected case, with the case id as the test id."""
    import pytest

    return [pytest.param(c, id=c["id"]) for c in select_cases(load_spec() if spec is None else spec, **filters)]


def check(runner, case, xfail=()):
    """Run ``case`` through ``runner`` (a :class:`softgauge.Runner`) and turn the verdict into
    a pytest outcome: pass, ``pytest.skip`` (unsupported), ``pytest.xfail`` (ids in
    ``xfail`` that fail) or ``pytest.fail``."""
    import pytest

    res = runner.case(case)
    j = judge(case, res)
    v = j["verdict"]
    if v == "unsupported":
        pytest.skip(res.get("error") or "unsupported")
    if v == "pass":
        if case["id"] in xfail:
            pytest.fail(f"{case['id']} passes but is listed as a known defect", pytrace=False)
        return
    if v == "error":
        msg = f"{case['id']}: error: {res.get('error')}"
    else:
        value = "NaN" if res.get("nan") else res.get("value")
        msg = (f"{case['id']}: value {value} vs reference {j['ref']} ({case['reference'].get('formula', '')}); "
               f"|error| {abs(j['error']) if j['error'] is not None else '—'} > limit {j['limit']} [{case.get('basis', '')}]")
    if case["id"] in xfail:
        pytest.xfail(msg)
    pytest.fail(msg, pytrace=False)
