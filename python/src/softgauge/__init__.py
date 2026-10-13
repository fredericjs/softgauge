"""softgauge — analytic soft gauges for surface texture software.

The package builds each gauge's height map, holds the frozen analytic references, runs
any implementation through a thin binding and judges the results. It never imports the
software under test. See SPEC.md for the contract.
"""
from __future__ import annotations

from .gauges import build_heights, fingerprint, verify_gauge
from .judge import judge, judged_reference
from .measures import ARRAY_MEASURES, ArrayResult, Unsupported, array_measure
from .run import Runner, run
from .spec import load_spec, select_cases

__version__ = "0.1.0"

__all__ = [
    "ARRAY_MEASURES", "ArrayResult", "Runner", "Unsupported", "array_measure", "build_heights", "fingerprint",
    "judge", "judged_reference", "load_spec", "run", "select_cases", "verify_gauge",
]
