#!/usr/bin/env python3
"""Evaluate the catalogue into the language-neutral spec.json.

    python generator/build.py           write spec/spec.json and the copies shipped by the packages
    python generator/build.py --check   recompute and compare with the committed spec.json

spec.json is the contract shared by every implementation: gauge recipes (+ the
fingerprints their builders must reproduce), the operations, the measures, the frozen
reference values and the tolerances. Only this directory knows how the references are
derived; the packages and the bindings read spec.json alone.

--check compares numbers, not text: references and fingerprints go through libm (sin,
acos…) and numpy's summation, which may differ in the last bit between platforms and
versions. References must agree to 1e-12 relative; fingerprint values to 1e-12 of the
gauge RMS (a zero-mean gauge's mean is ~1e-17, so a relative test would fail on
round-off alone).

Needs numpy; the gauge heights come from the Python package's own builder
(python/src/softgauge/gauges.py), which is put on the path here.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'python' / 'src'))
sys.path.insert(0, str(HERE))

from softgauge.gauges import fingerprint  # noqa: E402
from softgauge.measures import ARRAY_MEASURES  # noqa: E402

from catalogue import CASES, GAUGES  # noqa: E402
from fmt import dumps, exponential  # noqa: E402

SPEC_VERSION = '1.0.0'

# ── tiers (SPEC.md, Tiers) ────────────────────────────────────────────────────
# core       parameters of an unprocessed gauge: anything that can read the gauge can run it
# operators  standard operations (levelling, filtering, masking, geometry…) measured on the array
# extension  non-standard parameters, implementation-specific operations and analysis outputs
STANDARD_OPS = {'level', 'detrend', 'levelPoints', 'levelTLS', 'removeSphere', 'removeCylinder', 'filter',
                'center', 'zero', 'invert', 'threshold', 'removeOutliers', 'fill', 'crop', 'zoom', 'rotate',
                'maskRect', 'maskCircle', 'maskThreshold', 'maskInvert', 'maskToNaN'}
NON_STANDARD_PARAMS = {'period'}


def tier_of(c: dict) -> str:
    m = c['measure']
    standard_param = m['kind'] == 'param' and m['name'] not in NON_STANDARD_PARAMS
    if not c['ops'] and standard_param:
        return 'core'
    if all(o['op'] in STANDARD_OPS for o in c['ops']) and (standard_param or m['kind'] in ARRAY_MEASURES):
        return 'operators'
    return 'extension'


def build_spec() -> dict:
    gauges = {gid: {**g, 'fingerprint': fingerprint(g)} for gid, g in GAUGES.items()}
    ids = set()
    for c in CASES:
        if c['id'] in ids:
            raise ValueError(f"duplicate case id {c['id']}")
        ids.add(c['id'])
        if c['gauge'] not in gauges:
            raise ValueError(f"case {c['id']}: unknown gauge {c['gauge']}")
        for r in (c['reference'], c.get('discrete')):
            if r and r.get('value') is not None and not math.isfinite(r['value']):
                raise ValueError(f"case {c['id']}: non-finite reference")
    return {
        'name': 'softgauge: analytic soft gauges for surface texture software',
        'version': SPEC_VERSION,
        'conventions': {
            'units': 'lengths and heights in µm, material ratios in %, angles in degrees',
            'grid': 'row-major; pixel (r, c) at x = c·dx, y = r·dy (row 0 first); samples are points, so n samples span (n − 1)·dx; crop/mask boxes and circles use y measured from the bottom: yb = (ny − 1 − r)·dy',
            'judging': 'pass iff |value − ref| ≤ tol.abs + tol.rel·|ref|, where ref = discrete.value if present else reference.value; expectNaN cases pass iff the value is NaN/undefined',
            'tiers': 'core = parameters of an unprocessed gauge; operators = standard operations measured on the array; extension = non-standard parameters, implementation-specific operations and analysis outputs',
        },
        'gauges': gauges,
        'cases': [{**c, 'tier': tier_of(c), 'domain': 'profile' if gauges[c['gauge']].get('profile') else 'areal'} for c in CASES],
    }


def _rel(a: float, b: float, scale: float = 0.0) -> float:
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b), scale, 1e-300)


def diff(a, b, path: str = '', scale: float = 0.0) -> float:
    """Max relative difference between two JSON trees; raises if their structure or text differs.

    Inside a gauge fingerprint, differences are relative to the gauge RMS (``scale``).
    """
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return _rel(a, b, scale)
    if not isinstance(a, (dict, list)) or not isinstance(b, (dict, list)):
        if a != b:
            raise ValueError(f'spec differs at {path or "/"}: {json.dumps(a, ensure_ascii=False)} ≠ {json.dumps(b, ensure_ascii=False)}')
        return 0.0
    if isinstance(a, dict) != isinstance(b, dict):
        raise ValueError(f'spec differs at {path or "/"}: type')
    if isinstance(a, dict):
        if list(a) != list(b):
            raise ValueError(f'spec differs at {path or "/"}: keys')
        if path.endswith('/fingerprint'):
            scale = abs(b.get('rms', 0.0))
        return max((diff(a[k], b[k], f'{path}/{k}', scale) for k in a), default=0.0)
    if len(a) != len(b):
        raise ValueError(f'spec differs at {path or "/"}: length')
    return max((diff(x, y, f'{path}/{i}', scale) for i, (x, y) in enumerate(zip(a, b))), default=0.0)


def as_module(text: str) -> str:
    return f'// Generated by generator/build.py from spec/spec.json — do not edit.\nexport default {text.rstrip()};\n'


OUT = {
    'json': ROOT / 'spec' / 'spec.json',
    'js': ROOT / 'js' / 'src' / 'spec-data.js',
    'py': ROOT / 'python' / 'src' / 'softgauge' / 'spec.json',
}


def read(p: Path) -> str:
    return p.read_text(encoding='utf-8')


def write(p: Path, text: str) -> None:
    p.write_text(text, encoding='utf-8', newline='\n')


def main(argv: list[str]) -> int:
    spec = build_spec()
    text = dumps(spec) + '\n'
    if '--check' in argv:
        committed = read(OUT['json'])
        d = diff(json.loads(text), json.loads(committed))
        if d > 1e-12:
            print(f'spec.json is out of date (max relative difference {exponential(d, 2)}); run python generator/build.py', file=sys.stderr)
            return 1
        if read(OUT['py']) != committed:
            print(f"{OUT['py']} is not a copy of spec/spec.json", file=sys.stderr)
            return 1
        if read(OUT['js']) != as_module(committed):
            print(f"{OUT['js']} is not generated from spec/spec.json", file=sys.stderr)
            return 1
        print(f'spec.json up to date (max relative difference {exponential(d, 2)})')
        return 0
    write(OUT['json'], text)
    write(OUT['py'], text)
    write(OUT['js'], as_module(text))
    tiers: dict[str, int] = {}
    for c in spec['cases']:
        tiers[c['tier']] = tiers.get(c['tier'], 0) + 1
    print(f"spec {spec['version']}: {len(spec['gauges'])} gauges, {len(spec['cases'])} cases ({', '.join(f'{v} {k}' for k, v in tiers.items())})")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
