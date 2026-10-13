# Writing a binding

A **binding** connects softgauge to the software you want to validate. softgauge builds
each gauge, hands its height array to the binding, asks it to apply the case's
operations and evaluate the measure, then judges the number that comes back. The binding
only translates between softgauge's vocabulary (SPEC.md) and your software's API. It
contains no metrology of its own.

```
softgauge                                  your binding                     your software
─────────                                  ────────────                     ─────────────
build gauge, check fingerprint ── z ──────▶ load(z, gauge)  ──────────────▶ Surface(z, dx, dy)
for each op of the case        ── op ─────▶ apply(h, op)    ──────────────▶ s.level(), s.filter(…)
measure.kind = 'param'         ── name ───▶ parameter(h, name, args) ─────▶ s.Sq()
array measures (fieldError…)   ◀── array ── toArray(h)      ◀────────────── s.data
judge against the frozen reference
```

A binding can start small. Anything it does not handle is reported as `unsupported`,
which is not a failure. So you can begin with a handful of parameters and grow from
there.

## 1. The interface

| method | required | Python | JavaScript | returns |
|---|---|---|---|---|
| load the gauge | yes | `load(self, z, gauge)` | `load(z, gauge)` | a *handle*: whatever your software uses (a Surface object, an id…) |
| apply one operation | for `operators` cases | `apply(self, h, op, gauge)` | `apply(h, op, gauge)` | a new handle |
| one parameter | for `param` measures | `parameter(self, h, name, args, gauge)` | `parameter(h, name, args, gauge)` | a number |
| any other measure | optional | `measure(self, h, m, gauge)` | `measure(h, m, gauge)` | a number, or `NotImplemented` (Python) / `undefined` (JS) to fall back to softgauge's array measures |
| the output heights | for array measures | `to_array(self, h)` | `toArray(h)` | the heights, steps and mask (below) |

Python bindings are any object with these methods (duck typing). JavaScript bindings are
a plain object.

**What `load` receives**

- `z` is a fresh copy, so you may keep it or modify it.
  - Python: `numpy.ndarray` of shape `(ny, nx)`, float64 (float32 for the one
    storage-precision gauge). For a profile gauge, `(1, nx)`.
  - JS: `Float64Array` (or `Float32Array`) of length `ny·nx`, row-major. For a profile,
    length `nx`.
- Heights are in **µm**. NaN marks non-measured points.
- `gauge` is the gauge's entry in `spec.json`. The fields you need are `nx`, `ny`, `dx`,
  `dy` (µm) and `profile` (true for a profile, which is evaluated with R parameters).

**What `to_array` / `toArray` returns**

- Python: `softgauge.ArrayResult(z, dx, dy, mask=None)`, a dict with the same keys, or
  a tuple `(z, dx, dy[, mask])`. `z` may be 2-D `(rows, cols)` or 1-D for a profile.
- JS: `{ z, rows, cols, dx, dy, mask? }` with `z` **flat, row-major**.
- `mask`, if your software has one, is truthy for **masked (= excluded)** points.

**Saying "can't do that".** Raise / throw `softgauge.Unsupported("…reason…")`. A missing
method has the same effect. Any other exception is reported as `error`, which counts
against the implementation, so don't use errors for unsupported features.

## 2. Start with the core tier

`core` cases are standard parameters of an unprocessed gauge (461 of the 547 cases).
They need only `load` and `parameter`.

**Python**

```python
import softgauge
import mylib

class MyLibBinding:
    def load(self, z, gauge):
        if gauge.get("profile"):
            return mylib.Profile(z[0], gauge["dx"])
        return mylib.Surface(z, gauge["dx"], gauge["dy"])

    def parameter(self, s, name, args, gauge):
        fn = getattr(s, name, None)
        if fn is None:
            raise softgauge.Unsupported(f"parameter {name}")
        return float(fn(*args))

results = softgauge.run(MyLibBinding(), tier="core",
                        implementation={"name": "mylib", "version": mylib.__version__})
```

**JavaScript**

```js
import { loadSpec, selectCases, run, judge, Unsupported } from 'softgauge';
import * as mylib from 'mylib';

const binding = {
  load: (z, g) => (g.profile ? new mylib.Profile(z, g.dx) : new mylib.Surface(z, g.ny, g.nx, g.dx, g.dy)),
  parameter(s, name, args) {
    if (typeof s[name] !== 'function') throw new Unsupported(`parameter ${name}`);
    return s[name](...args);
  },
};

const spec = loadSpec();
const results = run(spec, binding, {
  implementation: { name: 'mylib', version: mylib.version },
  filter: (c) => c.tier === 'core',
});
```

If your software's parameter names differ from the ISO symbols, map them in
`parameter`, e.g. `{"Sq": s.rms_height, "Sdr": lambda: 100 * s.developed_ratio()}`.

## 3. Add operations

`operators` cases (54) chain operations before measuring. Translate each `op` object
(fields in SPEC.md, Operations) to your API, and throw `Unsupported` for the rest:

```python
    def apply(self, s, op, gauge):
        kind = op["op"]
        if kind == "level":
            return s.level()
        if kind == "detrend":
            return s.detrend(degree=op["degree"])
        if kind == "filter":
            if op.get("mode", "reflect") != "reflect":
                raise softgauge.Unsupported(f"filter border mode {op['mode']}")
            return s.gaussian_filter(op["type"], op["cutoff"], op.get("cutoff2"))
        if kind == "crop":
            x0, x1, y0, y1 = op["box"]          # µm, y measured from the bottom
            return s.crop(x0, x1, y0, y1)
        raise softgauge.Unsupported(f"operation {kind}")
```

Most operator cases are judged on the **output array** (`fieldError`, `sineAmplitude`,
`nanMismatch`, `planeGradient`, `stat`), which softgauge evaluates itself. Add
`to_array` / `toArray` so it can:

```python
    def to_array(self, s):
        return softgauge.ArrayResult(s.heights, s.dx, s.dy, mask=s.mask)   # mask: True = excluded
```

```js
  toArray: (s) => ({ z: s.data, rows: s.rows, cols: s.cols, dx: s.dx, dy: s.dy, mask: s.mask ?? null }),
```

## 4. Extension cases (optional)

`extension` cases (32) cover non-standard parameters and analysis outputs: form-fit
radius, radial PSD, ACF map, region volume, edge fits, cavities… Their measure objects
are described in SPEC.md, Measures, and carry their own fields. Implement the ones your
software offers in `measure` and return `NotImplemented` / `undefined` for the others:

```python
    def measure(self, s, m, gauge):
        if m["kind"] == "formRadius":
            return s.last_fit.radius
        return NotImplemented
```

## 5. Run it in your test suite

**pytest.** One test per case, unsupported cases skipped, known defects as xfail:

```python
# tests/test_softgauge.py
import pytest
import softgauge
from softgauge.testing import cases, check
from .binding import MyLibBinding

runner = softgauge.Runner(MyLibBinding())      # caches gauges and operation chains across tests

KNOWN_DEFECTS = {
    "hybrid.pyramids.Sdr",     # mylib issue 123: projected area over (n − 2) cells
}

@pytest.mark.parametrize("case", cases(tier=["core", "operators"]))
def test_softgauge(case):
    check(runner, case, xfail=KNOWN_DEFECTS)
```

A failure reports the value, the reference and its formula, the error, the limit and
the basis of the tolerance. A known defect that starts passing fails the test, so the
xfail list never goes stale.

**node --test** (or any JS runner):

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadSpec, run, judge } from 'softgauge';

test('softgauge', () => {
  const spec = loadSpec(), res = run(spec, binding);
  const failed = spec.cases.filter(c => ['fail', 'error'].includes(judge(c, res.cases[c.id]).verdict));
  assert.deepEqual(failed.map(c => c.id), []);
});
```

**A report.** Write the object `run` returns to `results/<name>.json`, one file per
implementation or version. Then:

```bash
npx softgauge report results/ --out report/
```

## 6. Conventions that trip bindings up

Get these right in the binding, not by loosening anything. Every one of them shows up
as a clean failure.

| | softgauge | if your software differs |
|---|---|---|
| units | µm for lengths and heights | convert in `load` and on the way out (e.g. a parameter in nm → ÷ 1000) |
| y axis | row r at y = r·dy; **boxes, circles and points in ops use y from the bottom** | convert `yb = (ny − 1)·dy − y` in `apply` |
| `Sdr` | % | × 100 if your software returns a ratio |
| `Spd`, `Svd` / `Spc`, `Svc` | 1/µm² / 1/µm | convert from 1/mm² etc. |
| `Smr(c)`, `Rmr(c)` | c in µm, in the frame of the data (no re-centring) | shift c if your software measures it from the mean or another datum |
| `Smc(mr)`, `Rmc(mr)` | mr in % | |
| `Std` | degrees, array frame (+x towards increasing row) | a y-up frame reads 180° − θ |
| `rotate` | counter-clockwise, y up; pixel pitch preserved | |
| masks | truthy = **excluded** from evaluation | invert if your mask marks the points to keep |
| NaN | non-measured | return NaN (Python `float("nan")`, JS `NaN`) for an undefined parameter; `expectNaN` cases require it |
| profiles | `gauge.profile`: one row, R parameters (ISO 21920-2) evaluated on the gauge as given (no λc filtering first); Rz is the mean over 5 sampling lengths | load as a profile, not a 1-row surface |

Two rules about handles:

- `apply` must **not modify its input**. Return a new object, or a copy. softgauge
  caches the result of each operation chain and evaluates several cases on it.
- `parameter` and `measure` must not modify the handle either. For example, don't
  level in place inside a parameter call.

## 7. Debugging a failure

- **Read the case.** `reference.formula`, `basis` (where the tolerance comes from) and
  `note` usually say what convention is being tested. `discrete.value` is the exact
  value for the sampled gauge, and that is what you are judged against.
  - Python: `softgauge.select_cases(only="hybrid.pyramids")`
  - JS: `selectCases(spec, { only: 'hybrid.pyramids' })`
- **Run one case.** Python: `softgauge.Runner(binding).case(case)`. JS:
  `run(spec, binding, { filter: (c) => c.id === '…' })`.
- **Check the gauge reached your software intact.** `results["gauges"][id]["ok"]` covers
  softgauge's builder. To check your loader, compute your software's mean, RMS, min and
  max of the loaded surface and compare them with `gauge["fingerprint"]`.
- **Look at the array.** For an operator case, compare `to_array(h)` with the field the
  measure expects:
  - Python: `softgauge.build_heights({**gauge, "parts": m["expect"]})`
  - JS: `buildHeights({ ...gauge, parts: m.expect })`

A failure that survives this is a finding: a defect in the software, or an
undocumented convention. Report it in your software's tracker. softgauge itself only
changes if the reference or the spec text is wrong.

## 8. No binding at all: files

Software that cannot be scripted (closed instrument or analysis packages) can still run
the `core` tier:

```bash
pip install "softgauge[sdf]"
softgauge export gauges/ --tier core          # SDF (ISO 25178-71) + <gauge>.json sidecars
```

1. Load each `.sdf` file. Check that the software's point count, mean, min and max match
   the sidecar's `fingerprint` (heights in the file are in metres, the sidecar is in µm).
2. Compute the parameters listed under `cases` in the sidecar.
3. Compare each one with `judgedReference` within `tol` (`|value − ref| ≤ abs + rel·|ref|`).

To produce a report, enter the values into a result file (SPEC.md, Results), one entry
per case id, and run `softgauge report`.
