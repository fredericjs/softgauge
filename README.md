# softgauge

**Analytic soft gauges for surface texture software.** softgauge validates areal
(ISO 25178) and profile (ISO 21920) surface texture software against synthetic surfaces
whose parameters and operation results are known in **closed form**: planes, sinusoids,
egg-crates, material-ratio ramps, spheres, cylinders, pockets, Gaussian bumps…
96 gauges and 547 cases. References are judged against the **sampled** surface where it
has a closed form, so tolerances are floating-point tight. Elsewhere every tolerance
carries a derived bound. None is tuned to make a result pass.

It is independent of the software it tests. softgauge builds the gauge, hands the
height array to a thin **binding** that you write for your software, and judges what
comes back. It never imports the implementation under test.

| | |
|---|---|
| `spec/` | The catalogue (gauges, cases, closed-form references) → **`spec.json`**, the language-neutral contract. |
| `js/` | npm package **`softgauge`**: gauge builder, runner, judge, report generator (ESM, no dependencies, Node and browsers). |
| `python/` | PyPI package **`softgauge`**: the same in numpy, a pytest helper, and SDF export through [sdfio](https://pypi.org/project/sdfio/). |
| [`SPEC.md`](SPEC.md) | The contract: conventions, gauge components, operations, measures, judging, result files, versioning. |

## Three ways to use it

**1. Through a binding (Python, e.g. in pytest).**

```python
import softgauge
from softgauge.testing import cases, check

class MyBinding:
    def load(self, z, gauge):                  # z: (ny, nx) heights in µm
        return mylib.Surface(z, gauge["dx"], gauge["dy"])
    def apply(self, s, op, gauge):             # one operation of the case (SPEC.md)
        if op["op"] == "level": return s.level()
        raise softgauge.Unsupported(op["op"])  # → 'unsupported', not 'fail'
    def parameter(self, s, name, args, gauge):
        return getattr(s, name)(*args)
    def to_array(self, s):                     # for the array measures
        return softgauge.ArrayResult(s.data, s.step_x, s.step_y)

runner = softgauge.Runner(MyBinding())

@pytest.mark.parametrize("case", cases(tier=["core", "operators"]))
def test_softgauge(case):
    check(runner, case, xfail={"…known defect ids…"})
```

**2. Through a binding (JavaScript).**

```js
import { loadSpec, run, judge, Unsupported } from 'softgauge';
const results = run(loadSpec(), {
  load: (z, g) => new MySurface(z, g.ny, g.nx, g.dx, g.dy),
  apply: (s, op) => { if (op.op === 'level') return s.level(); throw new Unsupported(op.op); },
  parameter: (s, name, args) => s[name](...args),
  toArray: (s) => ({ z: s.data, rows: s.rows, cols: s.cols, dx: s.dx, dy: s.dy }),
}, { implementation: { name: 'mylib', version: '1.0' } });
```

`run` returns a result file (SPEC.md, Results). `npx softgauge report results/` merges
several into JSON (for a web page), CSV and a LaTeX table.

**3. Through files**, for software you cannot script:

```bash
pip install "softgauge[sdf]"
softgauge export gauges/ --tier core
```

This writes every gauge as an ISO 25178-71 SDF file, plus a JSON sidecar with its
fingerprint (check it first, to prove the file was read correctly) and every case on
that gauge with its reference and tolerance.

## Development

```bash
node spec/build.js                # after editing spec/catalogue.js: regenerates spec.json + the package copies
node spec/build.js --check        # CI: spec.json is up to date, the copies are identical
cd js && node --test
cd python && pip install -e ".[test]" && pytest
```

A new gauge component goes in **both** builders (`js/src/gauges.js`,
`python/src/softgauge/gauges.py`). The fingerprints in `spec.json` catch any difference
between them.

## License

MIT
