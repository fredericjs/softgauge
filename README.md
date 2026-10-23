# softgauge

**Analytic soft gauges for surface texture software.** softgauge validates areal
(ISO 25178) and profile (ISO 21920) surface texture software against synthetic surfaces
whose parameters and operation results are known in **closed form**: planes, sinusoids,
egg-crates, material-ratio ramps, saddles, spheres, cylinders, pockets, Gaussian bumps…
97 gauges and 549 cases. References are judged against the **sampled** surface where it
has a closed form, so tolerances are floating-point tight. Elsewhere every tolerance
carries a derived bound. None is tuned to make a result pass.

It is independent of the software it tests. softgauge builds the gauge, hands the
height array to a thin **binding** that you write for your software, and judges what
comes back. It never imports the implementation under test.

## Layout

| | |
|---|---|
| [`spec/`](spec) | **The spec**: [`spec.json`](spec/spec.json) (gauges, cases, frozen references, tolerances), its [JSON Schema](spec/spec.schema.json), the [result-file schema](spec/results.schema.json) and [the contract](spec/README.md) that explains every field. Language-neutral and complete on its own. |
| [`generator/`](generator) | The Python source of `spec.json`: the catalogue of gauges and cases and the closed-form references. |
| [`js/`](js) | JavaScript adapter, npm package **`softgauge`**: gauge builder, runner, judge, report (ESM, no dependencies, Node and browsers). |
| [`python/`](python) | Python adapter, PyPI package **`softgauge`**: the same API in numpy, plus a pytest helper and SDF export through [sdfio](https://pypi.org/project/sdfio/). |
| [`docs/`](docs/writing-a-binding.md) | **How to connect your own software**, step by step, with the conventions that usually trip bindings up. |

Both adapters ship a byte-identical copy of `spec.json`, reproduce every gauge's
fingerprint and write identical reports.

## Three ways to use it

The full walkthrough is in [Writing a binding](docs/writing-a-binding.md).

**1. Through a binding (Python, e.g. in pytest).**

```python
import softgauge
from softgauge.testing import cases, check

class MyBinding:
    def load(self, z, gauge):                  # z: (ny, nx) heights in µm
        return mylib.Surface(z, gauge["dx"], gauge["dy"])
    def apply(self, s, op, gauge):             # one operation of the case (spec/README.md)
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
import { run, Unsupported } from 'softgauge';

const results = run({
  load: (z, g) => new MySurface(z, g.ny, g.nx, g.dx, g.dy),
  apply: (s, op) => { if (op.op === 'level') return s.level(); throw new Unsupported(op.op); },
  parameter: (s, name, args) => s[name](...args),
  toArray: (s) => ({ z: s.data, rows: s.rows, cols: s.cols, dx: s.dx, dy: s.dy }),
}, { implementation: { name: 'mylib', version: '1.0' } });
```

`run` returns a result file. `softgauge report results/` (either package) merges several
into JSON for a web page, CSV and a LaTeX table.

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
python generator/build.py         # after editing generator/: regenerates spec/spec.json + the package copies
python generator/build.py --check # CI: spec.json is up to date, the copies are identical
cd js && node --test
cd python && pip install -e ".[test]" && pytest
```

The generator needs numpy and builds the gauges with the Python adapter
(`python/src/softgauge/gauges.py`). A new gauge component goes in **both** builders
(`js/src/gauges.js` too); the fingerprints in `spec.json` catch any difference between
them. CI also checks that both adapters write the same report for the same results.

## Acknowledgements

The idea of judging software against mathematically defined surfaces instead of
reference software goes back to the work of Todhunter et al. on mathematical reference
standards for parameter calculation, filtration and form removal [1–5]. Blateyron and
Breton formalised it for material-ratio parameters as *mathematical softgauges* with
*mathematical true values* and published two families of Abbott curves [6], which are
included here in full. Several other gauges follow Todhunter et al.: the cubic
material-ratio curve and its reordered layouts [2], the parametric planes for Sdr [3],
the Fourier-series surfaces for Gaussian filtering [1, 4] and total least-squares form
removal [5]. Their references are defined on the continuous surface, and the sampling
error is estimated by extrapolating over resolutions. Here the gauges are laid out so
that the reference also holds for the sampled grid, so implementations are judged at
floating-point precision and the sampling error is reported exactly. The spec cites
these papers case by case (`cite`, `references` in `spec.json`).

1. L. Todhunter, R. Leach, S. Lawes, F. Blateyron, and P. Harris, “Development of mathematical reference standards for the validation of surface texture parameter calculation software,” *J. Phys.: Conf. Ser.*, vol. 1065, Art. no. 082004, 2018, doi: [10.1088/1742-6596/1065/8/082004](https://doi.org/10.1088/1742-6596/1065/8/082004).
2. L. Todhunter, R. Leach, S. Lawes, P. Harris, and F. Blateyron, “Mathematical approach to the validation of functional surface texture parameter software,” *Surf. Topogr.: Metrol. Prop.*, vol. 7, no. 1, Art. no. 015020, 2019, doi: [10.1088/2051-672X/ab07ca](https://doi.org/10.1088/2051-672X/ab07ca).
3. L. Todhunter, R. Leach, S. Lawes, P. Harris, and F. Blateyron, “Mathematical approach to the validation of field surface texture parameter software,” *Surf. Topogr.: Metrol. Prop.*, vol. 8, no. 1, Art. no. 015010, 2020, doi: [10.1088/2051-672X/ab7367](https://doi.org/10.1088/2051-672X/ab7367).
4. L. Todhunter, R. Leach, and F. Blateyron, “Mathematical approach to the validation of surface texture filtration software,” *Surf. Topogr.: Metrol. Prop.*, vol. 8, no. 4, Art. no. 045017, 2020, doi: [10.1088/2051-672X/abc0fb](https://doi.org/10.1088/2051-672X/abc0fb).
5. L. Todhunter, R. Leach, and F. Blateyron, “Mathematical approach to the validation of form removal surface texture software,” *Surf. Topogr.: Metrol. Prop.*, vol. 8, no. 4, Art. no. 045019, 2020, doi: [10.1088/2051-672X/abc737](https://doi.org/10.1088/2051-672X/abc737).
6. F. Blateyron and S. Breton, “Mathematical softgauges for material ratio parameters,” *Surf. Topogr.: Metrol. Prop.*, vol. 12, no. 3, Art. no. 035036, 2024, doi: [10.1088/2051-672X/ad6f2e](https://doi.org/10.1088/2051-672X/ad6f2e).

## Citing

See [`CITATION.cff`](CITATION.cff).

## License

MIT
