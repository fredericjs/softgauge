# softgauge (Python)

Analytic soft gauges for surface texture software (ISO 25178 areal, ISO 21920 profile).
softgauge builds synthetic surfaces whose parameters and operation results are known in
closed form, hands the height arrays to a thin binding you write for your software,
and judges what comes back against frozen analytic references. It never imports the
software under test.

```bash
pip install softgauge            # numpy only
pip install "softgauge[sdf]"     # + SDF export via sdfio
```

```python
import pytest
import softgauge
from softgauge.testing import cases, check

class MyBinding:
    def load(self, z, gauge):                   # z: (ny, nx) heights in µm
        return mylib.Surface(z, gauge["dx"], gauge["dy"])
    def apply(self, s, op, gauge):              # one operation of the case
        if op["op"] == "level":
            return s.level()
        raise softgauge.Unsupported(op["op"])   # reported 'unsupported', not 'fail'
    def parameter(self, s, name, args, gauge):
        return getattr(s, name)(*args)
    def to_array(self, s):                      # for the array measures
        return softgauge.ArrayResult(s.data, s.step_x, s.step_y)

runner = softgauge.Runner(MyBinding())

@pytest.mark.parametrize("case", cases(tier=["core", "operators"]))
def test_softgauge(case):
    check(runner, case)
```

`softgauge.run(binding)` runs everything at once and returns a result file;
`softgauge export <dir>` writes the gauges as SDF files with JSON sidecars, for software
that cannot be scripted. The contract (conventions, operations, measures, judging) is in
[SPEC.md](https://github.com/fredericjs/softgauge/blob/main/SPEC.md).

MIT licensed.
