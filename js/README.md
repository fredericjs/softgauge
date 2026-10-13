# softgauge (JavaScript)

Analytic soft gauges for surface texture software (ISO 25178 areal, ISO 21920 profile).
softgauge builds synthetic surfaces whose parameters and operation results are known in
closed form, hands the height arrays to a thin binding you write for your software,
and judges what comes back against frozen analytic references. It never imports the
software under test. Plain ESM, no dependencies, runs in Node and in browsers.

```bash
npm install softgauge
```

```js
import { loadSpec, run, judge, Unsupported } from 'softgauge';

const binding = {
  load: (z, g) => new MySurface(z, g.ny, g.nx, g.dx, g.dy),     // z: Float64Array, row-major, µm
  apply: (s, op) => { if (op.op === 'level') return s.level(); throw new Unsupported(op.op); },
  parameter: (s, name, args) => s[name](...args),
  toArray: (s) => ({ z: s.data, rows: s.rows, cols: s.cols, dx: s.dx, dy: s.dy }),
};
const spec = loadSpec();
const results = run(spec, binding, { implementation: { name: 'mylib', version: '1.0' } });
for (const c of spec.cases) console.log(c.id, judge(c, results.cases[c.id]).verdict);
```

`examples/naive-binding.js` is a complete small binding. `npx softgauge report <results-dir>`
merges result files into JSON, CSV and LaTeX. How to write a binding for your software:
[docs/writing-a-binding.md](https://github.com/fredericjs/softgauge/blob/main/docs/writing-a-binding.md). The contract (conventions, operations,
measures, judging) is in [SPEC.md](https://github.com/fredericjs/softgauge/blob/main/SPEC.md).

MIT licensed.
