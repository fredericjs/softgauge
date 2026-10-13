// run.js — run the spec against any implementation through a binding
// (mirrored in python/src/softgauge/run.py).
//
// softgauge knows nothing about the software under test. A binding translates:
//
//   binding = {
//     load(z, gauge)              → handle   z: a fresh Float64Array/Float32Array copy,
//                                            row-major ny × nx; gauge.profile = 1-row profile
//     apply(handle, op, gauge)    → handle   one operation of case.ops (SPEC.md, Operations);
//                                            must not modify `handle` (chains are cached)
//     parameter(handle, name, args, gauge) → number      measure.kind 'param'
//     measure?(handle, m, gauge)  → number | undefined   any measure; undefined = "use the
//                                            array measure" (or unsupported)
//     toArray?(handle)            → {z, rows, cols, dx, dy, mask?}   for the array measures
//   }
//
// Throw `Unsupported` (or leave a method out) for anything the software cannot do: the
// case is reported 'unsupported', never 'fail'.

import { buildHeights, verifyGauge } from './gauges.js';
import { arrayMeasure, ARRAY_MEASURES, Unsupported } from './measures.js';
import { judge } from './judge.js';

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

function evaluate(binding, h, m, g) {
  if (m.kind === 'param') {
    if (typeof binding.parameter !== 'function') throw new Unsupported('parameters');
    return binding.parameter(h, m.name, m.args || [], g);
  }
  if (typeof binding.measure === 'function') {
    const v = binding.measure(h, m, g);
    if (v !== undefined) return v;
  }
  if (ARRAY_MEASURES.has(m.kind) && typeof binding.toArray === 'function') return arrayMeasure(m, binding.toArray(h), g);
  throw new Unsupported(`measure ${m.kind}`);
}

/** JSON has no NaN/Infinity: NaN is flagged, ±Infinity (e.g. an unfilled hole) is clamped to ±1e308. */
function encode(v) {
  if (v == null) return { status: 'ok', value: null, nan: true };
  const num = Number(v);
  if (Number.isNaN(num)) return { status: 'ok', value: null, nan: true };
  return { status: 'ok', value: Number.isFinite(num) ? num : Math.sign(num) * 1e308 };
}

/**
 * Run the cases of `spec` (optionally only those accepted by `filter`) through `binding`.
 * @param {object} spec
 * @param {object} binding
 * @param {{implementation?: object, filter?: (c) => boolean, onCase?: (c, result, verdict) => void}} [opts]
 * @returns {{implementation, specVersion, gauges: {id: {ok, maxRelDiff}}, cases: {id: {status, value, nan?, error?, ms}}, date}}
 *          the result file of SPEC.md, Results
 */
export function run(spec, binding, { implementation = { name: 'unnamed' }, filter = null, onCase = null } = {}) {
  const heights = new Map(), gauges = {}, chains = new Map(), cases = {};
  const gaugeHeights = (id) => {
    if (!heights.has(id)) {
      const g = spec.gauges[id], z = buildHeights(g);
      gauges[id] = verifyGauge(g, z);
      heights.set(id, z);
    }
    return heights.get(id);
  };
  for (const c of spec.cases) {
    if (filter && !filter(c)) continue;
    const t0 = now();
    let res;
    try {
      const g = spec.gauges[c.gauge], z = gaugeHeights(c.gauge);
      if (!gauges[c.gauge].ok) throw new Error('gauge fingerprint mismatch');
      const key = c.gauge + '|' + JSON.stringify(c.ops);
      let h = chains.get(key);
      if (h === undefined) {
        h = binding.load(z.slice(), g);
        for (const o of c.ops) {
          if (typeof binding.apply !== 'function') throw new Unsupported(`operation ${o.op}`);
          h = binding.apply(h, o, g);
        }
        chains.set(key, h);
      }
      res = encode(evaluate(binding, h, c.measure, g));
    } catch (e) {
      res = e instanceof Unsupported || (e && e.name === 'Unsupported')
        ? { status: 'unsupported', error: e.message }
        : { status: 'error', error: String((e && e.message) || e) };
    }
    res.ms = Math.round((now() - t0) * 10) / 10;
    cases[c.id] = res;
    if (onCase) onCase(c, res, judge(c, res));
  }
  return { implementation, specVersion: spec.version, gauges, cases, date: new Date().toISOString() };
}
