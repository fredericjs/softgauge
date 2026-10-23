// run.js — run the spec against any implementation through a binding
// (mirrored in python/src/softgauge/run.py).
//
// softgauge knows nothing about the software under test. A binding translates:
//
//   binding = {
//     load(z, gauge)              → handle   z: a fresh Float64Array/Float32Array copy,
//                                            row-major ny × nx; gauge.profile = 1-row profile
//     apply(handle, op, gauge)    → handle   one operation of case.ops (spec/README.md, Operations);
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
import SPEC from './spec-data.js';

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

/** JSON has no NaN/Infinity: NaN is flagged, ±Infinity (e.g. an unfilled hole) is clamped to ±1e308. */
function encode(v) {
  if (v == null) return { status: 'ok', value: null, nan: true };
  const num = Number(v);
  if (Number.isNaN(num)) return { status: 'ok', value: null, nan: true };
  return { status: 'ok', value: Number.isFinite(num) ? num : Math.sign(num) * 1e308 };
}

/** Runs cases one at a time (e.g. one test per case), caching gauge heights and operation chains. */
export class Runner {
  /** @param {object} binding  @param {object} [spec]  defaults to the bundled spec */
  constructor(binding, spec = SPEC) {
    this.binding = binding;
    this.spec = spec;
    /** gauge id → {ok, maxRelDiff}: the fingerprint check of every gauge built so far */
    this.gauges = {};
    this._heights = new Map();
    this._chains = new Map();
  }

  /** Heights of a gauge (built once, checked against its fingerprint). */
  heights(gaugeId) {
    if (!this._heights.has(gaugeId)) {
      const g = this.spec.gauges[gaugeId], z = buildHeights(g);
      this.gauges[gaugeId] = verifyGauge(g, z);
      this._heights.set(gaugeId, z);
    }
    return this._heights.get(gaugeId);
  }

  _evaluate(h, m, g) {
    const b = this.binding;
    if (m.kind === 'param') {
      if (typeof b.parameter !== 'function') throw new Unsupported('parameters');
      return b.parameter(h, m.name, m.args || [], g);
    }
    if (typeof b.measure === 'function') {
      const v = b.measure(h, m, g);
      if (v !== undefined) return v;
    }
    if (ARRAY_MEASURES.has(m.kind) && typeof b.toArray === 'function') return arrayMeasure(m, b.toArray(h), g);
    throw new Unsupported(`measure ${m.kind}`);
  }

  /**
   * Result of one case.
   * @returns {{status: 'ok'|'unsupported'|'error', value?: number|null, nan?: boolean, error?: string, ms: number}}
   */
  case(c) {
    const t0 = now();
    let res;
    try {
      const g = this.spec.gauges[c.gauge], z = this.heights(c.gauge);
      if (!this.gauges[c.gauge].ok) throw new Error('gauge fingerprint mismatch');
      const key = c.gauge + '|' + JSON.stringify(c.ops);
      let h = this._chains.get(key);
      if (h === undefined) {
        h = this.binding.load(z.slice(), g);
        for (const o of c.ops) {
          if (typeof this.binding.apply !== 'function') throw new Unsupported(`operation ${o.op}`);
          h = this.binding.apply(h, o, g);
        }
        this._chains.set(key, h);
      }
      res = encode(this._evaluate(h, c.measure, g));
    } catch (e) {
      res = e instanceof Unsupported || (e && e.name === 'Unsupported')
        ? { status: 'unsupported', error: e.message }
        : { status: 'error', error: String((e && e.message) || e) };
    }
    res.ms = Math.round((now() - t0) * 10) / 10;
    return res;
  }
}

/**
 * Cases of `spec` filtered by tier ('core' | 'operators' | 'extension', or a list),
 * domain ('areal' | 'profile'), group, and a substring of the id.
 */
export function selectCases(spec = SPEC, { tier = null, domain = null, group = null, only = null } = {}) {
  const tiers = tier == null ? null : new Set([].concat(tier));
  return spec.cases.filter(c => (!tiers || tiers.has(c.tier)) && (!domain || c.domain === domain)
    && (!group || c.group === group) && (!only || c.id.includes(only)));
}

/**
 * Run `cases` (default: the spec's cases, filtered by tier / domain / group / only) through
 * `binding` and return the result file of spec/README.md, Results.
 * @param {object} binding
 * @param {{spec?: object, implementation?: object, cases?: object[], tier?: string|string[], domain?: string,
 *          group?: string, only?: string, onCase?: (c: object, result: object, judgement: object) => void}} [options]
 */
export function run(binding, { spec = SPEC, implementation = { name: 'unnamed' }, cases = null, onCase = null, ...filters } = {}) {
  const runner = new Runner(binding, spec), out = {};
  for (const c of cases ?? selectCases(spec, filters)) {
    const res = runner.case(c);
    out[c.id] = res;
    if (onCase) onCase(c, res, judge(c, res));
  }
  return { implementation, specVersion: spec.version, gauges: runner.gauges, cases: out, date: new Date().toISOString() };
}
