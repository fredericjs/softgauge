import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { loadSpec, selectCases, buildHeights, verifyGauge, judge, arrayMeasure, Runner, run, report, Unsupported } from '../src/index.js';
import { naiveBinding } from '../examples/naive-binding.js';

const here = dirname(fileURLToPath(import.meta.url));
const spec = loadSpec();

test('the bundled spec is the repository spec', { skip: !existsSync(join(here, '../../spec/spec.json')) }, () => {
  assert.deepEqual(spec, JSON.parse(readFileSync(join(here, '../../spec/spec.json'), 'utf8')));
});

test('every case has a gauge, a tier and a domain; ids are unique', () => {
  const ids = new Set();
  for (const c of spec.cases) {
    assert.ok(!ids.has(c.id), c.id); ids.add(c.id);
    assert.ok(spec.gauges[c.gauge], c.id);
    assert.ok(['core', 'operators', 'extension'].includes(c.tier), c.id);
    assert.equal(c.domain, spec.gauges[c.gauge].profile ? 'profile' : 'areal');
  }
});

test('every gauge reproduces its fingerprint', () => {
  for (const [id, g] of Object.entries(spec.gauges)) assert.ok(verifyGauge(g).ok, id);
});

test('a corrupted gauge fails its fingerprint', () => {
  const g = spec.gauges.plane, z = buildHeights(g);
  z[g.nx + 1] += 1e-6;
  assert.equal(verifyGauge(g, z).ok, true, 'probes do not see an interior pixel…');
  z[0] += 1e-6;
  assert.equal(verifyGauge(g, z).ok, false, '…but the first one is a probe');
});

test('judge', () => {
  const c = { reference: { value: 1 }, tol: { abs: 0.1, rel: 0.1 } };
  assert.equal(judge(c, { status: 'ok', value: 1.19 }).verdict, 'pass');
  assert.equal(judge(c, { status: 'ok', value: 1.21 }).verdict, 'fail');
  assert.equal(judge(c, { status: 'ok', value: null, nan: true }).verdict, 'fail');
  assert.equal(judge(c, { status: 'unsupported' }).verdict, 'unsupported');
  assert.equal(judge(c, null).verdict, 'unsupported');
  assert.equal(judge(c, { status: 'error' }).verdict, 'error');
  assert.equal(judge({ ...c, discrete: { value: 2 } }, { status: 'ok', value: 2.25 }).verdict, 'pass');
  assert.equal(judge({ ...c, expectNaN: true }, { status: 'ok', value: null, nan: true }).verdict, 'pass');
});

test('array measures on the unmodified gauge', () => {
  const g = spec.gauges.plane, z = buildHeights(g), a = { z, rows: g.ny, cols: g.nx, dx: g.dx, dy: g.dy };
  assert.equal(arrayMeasure({ kind: 'fieldError', expect: g.parts }, a, g), 0);
  assert.equal(arrayMeasure({ kind: 'stat', stat: 'cols' }, a), g.nx);
  const p = g.parts[0];
  assert.ok(Math.abs(arrayMeasure({ kind: 'planeGradient', component: 'mag' }, a) - Math.hypot(p.a, p.b)) < 1e-12);
  assert.throws(() => arrayMeasure({ kind: 'cavities' }, a), Unsupported);
});

test('selectCases', () => {
  const core = selectCases(spec, { tier: 'core', domain: 'profile' });
  assert.ok(core.length > 0 && core.every(c => c.tier === 'core' && c.domain === 'profile'));
  assert.equal(selectCases(spec, { tier: ['core', 'operators', 'extension'] }).length, spec.cases.length);
});

test('the naive binding passes every case it supports', () => {
  const res = run(naiveBinding, { implementation: { name: 'naive' } });
  const tally = { pass: 0, fail: 0, error: 0, unsupported: 0 }, bad = [];
  for (const c of spec.cases) {
    const v = judge(c, res.cases[c.id]).verdict;
    tally[v]++;
    if (v === 'fail' || v === 'error') bad.push(`${c.id}: ${JSON.stringify(res.cases[c.id])}`);
  }
  assert.deepEqual(bad, []);
  assert.ok(tally.pass >= 30, JSON.stringify(tally));
  assert.ok(Object.values(res.gauges).every(g => g.ok));
  const r = report(spec, [{ key: 'naive', ...res }]);
  assert.equal(r.json.cases.length, spec.cases.length);
  assert.equal(r.implementations[0].tally.pass, tally.pass);
  assert.ok(r.csv.split('\n').length > spec.cases.length);
});

test('run filters, reports every case through onCase and matches the Runner', () => {
  const seen = [];
  const res = run(naiveBinding, { tier: 'core', domain: 'profile', onCase: (c, r, j) => seen.push([c.id, j.verdict]) });
  const ids = selectCases(spec, { tier: 'core', domain: 'profile' }).map(c => c.id);
  assert.deepEqual(Object.keys(res.cases), ids);
  assert.deepEqual(seen.map(s => s[0]), ids);
  const runner = new Runner(naiveBinding);
  const c = spec.cases.find(k => k.id === 'height.sine.Sq');
  assert.equal(judge(c, runner.case(c)).verdict, 'pass');
  assert.ok(runner.gauges.sine.ok);
});

test('report escapes LaTeX specials once', () => {
  const g = { ...spec.gauges.plane, title: String.raw`a\b {c} 50 % x_y ~^` };
  const c = { ...spec.cases.find(k => k.id === 'height.plane.Sq'), group: 'G' };
  const r = report({ ...spec, gauges: { plane: g }, cases: [c] }, [{ key: 'k', implementation: { name: 'n' }, cases: {}, gauges: {} }]);
  assert.ok(r.tex.includes(String.raw`a\textbackslash{}b \{c\} 50 \% x\_y \textasciitilde{}\textasciicircum{}`), r.tex);
});
