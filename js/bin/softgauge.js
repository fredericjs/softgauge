#!/usr/bin/env node
// softgauge — command line.
//
//   softgauge info                                     gauges, cases and tiers of the bundled spec
//   softgauge report <results-dir|file…> [--out <dir>] [--order a,b] [--hide a,b]
//       merge result files (one per implementation, key = file name) into
//       <out>/results.json, results.csv, results.tex

import { readFileSync, writeFileSync, readdirSync, statSync, mkdirSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import { loadSpec, report } from '../src/index.js';

const [cmd, ...rest] = process.argv.slice(2);
const opt = (name) => { const i = rest.indexOf(name); return i >= 0 ? rest.splice(i, 2)[1] : null; };
const spec = loadSpec();

if (cmd === 'info') {
  const count = (key) => spec.cases.reduce((t, c) => ({ ...t, [c[key]]: (t[c[key]] || 0) + 1 }), {});
  console.log(`spec ${spec.version}: ${Object.keys(spec.gauges).length} gauges, ${spec.cases.length} cases`);
  console.log('tiers:  ', count('tier'));
  console.log('domains:', count('domain'));
} else if (cmd === 'report') {
  const out = resolve(opt('--out') || '.');
  const order = (opt('--order') || '').split(',').filter(Boolean);
  const hide = new Set((opt('--hide') || '').split(',').filter(Boolean));
  const files = rest.flatMap(p => (statSync(p).isDirectory() ? readdirSync(p).filter(f => f.endsWith('.json')).map(f => join(p, f)) : [p]));
  const rank = (k) => (order.includes(k) ? order.indexOf(k) : order.length);
  const runs = files.map(f => ({ key: basename(f, '.json'), ...JSON.parse(readFileSync(f, 'utf8')) }))
    .filter(r => !hide.has(r.key))
    .sort((a, b) => rank(a.key) - rank(b.key) || a.key.localeCompare(b.key));
  if (!runs.length) { console.error('no result files'); process.exit(1); }
  const r = report(spec, runs);
  mkdirSync(out, { recursive: true });
  writeFileSync(join(out, 'results.json'), JSON.stringify(r.json) + '\n');
  writeFileSync(join(out, 'results.csv'), r.csv);
  writeFileSync(join(out, 'results.tex'), r.tex);
  for (const i of r.implementations) {
    console.log(`${i.name} ${i.version ?? ''}: ${i.tally.pass} pass, ${i.tally.fail} fail, ${i.tally.error} error, ${i.tally.unsupported} unsupported — gauges ${i.gaugesOk ? 'verified' : 'MISMATCH'}`);
  }
  console.log(`wrote results.json, results.csv, results.tex to ${out}`);
} else {
  console.log('usage: softgauge info | softgauge report <results-dir|file…> [--out <dir>] [--order a,b] [--hide a,b]');
  process.exit(cmd ? 1 : 0);
}
