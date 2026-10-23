// report.js — merge the spec with any number of result files into
//   json  one document for a web page (cases × implementations, verdicts, tallies)
//   csv   one row per case × implementation
//   tex   a LaTeX longtable (compile with LuaLaTeX/XeLaTeX, UTF-8)

import { judge } from './judge.js';

// ── human-readable label of what a case measures ──────────────────────────────
const OPN = {
  level: 'level', levelTLS: 'level (total least squares)', detrend: 'detrend', levelPoints: 'level to points', removeSphere: 'remove sphere',
  removeCylinder: 'remove cylinder', filter: 'filter', center: 'center', zero: 'zero', invert: 'invert', threshold: 'threshold',
  removeOutliers: 'remove outliers', despike: 'despike', fill: 'fill', crop: 'crop', zoom: 'zoom', rotate: 'rotate',
  maskRect: 'mask rectangle', maskCircle: 'mask circle', maskThreshold: 'mask by height', maskInvert: 'invert mask', maskToNaN: 'mask → non-measured',
};

export function opLabel(o) {
  const b = OPN[o.op] || o.op;
  if (o.op === 'detrend') return `${b} (degree ${o.degree})`;
  if (o.op === 'filter') return `${o.mode && o.mode !== 'reflect' ? o.mode + ' ' : ''}${o.type} λc = ${o.cutoff}${o.cutoff2 ? '/' + o.cutoff2 : ''} µm`;
  if (o.op === 'removeOutliers' || o.op === 'despike' || o.op === 'fill') return `${b} (${o.method})`;
  if (o.op === 'rotate') return `${b} ${o.angle}°`;
  if (o.op === 'zoom') return `${b} ×${o.factor}`;
  return b;
}

const STATS = { mean: 'mean height', min: 'min height', max: 'max height', nanCount: 'non-measured points', maskedCount: 'masked points',
  validMean: 'mean of unmasked', rows: 'rows', cols: 'columns', stepX: 'Δx', stepY: 'Δy' };

export function measureLabel(m) {
  switch (m.kind) {
    case 'param': return m.args && m.args.length ? `${m.name}(${m.args.join(', ')})` : m.name;
    case 'stat': return STATS[m.stat] || m.stat;
    case 'fieldError': return 'max |Δz|';
    case 'sineAmplitude': return 'amplitude transmission';
    case 'nanMismatch': return 'misclassified points';
    case 'planeGradient': return m.component === 'mag' ? '|∇z|' : 'gradient direction';
    case 'formRadius': return 'fitted radius';
    case 'radialPsd': return 'Σ PSD·Δf² (radial PSD)';
    case 'fourierPeak': return 'peak frequency (2D DFT)';
    case 'acfMap': return `ACF(τ = ${m.tau[0]}, ${m.tau[1]} µm)`;
    case 'abbottCurveError': return 'max |Δh| of the displayed curve';
    case 'regionVolume': return m.what === 'nRegions' ? 'regions' : 'region volume';
    case 'fitCircle': return m.what === 'r' ? 'fitted radius' : 'centre error';
    case 'fitLine': return m.what === 'angle' ? 'fitted direction' : 'line offset';
    case 'sampleHeight': return 'z(x, y)';
    case 'profileLine': return 'max |Δz| along the line';
    case 'cavities': return m.what === 'count' ? 'cavities found' : `cavity ${m.what}`;
    default: return m.kind;
  }
}

export const caseLabel = (c) => [...c.ops.map(opLabel), measureLabel(c.measure)].join(' → ');

/**
 * @param {object} spec
 * @param {Array<{key: string} & object>} runs   result files (spec/README.md, Results), each with a short `key`;
 *                                               the order given is the column order
 * @returns {{json: object, csv: string, tex: string, implementations: object[]}}
 */
export function report(spec, runs) {
  const groups = [];
  const cases = spec.cases.map(c => {
    if (!groups.includes(c.group)) groups.push(c.group);
    const results = {};
    for (const r of runs) {
      const res = r.cases[c.id], j = judge(c, res);
      results[r.key] = {
        verdict: j.verdict, value: res ? (res.nan ? null : res.value ?? null) : null, nan: !!(res && res.nan),
        error: j.error, limit: j.limit, ms: res ? res.ms : null, message: res && res.error ? res.error : undefined,
      };
    }
    const j0 = judge(c, null);
    return {
      id: c.id, group: c.group, gauge: c.gauge, tier: c.tier, domain: c.domain, label: caseLabel(c), unit: c.unit,
      standards: c.standards, method: c.method, cite: c.cite, reference: c.reference, discrete: c.discrete,
      judgedRef: j0.ref, samplingError: j0.samplingError, tol: c.tol, basis: c.basis, note: c.note, expectNaN: !!c.expectNaN,
      series: c.series, results,
    };
  });
  const implementations = runs.map(r => {
    const tally = { pass: 0, fail: 0, unsupported: 0, error: 0 };
    for (const c of cases) tally[c.results[r.key].verdict]++;
    return { key: r.key, ...r.implementation, date: r.date, specVersion: r.specVersion, tally, gaugesOk: Object.values(r.gauges || {}).every(g => g.ok) };
  });
  const gauges = {};
  for (const [id, g] of Object.entries(spec.gauges)) {
    const { fingerprint, ...rest } = g;
    gauges[id] = { ...rest, cases: cases.filter(c => c.gauge === id).length };
  }
  const json = {
    name: spec.name, title: spec.title, specVersion: spec.version, conventions: spec.conventions, references: spec.references,
    generated: new Date().toISOString(), implementations, groups, gauges, cases,
  };

  // ── CSV ─────────────────────────────────────────────────────────────────────
  const q = (v) => (v == null ? '' : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
  const csv = [['case', 'group', 'tier', 'domain', 'gauge', 'quantity', 'unit', 'standards', 'reference', 'reference_formula', 'reference_text',
    'sampled_reference', 'tol_abs', 'tol_rel', 'implementation', 'value', 'error', 'limit', 'verdict'].join(',')];
  for (const c of cases) for (const r of runs) {
    const x = c.results[r.key];
    csv.push([c.id, c.group, c.tier, c.domain, c.gauge, c.label, c.unit, c.standards.join('; '), c.reference.value, c.reference.formula, c.reference.text,
      c.discrete ? c.discrete.value : '', c.tol.abs ?? '', c.tol.rel ?? '', r.key, x.nan ? 'NaN' : x.value, x.error, x.limit, x.verdict].map(q).join(','));
  }

  // ── LaTeX ───────────────────────────────────────────────────────────────────
  const TEX_ESC = { '\\': '\\textbackslash{}', '~': '\\textasciitilde{}', '^': '\\textasciicircum{}' };
  const tex = (s) => String(s ?? '').replace(/[\\~^&%$#_{}]/g, (ch) => TEX_ESC[ch] || '\\' + ch);
  const num = (v) => (v == null ? '--' : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(4).replace(/e([+-]\d+)/, (_, e) => `\\times10^{${+e}}`) : (+v.toPrecision(7)).toString());
  const vcell = (x) => x.verdict === 'unsupported' ? '\\textit{n/a}' : x.verdict === 'error' ? '\\textbf{error}' : `$${x.nan ? '\\mathrm{NaN}' : num(x.value)}$${x.verdict === 'fail' ? '\\,\\textbf{✗}' : '\\,✓'}`;
  let T = `% Generated by softgauge report — do not edit by hand.\n% Spec ${spec.version}; ${implementations.map(i => `${i.name} ${i.version ?? ''}`.trim()).join('; ')}.\n`;
  T += `\\begin{longtable}{@{}p{2.6cm}p{3.6cm}r${runs.map(() => 'r').join('')}@{}}\n\\toprule\nGauge & Quantity & Reference${implementations.map(i => ` & ${tex(i.name)}`).join('')}\\\\\n\\midrule\\endhead\n`;
  for (const gname of groups) {
    T += `\\multicolumn{${3 + runs.length}}{@{}l}{\\textbf{${tex(gname)}}}\\\\\n`;
    for (const c of cases.filter(k => k.group === gname)) {
      T += `${tex(gauges[c.gauge].title)} & ${tex(c.label)}${c.unit ? ` [${tex(c.unit)}]` : ''} & $${c.expectNaN ? '\\text{undef.}' : num(c.judgedRef)}$${runs.map(r => ` & ${vcell(c.results[r.key])}`).join('')}\\\\\n`;
    }
  }
  T += '\\bottomrule\n\\end{longtable}\n';

  return { json, csv: csv.join('\n') + '\n', tex: T, implementations };
}
