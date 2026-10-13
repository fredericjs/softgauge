// softgauge — analytic soft gauges for surface texture software.
//
// The package builds each gauge's height map, holds the frozen analytic references, runs
// any implementation through a thin binding and judges the results. It never imports the
// software under test. See SPEC.md for the contract.

import SPEC from './spec-data.js';

export { buildHeights, fingerprint, verifyGauge, tri } from './gauges.js';
export { judge, judgedReference } from './judge.js';
export { arrayMeasure, ARRAY_MEASURES, Unsupported } from './measures.js';
export { run } from './run.js';
export { report, caseLabel, opLabel, measureLabel } from './report.js';

/** The bundled spec (a plain object; treat it as read-only). */
export function loadSpec() {
  return SPEC;
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
