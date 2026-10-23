// softgauge — analytic soft gauges for surface texture software.
//
// The package builds each gauge's height map, holds the frozen analytic references, runs
// any implementation through a thin binding and judges the results. It never imports the
// software under test. See spec/README.md for the contract.

import SPEC from './spec-data.js';

export { buildHeights, fingerprint, verifyGauge, tri } from './gauges.js';
export { judge, judgedReference } from './judge.js';
export { arrayMeasure, ARRAY_MEASURES, Unsupported } from './measures.js';
export { Runner, run, selectCases } from './run.js';
export { report, caseLabel, opLabel, measureLabel } from './report.js';

/** The bundled spec (a plain object; treat it as read-only). */
export function loadSpec() {
  return SPEC;
}
