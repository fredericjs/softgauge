// judge.js — the single pass/fail rule of the suite (mirrored in python/src/softgauge/judge.py).

/** Reference a case is judged against: the sampled-gauge value when given, else the continuous one. */
export function judgedReference(c) {
  return c.discrete && c.discrete.value != null ? c.discrete.value : c.reference.value;
}

/**
 * @param {object} c       case from spec.json
 * @param {object} r       runner result {status, value, nan?}
 * @returns {{verdict:'pass'|'fail'|'unsupported'|'error', ref:number|null, error:number|null,
 *            limit:number|null, samplingError:number|null}}
 *   error = value − ref; limit = allowed |error|; samplingError = discrete − continuous.
 */
export function judge(c, r) {
  const ref = judgedReference(c);
  const out = { ref, error: null, limit: null, samplingError: null };
  if (c.discrete && c.discrete.value != null && c.reference.value != null) out.samplingError = c.discrete.value - c.reference.value;
  if (!r || r.status === 'unsupported') return { ...out, verdict: 'unsupported' };
  if (r.status !== 'ok') return { ...out, verdict: 'error' };
  const isNaN = r.nan || r.value == null || Number.isNaN(r.value);
  if (c.expectNaN) return { ...out, verdict: isNaN ? 'pass' : 'fail' };
  if (isNaN) return { ...out, verdict: 'fail' };
  const tol = c.tol || {};
  out.limit = (tol.abs || 0) + (tol.rel || 0) * Math.abs(ref);
  out.error = r.value - ref;
  return { ...out, verdict: Math.abs(out.error) <= out.limit ? 'pass' : 'fail' };
}
