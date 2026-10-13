// measures.js — the measures softgauge evaluates itself, from the height array a binding
// returns after the operations (mirrored in python/src/softgauge/measures.py).
//
// A binding only has to hand back its result as a plain array,
//   { z, rows, cols, dx, dy, mask? }   z row-major (NaN = non-measured), mask truthy = masked,
// and every implementation is then measured by the same code: fieldError, sineAmplitude,
// nanMismatch, planeGradient and stat. Parameters ('param') and the extension measures
// stay with the binding (SPEC.md, Measures).

import { buildHeights } from './gauges.js';

export class Unsupported extends Error {
  constructor(message) { super(message); this.name = 'Unsupported'; }
}

/** Measure kinds evaluated by arrayMeasure. */
export const ARRAY_MEASURES = new Set(['stat', 'fieldError', 'sineAmplitude', 'nanMismatch', 'planeGradient']);

/** Solve a small dense system (Gaussian elimination, partial pivoting). */
function solve(A, b) {
  const n = b.length, M = A.map((row, i) => [...row, b[i]]);
  for (let k = 0; k < n; k++) {
    let p = k;
    for (let i = k + 1; i < n; i++) if (Math.abs(M[i][k]) > Math.abs(M[p][k])) p = i;
    [M[k], M[p]] = [M[p], M[k]];
    for (let i = 0; i < n; i++) {
      if (i === k) continue;
      const f = M[i][k] / M[k][k];
      for (let j = k; j <= n; j++) M[i][j] -= f * M[k][j];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

/** Least-squares fit of Σ coef·basis_j(x, y) over the finite pixels selected by `use(r, c)`. */
function lsq(a, basis, use) {
  const { rows, cols, dx, dy, z } = a, m = basis(0, 0).length;
  const A = Array.from({ length: m }, () => new Array(m).fill(0)), b = new Array(m).fill(0);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const v = z[r * cols + c];
    if (v !== v || !use(r, c)) continue;
    const f = basis(c * dx, r * dy);
    for (let i = 0; i < m; i++) { b[i] += f[i] * v; for (let j = 0; j < m; j++) A[i][j] += f[i] * f[j]; }
  }
  return solve(A, b);
}

function stat(a, name) {
  const { z, mask } = a;
  switch (name) {
    case 'rows': return a.rows;
    case 'cols': return a.cols;
    case 'stepX': return a.dx;
    case 'stepY': return a.dy;
    case 'nanCount': { let n = 0; for (let i = 0; i < z.length; i++) if (z[i] !== z[i]) n++; return n; }
    case 'maskedCount': { let n = 0; if (mask) for (let i = 0; i < mask.length; i++) n += mask[i] ? 1 : 0; return n; }
    case 'mean': case 'min': case 'max': case 'validMean': {
      let n = 0, t = 0, mn = Infinity, mx = -Infinity;
      for (let i = 0; i < z.length; i++) {
        const v = z[i];
        if (v !== v || (name === 'validMean' && mask && mask[i])) continue;
        n++; t += v; if (v < mn) mn = v; if (v > mx) mx = v;
      }
      if (!n) return NaN;
      return name === 'min' ? mn : name === 'max' ? mx : t / n;
    }
    default: throw new Unsupported(`stat ${name}`);
  }
}

/**
 * Evaluate measure `m` of a case on a binding's output array.
 * @param {object} m      case.measure (kind ∈ ARRAY_MEASURES)
 * @param {{z: ArrayLike<number>, rows: number, cols: number, dx: number, dy: number, mask?: ArrayLike<any>}} a
 * @param {object} g      the case's gauge (for skipSpikes)
 * @returns {number}      ±Infinity where the spec asks for it (requireFinite on a NaN)
 */
export function arrayMeasure(m, a, g = {}) {
  const { rows, cols, dx, dy, z } = a;
  if (z.length !== rows * cols) throw new Error(`array has ${z.length} values, expected ${rows} × ${cols}`);
  switch (m.kind) {
    case 'stat': return stat(a, m.stat);
    case 'fieldError': {
      const ez = buildHeights({ nx: cols, ny: rows, dx, dy, parts: m.expect });
      const skip = new Set((m.skipSpikes ? g.spikes || [] : []).map(([r, c]) => r * cols + c));
      const mg = m.margin || 0, mr = m.marginRows || 0;
      let worst = 0;
      for (let r = mr; r < rows - mr; r++) for (let c = mg; c < cols - mg; c++) {
        const i = r * cols + c;
        if (skip.has(i)) continue;
        const v = z[i];
        if (v !== v) { if (m.requireFinite) return Infinity; continue; }
        worst = Math.max(worst, Math.abs(v - ez[i]));
      }
      return worst;
    }
    case 'sineAmplitude': {
      const k = (2 * Math.PI) / m.lambda, mg = m.margin || 0;
      const [, s, c] = m.axis === 'y'
        ? lsq(a, (x, y) => [1, Math.sin(k * y), Math.cos(k * y)], (r) => r >= mg && r < rows - mg)
        : lsq(a, (x) => [1, Math.sin(k * x), Math.cos(k * x)], (r, col) => col >= mg && col < cols - mg);
      return Math.hypot(s, c);
    }
    case 'nanMismatch': {
      const want = new Set(m.points.map(([r, c]) => r * cols + c));
      let n = 0;
      for (let i = 0; i < z.length; i++) if ((z[i] !== z[i]) !== want.has(i)) n++;
      return n;
    }
    case 'planeGradient': {
      const mg = m.margin || 0;
      const [, gx, gy] = lsq(a, (x, y) => [1, x, y], (r, c) => r >= mg && r < rows - mg && c >= mg && c < cols - mg);
      const gyUp = -gy;   // y measured upwards (rows run downwards)
      return m.component === 'mag' ? Math.hypot(gx, gyUp) : (Math.atan2(gyUp, gx) * 180) / Math.PI;
    }
    default: throw new Unsupported(`measure ${m.kind}`);
  }
}
