// reference.js — closed-form reference values for the analytic soft gauges.
//
// Pure mathematics: this module never imports an implementation under test. Every
// function evaluates a closed-form expression (or, where a closed form needs a
// special function or a root, evaluates it to ~1e-13 by a standard convergent
// method: AGM for elliptic integrals, bisection for monotone roots, Gauss–Legendre
// quadrature on polynomial pieces, where it is exact).

const TAU = 2 * Math.PI;

// ── special functions ──────────────────────────────────────────────────────────

/** Complete elliptic integral of the second kind E(m), parameter m = k², via AGM. */
export function ellipticE(m) {
  if (m === 1) return 1;
  let a = 1, b = Math.sqrt(1 - m), c = Math.sqrt(m), sum = 0.5 * c * c, p = 0.5;
  while (Math.abs(c) > 1e-16) {
    const an = 0.5 * (a + b), bn = Math.sqrt(a * b), cn = 0.5 * (a - b);
    p *= 2; sum += p * cn * cn;
    a = an; b = bn; c = cn;
  }
  return (Math.PI / (2 * a)) * (1 - sum);
}

/** Root of a continuous f on [a, b] with a sign change, by bisection to machine precision. */
export function bisect(f, a, b) {
  let fa = f(a);
  if (fa === 0) return a;
  for (let i = 0; i < 200; i++) {
    const m = 0.5 * (a + b);
    if (m === a || m === b) return m;
    const fm = f(m);
    if (fm === 0) return m;
    if ((fm < 0) === (fa < 0)) { a = m; fa = fm; } else b = m;
  }
  return 0.5 * (a + b);
}

// 10-point Gauss–Legendre nodes/weights on [-1, 1] (exact for polynomials ≤ degree 19).
const GL_X = [-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472, -0.1488743389816312,
  0.1488743389816312, 0.4333953941292472, 0.6794095682990244, 0.8650633666889845, 0.9739065285171717];
const GL_W = [0.0666713443086881, 0.1494513491505806, 0.2190863625159820, 0.2692667193099963, 0.2955242247147529,
  0.2955242247147529, 0.2692667193099963, 0.2190863625159820, 0.1494513491505806, 0.0666713443086881];

/** ∫_a^b f, composite 10-point Gauss–Legendre over `n` panels (smooth f). */
export function integrate(f, a, b, n = 64) {
  if (b === a) return 0;
  const h = (b - a) / n;
  let s = 0;
  for (let p = 0; p < n; p++) {
    const c = a + (p + 0.5) * h;
    for (let k = 0; k < 10; k++) s += GL_W[k] * f(c + 0.5 * h * GL_X[k]);
  }
  return 0.5 * h * s;
}

/** Σ_{i=0}^{M-1} cos(a + i·d) in closed form (Lagrange's identity). */
export function sumCos(a, d, M) {
  const s = Math.sin(0.5 * d);
  if (Math.abs(s) < 1e-300) return M * Math.cos(a);
  return (Math.sin(0.5 * M * d) / s) * Math.cos(a + 0.5 * (M - 1) * d);
}

// ── height statistics ─────────────────────────────────────────────────────────

/** Moments of the discrete uniform distribution on n points spaced h: {var, mu4}. */
export function discreteUniformMoments(n, h) {
  const v = (h * h * (n * n - 1)) / 12;
  const mu4 = (h ** 4 * (n * n - 1) * (3 * n * n - 7)) / 240;
  return { v, mu4 };
}

/** Two-level (Bernoulli) surface: fraction d at height h, 1-d at 0. */
export function bernoulliHeights(d, h) {
  const q = d * (1 - d);
  return {
    Sa: 2 * q * h, Sq: h * Math.sqrt(q), Sp: (1 - d) * h, Sv: d * h, Sz: h,
    Ssk: (1 - 2 * d) / Math.sqrt(q), Sku: (1 - 3 * q) / q,
  };
}

// ── hybrid ────────────────────────────────────────────────────────────────────

/** Developed area ratio (fraction, not %) of z = A sin(kx) with slope amplitude a = A·k. */
export function sineAreaRatio(a) {
  const m = (a * a) / (1 + a * a);
  return (2 / Math.PI) * Math.sqrt(1 + a * a) * ellipticE(m) - 1;
}

/**
 * Developed area ratio of the SAMPLED sinusoid over whole periods of P cells: each cell has
 * the secant slope a′·cos(2π(i+½)/P), a′ = 2A·sin(π/P)/Δx, so the ratio is the finite sum
 * (1/P)·Σᵢ √(1 + a′²cos²(2π(i+½)/P)) − 1 (exact; the elliptic form is its P → ∞ limit).
 */
export function sampledSineAreaRatio(ap, P) {
  let s = 0;
  for (let i = 0; i < P; i++) s += Math.sqrt(1 + (ap * Math.cos((2 * Math.PI * (i + 0.5)) / P)) ** 2);
  return s / P - 1;
}

// ── filtering (ISO 16610-21 / -61 Gaussian) ──────────────────────────────────

export const GAUSS_ALPHA = Math.sqrt(Math.LN2 / Math.PI);

/** Amplitude transmission of the Gaussian low-pass filter with cut-off λc at wavelength λ. */
export function gaussLowpassT(lambdaC, lambda) {
  return Math.exp(-Math.PI * (GAUSS_ALPHA * lambdaC / lambda) ** 2);
}

/**
 * Bound on |T_discrete − T| for a sampled, truncated (±4σ) and renormalised Gaussian:
 * truncation removes ε = erfc(4/√2) ≈ 6.3e-5 of the kernel mass, which can change the
 * transmission by at most 2ε after renormalisation; sampling aliasing ~exp(−2π²σ²) is
 * negligible for σ ≥ 1 px.
 */
export const GAUSS_TRUNC_BOUND = 2 * 6.334e-5;

// ── material-ratio (Abbott–Firestone) curve: ISO 25178-2 / ISO 13565-2 ────────

/**
 * Functional and volume parameters of a CONTINUOUS, monotonically non-increasing
 * material-ratio curve h(mr), mr in % ∈ [0, 100]. `breaks` lists the mr values where
 * h is not smooth (integration panels are split there, so piecewise-polynomial curves
 * integrate exactly). `mrOf(c)` is the inverse (material ratio at height c).
 */
export class BearingCurve {
  constructor(h, mrOf, breaks = []) {
    this.h = h; this.mrOf = mrOf;
    this.breaks = [0, ...breaks.filter(b => b > 0 && b < 100), 100].sort((a, b) => a - b);
    this.hmax = h(0); this.hmin = h(100);
    this._construct();
  }

  /** ∫_a^b f(mr) dmr, split at the breakpoints. */
  _int(f, a, b) {
    if (b <= a) return 0;
    const pts = [a, ...this.breaks.filter(x => x > a && x < b), b];
    let s = 0;
    for (let i = 0; i + 1 < pts.length; i++) s += integrate(f, pts[i], pts[i + 1], 16);
    return s;
  }

  _construct() {
    const W = 40, h = this.h;
    // 1. position of the 40 % secant with the smallest gradient (largest, i.e. least
    //    negative, slope); dense scan then golden-section refinement; first maximum wins.
    const slope = s => (h(s + W) - h(s)) / W;
    let best = 0, bv = -Infinity;
    const n = 60000;
    for (let i = 0; i <= n; i++) { const s = (60 * i) / n; const v = slope(s); if (v > bv + 1e-15) { bv = v; best = s; } }
    let lo = Math.max(0, best - 60 / n), hi = Math.min(60, best + 60 / n);
    for (let i = 0; i < 100; i++) {
      const a = lo + 0.381966 * (hi - lo), b = lo + 0.618034 * (hi - lo);
      if (slope(a) >= slope(b)) hi = b; else lo = a;
    }
    const s0 = 0.5 * (lo + hi);
    this.windowStart = s0;
    // 2. least-squares straight line through the curve on [s0, s0 + 40]
    const m0 = s0 + W / 2;
    const Sh = this._int(h, s0, s0 + W), Smh = this._int(m => (m - m0) * h(m), s0, s0 + W);
    const beta = Smh / (W ** 3 / 12), hbar = Sh / W;
    this.slope = beta;
    this.yupper = hbar + beta * (0 - m0);
    this.ylower = hbar + beta * (100 - m0);
  }

  mr(c) { return c >= this.hmax ? 0 : c <= this.hmin ? 100 : this.mrOf(c); }
  mc(p) { return this.h(p); }
  get k() { return this.yupper - this.ylower; }
  get mr1() { return this.mr(this.yupper); }
  get mr2() { return this.mr(this.ylower); }
  get A1() { return this._int(m => this.h(m) - this.yupper, 0, this.mr1); }
  get A2() { return this._int(m => this.ylower - this.h(m), this.mr2, 100); }
  get pk() { return 2 * this.A1 / this.mr1; }
  get vk() { return 2 * this.A2 / (100 - this.mr2); }
  get pkx() { return this.hmax - this.yupper; }
  get vkx() { return this.ylower - this.hmin; }
  /** Material volume at p (per unit area, µm = µm³/µm²; mr in % → /100). */
  Vm(p) { const hp = this.h(p); return this._int(m => this.h(m) - hp, 0, p) / 100; }
  /** Void volume at p. */
  Vv(p) { const hp = this.h(p); return this._int(m => hp - this.h(m), p, 100) / 100; }

  /** Every Abbott-derived parameter, keyed by the areal name. */
  all() {
    return {
      Sk: this.k, Spk: this.pk, Svk: this.vk, Spkx: this.pkx, Svkx: this.vkx,
      Sak1: this.A1, Sak2: this.A2, Smr1: this.mr1, Smr2: this.mr2,
      Sxp: this.h(2.5) - this.h(50), Sdc: this.h(2.5) - this.h(50),
      Vmp: this.Vm(10), Vmc: this.Vm(80) - this.Vm(10), Vvv: this.Vv(80), Vvc: this.Vv(10) - this.Vv(80),
    };
  }
}

/** Piecewise-linear material-ratio curve through knots [[mr, h], …] (mr ascending, h descending). */
export function piecewiseLinearCurve(knots) {
  const h = (m) => {
    if (m <= knots[0][0]) return knots[0][1];
    for (let i = 1; i < knots.length; i++) {
      const [m1, h1] = knots[i];
      if (m <= m1) { const [m0, h0] = knots[i - 1]; return h0 + (h1 - h0) * (m - m0) / (m1 - m0); }
    }
    return knots[knots.length - 1][1];
  };
  const mrOf = (c) => {
    for (let i = 1; i < knots.length; i++) {
      const [m0, h0] = knots[i - 1], [m1, h1] = knots[i];
      if (c <= h0 && c >= h1) return h0 === h1 ? m0 : m0 + (m1 - m0) * (h0 - c) / (h0 - h1);
    }
    return c > knots[0][1] ? 0 : 100;
  };
  return new BearingCurve(h, mrOf, knots.map(k => k[0]));
}

/** Material-ratio curve of a sinusoid of amplitude A: h(mr) = A·cos(π·mr/100). */
export function sineCurve(A) {
  return new BearingCurve(m => A * Math.cos(Math.PI * m / 100), c => 100 * Math.acos(Math.max(-1, Math.min(1, c / A))) / Math.PI);
}

/**
 * Polynomial material-ratio curve h(mr) = Σ coefs[i]·uⁱ, u = mr/100, which must be
 * strictly decreasing on [0, 100] (the inverse is solved by bisection). Todhunter et al.
 * (2019, eq. 1) use the cubic Z/10·(10 − 27u + 51u² − 34u³).
 */
export function polynomialCurve(coefs) {
  const h = (m) => { const u = m / 100; let s = 0; for (let i = coefs.length - 1; i >= 0; i--) s = s * u + coefs[i]; return s; };
  for (let i = 0; i < 10000; i++) if (!(h((i + 1) / 100) < h(i / 100))) throw new Error('polynomialCurve: h must decrease strictly');
  return new BearingCurve(h, (c) => bisect((m) => h(m) - c, 0, 100));
}

// ── total least squares (orthogonal-distance fit) ─────────────────────────────

/** Eigen-decomposition of a small symmetric matrix (cyclic Jacobi). Returns {values, vectors (columns)}. */
export function symmetricEigen(S) {
  const n = S.length, A = S.map(r => r.slice()), V = A.map((_, i) => A.map((__, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] ** 2;
    if (off < 1e-300) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (A[p][q] === 0) continue;
      const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  return { values: A.map((r, i) => r[i]), vectors: V };
}

/**
 * Total-least-squares hyperplane z = Σ aⱼ·vⱼ + c through the points (v₁, …, z): the
 * orthogonal-distance fit, i.e. through the centroid with the normal along the smallest
 * principal axis. Unlike an SVD of [X 1 z] (the ones column treated as noisy), it does
 * not depend on the coordinate origin. `vars` = [v₁, …, z] (equal-length arrays, NaN-free).
 * @returns {{coef: number[], c: number}}  z ≈ Σ coef[j]·vⱼ + c
 */
export function tlsHyperplane(vars) {
  const d = vars.length, n = vars[0].length;
  const mean = vars.map(v => { let s = 0; for (let i = 0; i < n; i++) s += v[i]; return s / n; });
  const S = Array.from({ length: d }, () => new Array(d).fill(0));
  for (let i = 0; i < n; i++) for (let a = 0; a < d; a++) { const da = vars[a][i] - mean[a]; for (let b = a; b < d; b++) S[a][b] += da * (vars[b][i] - mean[b]); }
  for (let a = 0; a < d; a++) for (let b = 0; b < a; b++) S[a][b] = S[b][a];
  const { values, vectors } = symmetricEigen(S);
  const k = values.indexOf(Math.min(...values)), nrm = vectors.map(r => r[k]);
  const coef = nrm.slice(0, d - 1).map(v => -v / nrm[d - 1]);
  return { coef, c: mean[d - 1] - coef.reduce((s, a, j) => s + a * mean[j], 0) };
}

// ── autocorrelation (ISO 25178-2 Sal / Str) ──────────────────────────────────

/**
 * Decay lengths of a normalised ACF R(τx, τy) (R(0)=1) to the level s: distance along
 * direction φ to the first crossing of s, minimised (Sal) and maximised over φ.
 * `smax` bounds the search; directions sampled every `dphi` rad then refined.
 */
export function acfDecayLengths(R, s, smax, { phiMax = Math.PI, nPhi = 3600 } = {}) {
  const firstCrossing = (phi) => {
    const c = Math.cos(phi), d = Math.sin(phi), g = (t) => R(t * c, t * d) - s;
    const n = 4000;
    let prev = 0;
    for (let i = 1; i <= n; i++) {
      const t = (smax * i) / n;
      if (g(t) <= 0) return bisect(g, prev, t);
      prev = t;
    }
    return Infinity;
  };
  let mn = Infinity, mx = -Infinity, phiMin = 0, phiMx = 0;
  for (let i = 0; i <= nPhi; i++) {
    const phi = (phiMax * i) / nPhi, L = firstCrossing(phi);
    if (L < mn) { mn = L; phiMin = phi; }
    if (L > mx) { mx = L; phiMx = phi; }
  }
  // golden refinement of both extrema
  const refine = (phi0, sign) => {
    let lo = phi0 - phiMax / nPhi, hi = phi0 + phiMax / nPhi;
    for (let i = 0; i < 80; i++) {
      const a = lo + 0.381966 * (hi - lo), b = lo + 0.618034 * (hi - lo);
      if (sign * firstCrossing(a) <= sign * firstCrossing(b)) hi = b; else lo = a;
    }
    return firstCrossing(0.5 * (lo + hi));
  };
  return { shortest: Math.min(mn, refine(phiMin, 1)), longest: Math.max(mx, refine(phiMx, -1)) };
}

export { TAU };
