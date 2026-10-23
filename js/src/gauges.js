// gauges.js — builds the height map of a soft gauge from its declarative spec.
//
// The SAME recipe is implemented in python/src/softgauge/gauges.py (numpy); both are
// checked against the fingerprints frozen into spec.json, so every implementation under
// test sees the same surface, whichever language its binding is written in.
//
// Grid convention: row-major, `ny` rows × `nx` columns, pixel (r, c) sits at
//   x = c·dx,  y = r·dy          (µm; row 0 is the FIRST row of the array)
// and for the components that are defined in the "y measured from the bottom"
// convention of crop/mask boxes (spec/README.md, Conventions), yb = (ny − 1 − r)·dy.
//
// A gauge is { nx, ny, dx, dy, parts: [component…], holes?, spikes?, dtype?, profile? }:
//   z(r, c) = Σ parts, then spikes are added and holes set to NaN (non-measured).

const TAU = 2 * Math.PI;

/** Triangle wave of period 1: T(0) = 1, T(½) = −1, linear in between. */
export const tri = (u) => 4 * Math.abs(u - Math.floor(u) - 0.5) - 1;

/**
 * Position of pixel (r, c) along a material-ratio ramp: raster order (default), every
 * odd row reversed ('serpentine'), or k ↦ mult·k mod N ('scramble', a bijection when
 * gcd(mult, N) = 1). Reordering the pixels leaves the height distribution unchanged.
 */
function rampIndex(p, g, r, c, k) {
  switch (p.order || 'raster') {
    case 'raster': return k;
    case 'serpentine': return r % 2 ? r * g.nx + (g.nx - 1 - c) : k;
    case 'scramble': return (p.mult * k) % (g.nx * g.ny);
    default: throw new Error(`Unknown ramp order '${p.order}'.`);
  }
}

/** Height of one component at pixel (r, c) / raster index k. */
function component(p, g, r, c, k) {
  const x = c * g.dx, y = r * g.dy, yb = (g.ny - 1 - r) * g.dy;
  switch (p.type) {
    case 'const': return p.value;
    case 'plane': return p.a * x + p.b * y + (p.c || 0);
    case 'poly': {
      let s = 0;
      const u = x - p.xc, v = y - p.yc;
      for (const [i, j, coef] of p.terms) s += coef * u ** i * v ** j;
      return s;
    }
    case 'sine': {
      const th = ((p.theta || 0) * Math.PI) / 180;
      return p.A * Math.sin((TAU * (x * Math.cos(th) + y * Math.sin(th))) / p.lambda + (p.phase || 0));
    }
    case 'eggcrate':
      return p.A * Math.cos((TAU * (x - (p.x0 || 0))) / p.lambdaX) * Math.cos((TAU * (y - (p.y0 || 0))) / p.lambdaY);
    case 'triangle': {
      let z = p.A * tri(x / p.lambda);
      if (p.axes === 'xy') z += p.A * tri(y / p.lambda);
      return z;
    }
    case 'columns': return c < p.split ? p.left : p.right;
    case 'cos': return p.A * Math.cos((TAU * (x - (p.x0 || 0))) / p.lambda);
    case 'disc': {   // sharp (w = 0) or tanh-edged; centre in the y-from-bottom frame unless frame 'top'
      const rho = Math.hypot(x - p.cx, (p.frame === 'top' ? y : yb) - p.cy);
      if (!p.w) return rho <= p.r ? p.h : 0;
      return 0.5 * p.h * (1 - Math.tanh((rho - p.r) / p.w));
    }
    case 'bearing': {   // raster-ordered ramp through a piecewise-linear material-ratio curve
      const N = g.nx * g.ny, m = (100 * k) / (N - 1), kn = p.knots;
      for (let i = 1; i < kn.length; i++) {
        if (m <= kn[i][0]) { const [m0, h0] = kn[i - 1], [m1, h1] = kn[i]; return h0 + ((h1 - h0) * (m - m0)) / (m1 - m0); }
      }
      return kn[kn.length - 1][1];
    }
    case 'bearingCos': {   // raster-ordered ramp through the material-ratio curve of a sinusoid
      const N = g.nx * g.ny;
      return p.A * Math.cos((Math.PI * k) / (N - 1));
    }
    case 'bearingPoly': {   // ramp through h(u) = Σ coefs[i]·uⁱ, u = mr/100 (Horner)
      const u = rampIndex(p, g, r, c, k) / (g.nx * g.ny - 1), a = p.coefs;
      let h = 0;
      for (let i = a.length - 1; i >= 0; i--) h = h * u + a[i];
      return h;
    }
    case 'dither': {   // deterministic, noise-like texture of amplitude ≤ amp (incommensurate frequencies)
      return p.amp * Math.sin(TAU * (0.7071067811865476 * c + 0.3183098861837907 * r)) * Math.cos(TAU * (0.4142135623730951 * r - 0.2718281828459045 * c));
    }
    case 'sphere': {
      const d2 = (x - p.xc) ** 2 + (y - p.yc) ** 2;
      return p.zc + Math.sqrt(p.R * p.R - d2);
    }
    case 'cylinder': {   // horizontal axis through (px, py, zc), direction phi (deg) in the x/y frame
      const ph = (p.phi * Math.PI) / 180;
      const d = -(x - p.px) * Math.sin(ph) + (y - p.py) * Math.cos(ph);
      return p.zc + Math.sqrt(p.R * p.R - d * d);
    }
    case 'gauss': return p.h * Math.exp(-((x - p.xc) ** 2 + (y - p.yc) ** 2) / (2 * p.s * p.s));
    case 'bump': {   // cos² bump of radius r0
      const rho = Math.hypot(x - p.xc, y - p.yc);
      return rho < p.r0 ? p.h * Math.cos((Math.PI * rho) / (2 * p.r0)) ** 2 : 0;
    }
    case 'edge': {   // tanh step across the line through (x0, y0) with direction phi (deg)
      const ph = (p.phi * Math.PI) / 180;
      const d = -(x - p.x0) * Math.sin(ph) + (y - p.y0) * Math.cos(ph);
      return 0.5 * p.h * (1 + Math.tanh(d / p.w));
    }
    case 'pockets': {   // flat-bottomed rectangular pockets (x/y frame), depth below 0
      for (const q of p.pockets) if (Math.abs(x - q.cx) < q.w / 2 && Math.abs(y - q.cy) < q.h / 2) return -q.depth;
      return 0;
    }
    default: throw new Error(`Unknown gauge component '${p.type}'.`);
  }
}

/**
 * Height map of a gauge as a typed array (Float64Array, or Float32Array for
 * dtype 'float32'). Values are always evaluated in double precision.
 */
export function buildHeights(g) {
  const N = g.nx * g.ny;
  const z = g.dtype === 'float32' ? new Float32Array(N) : new Float64Array(N);
  for (let r = 0; r < g.ny; r++) {
    for (let c = 0; c < g.nx; c++) {
      const k = r * g.nx + c;
      let s = 0;
      for (const p of g.parts) s += component(p, g, r, c, k);
      z[k] = s;
    }
  }
  for (const [r, c, dz] of g.spikes || []) z[r * g.nx + c] += dz;
  for (const [r0, r1, c0, c1] of g.holes || []) {
    for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) z[r * g.nx + c] = NaN;
  }
  return z;
}

/**
 * Fingerprint used to prove that two gauge builders agree: count of measured
 * points, their mean and RMS about zero, min, max, and z at fixed probe pixels.
 */
export function fingerprint(g, z = buildHeights(g)) {
  let n = 0, s = 0, s2 = 0, mn = Infinity, mx = -Infinity;
  for (let i = 0; i < z.length; i++) {
    const v = z[i];
    if (v !== v) continue;
    n++; s += v; s2 += v * v; if (v < mn) mn = v; if (v > mx) mx = v;
  }
  const probes = [[0, 0], [g.ny - 1, g.nx - 1], [g.ny >> 1, g.nx >> 1], [g.ny >> 2, (3 * g.nx) >> 2], [(3 * g.ny) >> 2, g.nx >> 3]];
  return {
    n, mean: s / n, rms: Math.sqrt(s2 / n), min: mn, max: mx,
    probes: probes.map(([r, c]) => [r, c, Number.isNaN(z[r * g.nx + c]) ? null : z[r * g.nx + c]]),
  };
}

/**
 * Check heights against the fingerprint frozen into the spec. Differences are scaled by
 * the gauge RMS: math libraries differ in the last bit (e.g. Math.sin in Node vs a
 * browser), and a zero-mean gauge's mean is ~1e−17.
 * @returns {{ok: boolean, maxRelDiff: number}}
 */
export function verifyGauge(g, z = buildHeights(g)) {
  const fp = fingerprint(g, z), ref = g.fingerprint;
  const scale = ref.rms + 1e-300, rel = (a, b) => (a === b ? 0 : Math.abs(a - b) / (scale + Math.abs(b)));
  let d = Math.max(rel(fp.mean, ref.mean), rel(fp.rms, ref.rms), rel(fp.min, ref.min), rel(fp.max, ref.max));
  fp.probes.forEach((p, i) => {
    const q = ref.probes[i][2];
    if (p[2] !== q) d = Math.max(d, p[2] == null || q == null ? Infinity : rel(p[2], q));
  });
  return { ok: fp.n === ref.n && d <= (g.dtype === 'float32' ? 1e-6 : 1e-11), maxRelDiff: d };
}
