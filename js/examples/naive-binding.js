// naive-binding.js — a complete binding for a tiny, self-contained "implementation":
// the areal height parameters (Sa, Sq, Sp, Sv, Sz, Ssk, Sku) and three operations
// (center, zero, invert) written directly against the array. Everything else it
// reports as unsupported.
//
// It is the template for a real binding: replace load/apply/parameter with calls into
// the software under test, and return its output from toArray.
//
//   import { run } from 'softgauge';
//   import { naiveBinding } from 'softgauge/examples/naive-binding.js';
//   const results = run(naiveBinding, { implementation: { name: 'naive' } });

import { Unsupported } from '../src/index.js';

const finite = (z) => Array.prototype.filter.call(z, (v) => v === v);

function moments(z) {
  const v = finite(z), n = v.length, m = v.reduce((a, b) => a + b, 0) / n;
  let a = 0, s2 = 0, s3 = 0, s4 = 0, lo = Infinity, hi = -Infinity;
  for (const x of v) {
    const d = x - m;
    a += Math.abs(d); s2 += d * d; s3 += d ** 3; s4 += d ** 4;
    if (x < lo) lo = x; if (x > hi) hi = x;
  }
  const sq = Math.sqrt(s2 / n);
  return { m, Sa: a / n, Sq: sq, Ssk: s3 / n / sq ** 3, Sku: s4 / n / sq ** 4, max: hi - m, min: lo - m };
}

const map = (h, f) => ({ ...h, z: Float64Array.from(h.z, f) });

export const naiveBinding = {
  load(z, g) {
    return { z, rows: g.ny, cols: g.nx, dx: g.dx, dy: g.dy, profile: !!g.profile };
  },
  apply(h, o) {
    const { m, min, max } = moments(h.z);
    switch (o.op) {
      case 'center': return map(h, (v) => v - m);
      case 'zero': return map(h, (v) => v - (min + m));
      case 'invert': return map(h, (v) => (min + max + 2 * m) - v);
      default: throw new Unsupported(`operation ${o.op}`);
    }
  },
  parameter(h, name) {
    if (h.profile) throw new Unsupported('profile parameters');
    if (!['Sa', 'Sq', 'Ssk', 'Sku', 'Sp', 'Sv', 'Sz'].includes(name)) throw new Unsupported(`parameter ${name}`);
    const p = moments(h.z);
    switch (name) {
      case 'Sa': case 'Sq': case 'Ssk': case 'Sku': return p[name];
      case 'Sp': return p.max;
      case 'Sv': return -p.min;
      case 'Sz': return p.max - p.min;
      default: throw new Unsupported(`parameter ${name}`);
    }
  },
  toArray(h) {
    return h;
  },
};
