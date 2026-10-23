"""Closed-form reference values for the analytic soft gauges.

Pure mathematics: this module never imports an implementation under test. Every
function evaluates a closed-form expression, or, where a closed form needs a special
function or a root, evaluates it to ~1e-13 by a standard convergent method: the AGM
for elliptic integrals, bisection for monotone roots, Gauss–Legendre quadrature on
polynomial pieces (where it is exact).
"""
from __future__ import annotations

import math

TAU = 2 * math.pi


def _div(a: float, b: float) -> float:
    """a / b with IEEE semantics (0/0 = NaN, x/0 = ±inf) instead of ZeroDivisionError."""
    if b == 0:
        return math.nan if a == 0 or math.isnan(a) else math.copysign(math.inf, a) * math.copysign(1, b)
    return a / b


# ── special functions ─────────────────────────────────────────────────────────

def elliptic_e(m: float) -> float:
    """Complete elliptic integral of the second kind E(m), parameter m = k², via the AGM."""
    if m == 1:
        return 1.0
    a, b, c = 1.0, math.sqrt(1 - m), math.sqrt(m)
    s, p = 0.5 * c * c, 0.5
    while abs(c) > 1e-16:
        an, bn, cn = 0.5 * (a + b), math.sqrt(a * b), 0.5 * (a - b)
        p *= 2
        s += p * cn * cn
        a, b, c = an, bn, cn
    return (math.pi / (2 * a)) * (1 - s)


def bisect(f, a: float, b: float) -> float:
    """Root of a continuous f on [a, b] with a sign change, by bisection to machine precision."""
    fa = f(a)
    if fa == 0:
        return a
    for _ in range(200):
        m = 0.5 * (a + b)
        if m == a or m == b:
            return m
        fm = f(m)
        if fm == 0:
            return m
        if (fm < 0) == (fa < 0):
            a, fa = m, fm
        else:
            b = m
    return 0.5 * (a + b)


# 10-point Gauss–Legendre nodes/weights on [-1, 1] (exact for polynomials ≤ degree 19).
GL_X = [-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472, -0.1488743389816312,
        0.1488743389816312, 0.4333953941292472, 0.6794095682990244, 0.8650633666889845, 0.9739065285171717]
GL_W = [0.0666713443086881, 0.1494513491505806, 0.2190863625159820, 0.2692667193099963, 0.2955242247147529,
        0.2955242247147529, 0.2692667193099963, 0.2190863625159820, 0.1494513491505806, 0.0666713443086881]


def integrate(f, a: float, b: float, n: int = 64) -> float:
    """∫_a^b f, composite 10-point Gauss–Legendre over n panels (smooth f)."""
    if b == a:
        return 0.0
    h = (b - a) / n
    s = 0.0
    for p in range(n):
        c = a + (p + 0.5) * h
        for k in range(10):
            s += GL_W[k] * f(c + 0.5 * h * GL_X[k])
    return 0.5 * h * s


def sum_cos(a: float, d: float, M: int) -> float:
    """Σ_{i=0}^{M-1} cos(a + i·d) in closed form (Lagrange's identity)."""
    s = math.sin(0.5 * d)
    if abs(s) < 1e-300:
        return M * math.cos(a)
    return (math.sin(0.5 * M * d) / s) * math.cos(a + 0.5 * (M - 1) * d)


# ── height statistics ─────────────────────────────────────────────────────────

def discrete_uniform_moments(n: int, h: float) -> dict:
    """Moments of the discrete uniform distribution on n points spaced h: {v, mu4}."""
    v = (h * h * (n * n - 1)) / 12
    mu4 = (h ** 4 * (n * n - 1) * (3 * n * n - 7)) / 240
    return {'v': v, 'mu4': mu4}


def bernoulli_heights(d: float, h: float) -> dict:
    """Two-level (Bernoulli) surface: fraction d at height h, 1 − d at 0."""
    q = d * (1 - d)
    return {
        'Sa': 2 * q * h, 'Sq': h * math.sqrt(q), 'Sp': (1 - d) * h, 'Sv': d * h, 'Sz': h,
        'Ssk': (1 - 2 * d) / math.sqrt(q), 'Sku': (1 - 3 * q) / q,
    }


# ── hybrid ────────────────────────────────────────────────────────────────────

def sine_area_ratio(a: float) -> float:
    """Developed area ratio (fraction, not %) of z = A·sin(kx) with slope amplitude a = A·k."""
    m = (a * a) / (1 + a * a)
    return (2 / math.pi) * math.sqrt(1 + a * a) * elliptic_e(m) - 1


def sampled_sine_area_ratio(ap: float, P: int) -> float:
    """Developed area ratio of the SAMPLED sinusoid over whole periods of P cells.

    Each cell has the secant slope a′·cos(2π(i+½)/P), a′ = 2A·sin(π/P)/Δx, so the ratio is
    the finite sum (1/P)·Σᵢ √(1 + a′²cos²(2π(i+½)/P)) − 1 (exact; the elliptic form is its
    P → ∞ limit).
    """
    s = 0.0
    for i in range(P):
        s += math.sqrt(1 + (ap * math.cos((2 * math.pi * (i + 0.5)) / P)) ** 2)
    return s / P - 1


def saddle_area_ratio(k: float, W: float, H: float) -> float:
    """Developed area ratio of the saddle z = k·x·y over [0, W] × [0, H]: mean of √(1 + k²(x² + y²)) − 1."""
    def g(x):
        return integrate(lambda y: math.sqrt(1 + k * k * (x * x + y * y)), 0, H)
    return integrate(g, 0, W) / (W * H) - 1


def sampled_saddle_area_ratio(k: float, nx: int, ny: int, dx: float, dy: float) -> float:
    """The same ratio for the SAMPLED saddle (nx × ny points, steps dx, dy), triangulated per ISO 25178-2 eq. (D.15).

    Along a grid line z is linear, so every edge difference is exact: the triangle at a
    cell corner (spanned by its x and y edges) has area ½·Δx·Δy·√(1 + |∇z|²) with the
    gradient (k·y, k·x) AT that corner, and the cell area is ¼·Σ_corners of the four
    triangles' doubled areas. Summed over the cells, the ratio is the trapezoid-weighted
    mean of √(1 + k²(x² + y²)) over the grid points (weight ½ on the edges, ¼ at the
    corners) − 1.
    """
    s = 0.0
    for r in range(ny):
        wy = 0.5 if r == 0 or r == ny - 1 else 1
        for c in range(nx):
            wx = 0.5 if c == 0 or c == nx - 1 else 1
            s += wx * wy * math.sqrt(1 + k * k * ((c * dx) ** 2 + (r * dy) ** 2))
    return s / ((nx - 1) * (ny - 1)) - 1


# ── filtering (ISO 16610-21 / -61 Gaussian) ──────────────────────────────────

GAUSS_ALPHA = math.sqrt(math.log(2) / math.pi)


def gauss_lowpass_t(lambda_c: float, lam: float) -> float:
    """Amplitude transmission of the Gaussian low-pass filter with cut-off λc at wavelength λ."""
    return math.exp(-math.pi * (GAUSS_ALPHA * lambda_c / lam) ** 2)


# Bound on |T_discrete − T| for a sampled, truncated (±4σ) and renormalised Gaussian:
# truncation removes ε = erfc(4/√2) ≈ 6.3e-5 of the kernel mass, which can change the
# transmission by at most 2ε after renormalisation; sampling aliasing ~exp(−2π²σ²) is
# negligible for σ ≥ 1 px.
GAUSS_TRUNC_BOUND = 2 * 6.334e-5


# ── material-ratio (Abbott–Firestone) curve: ISO 25178-2 / ISO 13565-2 ────────

class BearingCurve:
    """Functional and volume parameters of a CONTINUOUS, monotonically non-increasing material-ratio curve.

    h(mr) is the height at material ratio mr in % ∈ [0, 100]; ``breaks`` lists the mr
    values where h is not smooth (integration panels are split there, so
    piecewise-polynomial curves integrate exactly); ``mr_of(c)`` is the inverse (material
    ratio at height c).
    """

    def __init__(self, h, mr_of, breaks=()):
        self.h, self.mr_of = h, mr_of
        self.breaks = sorted([0, *(b for b in breaks if 0 < b < 100), 100])
        self.hmax, self.hmin = h(0), h(100)
        self._construct()

    def _int(self, f, a, b):
        """∫_a^b f(mr) dmr, split at the breakpoints."""
        if b <= a:
            return 0.0
        pts = [a, *(x for x in self.breaks if a < x < b), b]
        return sum(integrate(f, pts[i], pts[i + 1], 16) for i in range(len(pts) - 1))

    def _construct(self):
        W, h = 40, self.h

        # 1. position of the 40 % secant with the smallest gradient (largest, i.e. least
        #    negative, slope); dense scan then golden-section refinement; first maximum wins.
        def slope(s):
            return (h(s + W) - h(s)) / W

        best, bv, n = 0.0, -math.inf, 60000
        for i in range(n + 1):
            s = (60 * i) / n
            v = slope(s)
            if v > bv + 1e-15:
                bv, best = v, s
        lo, hi = max(0, best - 60 / n), min(60, best + 60 / n)
        for _ in range(100):
            a, b = lo + 0.381966 * (hi - lo), lo + 0.618034 * (hi - lo)
            if slope(a) >= slope(b):
                hi = b
            else:
                lo = a
        s0 = 0.5 * (lo + hi)
        self.window_start = s0
        # 2. least-squares straight line through the curve on [s0, s0 + 40]
        m0 = s0 + W / 2
        Sh, Smh = self._int(h, s0, s0 + W), self._int(lambda m: (m - m0) * h(m), s0, s0 + W)
        beta, hbar = Smh / (W ** 3 / 12), Sh / W
        self.slope = beta
        self.yupper = hbar + beta * (0 - m0)
        self.ylower = hbar + beta * (100 - m0)

    def mr(self, c):
        return 0 if c >= self.hmax else 100 if c <= self.hmin else self.mr_of(c)

    def mc(self, p):
        return self.h(p)

    @property
    def k(self):
        return self.yupper - self.ylower

    @property
    def mr1(self):
        return self.mr(self.yupper)

    @property
    def mr2(self):
        return self.mr(self.ylower)

    @property
    def A1(self):
        return self._int(lambda m: self.h(m) - self.yupper, 0, self.mr1)

    @property
    def A2(self):
        return self._int(lambda m: self.ylower - self.h(m), self.mr2, 100)

    @property
    def pk(self):
        return _div(2 * self.A1, self.mr1)

    @property
    def vk(self):
        return _div(2 * self.A2, 100 - self.mr2)

    @property
    def pkx(self):
        return self.hmax - self.yupper

    @property
    def vkx(self):
        return self.ylower - self.hmin

    def Vm(self, p):
        """Material volume at p (per unit area, µm = µm³/µm²; mr in % → /100)."""
        hp = self.h(p)
        return self._int(lambda m: self.h(m) - hp, 0, p) / 100

    def Vv(self, p):
        """Void volume at p."""
        hp = self.h(p)
        return self._int(lambda m: hp - self.h(m), p, 100) / 100

    def all(self) -> dict:
        """Every Abbott-derived parameter, keyed by the areal name."""
        return {
            'Sk': self.k, 'Spk': self.pk, 'Svk': self.vk, 'Spkx': self.pkx, 'Svkx': self.vkx,
            'Sak1': self.A1, 'Sak2': self.A2, 'Smr1': self.mr1, 'Smr2': self.mr2,
            'Sxp': self.h(2.5) - self.h(50), 'Sdc': self.h(2.5) - self.h(50),
            'Vmp': self.Vm(10), 'Vmc': self.Vm(80) - self.Vm(10), 'Vvv': self.Vv(80), 'Vvc': self.Vv(10) - self.Vv(80),
        }


def piecewise_linear_curve(knots) -> BearingCurve:
    """Piecewise-linear material-ratio curve through knots [[mr, h], …] (mr ascending, h descending)."""
    def h(m):
        if m <= knots[0][0]:
            return knots[0][1]
        for i in range(1, len(knots)):
            m1, h1 = knots[i]
            if m <= m1:
                m0, h0 = knots[i - 1]
                return h0 + (h1 - h0) * (m - m0) / (m1 - m0)
        return knots[-1][1]

    def mr_of(c):
        for i in range(1, len(knots)):
            (m0, h0), (m1, h1) = knots[i - 1], knots[i]
            if h0 >= c >= h1:
                return m0 if h0 == h1 else m0 + (m1 - m0) * (h0 - c) / (h0 - h1)
        return 0 if c > knots[0][1] else 100

    return BearingCurve(h, mr_of, [k[0] for k in knots])


def sine_curve(A: float) -> BearingCurve:
    """Material-ratio curve of a sinusoid of amplitude A: h(mr) = A·cos(π·mr/100)."""
    return BearingCurve(lambda m: A * math.cos(math.pi * m / 100),
                        lambda c: 100 * math.acos(max(-1, min(1, c / A))) / math.pi)


def polynomial_curve(coefs) -> BearingCurve:
    """Polynomial material-ratio curve h(mr) = Σ coefs[i]·uⁱ, u = mr/100, strictly decreasing on [0, 100].

    The inverse is solved by bisection. Todhunter et al. (2019, eq. 1) use the cubic
    Z/10·(10 − 27u + 51u² − 34u³).
    """
    def h(m):
        u, s = m / 100, 0.0
        for c in reversed(coefs):
            s = s * u + c
        return s

    for i in range(10000):
        if not h((i + 1) / 100) < h(i / 100):
            raise ValueError('polynomial_curve: h must decrease strictly')
    return BearingCurve(h, lambda c: bisect(lambda m: h(m) - c, 0, 100))


# ── total least squares (orthogonal-distance fit) ─────────────────────────────

def symmetric_eigen(S):
    """Eigen-decomposition of a small symmetric matrix (cyclic Jacobi) → (values, vectors as columns)."""
    n = len(S)
    A = [row[:] for row in S]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(100):
        off = sum(A[p][q] ** 2 for p in range(n) for q in range(p + 1, n))
        if off < 1e-300:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if A[p][q] == 0:
                    continue
                th = (A[q][q] - A[p][p]) / (2 * A[p][q])
                sgn = -1.0 if th < 0 else 1.0
                t = sgn / (abs(th) + math.sqrt(th * th + 1))
                c = 1 / math.sqrt(t * t + 1)
                s = t * c
                for k in range(n):
                    akp, akq = A[k][p], A[k][q]
                    A[k][p], A[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = A[p][k], A[q][k]
                    A[p][k], A[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = V[k][p], V[k][q]
                    V[k][p], V[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    return [A[i][i] for i in range(n)], V


def tls_hyperplane(vars_):
    """Total-least-squares hyperplane z = Σ aⱼ·vⱼ + c through the points (v₁, …, z).

    The orthogonal-distance fit: through the centroid, with the normal along the smallest
    principal axis. Unlike an SVD of [X 1 z] (the ones column treated as noisy), it does
    not depend on the coordinate origin. ``vars_`` = [v₁, …, z] (equal-length sequences,
    NaN-free). Returns (coef, c) with z ≈ Σ coef[j]·vⱼ + c.
    """
    d, n = len(vars_), len(vars_[0])
    mean = []
    for v in vars_:
        s = 0.0
        for i in range(n):
            s += v[i]
        mean.append(s / n)
    S = [[0.0] * d for _ in range(d)]
    for i in range(n):
        for a in range(d):
            da = vars_[a][i] - mean[a]
            for b in range(a, d):
                S[a][b] += da * (vars_[b][i] - mean[b])
    for a in range(d):
        for b in range(a):
            S[a][b] = S[b][a]
    values, vectors = symmetric_eigen(S)
    k = values.index(min(values))
    nrm = [row[k] for row in vectors]
    coef = [-v / nrm[d - 1] for v in nrm[:d - 1]]
    c = mean[d - 1]
    acc = 0.0
    for j, a in enumerate(coef):
        acc += a * mean[j]
    return coef, c - acc


# ── autocorrelation (ISO 25178-2 Sal / Str) ──────────────────────────────────

def acf_decay_lengths(R, s: float, smax: float, phi_max: float = math.pi, n_phi: int = 3600) -> dict:
    """Decay lengths of a normalised ACF R(τx, τy) (R(0) = 1) to the level s.

    The distance along direction φ to the first crossing of s, minimised (Sal) and
    maximised over φ. ``smax`` bounds the search; directions are sampled every
    phi_max/n_phi rad, then refined by golden-section search.
    """
    def first_crossing(phi):
        c, d = math.cos(phi), math.sin(phi)

        def g(t):
            return R(t * c, t * d) - s

        n, prev = 4000, 0.0
        for i in range(1, n + 1):
            t = (smax * i) / n
            if g(t) <= 0:
                return bisect(g, prev, t)
            prev = t
        return math.inf

    mn, mx, phi_min, phi_mx = math.inf, -math.inf, 0.0, 0.0
    for i in range(n_phi + 1):
        phi = (phi_max * i) / n_phi
        L = first_crossing(phi)
        if L < mn:
            mn, phi_min = L, phi
        if L > mx:
            mx, phi_mx = L, phi

    def refine(phi0, sign):
        lo, hi = phi0 - phi_max / n_phi, phi0 + phi_max / n_phi
        for _ in range(80):
            a, b = lo + 0.381966 * (hi - lo), lo + 0.618034 * (hi - lo)
            if sign * first_crossing(a) <= sign * first_crossing(b):
                hi = b
            else:
                lo = a
        return first_crossing(0.5 * (lo + hi))

    return {'shortest': min(mn, refine(phi_min, 1)), 'longest': max(mx, refine(phi_mx, -1))}
