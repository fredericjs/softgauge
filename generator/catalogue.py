"""The analytic soft gauges and the validation cases run on them.

Every case states the gauge, the operations applied to it, what is measured, the
analytic reference value (with its formula), optionally the exact value for the SAMPLED
gauge (when the sampling error has a closed form), and the tolerance with the reasoning
behind it. Implementations are judged against the sampled value when one exists (so the
tolerance is floating-point tight and the difference to the continuous value is reported
separately as sampling error), else against the continuous value with a stated
discretisation bound.

build.py evaluates this module into the language-neutral spec.json. Dictionary key order
is the key order of spec.json.
"""
from __future__ import annotations

import math

import numpy as np
from softgauge.gauges import build_heights

from fmt import exponential, fixed, num, precision
from reference import (
    GAUSS_TRUNC_BOUND, acf_decay_lengths, bernoulli_heights, discrete_uniform_moments, gauss_lowpass_t,
    piecewise_linear_curve, polynomial_curve, saddle_area_ratio, sampled_saddle_area_ratio,
    sampled_sine_area_ratio, sine_area_ratio, sine_curve, sum_cos, tls_hyperplane,
)

TAU = 2 * math.pi
ACOS02 = math.acos(0.2)
SQRT2 = math.sqrt(2)


def js_round(x: float) -> int:
    """Round half up (towards +∞), as JavaScript's Math.round."""
    return math.floor(x + 0.5)


# ── tolerance helpers ─────────────────────────────────────────────────────────

def FP(scale=1):
    return {'rel': 1e-9, 'abs': 1e-12 * scale}


FP_BASIS = 'floating-point round-off only (the reference is exact for the sampled gauge)'

# ── gauges ────────────────────────────────────────────────────────────────────

G: dict[str, dict] = {}


def gauge(gid: str, d: dict) -> str:
    G[gid] = {'id': gid, **d}
    return gid


# Tilted plane, anisotropic sampling (dx ≠ dy catches x/y mix-ups).
PL = {'a': 0.03, 'b': -0.02, 'c': 5}
gauge('plane', {
    'title': 'Tilted plane', 'formula': 'z = a·x + b·y + c',
    'description': 'A least-squares plane. Every height statistic of the sampled plane has a closed form (sum of two discrete uniform distributions); every finite difference is exact.',
    'nx': 201, 'ny': 151, 'dx': 0.5, 'dy': 0.25, 'parts': [{'type': 'plane', **PL}],
})

# Isotropic tilted plane for rotation.
gauge('plane-iso', {
    'title': 'Tilted plane (isotropic)', 'formula': 'z = a·x',
    'description': 'Plane sloping along x on a square-pixel grid, used to check that rotation preserves slope magnitude and turns its direction by the requested angle.',
    'nx': 201, 'ny': 201, 'dx': 0.5, 'dy': 0.5, 'parts': [{'type': 'plane', 'a': 0.04, 'b': 0, 'c': 0}],
})

# Egg-crate centred on the grid with an integer number of periods on both axes: it is
# exactly orthogonal (on the sampled grid) to every monomial of total degree ≤ 3.
EGG = {'A': 0.8, 'lambdaX': 12, 'lambdaY': 12, 'x0': 47.75, 'y0': 35.75}
EGG_GRID = {'nx': 192, 'ny': 144, 'dx': 0.5, 'dy': 0.5}
XC, YC = 47.75, 35.75
POLY1 = [[1, 0, 0.05], [0, 1, -0.03], [0, 0, 2]]
POLY2 = [*POLY1, [2, 0, 0.002], [1, 1, -0.001], [0, 2, 0.003]]
POLY3 = [*POLY2, [3, 0, 1e-4], [2, 1, -5e-5], [1, 2, 3e-5], [0, 3, -2e-5]]
gauge('egg', {
    'title': 'Centred egg-crate', 'formula': 'z = A·cos(2π(x−x₀)/λ)·cos(2π(y−y₀)/λ)',
    'description': 'Product of cosines centred on the grid with 8 × 6 full periods. Its sampled mean is 0 and its moments are exact: Sq = A/2, Sku = 9/4.',
    **EGG_GRID, 'parts': [{'type': 'eggcrate', **EGG}],
})
gauge('egg+poly1', {'title': 'Egg-crate + plane', 'formula': 'z = egg-crate + p₁(x, y)', 'description': 'Egg-crate plus a first-degree polynomial. Levelling must return the egg-crate exactly, because it is orthogonal to the polynomial on the sampled grid.', **EGG_GRID, 'parts': [{'type': 'eggcrate', **EGG}, {'type': 'poly', 'xc': XC, 'yc': YC, 'terms': POLY1}]})
gauge('egg+poly2', {'title': 'Egg-crate + quadratic', 'formula': 'z = egg-crate + p₂(x, y)', 'description': 'Egg-crate plus a full second-degree polynomial (with cross term).', **EGG_GRID, 'parts': [{'type': 'eggcrate', **EGG}, {'type': 'poly', 'xc': XC, 'yc': YC, 'terms': POLY2}]})
gauge('egg+poly3', {'title': 'Egg-crate + cubic', 'formula': 'z = egg-crate + p₃(x, y)', 'description': 'Egg-crate plus a full third-degree polynomial.', **EGG_GRID, 'parts': [{'type': 'eggcrate', **EGG}, {'type': 'poly', 'xc': XC, 'yc': YC, 'terms': POLY3}]})

# Plane + a cos² bump; reference circles for level-to-points avoid the bump.
LP_BUMP = {'type': 'bump', 'h': 3, 'r0': 15, 'xc': 70, 'yc': 50}
gauge('plane+bump', {
    'title': 'Plane with a bump', 'formula': 'z = a·x + b·y + c + h·cos²(πρ/2r₀)·[ρ < r₀]',
    'description': 'Tilted plane carrying a cos² bump. A plane fitted only around reference points on the flat part must remove the tilt exactly and leave the bump.',
    'nx': 161, 'ny': 121, 'dx': 0.5, 'dy': 0.5, 'parts': [{'type': 'plane', 'a': 0.02, 'b': 0.01, 'c': 1}, LP_BUMP],
})

# Two-level (Bernoulli) surface.
STEP = {'d': 0.25, 'h': 2}
gauge('step', {
    'title': 'Two-level step', 'formula': 'z = h for x < x_s, else 0',
    'description': 'A fraction d = 1/4 of the surface at height h, the rest at 0. Its height distribution is Bernoulli, so all height parameters including skewness and kurtosis are exact.',
    'nx': 200, 'ny': 100, 'dx': 1, 'dy': 1, 'parts': [{'type': 'columns', 'split': 50, 'left': STEP['h'], 'right': 0}],
})

# Sharp disc plateau for the circular mask (centre in the y-from-bottom frame).
DISC = {'cx': 50.3, 'cy': 48.7, 'r': 20.5, 'h': 1.5}
gauge('disc', {
    'title': 'Disc plateau', 'formula': 'z = h for (x−xc)² + (y−yc)² ≤ r²',
    'description': 'A sharp circular plateau. Masking the same circle must leave a perfectly flat surface. The centre is chosen so that no pixel lies exactly on the circle.',
    'nx': 101, 'ny': 101, 'dx': 1, 'dy': 1, 'parts': [{'type': 'disc', **DISC}],
})

# Sinusoids.
SINE = {'A': 1.5, 'lambda': 16}
SINE_P = 64  # samples per period (dx = 0.25)
gauge('sine', {
    'title': 'Sinusoid', 'formula': 'z = A·sin(2πx/λ)',
    'description': '10 full periods sampled with P = 64 points per period (a multiple of 4, so crests and troughs are sampled). Height moments over full periods are exact.',
    'nx': 10 * SINE_P, 'ny': 32, 'dx': 0.25, 'dy': 0.25, 'parts': [{'type': 'sine', **SINE}],
})
gauge('sine-f32', {**G['sine'], 'title': 'Sinusoid (float32 storage)', 'dtype': 'float32', 'description': 'The sinusoid gauge stored as 32-bit floats, the storage type of several instrument formats.'})
gauge('sine-acf', {
    'title': 'Sinusoid (ACF)', 'formula': 'z = A·sin(2πx/λ)',
    'description': '64 periods of λ = 16 µm at 1 µm sampling. Its ACF is A²/2·cos(2πτx/λ) and does not decay along y, so Str is undefined.',
    'nx': 1024, 'ny': 16, 'dx': 1, 'dy': 1, 'parts': [{'type': 'sine', 'A': 1, 'lambda': 16}],
})
gauge('sine-abbott', {
    'title': 'Sinusoid material-ratio curve', 'formula': 'z(k) = A·cos(π·k/(N−1))',
    'description': 'The material-ratio curve of a sinusoid, h(mr) = A·cos(π·mr/100), laid out as a raster ramp of N = 20 000 heights so the sampled curve is exact. (A sampled sinusoid repeats each height many times, so its empirical curve is a staircase.) It exercises the ISO 13565-2 construction on a smooth, curved material-ratio curve.',
    'nx': 200, 'ny': 100, 'dx': 1, 'dy': 1, 'parts': [{'type': 'bearingCos', 'A': 1}],
})
HYB_P = 32
gauge('sine-hybrid', {
    'title': 'Sinusoid (gradients)', 'formula': 'z = A·sin(2πx/λ)',
    'description': 'nx − 1 = 8 full periods of cells, so every forward difference set spans whole periods: the sampled Sdq and Sdr have closed forms with the secant slope amplitude a′ = 2A·sin(π/P)/Δx.',
    'nx': 8 * HYB_P + 1, 'ny': 17, 'dx': 0.5, 'dy': 0.5, 'parts': [{'type': 'sine', 'A': 1, 'lambda': 16}],
})

# Piecewise-linear "pyramid" field: every grid cell is planar, so slope and area are exact.
TRI = {'A': 1, 'lambda': 20}
gauge('pyramids', {
    'title': 'Pyramid field', 'formula': 'z = A·[T(x/λ) + T(y/λ)], T = triangle wave',
    'description': 'Sum of two triangle waves whose kinks fall on grid lines. Every grid cell is a plane with |∂z/∂x| = |∂z/∂y| = 4A/λ, so Sdq and the triangulated area are exact.',
    'nx': 201, 'ny': 161, 'dx': 0.5, 'dy': 0.5, 'parts': [{'type': 'triangle', **TRI, 'axes': 'xy'}],
})

# Saddle (hyperbolic paraboloid): every grid cell is twisted, so the two ways of splitting a
# cell into triangles differ. The corner at the origin sits on a grid corner; dx ≠ dy.
SAD = {'k': 0.05}
gauge('saddle', {
    'title': 'Saddle', 'formula': 'z = k·x·y',
    'description': 'Hyperbolic paraboloid with its centre on the first grid corner and anisotropic sampling. No grid cell is planar, so the triangulated area depends on how a cell is split into triangles (ISO 25178-2 eq. D.15 averages both diagonals); along every grid line z is linear, so every edge difference is exact and the D.15 area has a closed form.',
    'nx': 41, 'ny': 81, 'dx': 0.5, 'dy': 0.25, 'parts': [{'type': 'poly', 'xc': 0, 'yc': 0, 'terms': [[1, 1, SAD['k']]]}],
})

# Material-ratio gauge: the heights ARE the samples of a prescribed curve.
KNOTS = [[0, 4], [3, 2.2], [10, 1], [80, -1], [92, -2.2], [100, -4]]
gauge('bearing', {
    'title': 'Material-ratio gauge', 'formula': 'z(k) = h(100·k/(N−1)), h piecewise linear',
    'description': 'Heights placed in raster order along a prescribed piecewise-linear material-ratio curve (peak, core and valley zones), so the sampled material-ratio curve equals h exactly. ISO 13565-2 / ISO 25178-2 functional and volume parameters follow in closed form.',
    'nx': 200, 'ny': 100, 'dx': 1, 'dy': 1, 'parts': [{'type': 'bearing', 'knots': KNOTS}],
})
gauge('ramp', {
    'title': 'Uniform ramp', 'formula': 'z(k) = 1 − k/(N−1)',
    'description': 'Heights uniformly distributed on [0, 1] (raster ramp), used for material-ratio thresholding.',
    'nx': 200, 'ny': 100, 'dx': 1, 'dy': 1, 'parts': [{'type': 'bearing', 'knots': [[0, 1], [100, 0]]}],
})

# Feature egg-crate: border through extrema, K = 12 half-periods per axis.
# Layout: H = P/2 = 15 px per half-period (odd, so saddle lines fall between pixels);
# extremum i sits at column (H−1)/2 + i·H, so cell 0 spans columns 0…H−1 and the
# border runs just inside a complete ring of cells. Border cells are cut (excluded);
# every interior hill's catchment (its cell plus the facing triangles of the four
# neighbouring dale cells) stays ≥ (H−1)/2 px away from the border.
FE = {'A': 1, 'lambda': 15, 'P': 30, 'cells': 12}
FE_H = FE['P'] // 2
FE_N = FE_H * FE['cells']
FE_X0 = ((FE_H - 1) / 2) * 0.5
gauge('egg-feature', {
    'title': 'Egg-crate (features)', 'formula': 'z = A·cos(2πx/λ)·cos(2πy/λ)',
    'description': '12 × 12 complete half-period cells. Hills are the squares around the extrema with i + j even, dales the others, and saddles sit at height 0. The outer ring of cells touches the border and is excluded; the 10 × 10 interior cells hold 50 hills and 50 dales. With 15 px per half-period the zero-level saddle lines lie between pixels, so the segmentation has no ties.',
    'nx': FE_N, 'ny': FE_N, 'dx': 0.5, 'dy': 0.5, 'parts': [{'type': 'eggcrate', 'A': FE['A'], 'lambdaX': FE['lambda'], 'lambdaY': FE['lambda'], 'x0': FE_X0, 'y0': FE_X0}],
})

# ACF egg-crates (isotropic / anisotropic with dx ≠ dy).
gauge('egg-acf', {
    'title': 'Egg-crate (ACF, isotropic)', 'formula': 'z = A·cos(2πx/λ)·cos(2πy/λ)',
    'description': '32 × 32 periods. The normalised ACF is cos(2πτx/λ)·cos(2πτy/λ); its 0.2 contour is solved numerically from that closed form.',
    'nx': 512, 'ny': 512, 'dx': 1, 'dy': 1, 'parts': [{'type': 'eggcrate', 'A': 1, 'lambdaX': 16, 'lambdaY': 16}],
})
gauge('egg-acf-aniso', {
    'title': 'Egg-crate (ACF, anisotropic)', 'formula': 'z = A·cos(2πx/λx)·cos(2πy/λy)',
    'description': 'λx = 16 µm, λy = 24 µm on a grid with Δy = 1.5·Δx (16 px per period on both axes), so a mix-up between x and y would be visible.',
    'nx': 512, 'ny': 512, 'dx': 1, 'dy': 1.5, 'parts': [{'type': 'eggcrate', 'A': 1, 'lambdaX': 16, 'lambdaY': 24}],
})

# Filter sinusoids (one gauge per wavelength).
FILT_LAMBDAS = [4, 8, 16, 32, 64]
for L in FILT_LAMBDAS:
    gauge(f'sine-filter-{L}', {
        'title': f'Sinusoid λ = {L} µm', 'formula': 'z = A·sin(2πx/λ)',
        'description': f'Unit sinusoid of wavelength {L} µm (integer number of periods) for measuring the amplitude transmission of filters.',
        'nx': 512, 'ny': 8, 'dx': 0.5, 'dy': 0.5, 'parts': [{'type': 'sine', 'A': 1, 'lambda': L}],
    })
SPIKES_R = [[20, 30, 20], [40, 90, -20], [64, 64, 20], [90, 20, -20], [100, 100, 20], [30, 110, -20]]
DITHER = 0.01
gauge('dither-spiked', {
    'title': 'Flat surface with spikes', 'formula': 'z = ε(x, y) + Σ spikes, |ε| ≤ 0.01 µm',
    'description': 'A flat surface with a 0.01 µm noise-like texture carrying six isolated ±20 µm spikes. If the spikes get zero weight, every output of a weighted Gaussian mean is a convex combination of texture values, so it stays within ±0.01 µm, including at the border.',
    'nx': 256, 'ny': 128, 'dx': 0.5, 'dy': 0.5, 'parts': [{'type': 'dither', 'amp': DITHER}], 'spikes': SPIKES_R,
})
gauge('sine-filter-y', {
    'title': 'Sinusoid along y (Δy ≠ Δx)', 'formula': 'z = A·sin(2πy/λ)',
    'description': 'Unit sinusoid of wavelength 16 µm along y on a grid with Δy = 0.25 µm and Δx = 0.5 µm: the filter must use the y step for the y kernel.',
    'nx': 16, 'ny': 512, 'dx': 0.5, 'dy': 0.25, 'parts': [{'type': 'sine', 'A': 1, 'lambda': 16, 'theta': 90}],
})
gauge('const-holes', {
    'title': 'Constant with holes', 'formula': 'z = c, non-measured rectangles',
    'description': 'A constant height with two rectangular non-measured regions, one touching the border. Zero-order Gaussian regression must return c everywhere, at the border and around the holes.',
    'nx': 160, 'ny': 120, 'dx': 0.5, 'dy': 0.5, 'parts': [{'type': 'const', 'value': 3}], 'holes': [[30, 50, 40, 70], [0, 15, 120, 159]],
})

# Form gauges.
SPH = {'R': 500, 'xc': 100, 'yc': 100, 'zc': -480}
gauge('sphere', {
    'title': 'Spherical cap', 'formula': 'z = z_c + √(R² − (x−x_c)² − (y−y_c)²)',
    'description': 'Cap of a sphere of radius R = 500 µm over a 200 × 200 µm field.',
    'nx': 201, 'ny': 201, 'dx': 1, 'dy': 1, 'parts': [{'type': 'sphere', **SPH}],
})
CYL = {'R': 300, 'px': 100, 'py': 100, 'zc': -290, 'phi': 20}
gauge('cylinder', {
    'title': 'Cylinder section', 'formula': 'z = z_c + √(R² − d²), d = distance to the axis',
    'description': 'Cylinder of radius R = 300 µm whose horizontal axis is turned 20° from the x axis.',
    'nx': 201, 'ny': 201, 'dx': 1, 'dy': 1, 'parts': [{'type': 'cylinder', **CYL}],
})

# Fill gauges.
FILL_PLANE = {'a': 0.02, 'b': -0.01, 'c': 1}
gauge('plane-holes', {
    'title': 'Plane with holes', 'formula': 'z = a·x + b·y + c, non-measured rectangles',
    'description': 'Tilted plane with two interior non-measured rectangles. Linear (Delaunay) interpolation of a plane is exact.',
    'nx': 101, 'ny': 81, 'dx': 1, 'dy': 1, 'parts': [{'type': 'plane', **FILL_PLANE}], 'holes': [[20, 30, 20, 35], [50, 60, 60, 80]],
})
gauge('columns-hole', {
    'title': 'Step edge with hole', 'formula': 'z = [x ≥ x_s], non-measured square across the edge',
    'description': 'A vertical step edge with a 20 × 20 px hole centred on it (even width, so no pixel is equidistant from both sides). Nearest-neighbour filling must restore the edge exactly.',
    'nx': 200, 'ny': 100, 'dx': 1, 'dy': 1, 'parts': [{'type': 'columns', 'split': 100, 'left': 0, 'right': 1}], 'holes': [[40, 59, 90, 109]],
})

# Spikes on a smooth sinusoid (outlier removal / despiking).
SPIKES = [[20, 30, 15], [40, 90, -15], [64, 64, 12], [90, 20, -12], [100, 100, 15], [30, 110, -14]]
gauge('sine-spiked', {
    'title': 'Sinusoid with spikes', 'formula': 'z = A·sin(2πx/λ) + Σ spikes',
    'description': 'Smooth unit sinusoid (λ = 32 µm) with six isolated spikes of 12 to 15 µm. The six spikes, and nothing else, are the outliers.',
    'nx': 128, 'ny': 128, 'dx': 1, 'dy': 1, 'parts': [{'type': 'sine', 'A': 1, 'lambda': 32}], 'spikes': SPIKES,
})

# Profiles.
gauge('sine-profile', {
    'title': 'Sinusoidal profile', 'formula': 'z = A·sin(2πx/λ)',
    'description': 'Profile of 10 full periods with P = 64 samples per period. The 5 sampling lengths each hold 2 full periods with sampled crests.',
    'nx': 640, 'ny': 1, 'dx': 0.25, 'dy': 0.25, 'profile': True, 'parts': [{'type': 'sine', **SINE}],
})
gauge('triangle-profile', {
    'title': 'Triangular profile', 'formula': 'z = A·T(x/λ)',
    'description': 'Triangle wave with its kinks on samples, so every forward difference has slope magnitude exactly 4A/λ.',
    'nx': 401, 'ny': 1, 'dx': 0.5, 'dy': 0.5, 'profile': True, 'parts': [{'type': 'triangle', **TRI, 'axes': 'x'}],
})
gauge('bearing-profile', {
    'title': 'Material-ratio profile', 'formula': 'z(k) = h(100·k/(N−1))',
    'description': 'The material-ratio gauge as a profile of N = 20 000 points.',
    'nx': 20000, 'ny': 1, 'dx': 1, 'dy': 1, 'profile': True, 'parts': [{'type': 'bearing', 'knots': KNOTS}],
})
gauge('cos+line-profile', {
    'title': 'Cosine on a line (profile)', 'formula': 'z = a·x + c + A·cos(2π(x−x_c)/λ)',
    'description': 'Centred cosine with 8 full periods, which is orthogonal to every straight line on the sampled grid.',
    'nx': 400, 'ny': 1, 'dx': 0.25, 'dy': 0.25, 'profile': True, 'parts': [{'type': 'plane', 'a': 0.05, 'b': 0, 'c': 1}, {'type': 'cos', 'A': 0.7, 'lambda': 12.5, 'x0': 49.875}],
})
gauge('sine-filter-profile', {
    'title': 'Sinusoidal profile λ = 16 µm', 'formula': 'z = A·sin(2πx/λ)',
    'description': 'Unit sinusoidal profile for the profile Gaussian filter.',
    'nx': 512, 'ny': 1, 'dx': 0.5, 'dy': 0.5, 'profile': True, 'parts': [{'type': 'sine', 'A': 1, 'lambda': 16}],
})

# Spectral / ACF display.
gauge('sine-spectral', {
    'title': 'Sinusoid (spectra)', 'formula': 'z = A·sin(2πx/λ)',
    'description': '16 periods of a unit sinusoid. All of its power sits at f = (±1/λ, 0).',
    'nx': 256, 'ny': 256, 'dx': 1, 'dy': 1, 'parts': [{'type': 'sine', 'A': 1, 'lambda': 16}],
})
gauge('sine-90', {
    'title': 'Sinusoid along y', 'formula': 'z = A·sin(2πy/λ)',
    'description': 'Unit sinusoid whose wave vector points along y (16 whole periods).',
    'nx': 256, 'ny': 256, 'dx': 1, 'dy': 1, 'parts': [{'type': 'sine', 'A': 1, 'lambda': 16, 'theta': 90}],
})
gauge('sine-rot', {
    'title': 'Diagonal sinusoid', 'formula': 'z = A·sin(2π(x·cosθ + y·sinθ)/λ)',
    'description': 'Sinusoid with wave vector (16, 16) cycles per field, i.e. θ = 45° in the array frame (y pointing down the rows), λ = 8√2 µm. The wave vector sits exactly on a DFT bin.',
    'nx': 256, 'ny': 256, 'dx': 1, 'dy': 1, 'parts': [{'type': 'sine', 'A': 1, 'lambda': 8 * SQRT2, 'theta': 45}],
})

# Region volume: two Gaussian bumps (one small) and a Gaussian dimple.
GB = {'type': 'gauss', 'h': 2, 's': 3, 'xc': 25, 'yc': 30}
GS = {'type': 'gauss', 'h': 1, 's': 0.75, 'xc': 60, 'yc': 30}
gauge('gauss-bumps', {
    'title': 'Gaussian bumps', 'formula': 'z = Σ h·exp(−ρ²/2s²)',
    'description': 'A large (h = 2 µm, s = 3 µm) and a small (h = 1 µm, s = 0.75 µm) Gaussian bump. Riemann sums of Gaussians sampled finer than s converge exponentially, so the volumes are exact to round-off.',
    'nx': 161, 'ny': 121, 'dx': 0.5, 'dy': 0.5, 'parts': [GB, GS],
})
gauge('gauss-dimple', {
    'title': 'Gaussian dimple', 'formula': 'z = −h·exp(−ρ²/2s²)',
    'description': 'The large bump inverted, for the concave (void) volume.',
    'nx': 161, 'ny': 121, 'dx': 0.5, 'dy': 0.5, 'parts': [{**GB, 'h': -2}],
})

# Geometry metrology.
GDISC = {'cx': 30.3, 'cy': 29.6, 'r': 18, 'h': 2, 'w': 0.6, 'frame': 'top'}
gauge('boss', {
    'title': 'Circular boss', 'formula': 'z = h/2·(1 − tanh((ρ − R)/w))',
    'description': 'Cylindrical boss with a smooth (tanh) edge whose steepest point lies exactly on ρ = R.',
    'nx': 121, 'ny': 121, 'dx': 0.5, 'dy': 0.5, 'parts': [{'type': 'disc', **GDISC}],
})
GEDGE = {'h': 2, 'w': 0.6, 'x0': 30, 'y0': 30, 'phi': 25}
gauge('edge', {
    'title': 'Straight edge', 'formula': 'z = h/2·(1 + tanh(d/w))',
    'description': 'Straight step edge turned by 25°, with its steepest point on the line d = 0.',
    'nx': 121, 'ny': 121, 'dx': 0.5, 'dy': 0.5, 'parts': [{'type': 'edge', **GEDGE}],
})
SAMPLE_PLANE = {'a': 0.03, 'b': -0.02, 'c': 1}
gauge('plane-sample', {
    'title': 'Tilted plane (sampling)', 'formula': 'z = a·x + b·y + c',
    'description': 'Bilinear interpolation of a plane is exact at any point.',
    'nx': 121, 'ny': 101, 'dx': 0.5, 'dy': 0.25, 'parts': [{'type': 'plane', **SAMPLE_PLANE}],
})

# Cavity array.
POCKETS = [{'cx': cx, 'cy': cy, 'w': 40, 'h': 30, 'depth': 5} for cy in (50.5, 110.5) for cx in (50.5, 110.5, 170.5)]
gauge('pockets', {
    'title': 'Pocket array', 'formula': 'z = −d inside 40 × 30 µm rectangles',
    'description': 'Six flat-bottomed rectangular pockets of depth 5 µm in a flat top surface. The edges lie halfway between pixel centres, so the 50 %-depth edge positions are exact.',
    'nx': 221, 'ny': 161, 'dx': 1, 'dy': 1, 'parts': [{'type': 'pockets', 'pockets': POCKETS}],
})

gauge('pockets-dither', {
    **G['pockets'], 'title': 'Pocket array (dithered)', 'formula': 'pocket array + 0.1 nm deterministic texture',
    'description': 'The pocket array with a 0.1 nm noise-like texture (incommensurate sinusoids), as any real measurement has: the cavity detector estimates its threshold from the top-surface noise.',
    'parts': [{'type': 'pockets', 'pockets': POCKETS}, {'type': 'dither', 'amp': 1e-4}],
})

# ── Gauges after Todhunter et al. (2018–2020) ─────────────────────────────────
# The "mathematical reference" papers of L. Todhunter, R. Leach, S. Lawes, P. Harris and
# F. Blateyron (J. Phys. Conf. Ser. 1065 082004; Surf. Topogr. Metrol. Prop. 7 015020,
# 8 015010, 8 045017, 8 045019) define their references on continuous surfaces. Here the
# same surfaces are laid out so that the reference also holds for the sampled grid.

# Smooth cubic material-ratio curve (2019, eq. 1, Z = 1 µm), in raster order and in two
# reorderings of the same heights (2019, §5.1: the parameters must not depend on the order).
TOD_CUBIC = [1, -2.7, 5.1, -3.4]
TOD_SCRAMBLE = 7919  # prime, coprime with N = 20 000
TOD_CUBIC_DESC = 'The smooth material-ratio curve of Todhunter et al. (2019, eq. 1) with Z = 1 µm, laid out as a ramp of N = 20 000 heights so that the sampled curve is exact. Unlike the curve of a sinusoid its flattest 40 % window is unique (30–70 %), so the whole ISO 13565-2 construction is defined. The closed-form values reproduce the mathematical values published in that paper (Table 1) to every printed digit.'
gauge('bearing-cubic', {
    'title': 'Cubic material-ratio gauge', 'formula': 'z(k) = h(100·k/(N−1)), h = Z/10·(10 − 27u + 51u² − 34u³), u = mr/100',
    'description': TOD_CUBIC_DESC, 'nx': 200, 'ny': 100, 'dx': 1, 'dy': 1, 'parts': [{'type': 'bearingPoly', 'coefs': TOD_CUBIC}],
})
gauge('bearing-cubic-serpentine', {
    'title': 'Cubic material-ratio gauge, alternate rows reversed', 'formula': 'the cubic gauge with every odd row reversed',
    'description': 'The heights of the cubic material-ratio gauge with every second row reversed (Todhunter et al. 2019, §5.1). The height distribution, and so every material-ratio parameter, is unchanged.',
    'nx': 200, 'ny': 100, 'dx': 1, 'dy': 1, 'parts': [{'type': 'bearingPoly', 'coefs': TOD_CUBIC, 'order': 'serpentine'}],
})
gauge('bearing-cubic-scramble', {
    'title': 'Cubic material-ratio gauge, scrambled', 'formula': f'pixel k carries ramp sample {TOD_SCRAMBLE}·k mod N',
    'description': f'The heights of the cubic material-ratio gauge in a scrambled order (pixel k carries ramp sample {TOD_SCRAMBLE}·k mod N, a permutation because {TOD_SCRAMBLE} and N are coprime), the counterpart of the random order in Todhunter et al. (2019, §5.1). No two neighbours are close in height, yet every material-ratio parameter is unchanged.',
    'nx': 200, 'ny': 100, 'dx': 1, 'dy': 1, 'parts': [{'type': 'bearingPoly', 'coefs': TOD_CUBIC, 'order': 'scramble', 'mult': TOD_SCRAMBLE}],
})

# Parametric planes z = A·x + B·y (2020 field-parameter paper, §4.4–4.5, the five (A, B)
# pairs of its figure 11), on a grid with Δx = 2Δy.
TOD_PLANES = [[4, 3], [1, 2], [2, 6], [0.5, 0.125], [0.1, 0.05]]


def tod_plane_id(AB):
    return f'plane-A{num(AB[0])}-B{num(AB[1])}'


for A, B in TOD_PLANES:
    gauge(tod_plane_id([A, B]), {
        'title': f'Plane A = {num(A)}, B = {num(B)}', 'formula': 'z = A·x + B·y',
        'description': f'Parametric plane of Todhunter et al. (2020, §4.4) with slopes A = {num(A)}, B = {num(B)}. Steep planes catch area algorithms that assume small slopes. Its area ratio is √(1 + A² + B²) − 1; the reference given in that paper, √(1 + A²)·√(1 + B²) − 1 (eq. 35), is the area of a rectangle and holds only when A·B = 0 (here it reads {precision(100 * (math.sqrt(1 + A * A) * math.sqrt(1 + B * B) - 1), 5)} % instead of {precision(100 * (math.sqrt(1 + A * A + B * B) - 1), 5)} %).',
        'nx': 81, 'ny': 61, 'dx': 0.5, 'dy': 0.25, 'parts': [{'type': 'plane', 'a': A, 'b': B, 'c': 0}],
    })

# Oblique Fourier-series surface for Gaussian filtering (2018, eq. 1; 2020 filtration
# paper): every term is an integer number of cycles over the field, one of them close
# to the steep flank of the transmission curve (λ/λc ≈ 0.37).
FOURIER_GRID = {'nx': 512, 'ny': 256, 'dx': 0.5, 'dy': 0.5}
FOURIER_TERMS = [[16, 0, 1, 0.3], [8, 8, 0.5, 1.1], [32, -8, 0.7, 2.0], [80, 16, 0.3, 0.5], [4, 2, 1.2, -0.7]]  # [cycles x, cycles y, A, phase]


def fourier_term(t, gain=lambda lam: 1):
    m, n, A, phase = t
    fx, fy = m / (FOURIER_GRID['nx'] * FOURIER_GRID['dx']), n / (FOURIER_GRID['ny'] * FOURIER_GRID['dy'])
    lam = 1 / math.hypot(fx, fy)
    return {'type': 'sine', 'A': A * gain(lam), 'lambda': lam, 'theta': (math.atan2(fy, fx) * 180) / math.pi, 'phase': phase}


gauge('fourier-filter', {
    'title': 'Oblique Fourier series', 'formula': 'z = Σ Aᵢ·sin(2π(mᵢx/Lx + nᵢy/Ly) + φᵢ)',
    'description': f"Five oblique sinusoids with integer numbers of cycles over the field (Todhunter et al. 2018, eq. 1), wavelengths {', '.join(fixed(fourier_term(t)['lambda'], 2) for t in FOURIER_TERMS)} µm. A Gaussian filter must scale every term by its own transmission, which also tests the phase, the separable x/y implementation and the absence of cross-talk between terms.",
    **FOURIER_GRID, 'parts': [fourier_term(t) for t in FOURIER_TERMS],
})
# Egg-crate whose extrema sit half a pixel outside the border: it is symmetric under the
# half-sample reflection used to extend the surface (scipy mode 'reflect'), so the
# reflect-extended filter output is T·z right up to the edge.
BORDER_EGG = {'A': 1, 'lambdaX': 16, 'lambdaY': 12.8, 'x0': -0.25, 'y0': -0.25}
gauge('egg-filter-border', {
    'title': 'Reflection-symmetric egg-crate', 'formula': 'z = A·cos(2π(x+Δx/2)/λx)·cos(2π(y+Δy/2)/λy)',
    'description': 'Egg-crate with whole half-periods across the field (16 along x, 10 along y) and its extrema half a pixel outside every border. It is unchanged by the half-sample reflection at each border, so a Gaussian filter with reflect extension must return T(λx)·T(λy)·z everywhere, including at the border, where Todhunter et al. (2020) found the largest software deviations.',
    'nx': 256, 'ny': 128, 'dx': 0.5, 'dy': 0.5, 'parts': [{'type': 'eggcrate', **BORDER_EGG}],
})

# Material-ratio softgauges of Blateyron & Breton (Surf. Topogr. Metrol. Prop. 12 035036,
# 2024, annex A.2), written there as depth z(x), x = mr/100, z = 0 at the top and 1 at the
# bottom; here h = 1 − z (Z = 1 µm), laid out as ramps like the other material-ratio gauges.
# Abbott_01 is the cubic of Todhunter et al. (gauge 'bearing-cubic'); Abbott_22 (za = 0.5 >
# zb = 0.4) is not monotonic, so not a material-ratio curve, and is left out.
BB_POLY = {'02': [0.90, -1.10, 1.20], '03': [1.85, -2.56, 1.71], '04': [1.55, -3.95, 3.40], '05': [3.84, -6.24, 3.40], '06': [1.50, -1.50, 1.00],
           '07': [1.00, -1.50, 1.50], '08': [2.00, -3.50, 2.50], '09': [2.50, -3.50, 2.00], '10': [0.80, -2.00, 2.20], '11': [0.60, -2.00, 2.40], '12': [3.90, -6.00, 3.10],
           '13': [2.40, -2.00, 0.60], '14': [0, 0, 1], '15': [2.70, -4.50, 2.80], '16': [3.20, -5.50, 3.30]}  # z = αx + βx² + γx³
BB_SEG = {17: [0.1, 0.4, 0.8, 0.6], 18: [0.2, 0.5, 0.8, 0.6], 19: [0.3, 0.5, 0.8, 0.7], 20: [0.3, 0.5, 0.9, 0.6], 21: [0.2, 0.1, 0.8, 0.3],
          23: [0.5, 0.3, 0.9, 0.5], 24: [0.45, 0.4, 0.9, 0.5], 25: [0, 0, 1, 1]}  # (xa, za), (xb, zb)


def bb_knots(sg):
    xa, za, xb, zb = sg
    a = [[0, 1], [100 * xa, 1 - za], [100 * xb, 1 - zb], [100, 0]]
    return [k for i, k in enumerate(a) if i == 0 or k[0] > a[i - 1][0]]


def _JS_ORDER(d):  # the order of the former JS generator (Object.entries: integer keys first)
    return sorted(d.items(), key=lambda kv: (0, int(kv[0])) if kv[0].isdigit() and str(int(kv[0])) == kv[0] else (1, 0))


BB_DESC = 'Material-ratio softgauge of Blateyron & Breton (2024), written there as a depth z(x) below the highest point, x = mr/100; here h = 1 − z (Z = 1 µm), laid out as a ramp of N = 20 000 heights so the sampled curve is exact.'
for k, (a, b, g) in _JS_ORDER(BB_POLY):
    degenerate = ' Degenerate: the inflection lies at 0 % and the slope vanishes there.' if k == '14' else ''
    gauge(f'bb-abbott-{k}', {
        'title': f'Abbott_{k} (polynomial)', 'formula': f"z = {num(a)}x {'−' if b < 0 else '+'} {num(abs(b))}x² + {num(g)}x³, h = 1 − z",
        'description': f'{BB_DESC} Polynomial curve (α, β, γ) = ({num(a)}, {num(b)}, {num(g)}), inflection at mr = {fixed(-100 * b / (3 * g), 1)} %.{degenerate}',
        'nx': 200, 'ny': 100, 'dx': 1, 'dy': 1, 'parts': [{'type': 'bearingPoly', 'coefs': [1, -a, -b, -g]}],
    })
for k, sg in BB_SEG.items():
    extra = (' Degenerate: a single straight line, so the core spans the whole curve and there is no peak or valley zone.' if k == 25
             else ' The core segment spans exactly 40 %.' if k == 23 else '')
    gauge(f'bb-abbott-{k}', {
        'title': f'Abbott_{k} (segmented)', 'formula': f'z piecewise linear through (0, 0), ({num(sg[0])}, {num(sg[1])}), ({num(sg[2])}, {num(sg[3])}), (1, 1); h = 1 − z',
        'description': f'{BB_DESC} Three straight segments with control points (xa, za) = ({num(sg[0])}, {num(sg[1])}) and (xb, zb) = ({num(sg[2])}, {num(sg[3])}).{extra}',
        'nx': 200, 'ny': 100, 'dx': 1, 'dy': 1, 'parts': [{'type': 'bearing', 'knots': bb_knots(sg)}],
    })

# Convergence series gauges (sampling density P = samples per period).
SERIES_P = [8, 16, 32, 64, 128]
for P in SERIES_P:
    gauge(f'conv-sine-{P}', {
        'title': f'Sinusoid, P = {P}', 'formula': 'z = A·sin(2πx/λ)', 'description': f'Sinusoid with {P} samples per period (8 periods).',
        'nx': 8 * P, 'ny': 4, 'dx': 1, 'dy': 1, 'parts': [{'type': 'sine', 'A': 1, 'lambda': P}],
    })
    gauge(f'conv-hyb-{P}', {
        'title': f'Sinusoid (gradients), P = {P}', 'formula': 'z = A·sin(2πx/λ), A·k = 1/2', 'description': f'Sinusoid with slope amplitude A·k = 0.5 and {P} samples per period (8 whole periods of cells).',
        'nx': 8 * P + 1, 'ny': 3, 'dx': 1, 'dy': 1, 'parts': [{'type': 'sine', 'A': (0.5 * P) / TAU, 'lambda': P}],
    })
SERIES_SPC = [6, 10, 18, 34, 66]  # P/2 odd: saddle lines between pixels
for P in SERIES_SPC:
    h = P // 2
    x0 = (h - 1) / 2
    gauge(f'conv-egg-{P}', {
        'title': f'Egg-crate, P = {P}', 'formula': 'z = A·cos(2πx/λ)·cos(2πy/λ)', 'description': f'Egg-crate, 6 × 6 complete half-period cells, {P} samples per period.',
        'nx': 6 * h, 'ny': 6 * h, 'dx': 1, 'dy': 1, 'parts': [{'type': 'eggcrate', 'A': 1, 'lambdaX': P, 'lambdaY': P, 'x0': x0, 'y0': x0}],
    })

# ── cases ─────────────────────────────────────────────────────────────────────

CASES: list[dict] = []


def add(c: dict) -> None:
    r = c['reference']
    d = {'ops': [], 'unit': '', **c}
    d['reference'] = {'value': r} if isinstance(r, (int, float)) else r
    CASES.append(d)


def param(name, *args):
    return {'kind': 'param', 'name': name, 'args': list(args)}


def ref(value, formula):
    return {'value': value, 'formula': formula}


# ── 1. Height parameters ──────────────────────────────────────────────────────
def _height():
    g = G['plane']
    hx, hy = PL['a'] * g['dx'], PL['b'] * g['dy']
    ux, uy = discrete_uniform_moments(g['nx'], hx), discrete_uniform_moments(g['ny'], hy)
    sq = math.sqrt(ux['v'] + uy['v'])
    sz = abs(PL['a']) * (g['nx'] - 1) * g['dx'] + abs(PL['b']) * (g['ny'] - 1) * g['dy']
    sku = (ux['mu4'] + uy['mu4'] + 6 * ux['v'] * uy['v']) / (ux['v'] + uy['v']) ** 2
    grp, std = 'Height parameters', 'ISO 25178-2:2021'
    add({'id': 'height.plane.Sq', 'group': grp, 'gauge': 'plane', 'measure': param('Sq'), 'unit': 'µm', 'standard': std, 'reference': ref(sq, '√(a²Δx²(nx²−1) + b²Δy²(ny²−1))/√12'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'height.plane.Sz', 'group': grp, 'gauge': 'plane', 'measure': param('Sz'), 'unit': 'µm', 'standard': std, 'reference': ref(sz, '|a|·W + |b|·H'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'height.plane.Sp', 'group': grp, 'gauge': 'plane', 'measure': param('Sp'), 'unit': 'µm', 'standard': std, 'reference': ref(sz / 2, '(|a|·W + |b|·H)/2'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'height.plane.Sv', 'group': grp, 'gauge': 'plane', 'measure': param('Sv'), 'unit': 'µm', 'standard': std, 'reference': ref(sz / 2, '(|a|·W + |b|·H)/2'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'height.plane.Ssk', 'group': grp, 'gauge': 'plane', 'measure': param('Ssk'), 'standard': std, 'reference': ref(0, '0 (symmetric distribution)'), 'tol': {'abs': 1e-9}, 'basis': FP_BASIS})
    add({'id': 'height.plane.Sku', 'group': grp, 'gauge': 'plane', 'measure': param('Sku'), 'standard': std, 'reference': ref(sku, '(μ₄ₓ + μ₄ᵧ + 6σₓ²σᵧ²)/(σₓ² + σᵧ²)², discrete uniform moments'), 'tol': FP(), 'basis': FP_BASIS})

    b = bernoulli_heights(STEP['d'], STEP['h'])
    formulas = {'Sa': '2d(1−d)h', 'Sq': 'h√(d(1−d))', 'Sp': '(1−d)h', 'Sv': 'd·h', 'Sz': 'h', 'Ssk': '(1−2d)/√(d(1−d))', 'Sku': '(1−3d(1−d))/(d(1−d))'}
    for k in ['Sa', 'Sq', 'Sp', 'Sv', 'Sz', 'Ssk', 'Sku']:
        add({'id': f'height.step.{k}', 'group': grp, 'gauge': 'step', 'measure': param(k), 'unit': '' if k in ('Ssk', 'Sku') else 'µm', 'standard': std, 'reference': ref(b[k], formulas[k]), 'tol': FP(), 'basis': FP_BASIS})

    A, P = SINE['A'], SINE_P
    add({'id': 'height.sine.Sa', 'group': grp, 'gauge': 'sine', 'measure': param('Sa'), 'unit': 'µm', 'standard': std,
         'reference': ref((2 * A) / math.pi, '2A/π'), 'discrete': ref((2 * A) / P / math.tan(math.pi / P), '(2A/P)·cot(π/P)'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'height.sine.Sq', 'group': grp, 'gauge': 'sine', 'measure': param('Sq'), 'unit': 'µm', 'standard': std, 'reference': ref(A / SQRT2, 'A/√2'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'height.sine.Sp', 'group': grp, 'gauge': 'sine', 'measure': param('Sp'), 'unit': 'µm', 'standard': std, 'reference': ref(A, 'A'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'height.sine.Sv', 'group': grp, 'gauge': 'sine', 'measure': param('Sv'), 'unit': 'µm', 'standard': std, 'reference': ref(A, 'A'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'height.sine.Sz', 'group': grp, 'gauge': 'sine', 'measure': param('Sz'), 'unit': 'µm', 'standard': std, 'reference': ref(2 * A, '2A'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'height.sine.Ssk', 'group': grp, 'gauge': 'sine', 'measure': param('Ssk'), 'standard': std, 'reference': ref(0, '0'), 'tol': {'abs': 1e-9}, 'basis': FP_BASIS})
    add({'id': 'height.sine.Sku', 'group': grp, 'gauge': 'sine', 'measure': param('Sku'), 'standard': std, 'reference': ref(1.5, '3/2'), 'tol': FP(), 'basis': FP_BASIS})

    add({'id': 'height.egg.Sq', 'group': grp, 'gauge': 'egg', 'measure': param('Sq'), 'unit': 'µm', 'standard': std, 'reference': ref(EGG['A'] / 2, 'A/2'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'height.egg.Ssk', 'group': grp, 'gauge': 'egg', 'measure': param('Ssk'), 'standard': std, 'reference': ref(0, '0'), 'tol': {'abs': 1e-9}, 'basis': FP_BASIS})
    add({'id': 'height.egg.Sku', 'group': grp, 'gauge': 'egg', 'measure': param('Sku'), 'standard': std, 'reference': ref(9 / 4, '(3/8)²/(1/4)² = 9/4'), 'tol': FP(), 'basis': FP_BASIS})


_height()


# ── 2. Hybrid parameters ──────────────────────────────────────────────────────
def _hybrid():
    grp, std = 'Hybrid parameters', 'ISO 25178-2:2021'
    gr = math.hypot(PL['a'], PL['b'])
    add({'id': 'hybrid.plane.Sdq', 'group': grp, 'gauge': 'plane', 'measure': param('Sdq'), 'standard': std, 'reference': ref(gr, '√(a² + b²)'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'hybrid.plane.Sdr', 'group': grp, 'gauge': 'plane', 'measure': param('Sdr'), 'unit': '%', 'standard': std, 'reference': ref(100 * (math.sqrt(1 + gr * gr) - 1), '100·(√(1 + a² + b²) − 1)'), 'tol': FP(), 'basis': FP_BASIS})
    s = (4 * TRI['A']) / TRI['lambda']
    add({'id': 'hybrid.pyramids.Sdq', 'group': grp, 'gauge': 'pyramids', 'measure': param('Sdq'), 'standard': std, 'reference': ref(s * SQRT2, '√2·4A/λ'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'hybrid.pyramids.Sdr', 'group': grp, 'gauge': 'pyramids', 'measure': param('Sdr'), 'unit': '%', 'standard': std, 'reference': ref(100 * (math.sqrt(1 + 2 * s * s) - 1), '100·(√(1 + 2(4A/λ)²) − 1)'), 'tol': FP(), 'basis': FP_BASIS})
    # parametric planes after Todhunter et al. (2020): steep slopes, every cell planar
    for A, B in TOD_PLANES:
        gid = tod_plane_id([A, B])
        note = 'Gauge after Todhunter et al. (2020), §4.4. Their eq. (35), √(1 + A²)·√(1 + B²) − 1, is not used: it is the area of a rectangle, while the plane element spanned by (1, 0, A) and (0, 1, B) is a parallelogram of area √(1 + A² + B²).'
        add({'id': f'hybrid.{gid}.Sdq', 'group': grp, 'gauge': gid, 'measure': param('Sdq'), 'standard': std, 'reference': ref(math.hypot(A, B), '√(A² + B²)'), 'tol': FP(), 'basis': FP_BASIS})
        add({'id': f'hybrid.{gid}.Sdr', 'group': grp, 'gauge': gid, 'measure': param('Sdr'), 'unit': '%', 'standard': std, 'note': note,
             'reference': ref(100 * (math.sqrt(1 + A * A + B * B) - 1), '100·(√(1 + A² + B²) − 1)'), 'tol': FP(), 'basis': FP_BASIS})
    # saddle: twisted cells (ISO 25178-2 D.15 triangulation)
    g, k = G['saddle'], SAD['k']
    W, H = (g['nx'] - 1) * g['dx'], (g['ny'] - 1) * g['dy']

    def msq(n, d):
        t = 0.0
        for i in range(n):
            t += (i * d) ** 2
        return t / n

    add({'id': 'hybrid.saddle.Sdq', 'group': grp, 'gauge': 'saddle', 'measure': param('Sdq'), 'standard': std,
         'reference': ref(k * math.sqrt((W * W + H * H) / 3), 'k·√((W² + H²)/3)'),
         'discrete': ref(k * math.sqrt(msq(g['ny'], g['dy']) + msq(g['nx'], g['dx'])), 'k·√(⟨y²⟩ + ⟨x²⟩) over the rows / columns (forward differences are exact)'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'hybrid.saddle.Sdr', 'group': grp, 'gauge': 'saddle', 'measure': param('Sdr'), 'unit': '%', 'standard': std,
         'note': 'ISO 25178-2:2021 D.3: each cell is triangulated along both diagonals and the two areas are averaged (eq. D.15). Taking a single triangle per cell is exact only for planar cells.',
         'reference': ref(100 * saddle_area_ratio(k, W, H), '100·[(1/(W·H))·∬ √(1 + k²(x² + y²)) dx dy − 1]'),
         'discrete': ref(100 * sampled_saddle_area_ratio(k, g['nx'], g['ny'], g['dx'], g['dy']), '100·[trapezoid-weighted mean of √(1 + k²(x² + y²)) over the grid points − 1]'), 'tol': FP(), 'basis': FP_BASIS})
    # smooth sinusoid: continuous vs sampled closed forms
    g, A, k = G['sine-hybrid'], 1, TAU / 16
    a = A * k
    ap = (2 * A * math.sin(math.pi / HYB_P)) / g['dx']
    add({'id': 'hybrid.sine.Sdq', 'group': grp, 'gauge': 'sine-hybrid', 'measure': param('Sdq'), 'standard': std,
         'reference': ref(a / SQRT2, 'A·k/√2'), 'discrete': ref(ap / SQRT2, 'a′/√2, a′ = 2A·sin(π/P)/Δx'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'hybrid.sine.Sdr', 'group': grp, 'gauge': 'sine-hybrid', 'measure': param('Sdr'), 'unit': '%', 'standard': std,
         'reference': ref(100 * sine_area_ratio(a), '100·[(2/π)√(1+a²)·E(a²/(1+a²)) − 1], a = A·k'),
         'discrete': ref(100 * sampled_sine_area_ratio(ap, HYB_P), '100·[(1/P)·Σᵢ √(1 + a′²cos²(2π(i+½)/P)) − 1], a′ = 2A·sin(π/P)/Δx'), 'tol': {'rel': 1e-9, 'abs': 1e-11}, 'basis': FP_BASIS})


_hybrid()

# ── 3./4. Functional + volume parameters (material ratio) ────────────────────
ABBOTT_NAMES = ['Sk', 'Spk', 'Svk', 'Spkx', 'Svkx', 'Sak1', 'Sak2', 'Smr1', 'Smr2', 'Sxp', 'Sdc', 'Vmp', 'Vmc', 'Vvv', 'Vvc']
VOLUME = {'Vmp', 'Vmc', 'Vvv', 'Vvc'}
ABBOTT_UNITS = {'Smr1': '%', 'Smr2': '%', 'Sak1': '%·µm', 'Sak2': '%·µm'}
ABBOTT_FORMULA = {
    'Sk': 'y_upper − y_lower of the least-squares line in the flattest 40 % window',
    'Spk': '2·A₁/Smr1', 'Svk': '2·A₂/(100 − Smr2)', 'Spkx': 'h(0) − y_upper', 'Svkx': 'y_lower − h(100)',
    'Sak1': 'A₁ = ∫₀^Smr1 (h − y_upper) dmr', 'Sak2': 'A₂ = ∫_Smr2^100 (y_lower − h) dmr',
    'Smr1': 'h⁻¹(y_upper)', 'Smr2': 'h⁻¹(y_lower)', 'Sxp': 'h(2.5 %) − h(50 %)', 'Sdc': 'h(2.5 %) − h(50 %)',
    'Vmp': '∫₀^10 (h − h(10)) dmr / 100', 'Vmc': 'Vm(80) − Vm(10)', 'Vvv': '∫_80^100 (h(80) − h) dmr / 100', 'Vvc': 'Vv(10) − Vv(80)',
}
# Tolerances follow from the implementation's material-ratio resolution: classes of
# Δ = 100/9999 % (10 000 samples of the curve). Allowing two classes, a height read at
# material ratio p is uncertain by 2Δ·|h′(p)|, a material ratio by 2Δ, and an area or
# volume by the strip it may gain or lose at its limit.
DMR = 100 / 9999


def abbott_tol(curve, name):
    d = 2 * DMR

    def sl(m):
        e = 1e-6
        a, b = max(0, m - e), min(100, m + e)
        return max(abs((curve.h(m) - curve.h(a)) / max(m - a, 1e-300)), abs((curve.h(b) - curve.h(m)) / max(b - m, 1e-300)))

    core, mr1, mr2 = abs(curve.slope), curve.mr1, curve.mr2
    # A₁ moves with its limit Smr1 (strip mr1·|h′|·d) and with y_upper (mr1·|β|·d).
    tA1, tA2 = d * mr1 * (sl(mr1) + core), d * (100 - mr2) * (sl(mr2) + core)

    def vm(p):
        return d * p * sl(p) / 100

    def vv(p):
        return d * (100 - p) * sl(p) / 100

    t = {
        'Sk': lambda: d * core * 100, 'Spkx': lambda: d * core * 100, 'Svkx': lambda: d * core * 100,
        'Sxp': lambda: d * (sl(2.5) + sl(50)), 'Sdc': lambda: d * (sl(2.5) + sl(50)),
        'Smr1': lambda: d + d * core * 100 / max(sl(mr1), 1e-12), 'Smr2': lambda: d + d * core * 100 / max(sl(mr2), 1e-12),
        'Sak1': lambda: tA1, 'Sak2': lambda: tA2,
        'Spk': lambda: 2 * tA1 / mr1 + (2 * curve.A1 / mr1 ** 2) * d if mr1 > 0 else 0,
        'Svk': lambda: 2 * tA2 / (100 - mr2) + (2 * curve.A2 / (100 - mr2) ** 2) * d if mr2 < 100 else 0,
        'Vmp': lambda: vm(10), 'Vmc': lambda: vm(10) + vm(80), 'Vvv': lambda: vv(80), 'Vvc': lambda: vv(10) + vv(80),
    }[name]()
    return {'tol': {'abs': t + 1e-12}, 'basis': 'material-ratio resolution of the implementation: two classes of 0.01 %, propagated through the local slope of the curve'}


def abbott_cases(curve, gauge_id, prefix, profile, tol_of, names=ABBOTT_NAMES, group=None):
    vals = curve.all()
    for name in names:
        # Spk = 2A₁/Smr1 (Svk alike) is 0/0 when the curve never rises above the core line
        # (e.g. a sinusoid): undefined by ISO 13565-2, so not a valid test value.
        if not math.isfinite(vals[name]):
            continue
        vol = name in VOLUME
        pname = 'R' + name[1:] if profile and not vol else name
        add({
            'id': f'{prefix}.{pname}', 'group': group or ('Volume parameters' if vol else 'Functional parameters'), 'gauge': gauge_id,
            'measure': param(pname), 'unit': ABBOTT_UNITS.get(name, 'µm'),
            'standard': 'ISO 25178-2:2021' if vol else ('ISO 13565-2:1996 / ISO 21920-2:2021' if profile else 'ISO 25178-2:2021 / ISO 13565-2'),
            'reference': ref(vals[name], ABBOTT_FORMULA[name]), **(tol_of(name, vals[name]) if tol_of else abbott_tol(curve, name)),
        })


def _abbott():
    # Piecewise-linear curve: exact except for the implementation's own material-ratio
    # resolution (classes of Δmr = 0.01 %), which bounds every derived quantity by the
    # curve slope over one class.
    curve = piecewise_linear_curve(KNOTS)
    abbott_cases(curve, 'bearing', 'abbott.pl', False, None)
    add({'id': 'abbott.pl.Smr(0)', 'group': 'Functional parameters', 'gauge': 'bearing', 'measure': param('Smr', 0), 'unit': '%', 'standard': 'ISO 25178-2:2021', 'reference': ref(curve.mr(0), 'h⁻¹(0) = 45 %'), 'tol': {'abs': 2 * DMR}, 'basis': 'two material-ratio classes'})
    add({'id': 'abbott.pl.Smc(30)', 'group': 'Functional parameters', 'gauge': 'bearing', 'measure': param('Smc', 30), 'unit': 'µm', 'standard': 'ISO 25178-2:2021', 'reference': ref(curve.mc(30), 'h(30 %)'), 'tol': {'abs': 2 * DMR * abs(curve.slope)}, 'basis': 'two classes × local slope'})
    for nm, v, f in [['Smrk1', curve.mr1, 'h⁻¹(y_upper) (ISO 25178-2:2021 name of Smr1)'], ['Smrk2', curve.mr2, 'h⁻¹(y_lower) (ISO 25178-2:2021 name of Smr2)']]:
        add({'id': f'abbott.pl.{nm}', 'group': 'Functional parameters', 'gauge': 'bearing', 'measure': param(nm), 'unit': '%', 'standard': 'ISO 25178-2:2021', 'reference': ref(v, f), **abbott_tol(curve, 'Smr1' if nm == 'Smrk1' else 'Smr2')})
    abbott_cases(curve, 'bearing-profile', 'abbott.profile', True, None)
    add({'id': 'abbott.profile.Rmr(0)', 'group': 'Functional parameters', 'gauge': 'bearing-profile', 'measure': param('Rmr', 0), 'unit': '%', 'standard': 'ISO 21920-2:2021', 'reference': ref(curve.mr(0), 'h⁻¹(0) = 45 %'), 'tol': {'abs': 2 * DMR}, 'basis': 'two material-ratio classes'})
    add({'id': 'abbott.profile.Rmc(30)', 'group': 'Functional parameters', 'gauge': 'bearing-profile', 'measure': param('Rmc', 30), 'unit': 'µm', 'standard': 'ISO 21920-2:2021', 'reference': ref(curve.mc(30), 'h(30 %)'), 'tol': {'abs': 2 * DMR * abs(curve.slope)}, 'basis': 'two classes × local slope'})

    # Sinusoid: h(mr) = A·cos(π·mr/100). The cosine curve is flattest at BOTH ends: the
    # 0–40 % and 60–100 % windows tie exactly, so which one is used is decided by
    # round-off. Only quantities that do not depend on that choice are valid references.
    sc = sine_curve(1)
    abbott_cases(sc, 'sine-abbott', 'abbott.sine', False, None, ['Sk', 'Sxp', 'Sdc', 'Vmp', 'Vmc', 'Vvv', 'Vvc'])

    # Smooth cubic curve of Todhunter et al. (2019): the full construction, then the same
    # heights in two other orders, which must give the same parameters.
    cc = polynomial_curve(TOD_CUBIC)
    abbott_cases(cc, 'bearing-cubic', 'abbott.cubic', False, None)
    order_names = ['Sk', 'Spk', 'Svk', 'Smr1', 'Smr2', 'Vmp', 'Vmc', 'Vvv', 'Vvc']
    abbott_cases(cc, 'bearing-cubic-serpentine', 'abbott.cubic-serpentine', False, None, order_names)
    abbott_cases(cc, 'bearing-cubic-scramble', 'abbott.cubic-scramble', False, None, order_names)

    # The softgauge families of Blateyron & Breton (2024): Rk construction + volume parameters.
    bb_names = ['Sk', 'Spk', 'Svk', 'Spkx', 'Svkx', 'Sak1', 'Sak2', 'Smr1', 'Smr2', 'Vmp', 'Vmc', 'Vvv', 'Vvc']
    bb_group = 'Material-ratio softgauges (Blateyron & Breton)'
    for k, (a, b, g) in _JS_ORDER(BB_POLY):
        abbott_cases(polynomial_curve([1, -a, -b, -g]), f'bb-abbott-{k}', f'bb.abbott-{k}', False, None, bb_names, bb_group)
    for k, sg in BB_SEG.items():
        abbott_cases(piecewise_linear_curve(bb_knots(sg)), f'bb-abbott-{k}', f'bb.abbott-{k}', False, None, bb_names, bb_group)


_abbott()


# ── 5. Spatial parameters ─────────────────────────────────────────────────────
def _spatial():
    grp, std = 'Spatial parameters', 'ISO 25178-2:2021'

    # Bias of the unbiased finite-field ACF estimator for a sinusoid of k rad/px over
    # M overlapping samples: |ΔR| ≤ 1/(M·|sin k|), mapped to lag through |dR/dτ|; plus
    # the cubic-spline interpolation error 5/384·k⁴.
    def sal_bound(kpx, M, tau0px, step_um):
        dR = 1 / (M * abs(math.sin(kpx))) + (5 / 384) * kpx ** 4
        return (dR / (kpx * abs(math.sin(kpx * tau0px)))) * step_um

    L, kpx = 16, TAU / 16
    tau0 = (L * ACOS02) / TAU
    bS = sal_bound(kpx, 1024 - tau0, tau0, 1)
    add({'id': 'spatial.sine.Sal', 'group': grp, 'gauge': 'sine-acf', 'measure': param('Sal'), 'unit': 'µm', 'standard': std,
         'reference': ref(tau0, 'λ·arccos(0.2)/(2π)'), 'tol': {'abs': bS}, 'basis': f'finite-field ACF bias ≤ 1/(M·|sin k|) + spline error, mapped to lag: {exponential(bS, 2)} µm'})
    add({'id': 'spatial.sine.Str', 'group': grp, 'gauge': 'sine-acf', 'measure': param('Str'), 'standard': std,
         'reference': {'value': None, 'formula': 'undefined: the ACF does not decay along the crests'}, 'expectNaN': True, 'tol': {}, 'basis': 'must be reported as undefined (NaN)'})

    iso = acf_decay_lengths(lambda tx, ty: math.cos(TAU * tx / 16) * math.cos(TAU * ty / 16), 0.2, 8, phi_max=math.pi / 2, n_phi=900)
    bE = 2 * sal_bound(kpx, 512 - tau0, tau0, 1)
    add({'id': 'spatial.egg.Sal', 'group': grp, 'gauge': 'egg-acf', 'measure': param('Sal'), 'unit': 'µm', 'standard': std,
         'reference': ref(iso['shortest'], 'min over φ of the first 0.2 crossing of cos(kτcosφ)·cos(kτsinφ)'), 'tol': {'abs': bE}, 'basis': f'two-axis finite-field ACF bias + spline error: {exponential(bE, 2)} µm'})
    add({'id': 'spatial.egg.Str', 'group': grp, 'gauge': 'egg-acf', 'measure': param('Str'), 'standard': std,
         'reference': ref(iso['shortest'] / iso['longest'], 'Sal / longest 0.2 decay length (diagonal)'), 'tol': {'abs': (bE / iso['shortest'] + bE / iso['longest']) * (iso['shortest'] / iso['longest'])}, 'basis': 'propagated decay-length bound'})

    an = acf_decay_lengths(lambda tx, ty: math.cos(TAU * tx / 16) * math.cos(TAU * ty / 24), 0.2, 12, phi_max=math.pi / 2, n_phi=900)
    bA = 2 * sal_bound(kpx, 512 - tau0, tau0, 1.5)
    add({'id': 'spatial.egg-aniso.Sal', 'group': grp, 'gauge': 'egg-acf-aniso', 'measure': param('Sal'), 'unit': 'µm', 'standard': std,
         'reference': ref(an['shortest'], 'λx·arccos(0.2)/(2π) (fastest decay along x)'), 'tol': {'abs': bA}, 'basis': f'two-axis finite-field ACF bias + spline error: {exponential(bA, 2)} µm'})
    add({'id': 'spatial.egg-aniso.Str', 'group': grp, 'gauge': 'egg-acf-aniso', 'measure': param('Str'), 'standard': std,
         'reference': ref(an['shortest'] / an['longest'], 'Sal / longest 0.2 decay length (solved on the closed-form ACF)'),
         'tol': {'abs': (bA / an['shortest'] + bA / an['longest']) * (an['shortest'] / an['longest']) + 0.005}, 'basis': 'propagated decay-length bound + angular sampling of the longest decay by edge-pixel rays'})

    # Texture direction / dominant wavelength (FFT based).
    add({'id': 'spatial.sine.Ssw', 'group': grp, 'gauge': 'sine-acf', 'measure': param('Ssw'), 'unit': 'µm', 'standard': std, 'reference': ref(16, 'λ'), 'tol': FP(), 'basis': 'the wavelength falls on an FFT bin (64 whole periods)'})
    add({'id': 'spatial.sine.period', 'group': grp, 'gauge': 'sine-acf', 'measure': param('period'), 'unit': 'µm', 'standard': 'non-standard (dominant wavelength of the power spectrum)', 'reference': ref(16, 'λ'), 'tol': FP(), 'basis': 'the wavelength falls on an FFT bin'})
    std_tol = {'tol': {'abs': 0.25 + 1e-9}, 'basis': 'half the angular resolution of the angular power spectrum (0.5°); the wave vector lies on a DFT bin'}
    add({'id': 'spatial.sine.Std', 'group': grp, 'gauge': 'sine-acf', 'measure': param('Std'), 'unit': '°', 'standard': std,
         'reference': ref(0, 'direction of the wave vector: 0° (independent of the y-axis convention)'), **std_tol})
    add({'id': 'spatial.sine-90.Std', 'group': grp, 'gauge': 'sine-90', 'measure': param('Std'), 'unit': '°', 'standard': std,
         'reference': ref(90, 'direction of the wave vector: 90° (independent of the y-axis convention)'), **std_tol})
    add({'id': 'spatial.sine-rot.Std', 'group': grp, 'gauge': 'sine-rot', 'measure': param('Std'), 'unit': '°', 'standard': std,
         'reference': ref(45, 'wave-vector direction measured from +x towards increasing row index (array frame); the y-up frame of crop/mask would read 135°'),
         'note': 'ISO 25178-2 fixes no orientation for Std. This case pins the array-frame convention of the spec (SPEC.md, Conventions); a binding whose software measures angles in a y-up frame converts (θ ↦ 180° − θ).', **std_tol})


_spatial()


# ── 6. Feature parameters ─────────────────────────────────────────────────────
def _feature():
    grp, std = 'Feature parameters', 'ISO 25178-2:2021 (watershed, Wolf pruning 5 %)'
    g = G['egg-feature']
    W, H, n = (g['nx'] - 1) * g['dx'], (g['ny'] - 1) * g['dy'], FE['cells'] - 2
    peaks = sum(1 for i in range(1, n + 1) for j in range(1, n + 1) if (i + j) % 2 == 0)
    pits = n * n - peaks
    kdx = (TAU * g['dx']) / FE['lambda']
    spcD, spcC = (2 * FE['A'] * (1 - math.cos(kdx))) / g['dx'] ** 2, FE['A'] * (TAU / FE['lambda']) ** 2
    sx = sum_cos(-kdx * (FE_H - 1) / 2, kdx, g['nx'])
    mean = FE['A'] * (sx / g['nx']) ** 2  # = 0: whole periods
    add({'id': 'feature.egg.Spd', 'group': grp, 'gauge': 'egg-feature', 'measure': param('Spd'), 'unit': '1/µm²', 'standard': std, 'reference': ref(peaks / (W * H), f'{peaks} complete hills / (W·H)'), 'tol': FP(), 'basis': 'exact count'})
    add({'id': 'feature.egg.Svd', 'group': grp, 'gauge': 'egg-feature', 'measure': param('Svd'), 'unit': '1/µm²', 'standard': std, 'reference': ref(pits / (W * H), f'{pits} complete dales / (W·H)'), 'tol': FP(), 'basis': 'exact count'})
    add({'id': 'feature.egg.Spc', 'group': grp, 'gauge': 'egg-feature', 'measure': param('Spc'), 'unit': '1/µm', 'standard': std,
         'reference': ref(spcC, '−½∇²z at a peak = A·k²'), 'discrete': ref(spcD, '2A·(1 − cos kΔx)/Δx² (central differences)'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'feature.egg.Svc', 'group': grp, 'gauge': 'egg-feature', 'measure': param('Svc'), 'unit': '1/µm', 'standard': std,
         'reference': ref(-spcC, '−½∇²z at a pit = −A·k² (pits are concave)'), 'discrete': ref(-spcD, '−2A·(1 − cos kΔx)/Δx²'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'feature.egg.S5p', 'group': grp, 'gauge': 'egg-feature', 'measure': param('S5p'), 'unit': 'µm', 'standard': std, 'reference': ref(FE['A'] - mean, 'A (the mean is 0 over whole periods)'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'feature.egg.S5v', 'group': grp, 'gauge': 'egg-feature', 'measure': param('S5v'), 'unit': 'µm', 'standard': std, 'reference': ref(FE['A'] + mean, 'A'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'feature.egg.S10z', 'group': grp, 'gauge': 'egg-feature', 'measure': param('S10z'), 'unit': 'µm', 'standard': std, 'reference': ref(2 * FE['A'], '2A'), 'tol': FP(), 'basis': FP_BASIS})


_feature()


# ── 7. Profile parameters ─────────────────────────────────────────────────────
def _profile():
    grp, std = 'Profile parameters', 'ISO 21920-2:2021 / ISO 4287:1997'
    A, P, g, k = SINE['A'], SINE_P, G['sine-profile'], TAU / SINE['lambda']
    M = g['nx'] - 1
    kdx = k * g['dx']
    amp = (2 * A * math.sin(kdx / 2)) / g['dx']
    S = M / 2 + 0.5 * sum_cos(kdx, 2 * kdx, M)  # Σ cos²(k(i+½)Δx), i < M
    add({'id': 'profile.sine.Ra', 'group': grp, 'gauge': 'sine-profile', 'measure': param('Ra'), 'unit': 'µm', 'standard': std, 'reference': ref((2 * A) / math.pi, '2A/π'), 'discrete': ref((2 * A) / P / math.tan(math.pi / P), '(2A/P)·cot(π/P)'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'profile.sine.Rq', 'group': grp, 'gauge': 'sine-profile', 'measure': param('Rq'), 'unit': 'µm', 'standard': std, 'reference': ref(A / SQRT2, 'A/√2'), 'tol': FP(), 'basis': FP_BASIS})
    for k2, v, f in [['Rp', A, 'A'], ['Rv', A, 'A'], ['Rz', 2 * A, '2A (mean over 5 sampling lengths)'], ['Rt', 2 * A, '2A']]:
        add({'id': f'profile.sine.{k2}', 'group': grp, 'gauge': 'sine-profile', 'measure': param(k2), 'unit': 'µm', 'standard': std, 'reference': ref(v, f), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'profile.sine.Rsk', 'group': grp, 'gauge': 'sine-profile', 'measure': param('Rsk'), 'standard': std, 'reference': ref(0, '0'), 'tol': {'abs': 1e-9}, 'basis': FP_BASIS})
    add({'id': 'profile.sine.Rku', 'group': grp, 'gauge': 'sine-profile', 'measure': param('Rku'), 'standard': std, 'reference': ref(1.5, '3/2'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'profile.sine.Rdq', 'group': grp, 'gauge': 'sine-profile', 'measure': param('Rdq'), 'standard': std,
         'reference': ref((A * k) / SQRT2, 'A·k/√2'), 'discrete': ref(amp * math.sqrt(S / M), '(2A·sin(kΔx/2)/Δx)·√(Σcos²/M), closed-form sum over the N−1 differences'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'profile.triangle.Rdq', 'group': grp, 'gauge': 'triangle-profile', 'measure': param('Rdq'), 'standard': std, 'reference': ref((4 * TRI['A']) / TRI['lambda'], '4A/λ'), 'tol': FP(), 'basis': FP_BASIS})


_profile()


# ── 8. Levelling & form removal ───────────────────────────────────────────────
def _levelling():
    grp = 'Levelling & form removal'

    def field(expect, margin=0):
        return {'kind': 'fieldError', 'expect': expect, 'margin': margin}

    zero = {'type': 'const', 'value': 0}
    add({'id': 'level.plane', 'group': grp, 'gauge': 'plane', 'ops': [{'op': 'level'}], 'measure': field([zero]), 'unit': 'µm', 'standard': 'ordinary least-squares plane (vertical residuals)',
         'reference': ref(0, 'max |residual| = 0'), 'tol': {'abs': 1e-10}, 'basis': FP_BASIS})
    egg_only = [{'type': 'eggcrate', **EGG}]
    add({'id': 'level.egg+poly1', 'group': grp, 'gauge': 'egg+poly1', 'ops': [{'op': 'level'}], 'measure': field(egg_only), 'unit': 'µm', 'standard': 'ordinary least-squares plane (vertical residuals)',
         'reference': ref(0, 'max |result − egg-crate| = 0 (orthogonality on the grid)'), 'tol': {'abs': 1e-9}, 'basis': FP_BASIS})
    add({'id': 'detrend.egg+poly2', 'group': grp, 'gauge': 'egg+poly2', 'ops': [{'op': 'detrend', 'degree': 2}], 'measure': field(egg_only), 'unit': 'µm', 'standard': 'ordinary least-squares polynomial (degree 2)',
         'reference': ref(0, 'max |result − egg-crate| = 0'), 'tol': {'abs': 1e-9}, 'basis': FP_BASIS})
    add({'id': 'detrend.egg+poly3', 'group': grp, 'gauge': 'egg+poly3', 'ops': [{'op': 'detrend', 'degree': 3}], 'measure': field(egg_only), 'unit': 'µm', 'standard': 'ordinary least-squares polynomial (degree 3)',
         'reference': ref(0, 'max |result − egg-crate| = 0'), 'tol': {'abs': 1e-8}, 'basis': 'floating-point round-off of a cubic fit (|p₃| ≈ 10 µm)'})
    pts = [[10, 50], [40, 55], [15, 15]]
    add({'id': 'levelpoints.plane+bump', 'group': grp, 'gauge': 'plane+bump', 'ops': [{'op': 'levelPoints', 'points': pts, 'radius': 4}], 'measure': field([LP_BUMP]), 'unit': 'µm', 'standard': 'ordinary least-squares plane fitted to reference regions',
         'reference': ref(0, 'max |result − bump| = 0'), 'tol': {'abs': 1e-10}, 'basis': FP_BASIS})
    add({'id': 'level.profile', 'group': grp, 'gauge': 'cos+line-profile', 'ops': [{'op': 'level'}], 'measure': field([{'type': 'cos', 'A': 0.7, 'lambda': 12.5, 'x0': 49.875}]), 'unit': 'µm', 'standard': 'ordinary least-squares line (vertical residuals)',
         'reference': ref(0, 'max |result − cosine| = 0'), 'tol': {'abs': 1e-10}, 'basis': FP_BASIS})

    # Total least squares (orthogonal distances), the association ISO 21920 names for the
    # straight-line form (Todhunter et al. 2020, form removal): the sampled gauge's TLS
    # hyperplane, subtracted vertically. Lateral and vertical axes are both in µm.
    def tls_case(cid, gauge_id, standard, ols_truth):
        g = G[gauge_id]
        z = build_heights(g).ravel().tolist()
        rr, cc = np.divmod(np.arange(g['ny'] * g['nx']), g['nx'])
        xs, ys = (cc * g['dx']).tolist(), (rr * g['dy']).tolist()
        coef, c = tls_hyperplane([xs, z] if g.get('profile') else [xs, ys, z])
        a, b = coef[0], coef[1] if len(coef) > 1 else 0
        tls_plane = {'type': 'plane', 'a': -a, 'b': -b, 'c': -c}
        # how far the OLS result (= ols_truth exactly, see the OLS case) lies from the TLS one
        ez, oz = build_heights({**g, 'parts': [*g['parts'], tls_plane]}), build_heights({**g, 'parts': ols_truth})
        d = float(np.max(np.abs(ez - oz)))
        yterm = '' if g.get('profile') else f" {'−' if b < 0 else '+'} {precision(abs(b), 10)}·y"
        add({'id': cid, 'group': grp, 'gauge': gauge_id, 'ops': [{'op': 'levelTLS'}], 'measure': field([*g['parts'], tls_plane]), 'unit': 'µm', 'standard': standard,
             'reference': ref(0, f"max |result − (z − TLS fit)| = 0; TLS fit z = {precision(a, 10)}·x{yterm} {'−' if c < 0 else '+'} {precision(abs(c), 10)}"),
             'tol': {'abs': 1e-9}, 'basis': 'floating-point round-off of the centred eigen-solution',
             'note': f'Orthogonal-distance fit (through the centroid, normal along the smallest principal axis), which does not depend on the coordinate origin; an SVD of [X 1 z] with the ones column treated as noisy (Todhunter et al. 2020, eqs. 9–13 as written) does. An ordinary least-squares implementation is off by up to {exponential(d, 2)} µm here, so it fails this case.'})

    tls_case('levelTLS.egg+poly1', 'egg+poly1', 'total least-squares plane (orthogonal residuals), removed vertically', egg_only)
    tls_case('levelTLS.profile', 'cos+line-profile', 'total least-squares line (orthogonal residuals, ISO 21920), removed vertically', [{'type': 'cos', 'A': 0.7, 'lambda': 12.5, 'x0': 49.875}])

    add({'id': 'form.sphere.radius', 'group': grp, 'gauge': 'sphere', 'ops': [{'op': 'removeSphere'}], 'measure': {'kind': 'formRadius'}, 'unit': 'µm', 'standard': 'least-squares sphere (radial residuals)',
         'reference': ref(SPH['R'], 'R'), 'tol': {'rel': 1e-7}, 'basis': 'Levenberg–Marquardt convergence on exact data'})
    add({'id': 'form.sphere.residual', 'group': grp, 'gauge': 'sphere', 'ops': [{'op': 'removeSphere'}], 'measure': field([zero]), 'unit': 'µm', 'standard': 'vertical removal of the fitted sphere',
         'reference': ref(0, 'max |residual| = 0'), 'tol': {'abs': 1e-6}, 'basis': 'Levenberg–Marquardt convergence on exact data'})
    add({'id': 'form.cylinder.radius', 'group': grp, 'gauge': 'cylinder', 'ops': [{'op': 'removeCylinder'}], 'measure': {'kind': 'formRadius'}, 'unit': 'µm', 'standard': 'least-squares cylinder (radial residuals)',
         'reference': ref(CYL['R'], 'R'), 'tol': {'rel': 1e-7}, 'basis': 'Levenberg–Marquardt convergence on exact data'})
    add({'id': 'form.cylinder.residual', 'group': grp, 'gauge': 'cylinder', 'ops': [{'op': 'removeCylinder'}], 'measure': field([zero]), 'unit': 'µm', 'standard': 'vertical removal of the fitted cylinder',
         'reference': ref(0, 'max |residual| = 0'), 'tol': {'abs': 1e-6}, 'basis': 'Levenberg–Marquardt convergence on exact data'})


_levelling()


# ── 9. Filtering ──────────────────────────────────────────────────────────────
def _filtering():
    grp, std = 'Filtering', 'ISO 16610-21 / ISO 16610-61 (Gaussian)'
    LC, tB = 8, GAUSS_TRUNC_BOUND

    def amp(lam, margin):
        return {'kind': 'sineAmplitude', 'lambda': lam, 'margin': margin}

    for L in FILT_LAMBDAS:
        T = gauss_lowpass_t(LC, L)
        add({'id': f'filter.lowpass.λ{L}', 'group': grp, 'gauge': f'sine-filter-{L}', 'ops': [{'op': 'filter', 'type': 'lowpass', 'cutoff': LC}], 'measure': amp(L, 16),
             'standard': std, 'reference': ref(T, f'exp(−π(α·λc/λ)²), λc = {LC} µm, α = √(ln2/π)'), 'tol': {'abs': tB}, 'basis': 'kernel truncated at ±4σ: |ΔT| ≤ 2·erfc(2√2) = 1.3e-4'})
    for L in [8, 16]:
        T = gauss_lowpass_t(LC, L)
        add({'id': f'filter.highpass.λ{L}', 'group': grp, 'gauge': f'sine-filter-{L}', 'ops': [{'op': 'filter', 'type': 'highpass', 'cutoff': LC}], 'measure': amp(L, 16),
             'standard': std, 'reference': ref(1 - T, f'1 − exp(−π(α·λc/λ)²), λc = {LC} µm'), 'tol': {'abs': tB}, 'basis': 'kernel truncation bound'})
    L, ls, lc = 8, 2.5, 25
    T = gauss_lowpass_t(ls, L) * (1 - gauss_lowpass_t(lc, L))
    add({'id': 'filter.bandpass.λ8', 'group': grp, 'gauge': 'sine-filter-8', 'ops': [{'op': 'filter', 'type': 'bandpass', 'cutoff': ls, 'cutoff2': lc}], 'measure': amp(L, 56),
         'standard': std, 'reference': ref(T, 'T_low(λs = 2.5 µm)·(1 − T_low(λc = 25 µm))'), 'tol': {'abs': 2 * tB}, 'basis': 'two truncated kernels'})
    add({'id': 'filter.lowpass.y-aniso', 'group': grp, 'gauge': 'sine-filter-y', 'ops': [{'op': 'filter', 'type': 'lowpass', 'cutoff': LC}], 'measure': {'kind': 'sineAmplitude', 'lambda': 16, 'axis': 'y', 'margin': 32},
         'standard': std, 'reference': ref(gauss_lowpass_t(LC, 16), 'exp(−π(α·λc/λ)²) along y with Δy ≠ Δx'), 'tol': {'abs': tB}, 'basis': 'kernel truncation bound (σ is set per axis from Δx and Δy)'})
    add({'id': 'filter.profile.lowpass.λ16', 'group': grp, 'gauge': 'sine-filter-profile', 'ops': [{'op': 'filter', 'type': 'lowpass', 'cutoff': LC}], 'measure': amp(16, 16),
         'standard': 'ISO 16610-21 (profile Gaussian)', 'reference': ref(gauss_lowpass_t(LC, 16), 'exp(−π(α·λc/λ)²)'), 'tol': {'abs': tB}, 'basis': 'kernel truncation bound'})
    # Whole filtered field of an oblique Fourier series (Todhunter et al. 2020, filtration):
    # each term is scaled by T(|f|) = Tx·Ty. Away from the border (16 px > 4σ = 12 px).
    sumA = 0
    for t in FOURIER_TERMS:
        sumA = sumA + abs(t[2])
    tol = {'abs': 2 * tB * sumA}
    basis = f'separable kernels, each truncated at ±4σ: |ΔT| ≤ 2·{exponential(tB, 1)} per term, times Σ|Aᵢ| = {num(sumA)}'

    def field_of(gain):
        return {'kind': 'fieldError', 'expect': [fourier_term(t, gain) for t in FOURIER_TERMS], 'margin': 16, 'marginRows': 16}

    add({'id': 'filter.fourier.lowpass', 'group': grp, 'gauge': 'fourier-filter', 'ops': [{'op': 'filter', 'type': 'lowpass', 'cutoff': LC}], 'measure': field_of(lambda lam: gauss_lowpass_t(LC, lam)), 'unit': 'µm',
         'standard': 'ISO 16610-61 (areal Gaussian)', 'reference': ref(0, 'max |result − Σ T(λᵢ)·Aᵢ·sin(…)| = 0, T = exp(−π(α·λc/λ)²), λc = 8 µm'), 'tol': tol, 'basis': basis})
    add({'id': 'filter.fourier.highpass', 'group': grp, 'gauge': 'fourier-filter', 'ops': [{'op': 'filter', 'type': 'highpass', 'cutoff': LC}], 'measure': field_of(lambda lam: 1 - gauss_lowpass_t(LC, lam)), 'unit': 'µm',
         'standard': 'ISO 16610-61 (areal Gaussian)', 'reference': ref(0, 'max |result − Σ (1 − T(λᵢ))·Aᵢ·sin(…)| = 0'), 'tol': tol, 'basis': basis})
    T2 = gauss_lowpass_t(LC, BORDER_EGG['lambdaX']) * gauss_lowpass_t(LC, BORDER_EGG['lambdaY'])
    add({'id': 'filter.border.reflect', 'group': grp, 'gauge': 'egg-filter-border', 'ops': [{'op': 'filter', 'type': 'lowpass', 'cutoff': LC}],
         'measure': {'kind': 'fieldError', 'expect': [{'type': 'eggcrate', **BORDER_EGG, 'A': BORDER_EGG['A'] * T2}], 'margin': 0}, 'unit': 'µm',
         'standard': 'Gaussian filter, half-sample reflect extension', 'reference': ref(0, 'max |result − T(λx)·T(λy)·z| = 0 over the whole field, border included'),
         'tol': {'abs': 2 * tB * BORDER_EGG['A']}, 'basis': 'two truncated kernels (the reflected extension is the surface itself)',
         'note': 'ISO 16610-61 does not prescribe how the border is handled. This case fixes the reflect convention: an implementation that pads with zeros, crops or extrapolates the border differs here by up to the surface amplitude.'})
    add({'id': 'filter.normalized.λ16', 'group': grp, 'gauge': 'sine-filter-16', 'ops': [{'op': 'filter', 'type': 'lowpass', 'cutoff': LC, 'mode': 'normalized'}], 'measure': amp(16, 16),
         'standard': 'Gaussian regression, order 0', 'reference': ref(gauss_lowpass_t(LC, 16), 'exp(−π(α·λc/λ)²) away from the border'), 'tol': {'abs': tB}, 'basis': 'kernel truncation bound'})
    add({'id': 'filter.normalized.constant', 'group': grp, 'gauge': 'const-holes', 'ops': [{'op': 'filter', 'type': 'lowpass', 'cutoff': LC, 'mode': 'normalized'}], 'measure': {'kind': 'fieldError', 'expect': [{'type': 'const', 'value': 3}], 'margin': 0},
         'unit': 'µm', 'standard': 'Gaussian regression, order 0', 'reference': ref(0, 'a constant is reproduced exactly, including the border and around the holes'), 'tol': {'abs': 1e-12}, 'basis': FP_BASIS})
    add({'id': 'filter.robust.spikes', 'group': grp, 'gauge': 'dither-spiked', 'ops': [{'op': 'filter', 'type': 'lowpass', 'cutoff': LC, 'mode': 'robust'}],
         'measure': {'kind': 'fieldError', 'expect': [{'type': 'const', 'value': 0}], 'skipSpikes': True}, 'unit': 'µm',
         'standard': 'robust Gaussian regression (ISO 16610-71 style)', 'reference': ref(0, 'max |result| at non-spike points: a convex combination of the texture once the ±20 µm spikes carry no weight'),
         'tol': {'abs': DITHER}, 'basis': 'convex-combination bound: |mean| ≤ max |texture| = 0.01 µm (a spike leaking in with weight w adds ≈ 20·w µm)'})


_filtering()


# ── 10. Correction operators ──────────────────────────────────────────────────
def _correction():
    grp = 'Correction operators'

    def stat(s):
        return {'kind': 'stat', 'stat': s}

    g = G['plane']
    zmin = PL['c'] + min(0, PL['a'] * (g['nx'] - 1) * g['dx']) + min(0, PL['b'] * (g['ny'] - 1) * g['dy'])
    zmax = PL['c'] + max(0, PL['a'] * (g['nx'] - 1) * g['dx']) + max(0, PL['b'] * (g['ny'] - 1) * g['dy'])
    add({'id': 'center.plane', 'group': grp, 'gauge': 'plane', 'ops': [{'op': 'center'}], 'measure': stat('mean'), 'unit': 'µm', 'standard': 'z − z̄', 'reference': ref(0, 'mean = 0'), 'tol': {'abs': 1e-12}, 'basis': FP_BASIS})
    add({'id': 'zero.plane', 'group': grp, 'gauge': 'plane', 'ops': [{'op': 'zero'}], 'measure': stat('min'), 'unit': 'µm', 'standard': 'z − min z', 'reference': ref(0, 'min = 0'), 'tol': {'abs': 1e-12}, 'basis': FP_BASIS})
    add({'id': 'invert.plane', 'group': grp, 'gauge': 'plane', 'ops': [{'op': 'invert'}], 'measure': {'kind': 'fieldError', 'expect': [{'type': 'plane', 'a': -PL['a'], 'b': -PL['b'], 'c': zmin + zmax - PL['c']}]}, 'unit': 'µm',
         'standard': 'min + max − z', 'reference': ref(0, 'max |result − (min + max − z)| = 0'), 'tol': {'abs': 1e-12}, 'basis': FP_BASIS})
    add({'id': 'invert.step.Ssk', 'group': grp, 'gauge': 'step', 'ops': [{'op': 'invert'}], 'measure': param('Ssk'), 'standard': 'min + max − z', 'reference': ref(-bernoulli_heights(STEP['d'], STEP['h'])['Ssk'], '−(1−2d)/√(d(1−d))'), 'tol': FP(), 'basis': FP_BASIS})

    add({'id': 'threshold.ramp.max', 'group': grp, 'gauge': 'ramp', 'ops': [{'op': 'threshold', 'upper': 5, 'lower': 10}], 'measure': stat('max'), 'unit': 'µm', 'standard': 'material-ratio threshold',
         'reference': ref(0.95, 'h(5 %) of a uniform distribution'), 'tol': {'abs': 1.0001 / 19999}, 'basis': 'one height step of the sampled ramp (1/(N−1)): the side of the 5 % sample is a convention'})
    add({'id': 'threshold.ramp.min', 'group': grp, 'gauge': 'ramp', 'ops': [{'op': 'threshold', 'upper': 5, 'lower': 10}], 'measure': stat('min'), 'unit': 'µm', 'standard': 'material-ratio threshold',
         'reference': ref(0.10, 'h(90 %)'), 'tol': {'abs': 1.0001 / 19999}, 'basis': 'one height step of the sampled ramp'})
    add({'id': 'threshold.ramp.removed', 'group': grp, 'gauge': 'ramp', 'ops': [{'op': 'threshold', 'upper': 5, 'lower': 10}], 'measure': stat('nanCount'), 'unit': 'points', 'standard': 'material-ratio threshold',
         'reference': ref(3000, '15 % of N = 20 000'), 'tol': {'abs': 2}, 'basis': 'one sample at each threshold'})

    spikes = [[r, c] for r, c, _ in SPIKES]
    nan_set = {'kind': 'nanMismatch', 'points': spikes}
    exact_set = {'reference': ref(0, 'flagged set = the 6 spikes (mismatched pixels = 0)'), 'tol': {'abs': 0}, 'basis': 'exact set comparison', 'unit': 'points'}
    add({'id': 'outliers.mean', 'group': grp, 'gauge': 'sine-spiked', 'ops': [{'op': 'removeOutliers', 'method': 'mean', 'n': 3}], 'measure': nan_set, 'standard': 'mean ± 3σ', **exact_set})
    add({'id': 'outliers.median', 'group': grp, 'gauge': 'sine-spiked', 'ops': [{'op': 'removeOutliers', 'method': 'median', 'n': 3}], 'measure': nan_set, 'standard': 'median ± 3·MAD', **exact_set})
    despike_std = {'hampel': 'Hampel identifier, 5 × 5', 'local': 'local mean ± 3σ (leave-one-out), 5 × 5', 'residual': 'residual to the 8-neighbour median, robust z-score'}
    for method in ['hampel', 'local', 'residual']:
        add({'id': f'despike.{method}', 'group': grp, 'gauge': 'sine-spiked', 'ops': [{'op': 'despike', 'method': method, 'n': 3, 'size': 5, 'minHeight': 0.5}], 'measure': nan_set,
             'standard': despike_std[method], **exact_set})

    add({'id': 'fill.linear.plane', 'group': grp, 'gauge': 'plane-holes', 'ops': [{'op': 'fill', 'method': 'linear'}], 'measure': {'kind': 'fieldError', 'expect': [{'type': 'plane', **FILL_PLANE}], 'requireFinite': True}, 'unit': 'µm',
         'standard': 'linear (Delaunay) interpolation', 'reference': ref(0, 'max |filled − plane| = 0'), 'tol': {'abs': 1e-9}, 'basis': FP_BASIS})
    add({'id': 'fill.nearest.edge', 'group': grp, 'gauge': 'columns-hole', 'ops': [{'op': 'fill', 'method': 'nearest'}], 'measure': {'kind': 'fieldError', 'expect': [{'type': 'columns', 'split': 100, 'left': 0, 'right': 1}], 'requireFinite': True}, 'unit': 'µm',
         'standard': 'nearest measured point (Euclidean)', 'reference': ref(0, 'max |filled − step| = 0'), 'tol': {'abs': 0}, 'basis': 'exact (no ties by construction)'})


_correction()


# ── 11. Geometry operators ────────────────────────────────────────────────────
def _geometry_ops():
    grp, g = 'Geometry operators', G['plane']
    # crop box (µm, y measured from the bottom: the spec's box convention)
    box = [10, 60, 8, 30]
    c0, c1 = js_round(box[0] / g['dx']), js_round(box[1] / g['dx'])
    r0, r1 = g['ny'] - js_round(box[3] / g['dy']) - 1, g['ny'] - js_round(box[2] / g['dy']) - 1
    cc = PL['a'] * c0 * g['dx'] + PL['b'] * r0 * g['dy'] + PL['c']
    add({'id': 'crop.plane.values', 'group': grp, 'gauge': 'plane', 'ops': [{'op': 'crop', 'box': box}], 'measure': {'kind': 'fieldError', 'expect': [{'type': 'plane', 'a': PL['a'], 'b': PL['b'], 'c': cc}]}, 'unit': 'µm',
         'standard': 'crop to [x0, x1] × [y0, y1] µm, y from the bottom', 'reference': ref(0, f'the cropped field is the plane on columns {c0}–{c1}, rows {r0}–{r1}'), 'tol': {'abs': 1e-12}, 'basis': FP_BASIS})
    add({'id': 'crop.plane.cols', 'group': grp, 'gauge': 'plane', 'ops': [{'op': 'crop', 'box': box}], 'measure': {'kind': 'stat', 'stat': 'cols'}, 'unit': 'px', 'standard': 'crop', 'reference': ref(c1 - c0 + 1, '(x1 − x0)/Δx + 1'), 'tol': {'abs': 0}, 'basis': 'exact'})
    add({'id': 'crop.plane.rows', 'group': grp, 'gauge': 'plane', 'ops': [{'op': 'crop', 'box': box}], 'measure': {'kind': 'stat', 'stat': 'rows'}, 'unit': 'px', 'standard': 'crop', 'reference': ref(r1 - r0 + 1, '(y1 − y0)/Δy + 1'), 'tol': {'abs': 0}, 'basis': 'exact'})

    gi = G['plane-iso']
    zc = 0.04 * 50 * gi['dx']
    add({'id': 'zoom.plane.values', 'group': grp, 'gauge': 'plane-iso', 'ops': [{'op': 'zoom', 'factor': 2}], 'measure': {'kind': 'fieldError', 'expect': [{'type': 'plane', 'a': 0.04, 'b': 0, 'c': zc}]}, 'unit': 'µm',
         'standard': 'centred magnification', 'reference': ref(0, 'the central half of the field (columns 50–150)'), 'tol': {'abs': 1e-12}, 'basis': FP_BASIS})
    add({'id': 'zoom.plane.cols', 'group': grp, 'gauge': 'plane-iso', 'ops': [{'op': 'zoom', 'factor': 2}], 'measure': {'kind': 'stat', 'stat': 'cols'}, 'unit': 'px', 'standard': 'centred magnification', 'reference': ref(101, '(nx − 1)/f + 1'), 'tol': {'abs': 0}, 'basis': 'exact'})
    add({'id': 'rotate.plane.stepX', 'group': grp, 'gauge': 'plane-iso', 'ops': [{'op': 'rotate', 'angle': 30}], 'measure': {'kind': 'stat', 'stat': 'stepX'}, 'unit': 'µm',
         'standard': 'rotation (rigid motion)', 'reference': ref(gi['dx'], 'Δx: the image is resampled on the original pitch'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'rotate.plane.slope', 'group': grp, 'gauge': 'plane-iso', 'ops': [{'op': 'rotate', 'angle': 30}], 'measure': {'kind': 'planeGradient', 'component': 'mag', 'margin': 12},
         'standard': 'rotation (cubic spline) + crop', 'reference': ref(0.04, '|∇z| is invariant under rotation'), 'tol': {'rel': 1e-6}, 'basis': 'a cubic spline reproduces a plane exactly ≥ 12 px from the border; the physical pixel size must be preserved'})
    add({'id': 'rotate.plane.direction', 'group': grp, 'gauge': 'plane-iso', 'ops': [{'op': 'rotate', 'angle': 30}], 'measure': {'kind': 'planeGradient', 'component': 'angle', 'margin': 12}, 'unit': '°',
         'standard': 'counter-clockwise rotation', 'reference': ref(30, 'gradient direction turns by +30° (counter-clockwise, y up)'), 'tol': {'abs': 1e-6}, 'basis': 'exact for a plane ≥ 12 px from the border'})


_geometry_ops()


# ── 12. Masking ───────────────────────────────────────────────────────────────
def _masking():
    grp = 'Masking'

    def both(cid, ops, gauge_id='step', count=None, count_formula=None):
        add({'id': f'{cid}.Sq', 'group': grp, 'gauge': gauge_id, 'ops': ops, 'measure': param('Sq'), 'unit': 'µm', 'standard': 'masked points excluded from evaluation', 'reference': ref(0, 'only one height level remains → Sq = 0'), 'tol': {'abs': 1e-12}, 'basis': FP_BASIS})
        if count is not None:
            add({'id': f'{cid}.count', 'group': grp, 'gauge': gauge_id, 'ops': ops, 'measure': {'kind': 'stat', 'stat': 'maskedCount'}, 'unit': 'points', 'standard': 'mask', 'reference': ref(count, count_formula), 'tol': {'abs': 0}, 'basis': 'exact'})

    both('mask.rect', [{'op': 'maskRect', 'box': [0, 49, 0, 99]}], count=5000, count_formula='50 × 100 points of the upper level')
    both('mask.threshold', [{'op': 'maskThreshold', 'above': 1}], count=5000, count_formula='every point above h/2')
    both('mask.invert', [{'op': 'maskThreshold', 'above': 1}, {'op': 'maskInvert'}], count=15000, count_formula='the complement: 150 × 100 points')
    add({'id': 'mask.invert.mean', 'group': grp, 'gauge': 'step', 'ops': [{'op': 'maskThreshold', 'above': 1}, {'op': 'maskInvert'}], 'measure': {'kind': 'stat', 'stat': 'validMean'}, 'unit': 'µm', 'standard': 'mask', 'reference': ref(STEP['h'], 'only the upper level (h) remains'), 'tol': {'abs': 1e-12}, 'basis': FP_BASIS})
    both('mask.circle', [{'op': 'maskCircle', 'center': [DISC['cx'], DISC['cy']], 'radius': DISC['r']}], gauge_id='disc')
    add({'id': 'mask.toNaN.count', 'group': grp, 'gauge': 'step', 'ops': [{'op': 'maskThreshold', 'above': 1}, {'op': 'maskToNaN'}], 'measure': {'kind': 'stat', 'stat': 'nanCount'}, 'unit': 'points', 'standard': 'masked → non-measured', 'reference': ref(5000, 'the 5000 masked points'), 'tol': {'abs': 0}, 'basis': 'exact'})


_masking()


# ── 13. Spectral & ACF analyses ───────────────────────────────────────────────
def _spectral():
    grp = 'Spectral & ACF analyses'
    add({'id': 'rpsd.parseval', 'group': grp, 'gauge': 'sine-spectral', 'measure': {'kind': 'radialPsd', 'what': 'parseval'}, 'unit': 'µm²', 'standard': 'Parseval: Σ C·Δf² = Sq²',
         'reference': ref(0.5, 'A²/2'), 'tol': FP(), 'basis': 'all power inside the Nyquist circle; exact for whole periods'})
    add({'id': 'fft.peak', 'group': grp, 'gauge': 'sine-spectral', 'measure': {'kind': 'fourierPeak'}, 'unit': '1/µm', 'standard': '2D DFT', 'reference': ref(1 / 16, '1/λ'), 'tol': FP(), 'basis': 'the frequency falls on a DFT bin'})

    def acf(tx, ty):
        return {'kind': 'acfMap', 'tau': [tx, ty]}

    add({'id': 'acfmap.zero', 'group': grp, 'gauge': 'sine-spectral', 'measure': acf(0, 0), 'unit': 'µm²', 'standard': 'unbiased linear ACF', 'reference': ref(0.5, 'R(0) = Sq² = A²/2'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'acfmap.lambda', 'group': grp, 'gauge': 'sine-spectral', 'measure': acf(16, 0), 'unit': 'µm²', 'standard': 'unbiased linear ACF', 'reference': ref(0.5, 'R(λ, 0) = A²/2 (overlap of whole periods)'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'acfmap.halflambda', 'group': grp, 'gauge': 'sine-spectral', 'measure': acf(8, 0), 'unit': 'µm²', 'standard': 'unbiased linear ACF', 'reference': ref(-0.5, 'R(λ/2, 0) = −A²/2 (sin² has period λ/2)'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'abbottcurve.display', 'group': grp, 'gauge': 'bearing', 'measure': {'kind': 'abbottCurveError', 'knots': KNOTS}, 'unit': 'µm', 'standard': 'material-ratio curve (display)',
         'reference': ref(0, 'max |curve − h(mr)|'), 'tol': {'abs': 0.6 * 100 / 19999 * 1.01}, 'basis': 'one sample of the ramp × steepest slope'})


_spectral()


# ── 14. Region volume ─────────────────────────────────────────────────────────
def _volume():
    grp = 'Region volume'

    # Above z = 0 the Riemann sum of a Gaussian sampled at s/Δ ≥ 1.5 equals 2πhs² to
    # < 1e−15 (Poisson summation: error ~ exp(−2π²s²/Δ²)); the field border cuts tails
    # below exp(−34). Separating the two bumps into regions needs a cut plane z = ε > 0,
    # whose kink adds a lattice error ≤ ε·(perimeter of the cut)·Δ.
    def V(h, s):
        return TAU * h * s * s

    vB, vS = V(GB['h'], GB['s']), V(GS['h'], GS['s'])
    eps = 1e-6

    def Ve(h, s):
        return TAU * s * s * (h - eps - eps * math.log(h / eps))

    def rs(h, s):
        return s * math.sqrt(2 * math.log(h / eps))

    kink = eps * TAU * (rs(GB['h'], GB['s']) + rs(GS['h'], GS['s'])) * 0.5
    F0 = '2π·h·s² per bump'
    Fe = '2πs²·[h − ε − ε·ln(h/ε)] per bump above the cut plane z = ε'

    def R(opts, what='volume'):
        return {'kind': 'regionVolume', 'what': what, 'opts': {'plane': 0, **opts}}

    exact = {'tol': {'rel': 1e-12}, 'basis': 'exponentially convergent lattice sum of a Gaussian (< 1e−15)'}
    add({'id': 'volume.all', 'group': grp, 'gauge': 'gauss-bumps', 'measure': R({}), 'unit': 'µm³', 'standard': 'material volume above a plane', 'reference': ref(vB + vS, F0), **exact})
    add({'id': 'volume.circle', 'group': grp, 'gauge': 'gauss-bumps', 'measure': R({'region': {'kind': 'circle', 'cx': GB['xc'], 'cy': GB['yc'], 'r': 20}}), 'unit': 'µm³', 'standard': 'circular region', 'reference': ref(vB, 'large bump only'),
         'tol': {'rel': 1e-9}, 'basis': 'Gaussian mass outside r = 20 µm: exp(−r²/2s²) = 2e−10'})
    add({'id': 'volume.rect', 'group': grp, 'gauge': 'gauss-bumps', 'measure': R({'region': {'kind': 'rect', 'x0': 50, 'x1': 70, 'y0': 20, 'y1': 40}}), 'unit': 'µm³', 'standard': 'rectangular region', 'reference': ref(vS, 'small bump only'), **exact})
    add({'id': 'volume.polygon', 'group': grp, 'gauge': 'gauss-bumps', 'measure': R({'region': {'kind': 'polygon', 'pts': [{'x': 1, 'y': 1}, {'x': 52, 'y': 3}, {'x': 50, 'y': 58}, {'x': 2, 'y': 57}]}}), 'unit': 'µm³', 'standard': 'polygonal region', 'reference': ref(vB, 'large bump only'), **exact})
    add({'id': 'volume.concave', 'group': grp, 'gauge': 'gauss-dimple', 'measure': {'kind': 'regionVolume', 'what': 'volume', 'opts': {'plane': 0, 'mode': 'concave'}}, 'unit': 'µm³', 'standard': 'void volume below a plane', 'reference': ref(vB, F0 + ' (mirrored)'), **exact})
    cut = {'tol': {'abs': kink}, 'basis': f'lattice error of the kink at z = ε: ε·perimeter·Δ ≤ {exponential(kink, 1)} µm³'}
    add({'id': 'volume.cut', 'group': grp, 'gauge': 'gauss-bumps', 'measure': R({'plane': eps}), 'unit': 'µm³', 'standard': 'material volume above a plane', 'reference': ref(Ve(GB['h'], GB['s']) + Ve(GS['h'], GS['s']), Fe), **cut})
    add({'id': 'volume.regions', 'group': grp, 'gauge': 'gauss-bumps', 'measure': R({'plane': eps}, 'nRegions'), 'unit': '', 'standard': 'connected regions above z = ε', 'reference': ref(2, 'two separate bumps'), 'tol': {'abs': 0}, 'basis': 'exact'})
    add({'id': 'volume.minArea', 'group': grp, 'gauge': 'gauss-bumps', 'measure': R({'plane': eps, 'minArea': 100}), 'unit': 'µm³', 'standard': 'small regions excluded', 'reference': ref(Ve(GB['h'], GB['s']), 'large bump only (the small one covers 49 µm² < 100 µm²)'), **cut})


_volume()


# ── 15. Geometry metrology ────────────────────────────────────────────────────
def _geometry():
    grp = 'Geometry metrology'
    step = 0.25  # caliper sampling step for these gauges (max(Δ/2, band/60))
    add({'id': 'geometry.circle.radius', 'group': grp, 'gauge': 'boss', 'measure': {'kind': 'fitCircle', 'guess': {'cx': 31, 'cy': 29, 'r': 16}, 'what': 'r'}, 'unit': 'µm', 'standard': 'caliper edge detection + least-squares circle',
         'reference': ref(GDISC['r'], 'R (steepest point of the edge)'), 'tol': {'abs': step / 2}, 'basis': 'half the caliper sampling step'})
    add({'id': 'geometry.circle.centre', 'group': grp, 'gauge': 'boss', 'measure': {'kind': 'fitCircle', 'guess': {'cx': 31, 'cy': 29, 'r': 16}, 'what': 'centreError', 'truth': [GDISC['cx'], GDISC['cy']]}, 'unit': 'µm', 'standard': 'caliper edge detection + least-squares circle',
         'reference': ref(0, '|fitted centre − (x_c, y_c)|'), 'tol': {'abs': step / 2}, 'basis': 'half the caliper sampling step'})
    ph, Lh = (GEDGE['phi'] * math.pi) / 180, 15
    seg = {'x0': GEDGE['x0'] - Lh * math.cos(ph), 'y0': GEDGE['y0'] - Lh * math.sin(ph), 'x1': GEDGE['x0'] + Lh * math.cos(ph), 'y1': GEDGE['y0'] + Lh * math.sin(ph)}
    add({'id': 'geometry.line.angle', 'group': grp, 'gauge': 'edge', 'measure': {'kind': 'fitLine', 'seg': seg, 'what': 'angle'}, 'unit': '°', 'standard': 'caliper edge detection + total least squares',
         'reference': ref(GEDGE['phi'], 'φ'), 'tol': {'abs': (math.atan(step / (2 * Lh)) * 180) / math.pi}, 'basis': 'half a sampling step over the segment half-length'})
    add({'id': 'geometry.line.offset', 'group': grp, 'gauge': 'edge', 'measure': {'kind': 'fitLine', 'seg': seg, 'what': 'offset', 'truth': [GEDGE['x0'], GEDGE['y0']]}, 'unit': 'µm', 'standard': 'caliper edge detection + total least squares',
         'reference': ref(0, 'distance of (x₀, y₀) to the fitted line'), 'tol': {'abs': step / 2}, 'basis': 'half the caliper sampling step'})
    sp = [12.34, 7.89]
    add({'id': 'geometry.sampleHeight', 'group': grp, 'gauge': 'plane-sample', 'measure': {'kind': 'sampleHeight', 'x': sp[0], 'y': sp[1]}, 'unit': 'µm', 'standard': 'bilinear interpolation',
         'reference': ref(SAMPLE_PLANE['a'] * sp[0] + SAMPLE_PLANE['b'] * sp[1] + SAMPLE_PLANE['c'], 'a·x + b·y + c'), 'tol': FP(), 'basis': FP_BASIS})
    add({'id': 'geometry.profileLine', 'group': grp, 'gauge': 'plane-sample', 'measure': {'kind': 'profileLine', 'line': {'x0': 5, 'y0': 5, 'x1': 40, 'y1': 20}, 'plane': SAMPLE_PLANE}, 'unit': 'µm', 'standard': 'line profile extraction',
         'reference': ref(0, 'max |profile − plane along the line|'), 'tol': {'abs': 1e-12}, 'basis': FP_BASIS})


_geometry()


# ── 16. Cavity metrology ──────────────────────────────────────────────────────
def _cavities():
    grp = 'Cavity metrology'

    def cav(what, truth=None):
        return {'kind': 'cavities', 'what': what} if truth is None else {'kind': 'cavities', 'what': what, 'truth': truth}

    D = 1e-4  # dither amplitude (µm) of the 'pockets-dither' gauge
    add({'id': 'cavities.ideal.count', 'group': grp, 'gauge': 'pockets', 'measure': cav('count'), 'unit': '', 'standard': 'automatic cavity detection on noise-free data', 'reference': ref(6, '2 × 3 pockets'), 'tol': {'abs': 0}, 'basis': 'exact'})
    add({'id': 'cavities.count', 'group': grp, 'gauge': 'pockets-dither', 'measure': cav('count'), 'unit': '', 'standard': 'automatic cavity detection', 'reference': ref(6, '2 × 3 pockets'), 'tol': {'abs': 0}, 'basis': 'exact'})
    add({'id': 'cavities.stepHeight', 'group': grp, 'gauge': 'pockets-dither', 'measure': cav('stepHeight', 5), 'unit': 'µm', 'standard': 'step height to the top plane (worst cavity)', 'reference': ref(5, 'd'), 'tol': {'abs': 2 * D}, 'basis': 'twice the dither amplitude'})
    add({'id': 'cavities.sizeX', 'group': grp, 'gauge': 'pockets-dither', 'measure': cav('sizeX', 40), 'unit': 'µm', 'standard': '50 %-depth edges (worst cavity)', 'reference': ref(40, 'w'), 'tol': {'abs': 1e-4}, 'basis': 'edges halfway between pixel centres; dither shifts a 50 % crossing by ≤ 2·D/d px'})
    add({'id': 'cavities.sizeY', 'group': grp, 'gauge': 'pockets-dither', 'measure': cav('sizeY', 30), 'unit': 'µm', 'standard': '50 %-depth edges (worst cavity)', 'reference': ref(30, 'h'), 'tol': {'abs': 1e-4}, 'basis': 'as sizeX'})
    add({'id': 'cavities.volume', 'group': grp, 'gauge': 'pockets-dither', 'measure': cav('volume', 6000), 'unit': 'µm³', 'standard': 'void volume below the top plane (worst cavity)', 'reference': ref(6000, 'd·w·h'), 'tol': {'abs': 2 * D * 6000 / 5 + 1e-6}, 'basis': 'dither amplitude over the footprint'})


_cavities()


# ── 17. Storage precision ─────────────────────────────────────────────────────
def _storage():
    grp = 'Storage precision (float32)'
    A, P = SINE['A'], SINE_P
    f32 = {'tol': {'rel': 1e-6}, 'basis': 'float32 heights: relative quantisation 6e−8 per sample, accumulated in double precision'}
    add({'id': 'f32.sine.Sq', 'group': grp, 'gauge': 'sine-f32', 'measure': param('Sq'), 'unit': 'µm', 'standard': 'ISO 25178-2:2021', 'reference': ref(A / SQRT2, 'A/√2'), **f32})
    add({'id': 'f32.sine.Sa', 'group': grp, 'gauge': 'sine-f32', 'measure': param('Sa'), 'unit': 'µm', 'standard': 'ISO 25178-2:2021', 'reference': ref((2 * A) / math.pi, '2A/π'), 'discrete': ref((2 * A) / P / math.tan(math.pi / P), '(2A/P)·cot(π/P)'), **f32})
    add({'id': 'f32.sine.Sz', 'group': grp, 'gauge': 'sine-f32', 'measure': param('Sz'), 'unit': 'µm', 'standard': 'ISO 25178-2:2021', 'reference': ref(2 * A, '2A'), **f32})


_storage()


# ── 18. Convergence series (sampling error vs samples per period) ─────────────
def _convergence():
    grp = 'Convergence with sampling density'
    for P in SERIES_P:
        add({'id': f'conv.Sa.P{P}', 'group': grp, 'gauge': f'conv-sine-{P}', 'measure': param('Sa'), 'unit': 'µm', 'standard': 'ISO 25178-2:2021', 'series': {'id': 'Sa', 'label': 'Sa of a sinusoid', 'x': P},
             'reference': ref(2 / math.pi, '2A/π'), 'discrete': ref(2 / P / math.tan(math.pi / P), '(2A/P)·cot(π/P)'), 'tol': FP(), 'basis': FP_BASIS})
        ap = (2 * ((0.5 * P) / TAU) * math.sin(math.pi / P)) / 1
        add({'id': f'conv.Sdr.P{P}', 'group': grp, 'gauge': f'conv-hyb-{P}', 'measure': param('Sdr'), 'unit': '%', 'standard': 'ISO 25178-2:2021', 'series': {'id': 'Sdr', 'label': 'Sdr of a sinusoid (A·k = 0.5)', 'x': P},
             'reference': ref(100 * sine_area_ratio(0.5), '100·[(2/π)√(1+a²)·E(a²/(1+a²)) − 1]'), 'discrete': ref(100 * sampled_sine_area_ratio(ap, P), '100·[(1/P)·Σᵢ √(1 + a′²cos²(2π(i+½)/P)) − 1]'), 'tol': {'rel': 1e-9, 'abs': 1e-11}, 'basis': FP_BASIS})
        add({'id': f'conv.Sdq.P{P}', 'group': grp, 'gauge': f'conv-hyb-{P}', 'measure': param('Sdq'), 'unit': '', 'standard': 'ISO 25178-2:2021', 'series': {'id': 'Sdq', 'label': 'Sdq of a sinusoid (A·k = 0.5)', 'x': P},
             'reference': ref(0.5 / SQRT2, 'A·k/√2'), 'discrete': ref(ap / SQRT2, 'a′/√2'), 'tol': FP(), 'basis': FP_BASIS})
    for P in SERIES_SPC:
        kdx = TAU / P
        add({'id': f'conv.Spc.P{P}', 'group': grp, 'gauge': f'conv-egg-{P}', 'measure': param('Spc'), 'unit': '1/µm', 'standard': 'ISO 25178-2:2021', 'series': {'id': 'Spc', 'label': 'Spc of an egg-crate', 'x': P},
             'reference': ref(kdx ** 2, 'A·k²'), 'discrete': ref(2 * (1 - math.cos(kdx)), '2A·(1 − cos kΔx)/Δx²'), 'tol': FP(), 'basis': FP_BASIS})


_convergence()

GAUGES = G
