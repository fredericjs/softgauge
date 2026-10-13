# The softgauge contract

This is what an implementation has to understand to be validated with softgauge: how a
gauge is built, what every operation and measure means, how a result is judged, and
what a result file looks like. `spec.json` is the machine-readable form; this document
is the human one. Spec version: **1.0.0**.

## Conventions

- **Units.** Lengths and heights in µm, material ratios in %, angles in degrees.
- **Grid.** A gauge is `ny` rows × `nx` columns, row-major. Pixel (r, c) sits at
  x = c·dx, y = r·dy. Row 0 is the first row of the array. Samples are points, so n
  samples span (n − 1)·dx.
- **Boxes and circles** (crop, masks, levelling points) use **y measured from the
  bottom**: yb = (ny − 1 − r)·dy. µm → pixel conversion rounds to the nearest pixel
  (the cases avoid ties, so the tie rule does not matter).
- **Angles in the array frame** (Std) are measured from +x towards increasing row
  index. Rotation (`rotate`) is counter-clockwise with y up.
- **Non-measured points** are NaN.
- **Profiles.** A gauge with `profile: true` has `ny = 1`. It is a profile with
  spacing dx, and is evaluated with the R parameters (ISO 21920-2).

## Gauges

`z(r, c) = Σ parts`. Spikes `[r, c, dz]` are added afterwards, and holes
`[r0, r1, c0, c1]` (inclusive) are then set to NaN. All values are evaluated in double
precision; `dtype: 'float32'` rounds the stored result to float32 (a storage-precision
gauge). x, y and yb are defined as above, and k = r·nx + c is the raster index.

| type | z |
|---|---|
| `const` | value |
| `plane` | a·x + b·y + c |
| `poly` | Σ coef·(x − xc)ⁱ·(y − yc)ʲ over `terms: [[i, j, coef]…]` |
| `sine` | A·sin(2π(x cos θ + y sin θ)/λ + phase), θ in degrees |
| `cos` | A·cos(2π(x − x0)/λ) |
| `eggcrate` | A·cos(2π(x − x0)/λx)·cos(2π(y − y0)/λy) |
| `triangle` | A·T(x/λ) (+ A·T(y/λ) when `axes: 'xy'`), T = unit triangle wave, T(0) = 1, T(½) = −1 |
| `columns` | `left` for c < split, else `right` |
| `disc` | h inside radius r around (cx, cy) in the yb frame (y frame with `frame: 'top'`); with a width w: ½h(1 − tanh((ρ − r)/w)) |
| `bearing` | ramp in raster order through a piecewise-linear material-ratio curve: z(k) = h(100·k/(N − 1)), `knots: [[mr %, h]…]` |
| `bearingCos` | A·cos(π·k/(N − 1)): the material-ratio curve of a sinusoid |
| `bearingPoly` | Σ coefs[i]·uⁱ with u = index/(N − 1); the index is the raster index, odd rows reversed (`order: 'serpentine'`), or mult·k mod N (`'scramble'`) |
| `dither` | amp·sin(2π(0.7071067811865476·c + 0.3183098861837907·r))·cos(2π(0.4142135623730951·r − 0.2718281828459045·c)) |
| `sphere` | zc + √(R² − (x − xc)² − (y − yc)²) |
| `cylinder` | zc + √(R² − d²), d = −(x − px)·sin φ + (y − py)·cos φ |
| `gauss` | h·exp(−((x − xc)² + (y − yc)²)/(2s²)) |
| `bump` | h·cos²(πρ/(2·r0)) for ρ < r0, else 0 |
| `edge` | ½h(1 + tanh(d/w)), d = −(x − x0)·sin φ + (y − y0)·cos φ |
| `pockets` | −depth inside each axis-aligned rectangle (cx, cy, w, h), else 0 |

Each gauge carries a **fingerprint**: the number of measured points, their mean, RMS
about zero, min and max, and z at five probe pixels. A binding never builds a gauge
itself. softgauge's builder hands it the heights, after checking them against the
fingerprint. The JS and Python builders both reproduce the fingerprints, to 1e−11
relative to the gauge RMS (1e−6 for float32 gauges).

## Cases

```json
{ "id": "height.plane.Sq", "gauge": "plane", "tier": "core", "domain": "areal",
  "group": "Height parameters", "ops": [], "measure": { "kind": "param", "name": "Sq", "args": [] },
  "unit": "µm", "standard": "ISO 25178-2:2021",
  "reference": { "value": 0.8972179, "formula": "√(a²Δx²(nx²−1) + b²Δy²(ny²−1))/√12" },
  "discrete": { "value": … },           // optional: exact value for the SAMPLED gauge
  "tol": { "abs": 1e-12, "rel": 1e-9 }, "basis": "where the tolerance comes from",
  "expectNaN": false, "note": "…", "series": { … } }
```

### Tiers

| tier | meaning |
|---|---|
| `core` | A standard parameter of an unprocessed gauge. Anything that can read the gauge can run it, including software that only opens files (see Files). |
| `operators` | Standard operations (levelling, form removal, filtering, correction, geometry, masking), measured on the output array or with a standard parameter. |
| `extension` | Non-standard parameters, implementation-specific operations and analysis outputs. |

## Operations (`case.ops`, applied in order)

| op | fields | meaning |
|---|---|---|
| `level` | | subtract the least-squares plane (vertical residuals; a line for profiles) |
| `levelTLS` | | subtract the total-least-squares plane/line (orthogonal residuals, through the centroid), vertically |
| `detrend` | degree | subtract the least-squares polynomial of total degree `degree` in x, y |
| `levelPoints` | points `[[x, y]…]`, radius | fit the plane to the discs of `radius` µm around the points (yb frame) and subtract it |
| `removeSphere` / `removeCylinder` | radius? | least-squares sphere / cylinder (radial residuals; `radius` fixed or free when null), removed vertically |
| `filter` | type `lowpass`\|`highpass`\|`bandpass`, cutoff (µm), cutoff2?, mode? | Gaussian filter (ISO 16610-21/-61, α = √(ln 2/π)). `bandpass`: cutoff = λs, cutoff2 = λc. mode `reflect` (default: half-sample symmetric extension at the border), `normalized` (order-0 Gaussian regression, i.e. weights renormalised over the measured points), `robust` (robust Gaussian regression, ISO 16610-71 style) |
| `center` | | z − mean |
| `zero` | | z − min |
| `invert` | | min + max − z |
| `threshold` | upper, lower (%) | set the top `upper` % and the bottom `lower` % of the material ratio to NaN |
| `removeOutliers` | method `mean`\|`median`, n | set points outside mean ± n·σ / median ± n·MAD to NaN |
| `despike` | method, n, size, minHeight | *extension:* local spike detectors (`hampel`, `local` leave-one-out mean ± nσ, `residual` to the 8-neighbour median) |
| `fill` | method `nearest`\|`linear` | fill non-measured points (nearest neighbour / linear interpolation) |
| `crop` | box `[x0, x1, y0, y1]` µm | keep the pixels inside the box (yb frame, bounds included) |
| `zoom` | factor | centred magnification: keep the central (nx − 1)/f + 1 columns (and rows) |
| `rotate` | angle (°) | rotate counter-clockwise about the centre (cubic spline) and crop; the pixel pitch is preserved |
| `maskRect` | box `[x0, x1, y0, y1]` µm | mask the box (masked points are excluded from evaluation) |
| `maskCircle` | center `[x, y]`, radius (µm) | mask the disc (yb frame) |
| `maskThreshold` | below?, above? | mask the points below / above a height |
| `maskInvert` | | invert the mask |
| `maskToNaN` | | set masked points to NaN |

## Measures (`case.measure`)

**Evaluated by the binding:**

- `param` — `{name, args}`: the ISO 25178-2 / ISO 21920-2 parameter `name` (Sa, Sq, Sk,
  Vmp, Sdr, Sal, Spd, Ra, Rk…), with the standard defaults unless `args` are given.
  Smr(c) / Rmr(c) take a height c in µm in the frame of the data (no re-centring).
  Smc(mr) / Rmc(mr) take a material ratio in %.
  `period` (dominant wavelength of the power spectrum) is non-standard, which makes its
  case an extension case.

**Evaluated by softgauge from the binding's output array** (`toArray` / `to_array`):

- `stat` — `{stat}`: `mean`, `min`, `max` (over measured points), `validMean` (over
  measured, unmasked points), `nanCount`, `maskedCount`, `rows`, `cols`, `stepX`, `stepY`.
  A binding may compute them itself (`measure`), e.g. with the software's own reductions.
- `fieldError` — `{expect: [parts], margin?, marginRows?, skipSpikes?, requireFinite?}`:
  max |z − expected| over the interior, where the expected field is built from `expect`
  on the output grid. With `requireFinite`, any NaN in the interior gives +∞.
- `sineAmplitude` — `{lambda, axis?, margin?}`: amplitude of the least-squares sinusoid
  of wavelength λ along x (or y).
- `nanMismatch` — `{points}`: the number of pixels whose NaN state differs from "NaN
  exactly at `points`".
- `planeGradient` — `{component: 'mag'|'angle', margin?}`: |∇z|, or its direction (° from
  +x, y up), of the least-squares plane.

**Extension measures** (analysis outputs, evaluated by the binding): `formRadius`,
`radialPsd`, `fourierPeak`, `acfMap`, `abbottCurveError`, `regionVolume`, `fitCircle`,
`fitLine`, `sampleHeight`, `profileLine`, `cavities`. Their fields are in the case.

## Judging

`ref = discrete.value` when present, else `reference.value`. A case passes iff
`|value − ref| ≤ tol.abs + tol.rel·|ref|`. An `expectNaN` case passes iff the value is
NaN/undefined. Anything else returning NaN fails.

Verdicts: `pass`, `fail`, `unsupported` (the software lacks the operation, parameter or
measure; this is not a failure), `error` (the software threw).

Every tolerance states its `basis`. It is either floating-point round-off (the reference
is exact for the sampled gauge) or a derived discretisation bound. No tolerance is tuned
to make a result pass.

## Results

A run produces one result file per implementation:

```json
{ "implementation": { "name": "…", "version": "…", "runtime": "…" },
  "specVersion": "1.0.0",
  "gauges": { "<gauge id>": { "ok": true, "maxRelDiff": 0 } },
  "cases": { "<case id>": { "status": "ok"|"unsupported"|"error", "value": 0.897, "nan": false, "error": "…", "ms": 0.4 } },
  "date": "2026-10-13T12:00:00Z" }
```

JSON has no NaN or Infinity: NaN is `value: null, nan: true`, and ±∞ is clamped to ±1e308.
`softgauge report` (npm) merges any number of result files into one web/CSV/LaTeX report.

## Files

`softgauge export` (PyPI, `pip install softgauge[sdf]`) writes every gauge as an
ISO 25178-71 SDF file (binary64, or ASCII with 17 significant digits; heights and steps
in metres as the format requires; NaN → the format's non-measured marker). The writer is
[sdfio](https://pypi.org/project/sdfio/), an independent implementation, so the files
carry no code shared with any implementation under test. Next to every file is a
`<gauge>.json` sidecar with the fingerprint and every case on that gauge. Check the
fingerprint first: it proves the software read the gauge correctly, before any
parameter is judged.

## Versioning

- Case and gauge ids are permanent. A removed case is never reused.
- A changed reference, tolerance or gauge, or a changed meaning of an operation or
  measure, is a **major** spec version. New cases, gauges, operations or measures are
  **minor**. Wording is a **patch**.
- Result files record `specVersion`. The packages' own versions are independent of the
  spec version.
