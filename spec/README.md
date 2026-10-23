# The softgauge spec

This folder is the spec: everything an implementation has to understand to be validated
with softgauge. It is language-neutral and complete on its own; the JavaScript and Python
packages are two adapters that read it.

| file | |
|---|---|
| [`spec.json`](spec.json) | the spec: gauges, cases, frozen references and tolerances (**version 1.0.0**) |
| [`spec.schema.json`](spec.schema.json) | its JSON Schema (draft 2020-12) |
| [`results.schema.json`](results.schema.json) | the JSON Schema of a result file |
| this document | what the fields mean |

`spec.json` is generated from [`generator/`](../generator) (Python), where every
reference value is derived. Edit the generator, never the JSON.

## Conventions

- **Units.** Lengths and heights in µm, material ratios in %, angles in degrees.
- **Grid.** A gauge is `ny` rows × `nx` columns, row-major. Pixel (r, c) sits at
  x = c·dx, y = r·dy. Row 0 is the first row of the array. Samples are points, so n
  samples span (n − 1)·dx.
- **Boxes and circles** (crop, masks, levelling points) use **y measured from the
  bottom**: yb = (ny − 1 − r)·dy. µm → pixel conversion rounds to the nearest pixel
  (the cases avoid ties, so the tie rule does not matter).
- **Angles.** Angles that define a gauge component (θ, φ) are in the (x, y) frame of the
  gauge. **Std** is measured counter-clockwise from +x with y pointing up the map, as
  ISO 25178-2:2021, 3.2.15.1 defines it. **rotate** turns counter-clockwise with y up.
- **Non-measured points** are NaN.
- **Profiles.** A gauge with `profile: true` has `ny = 1`. It is a profile with
  spacing dx, and is evaluated with the R parameters (ISO 21920-2).

### Formulas

Every `formula` field is **LaTeX math without delimiters, in plain ASCII**, e.g.
`100 \left(\sqrt{1 + a^2 + b^2} - 1\right)`. It renders with KaTeX or MathJax and pastes
into a paper as `$…$`. Prose never goes into a formula: a reference value carries it
in `text`, a gauge in its `description` (Unicode is fine there).

## Gauges

```json
"saddle": {
  "id": "saddle", "title": "Saddle", "formula": "z = k x y",
  "description": "Hyperbolic paraboloid with its centre on the first grid corner …",
  "nx": 41, "ny": 81, "dx": 0.5, "dy": 0.25,
  "parts": [{ "type": "poly", "xc": 0, "yc": 0, "terms": [[1, 1, 0.05]] }],
  "fingerprint": { "n": 3321, "mean": 5, "rms": …, "min": 0, "max": 20, "probes": [[0, 0, 0], …] }
}
```

Optional fields: `cite` (keys into `references`), `profile`, `dtype` (`float32` rounds
the stored heights: a storage-precision gauge), `spikes` `[r, c, dz]` (added after the
parts) and `holes` `[r0, r1, c0, c1]` (inclusive, set to NaN last).

`z(r, c) = Σ parts`, evaluated in double precision. x, y and yb are defined as above, and
k = r·nx + c is the raster index.

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
{ "id": "hybrid.saddle.Sdr", "group": "Hybrid parameters", "tier": "core", "domain": "areal",
  "gauge": "saddle", "ops": [], "measure": { "kind": "param", "name": "Sdr", "args": [] }, "unit": "%",
  "standards": ["ISO 25178-2:2021"],
  "reference": { "value": 28.0789…, "formula": "100 \\left(\\frac{1}{W H} \\int_0^W …\\right)" },
  "discrete":  { "value": 28.0832…, "formula": "…", "text": "trapezoid weights: …" },
  "tol": { "rel": 1e-9, "abs": 1e-12 }, "basis": "floating-point round-off only (…)",
  "note": "ISO 25178-2:2021 D.3: each cell is triangulated along both diagonals …" }
```

| field | |
|---|---|
| `id` | permanent identifier, `<group>.<gauge>.<quantity>` |
| `group`, `tier`, `domain` | for selecting and reporting (tiers below; domain `areal` or `profile`) |
| `gauge`, `ops`, `measure` | what to load, what to apply (in order) and what to evaluate |
| `unit` | of the value |
| `standards` | the normative documents the quantity is defined by (may be empty) |
| `method` | what an operation or a non-standard measure computes, in words |
| `cite` | keys into `references`: the literature the case is taken from |
| `reference` | `{value, formula?, text?}`: the analytic value, on the continuous surface |
| `discrete` | the same for the **sampled** gauge, when it has a closed form; judged against when present |
| `expectNaN` | the quantity is undefined and must be reported as NaN |
| `tol`, `basis` | `{abs?, rel?}` and where the tolerance comes from |
| `note`, `series` | remarks; the position of a case in a convergence series |

### Tiers

| tier | meaning |
|---|---|
| `core` | A standard parameter of an unprocessed gauge. Anything that can read the gauge can run it, including software that only opens files (see Files). |
| `operators` | Standard operations (levelling, form removal, filtering, correction, geometry, masking), measured on the output array or with a standard parameter. |
| `extension` | Non-standard parameters, implementation-specific operations and analysis outputs. |

### References

`references` is the bibliography of the spec, keyed by citation key (`todhunter2019`,
`blateyron2024`, …), with authors, title, journal, volume, issue, article number, year
and DOI. Gauges and cases taken from a paper list its key in `cite`.

## Operations (`case.ops`, applied in order)

| op | fields | meaning |
|---|---|---|
| `level` | | subtract the least-squares plane (vertical residuals; a line for profiles) |
| `levelTLS` | | subtract the total-least-squares plane/line (orthogonal residuals, through the centroid), vertically |
| `detrend` | degree | subtract the least-squares polynomial of total degree `degree` in x, y |
| `levelPoints` | points `[[x, y]…]`, radius | fit the plane to the discs of `radius` µm around the points (yb frame) and subtract it |
| `removeSphere` / `removeCylinder` | radius? | least-squares sphere / cylinder (radial residuals; `radius` fixed or free when null), removed vertically |
| `filter` | type `lowpass`\|`highpass`\|`bandpass`, cutoff (µm), cutoff2?, mode? | Gaussian filter (ISO 16610-21/-61, α = √(ln 2/π)). `bandpass`: cutoff = λs, cutoff2 = λc. mode `reflect` (default: half-sample symmetric extension at the border), `normalized` (order-0 Gaussian regression, i.e. weights renormalised over the measured points), `robust` (robust Gaussian regression, ISO 16610-71) |
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
`fitLine`, `sampleHeight`, `profileLine`, `cavities`. Their fields are in the case; the
geometry they take (points, lines, boxes) is in the gauge frame.

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

A run produces one result file per implementation ([`results.schema.json`](results.schema.json)):

```json
{ "implementation": { "name": "…", "version": "…", "runtime": "…" },
  "specVersion": "1.0.0",
  "gauges": { "<gauge id>": { "ok": true, "maxRelDiff": 0 } },
  "cases": { "<case id>": { "status": "ok"|"unsupported"|"error", "value": 0.897, "nan": false, "error": "…", "ms": 0.4 } },
  "date": "2026-10-13T12:00:00.000Z" }
```

JSON has no NaN or Infinity: NaN is `value: null, nan: true`, and ±∞ is clamped to ±1e308.
`softgauge report` (in both packages, with identical output) merges any number of result
files into one web/CSV/LaTeX report.

## Files

`softgauge export` (PyPI, `pip install softgauge[sdf]`) writes every gauge as an
ISO 25178-71 SDF file (binary64, or ASCII with 17 significant digits; heights and steps
in metres as the format requires; NaN → the format's non-measured marker). The writer is
[sdfio](https://pypi.org/project/sdfio/), an independent implementation, so the files
carry no code shared with any implementation under test. Next to every file is a
`<gauge>.json` sidecar with the fingerprint, every case on that gauge and the references
they cite. Check the fingerprint first: it proves the software read the gauge correctly,
before any parameter is judged.

## Versioning

- Case and gauge ids are permanent. A removed case is never reused.
- A changed reference, tolerance or gauge, or a changed meaning of an operation or
  measure, is a **major** spec version. New cases, gauges, operations or measures are
  **minor**. Wording is a **patch**.
- Result files record `specVersion`. The packages' own versions are independent of the
  spec version.
