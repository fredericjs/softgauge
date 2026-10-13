# Changelog

## 0.1.0 (unreleased)

- First standalone release, extracted from the surfalize webapp's validation suite
  (spec 1.0.0: the same gauges, cases and references; cases gain `tier` and `domain`).
- JS and Python packages: gauge builders checked against shared fingerprints, a
  binding-based runner, array measures evaluated by softgauge itself, the judge.
- JS: `softgauge report` (results → JSON / CSV / LaTeX). Python: pytest helper
  (`softgauge.testing`), `softgauge export` (SDF via sdfio, with JSON sidecars).
