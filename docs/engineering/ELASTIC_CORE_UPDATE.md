# Elastic core correction — 23 September 2026

The earlier n-cable fabrication core had constant diameter, and the two-cable core had constant X width. Both contradicted the requested local-width scaling. The default is now a 5% core, continuously tapered with the reference width used by the straightened model. STEP, STL and IGES use the same fused CAD shape.

## Parameter contract

- `build.elastic_core_percent` / `--elastic-core-percent`: finite number strictly between 0 and 100; 5 means 5%.
- The reference is the continuous pre-notch, pre-slit straightened width envelope, normalized at the base and extended to the geometric virtual apex.
- Two cables: core X width varies; core Y thickness follows the chosen link thickness and ridge profile.
- n cables and plain circular comparison: conical core with diameter proportional to reference width.
- Legacy `neck_width_mm` is migrated to a percentage preserving the base dimension. It no longer creates a uniform core. Conflicting definitions are rejected.
- Existing joint coefficients, their exponential index law and the protected base remain unchanged. The README explains both the geometric scaling motivation and the need for physical identification.
- Simulation mass/inertia and contact geoms do not automatically include fabrication cores or drilled channels.

## Original source definition

Upstream `ZhanchiWang/Open-Spiral-Robots` at `f563e69f292a6b8cac1946dc45ad9e60d8795391`, DesignTool's elastic-geometry definition, scales the half-taper angle. Its effective width ratio is tan(p φ/2)/tan(φ/2), with p a fraction. This update follows the user's explicit width-percentage definition instead. The algorithms here are independently implemented, not copied from upstream.

## Verification

`tests/test_elastic_core.py` covers continuous axial taper, geometric complete-link ratios, the partial-base case, input migration, browser/Python dimensions, and actual section measurements on reimported STEP, IGES and STL. The 2-, 3- and 4-cable export fixtures include drilled channels; their XML joint gains are checked against the existing exponential law. IGES is a surface representation and is intersected directly with a plane, not treated as a solid.

Validation result: **380 passed, 19 skipped, 1 expected failure** in the full Python suite; the 22 focused core tests passed independently. Browser workflow checks passed, including live percentage editing, narrowing sections, legacy migration and actual XML/STL/STEP/IGES ZIP download. Browser/Python geometry parity passed across 24 configurations. The skips and expected tendon-routing failure are existing documented cases.

Full-sized 21-link 2-hex, 3- and 4-cable examples have also been generated with STEP/STL/IGES. At 5%: base c = 1.554378713 mm; tip c = 0.346183139 mm. No mass, damping identification or print-strength conclusion follows from successful CAD topology checks.

Reproduction:

```bash
uv run pytest -q
uv run python dev/check_web_geometry.py
node dev/browser/check.mjs
uv run python dev/render_core_figures.py
```

`render_core_figures.py` expects `build/core-two`, `build/core-three`, and `build/core-four` generated with `--cad --iges`. All tools, examples and verification logs are included in the handoff. GitHub Actions and Pages must run again after the update reaches the chosen deployment branch; no remote publication is performed by the local bundle.
