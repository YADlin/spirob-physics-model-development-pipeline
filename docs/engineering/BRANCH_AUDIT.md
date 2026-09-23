> Historical audit for 090665f. For the 23 September tapered-core correction, see ELASTIC_CORE_UPDATE.md and the current README.

# Branch audit and designer handoff

Review base: `ad64c019504bbfcd32a7e879e224cb3c8105b9d3` on `fix/consolidation-cad-workflows`.

This report records the current cleanup. The [README](../../README.md) is the user guide; other reports in this folder describe earlier work. The review is prepared for the branch, without merging `main` or publishing the site.

## Main and branch inventory

Remote refs checked on 19 September 2026: `main` is `55cc5f10594f9ada18d21b872e4d13fbc266e064`; the published working branch is `ad64c019504bbfcd32a7e879e224cb3c8105b9d3`. Main is an ancestor of the working branch (13 earlier feature/fix commits), with no main-only commits at that check. Its scattered root generators, duplicate guides and touch utilities are covered by this cleanup. There is no second independent set of changes to reconcile.

This update changes the review branch. Main acquires the organized layout when that branch is merged; it has not been rewritten remotely. Re-fetch before merging because later remote work may change this relationship.

## Findings addressed

| Finding | Change | Evidence / limit |
| --- | --- | --- |
| Hidden target site remained in XML | Removed target emission, configuration and marker CLI; legacy JSON is migrated with a notice. | Generated MuJoCo models have no named `target`; rejection of removed flag is tested. |
| Root mixed generators, tools, duplicate helpers and documentation | Production stages moved to `spirob/pipeline`; tools consolidated under `tools`; old geometry oracle retained under `tests/reference` only. | Imports and regression fixtures updated; full suite runs from a fresh locked environment. |
| Several dependency files and locally ignored packaging metadata | One `pyproject.toml` and public-PyPI `uv.lock`. | Python 3.12, MuJoCo 3.3.5, CADQuery 2.6.1. Clone/editable workflow; arbitrary wheel distribution is not claimed. |
| Optional build settings were not all reproducible from JSON | Added a validated `build` object, explicit CLI precedence, and resolved build settings in the output JSON. | Tests cover CLI override of CAD/IGES and collider settings. |
| Weak type/range checks and unknown JSON keys | Shared schema/field catalogue plus Python validation; cross-field geometry validation retained. | Boolean-as-number, non-finite values, bad vectors, misspellings and unsupported combinations rejected. |
| Repeated array copies inside every STL link adapter | Get all inverted quads once outside the loop. | Removes repeated whole-chain copying without changing CSV numeric provenance. |
| Preview cache depended on object identity and retained stale geometry | Bounded value-keyed cache. | Mutated parameter dictionaries no longer reuse the previous geometry. |
| Two-cable CAD rebuilt every geometrically similar complete link | Construct the two source solids and scale/translate complete links; special handling for the partial base. | Measured 21 → 2 constructions for a standard two-cable model. n-cable reuse deliberately deferred; see below. |
| A valid zero-notch input attempted a zero-radius CAD cut | Skip the cutter when notch ratio is zero. | Compiled simulation plus STEP/STL/IGES workflow tested with a zero-notch three-cable robot. |
| Old CAD could survive a later non-CAD build | Successful rebuild removes superseded optional CAD/MJB products transactionally. | Regression rebuild verifies no stale CAD and a loadable replacement XML. |
| Original touch editing script discarded existing sensors | New `tools/add_touch_sensors.py` preserves sensors, validates unique names, rebases paths and verifies physical properties. | Existing frame-position sensor and masses/inertias remain unchanged; duplicate sensor insertion fails without writing output. |
| CLI flags omitted useful explanations | Added descriptions for all explicit arguments. | Every user/pipeline entry point was exercised with `--help`; no undocumented explicit arguments found. |
| Manuals and GUI did not connect inputs to shape | One current illustrated README and a schema-driven web designer. | Front/side slices, selected-link XY section, polar view, constants, gains, 3D view, valid JSON import/export. |
| Export button could imply unsupported in-browser CAD | Local Python service and explicit static-hosting mode; optional manual GitHub Actions generator. | Real browser-driven download contains compiled XML/assets and STEP/STL/IGES. Pages hosting itself executes no Python. |
| Geom-count audit counted visual and contact meshes as separate links | `inspect_model` distinguishes unique links from mesh geoms checked. | Report semantics corrected without changing model geometry. |

## Measured efficiency result

One local run of **link-solid assembly only**, with the standard 21-link two-cable configuration:

| Metric | Prior construction | Shared construction |
| --- | ---: | ---: |
| OpenCascade link-solid constructions | 21 | 2 |
| Wall time | 2.681 s | 0.370 s |
| Sum of link volumes | 64193.98983675828 mm³ | 64193.98983675822 mm³ |

Maximum bounding-coordinate difference: 5.7e−14 mm; relative summed-volume difference: 9.1e−16. These are single-run observations, not a cross-platform benchmark or a claim about total export / simulation speed.

An analogous experiment on n-lobe CAD showed relative volume differences around 6.1e−6 (3 cables) and 1.1e−5 (4 cables), despite essentially identical bounds. That optimization is **not enabled**. Native n-lobe solids continue to be built individually until the difference is explained.

Shared simulation STL generation and cached convex-hull construction remain in use. Reducing the number of colliders helps contact-pair complexity, but this change does not remove physically necessary manifold contacts, alter friction, or guarantee real-time execution at a 0.0001 s timestep.

## Verification

Local locked-environment result: **358 passed, 19 skipped, 1 xfailed** in 250.25 s. Fifteen warnings are CADQuery/pyparsing deprecations, not numerical-instability reports.

- **1 strict xfail:** F08, the retained distal-site `−dz tan(φ/2)` correction does not satisfy the alternative constant-offset property. It is documented and should remain visible until the routing rule is deliberately changed and calibrated.
- **16 skips:** n-cable combinations outside the legacy two-cable compound collider's supported domain.
- **3 skips:** non-legacy profiles excluded from a benchmark defined for stepped rectangular links. Separate convex tests cover the newer shapes.
- **24 browser/Python configurations:** constants, requested/effective/chord lengths, all quad coordinates, all cable routes, and two-cable thickness endpoints agree within the declared tolerances (2e−13 m for coordinates).
- **2,412 CAD probes:** points ±0.003 mm around browser cross-section boundaries classify correctly inside/outside OpenCascade solids. Covers two-cable hex, three-cable notched and four-cable zero-notch models; base, complete and tip links at three stations. These samples do not certify every permitted combination.
- **Real Chromium interaction:** parameter edits, invalid input, selected-link slices, JSON download/import, persistence, old-target removal, mobile overflow, a static project subpath, and a completed XML/STL/STEP/IGES download.
- **CAD checks:** STEP solid reimport, positive/closed STL, volume/bounds reports, IGES surface reimport with mm units and 0.01 mm bounding-length tolerance.

Reproduce with `uv run pytest -q`, `uv run python dev/check_web_geometry.py`, `uv run python dev/check_web_sections.py`, and `node dev/browser/check.mjs` after installing its pinned browser dependencies. The README gives the exact commands. GitHub workflows are prepared but have not been run or deployed on the remote repository as part of this handoff.

## Remaining engineering work

1. **Physical calibration:** the exponential gain law, density and motor assumptions are not identified from printed hardware. Fabrication channels/cores are not automatically folded into simulation inertia. No new calibration claims are made.
2. **Concave contact:** n-cable convex hulls bridge notches. One hull can yield several contact points; regional force aggregation is available but cannot infer a unique point-load distribution.
3. **Preset cleanup:** legacy preset names and values remain for compatibility, including coarse timesteps if `post_gen.timestep` is omitted. Current examples explicitly set 0.0001 s. A future breaking release can simplify these presets after comparing dynamics.
4. **Separate elastic layer:** current fabrication exposes core width/diameter, not an independent axial layer thickness. The UI labels the actual supported dimension.
5. **n-lobe CAD reuse:** investigate the numerical volume differences before enabling the optimization.
6. **Preview limits:** the site samples surfaces and XY clearance, with a 32-cable / 400-link interaction limit. Exact solids, full 3D wall thickness and arbitrary deformed-contact states require the appropriate CAD/simulation tools.
7. **Cross-platform support:** automated workflow targets Ubuntu; the browser was exercised in Linux Chromium. Windows/macOS package and GUI behavior were not validated here.

The audit separates measured improvements from those deferred items; it does not assert that every possible configuration is optimized or physically validated.
