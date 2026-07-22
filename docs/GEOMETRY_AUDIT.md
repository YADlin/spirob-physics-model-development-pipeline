# SpiRob Geometry Audit

**Scope:** `spirob-physics-model-development-pipeline` (primary), with cross-checks
against `spirob-assembly` and behavioural reference to `Open-Spiral-Robots`.

**Status:** Phase 1 complete. **Phase 2 complete** — see
`docs/CANONICAL_GEOMETRY.md`. Findings below carry their post-Phase-2
disposition. Three Phase 1 findings were revised on further evidence; those
revisions are marked **REVISED** and explained in place.

**Audited commits**

| Repository | Commit | Date |
|---|---|---|
| `spirob-physics-model-development-pipeline` | `a6ea28d` | 2026-05-14 |
| `spirob-assembly` | `f95d791` | 2026-05-16 |
| `Open-Spiral-Robots` (reference only) | `a03ef17` | 2026-05-10 |

**Provenance note.** No source was copied from `Open-Spiral-Robots`. That
repository is PolyForm Noncommercial; the primary and assembly repositories are
MIT. It was read for behavioural reference only — specifically the existence and
rough shape of its `fabrication/splitter_generator.py` and `design-tool` feature
set. Every equation in this document is transcribed from the task specification
and from Wang et al. 2024, and every implementation in `tools/` was written
against those equations. See `docs/LICENSE_PROVENANCE.md` (Phase 11) for the
formal statement.

---

## 1. Reference equation set

With `E = exp(2*pi*b)`:

| Quantity | Equation |
|---|---|
| Full included taper angle | `phi(b) = 2*atan( b*(E-1) / (sqrt(1+b^2)*(E+1)) )` |
| Tip width relation | `a = d_tip/(E-1)` |
| Central spiral | `r_c(theta) = 0.5*a*(E+1)*exp(b*theta)` |
| Local width | `d(theta) = a*(E-1)*exp(b*theta)` |
| Continuous length | `L_arc(q) = sqrt(1+b^2)/b * 0.5*a*(E+1) * (exp(b*q)-1)` |
| Adjacent-unit scale | `beta = exp(b*Delta_theta)` |
| Root width | `d_root = d_tip*exp(b*q0)` |

These are implemented independently in `tools/geometry_audit.py` so that
agreement with `helper_functions.py` is a genuine cross-check rather than a
tautology.

### 1.1 Verification against the committed defaults

`params.json` at `a6ea28d`: `L = 0.22628`, `d_tip = 0.007139`,
`phi_deg = 6.3`, `Delta_theta_deg = 30.78`, `n_cables = 3`.

| Quantity | Specification | Computed | Agreement |
|---|---|---|---|
| `b` | 0.1369640839 | 0.1369640839 | exact to 10 s.f. |
| `a` | 0.0052319156 m | 0.0052319156 m | exact to 10 s.f. |
| `q0` | 10.9632013369 rad | 10.9632013369 rad | exact to 10 s.f. |
| `d_root` | 0.0320448838 m | 0.0320448838 m | exact to 10 s.f. |
| `beta` | 1.076353346 | 1.076353346 | exact to 10 s.f. |

**Tolerances used in `tests/test_analytical.py`:**

| Constant | Value | Applies to |
|---|---|---|
| `REL_TIGHT` | `1e-9` | closed-form identities (round trips, quadrature) |
| `REL_REF` | `5e-9` | the five reference constants above, quoted to 10 s.f. |
| `ABS_GEOM` | `1e-12` | metre-scale geometric identities |

The reference constants live only in the test module. Production code derives
them; nothing hard-codes them.

### 1.2 Implementation agreement

`helper_functions.generate_spiral_pose()` computes

```python
A  = (a/b)*sqrt(b**2 + 1)*(e2pb + 1)/2
Q0 = (1/b)*log(1 + Length/A)
r1 = (a/2)*(exp(2*pi*b) + 1)*exp(b*theta)     # side_A, the central spiral
r2 = a*exp(b*theta)                            # side_B, the inner edge
```

All four agree with the reference set. `A` is exactly the bracketed prefactor of
`L_arc`, so `Q0` is its exact closed-form inverse. `r1` is `r_c`. `r2` is
`r_c - d/2`, since

```
r_c - d/2 = 0.5*a*exp(b*theta)*[(E+1) - (E-1)] = a*exp(b*theta)
```

and the corresponding outer edge `r_c + d/2 = a*exp(b*(theta + 2*pi))` — the
self-similarity property that makes the design tile. Confirmed by
`test_outer_edge_is_inner_edge_one_turn_later`.

`solve_b_for_phi()` agrees with an independent bisection to `1e-8` relative over
`phi_deg` in [1, 40]. Two robustness notes, neither currently triggered:

- The fallback `return max(1e-6, tan(phi_target/2))` on bracketing failure
  silently returns a *different, wrong* `b` rather than raising. It is
  unreachable for `phi_deg` in the validated range `(0, 45)`, but it is a
  latent silent-failure path.
- `atan2(num, den)` is used where `2*atan(num/den)` is specified. Identical for
  `b > 0` because `den > 0`; harmless, but it obscures the intent.

**Verdict: the analytical core is correct.** Every discrepancy found in this
audit is in documentation, discretization semantics, or downstream consumers.

---

## 2. Findings

Severity: **A** = wrong output today · **B** = wrong documentation or a
correctness trap · **C** = maintainability risk.

| ID | Sev | Summary |
|---|---|---|
| F-01 | B | `phi_deg` is documented as a half-angle. It is the full included angle. |
| F-02 | B | `README.md` quick start uses `python build.py --nlobe`; `build.py` has no such flag. |
| F-03 | A | Requested `L` is a continuous arc length, but the built model realises a chord sum 1.16 % shorter. |
| F-04 | ~~A~~ → **not a defect** | **REVISED.** `Invert_pose()` anchors the TIP at the requested length. Intentional frame convention. |
| F-05 | ~~B~~ → **not a defect** | **REVISED.** The partial unit becomes `link_001` by design, preserving large-at-base ordering. |
| F-06 | B | Straightened elements have exactly zero intra-element taper. Undocumented, and downstream code assumes otherwise. |
| F-07 | A — **FIXED** | `preview.py` and `csv2xml.py` disagreed about the tendon path by up to 1.2956 mm — 86 % of `tendon_inward_shift`. Now agree to 1.1e-15 m. |
| F-08 | B — **deferred by decision** | The `-dz*tan(phi/2)` tendon correction applies a taper the discretised geometry does not have. |
| F-09 | B | Realised tip/root widths differ from `d_tip`/`d_root` by −0.92 % / −2.99 %. |
| F-10 | C — **FIXED** | `preview.py` carried a full private fork of the spiral maths. Deleted in Phase 2. |
| F-11 | C | `csv2xml.py` defaults `--phi-deg` to 5.7 while `params.json` ships 6.3. |
| F-12 | C | Three modules use three mutually inconsistent vertex-naming conventions. |
| F-13 | C | `requirements.txt` pins CadQuery to a git SHA; PyPI now ships a working 2.8.0. |

---

### F-01 · `phi_deg` is the full included taper angle — B

`README.md` describes `phi_deg` as *"Taper half-angle of the logarithmic
spiral"*. This is wrong. The code is right; only the documentation is wrong.

Proof. The half-width is `h(theta) = d(theta)/2`, and arc length satisfies
`ds/dtheta = sqrt(1+b^2)*r_c`. Therefore

```
dh/ds = [0.5*a*(E-1)*b*exp(b*t)] / [0.5*a*(E+1)*exp(b*t)*sqrt(1+b^2)]
      = b*(E-1) / ((E+1)*sqrt(1+b^2))
      = tan(phi/2)
```

So `tan(phi/2)` — not `tan(phi)` — is the half-width slope, which is exactly the
definition of `phi` as the **full included** angle between the two boundary
edges. Both consumers already use `phi/2` correctly:

- `csv2xml.py:183` — `HALF_PHI = math.radians(config.phi_deg / 2)`
- `csv2geom_nlobe.py:241` — `draft_angle_deg = phi_deg / 2.0`

Locked by `test_phi_is_the_full_included_angle`, which fails by a factor of two
if the semantics are ever flipped.

**Action.** Correct the README wording. Keep the parameter name. Accept
`taper_angle_deg` as an alias with an explicit conflict check that errors if
both keys are present with different values.

---

### F-02 · Documented quick-start command fails — B

`README.md` instructs:

```bash
python build.py --nlobe
```

`build.py` defines only `--params`, `--noclean`, `--no-preview`, `--plain`,
`--safe`, `--fast`, `--high`. `argparse` exits 2 on the unknown flag. `--nlobe`
exists on `preview.py`, not `build.py`; in `build.py` n-lobe is implicit
(`use_nlobe = (not args.plain) and (n_cables >= 3)`).

**Action.** Add `--nlobe` as an accepted no-op alias for backward compatibility
with any user scripts, and fix the README.

---

### F-03 · Continuous arc length vs discrete backbone length — A

`q0` is solved so the *continuous* arc length equals `L`. The pipeline then
straightens each sampled interval as a **chord**. Chords are shorter than arcs,
so the delivered robot is short:

| Quantity | Value |
|---|---|
| Requested `L` | 226.2800 mm |
| Continuous arc length | 226.2800 mm |
| Discrete backbone length | **223.6554 mm** |
| Absolute deficit | 2.6246 mm |
| Relative error | **−1.1599 %** |

This matches the specification's expected 223.6554 mm exactly. The error is
second order in `Delta_theta`; `test_discrete_converges_to_continuous` confirms
`O(Delta_theta^2)` convergence across 0.5°–60°.

**Action (Phase 2).** Add `"length_definition"`, default `"continuous_arc"` so
existing `params.json` files are bit-identical. `"discrete_backbone"` invokes
the deterministic bracketed bisection in
`tools/geometry_audit.solve_q0_for_discrete_length()` (explicit `tol`, explicit
`max_iter`, raises with an actionable message on failure to bracket or
converge). Verified: it drives the relative error to `< 1e-10`, raising `q0`
from 10.9632 to a value whose continuous length is 228.9254 mm.

---

### F-04 · Tip anchoring at the requested length — REVISED: not a defect

**Phase 1 claimed this was a severity-A defect. That was wrong, and the
correction is recorded here rather than quietly deleted.**

```python
def Invert_pose(quads, Length):
    q[:, 1] = -q[:, 1] + Length      # Length is the REQUESTED L
```

Phase 1 read the resulting `z_min = 0.0026246` as the base failing to sit at
the origin. Tracing the pipeline shows the opposite reading is the right one:
the reflection anchors the **tip** at exactly `y = L`, and the base then
necessarily sits at `L - L_discrete`. Verified to `1e-12`:

```
inverted span y = 0.0026246 .. 0.2262800
TIP  y = 0.2262800 == requested L      -> anchored exactly
BASE y = 0.0026246 == L - L_discrete
```

Two further facts settle it:

1. **The offset never reaches the simulation.** `csv2xml.py:313` overrides
   `rel[0]` from `post_gen.robot_pos`, discarding the CSV-frame root position
   entirely. Confirmed against the built model: `link_001` world position is
   exactly `[0, 0, 0.22628]`.
2. **The default `post_gen` block is consistent with tip anchoring.**
   `robot_pos = [0, 0, 0.22628]` with `robot_quat = [0, 0, -1, 0]` (180° about
   Y) hangs the robot from `z = L` so the tip descends to `z ≈ 0`. The
   2.6246 mm is the resulting tip-to-floor clearance, not a stray offset.

**Disposition.** Documented as a frame convention in `spirob/geometry.py` and
exposed as `BaseFrame.origin_offset_m` so a future CAD exporter can consume it
explicitly. The Phase 1 xfail `test_base_of_robot_sits_at_origin` was replaced
by `test_inversion_anchors_the_tip_at_the_requested_length`, which asserts the
correct convention — a stronger assertion, not a weakened one.

### F-05 · The partial unit is the base link — REVISED: correct by design

**Phase 1 flagged this as surprising and implicitly wrong. It is neither.**

`theta[i] = min(i*Delta_theta, q0)` truncates the last spiral interval, and
`Invert_pose()` then reverses the chain, so the partial unit becomes
`link_001`. With the committed defaults, `q0/Delta_theta = 20.4076`, so unit 21
spans 12.5452° instead of 30.78°.

**The requested length creates a partial terminal unit. After
`Invert_pose()`, this unit becomes the base link by design, preserving the
intended large-at-base and small-at-tip ordering.**

The spiral is generated tip-first: `theta = 0` is the smallest radius.
Truncation therefore lands on the largest-radius end, which after inversion is
the base. Moving the partial unit to the tip would invert the size ordering the
whole design depends on. It must not be "fixed".

Confirmed numerically: `link_001` is simultaneously the partial unit
(12.5452° span) *and* the largest unit (31.0876 mm realized width). It is short
in **angle**, not small in **radius**.

**Remaining fabrication note, not a geometry defect.** A printed robot built
from the default parameters will have a partial unit at its base. If a whole
number of units is wanted, select the Phase 2 policy:

```json
"terminal_unit_policy": "whole_units"
```

which extends the spiral to the next complete boundary and reports the excess
length. It never shortens.

### F-06 · Straightened elements have exactly zero intra-element taper — B

This is a *property of the model* that no module documents and at least one
module contradicts.

Because the inner edge is a constant radial scaling of the centreline —
`B(theta) = k * A(theta)` with `k = 2/(E+1)`, independent of `theta` — the
straightening step maps both inner-edge endpoints to the **same** lateral
offset:

```
x(B1 - A0) = k*x(R*A1) - x(R*A0) = (k-1)*x(R*A0) = x(B0 - A0)
```

using `x(R*A1) = x(R*A0)`, which holds because `R` rotates `A1 - A0` onto `+Y`.

Measured: `max |B0x − B1x| = 3.47e-18 m` across all 21 elements — machine zero.

So each straightened element is a **constant-width block with two slanted end
faces**. The taper is a 21-step staircase between elements, not a continuous
loft. The revolved solid is a cylinder with conical end caps, and the wedge
between caps is what provides bending clearance. This is a legitimate
discretisation — but it means:

- the realised width of unit `i` is `d(theta_i) * dist(O, chord_i) / r_c(theta_i)`
  (proved to `1e-17` by the audit tool), **not** `d(theta_i)`;
- any downstream code that assumes an intra-element taper is compensating for
  something that is not there. See F-08.

---

### F-07 · Preview and MJCF disagree about the tendon path — A

`preview.py:122` claims:

> *"The inward shift and taper correction from csv2xml are replicated here so
> the preview matches what the XML will actually use."*

It does not. The two modules unpack the **same array** with **different names**:

| Module | Unpacking | Meaning of slot 2 / slot 3 |
|---|---|---|
| `helper_functions.generate_cable_sites_csv_zrot_from_P` | `A0, A1, B1, B0 = quad` | `B1` = slot 2, `B0` = slot 3 |
| `preview._tendon_path` | `A1, A0, B0, B1 = quad` | `B0` = slot 2, `B1` = slot 3 |

`B0` and `B1` are therefore **swapped** between the producer and the preview.
Consequences:

- `preview` computes `dz = B1[1] - B0[1]`, which has the **opposite sign** to
  `csv2xml`'s `dz = s2_local[2] - s1_local[2]`;
- the taper correction is anchored at the opposite end of every link;
- the two per-link points are emitted in reversed `z` order.

Measured, cable 0, worst case:

```
k    MJCF r(mm)   z(mm)   |  PREVIEW r(mm)  z(mm)   |   dr(mm)
38     12.1653   29.0555  |     13.4609   17.2841   |  -1.2956
```

**Worst radial disagreement: 1.2956 mm**, against a `tendon_inward_shift` of
1.5 mm — an 86 % error in the quantity the preview exists to let you check. The
README explicitly tells users to trust it:

> *"The preview shows the corrected tendon path; verify visually before
> generating STL."*

The MJCF path is the self-consistent one (monotonic in both radius and height,
verified). **The preview is wrong.**

Guarded by strict-xfail `test_preview_tendon_path_matches_mjcf`, with
`test_preview_mjcf_disagreement_is_the_documented_magnitude` pinning the
1.2956 mm figure so a partial fix cannot slip through.

**This is a naming defect that produced a real numerical error in one consumer.**
The intended geometry was established numerically first: the MJCF path was
independently checked for monotonicity and containment before being declared
the reference.

**FIXED in Phase 2.** `preview.py` no longer computes tendon geometry. It reads
`SpiRobGeometry.tendon_path(0)`, the single canonical definition, which the
test suite proves equals `csv2xml`'s independent computation. Measured after
the change: worst radial disagreement **1.102e-15 m**, worst height
disagreement **1.804e-15 m** (down from 1.2956e-3 m). Guarded by
`test_preview_mjcf_agreement_is_at_float_precision`.

---

### F-08 · The tendon taper correction models a taper that is not there — B

```python
r2_new = max(r1_old - TENDON_INWARD_SHIFT - dz*math.tan(HALF_PHI), 1e-6)
```

Per F-06 the link's outer radius is *constant* along its length
(`max |r1_old − r2_old| = 3.5e-18 m`). Subtracting `dz*tan(phi/2)` within the
link therefore pulls the second site inboard of a surface that never moved, and
the next link's `s1` re-anchors at its own `r_old − shift`. The routed tendon is
a sawtooth rather than a constant offset from the surface:

| Link | intra-link taper applied | jump at boundary into next link |
|---|---|---|
| 1 → 2 | 0.2808 mm | −0.9499 mm |
| 11 → 12 | 0.3341 mm | −0.1895 mm |
| 20 → 21 | 0.1723 mm | −0.0977 mm |

The path stays monotonic, and the per-link error is bounded (0.16–0.65 mm), so
this is not catastrophic — but at the tip the tendon sits at 1.8766 mm radius
where a constant-offset rule gives 2.0366 mm. That 0.16 mm is 11 % of the shift,
and it will not agree with a CAD channel generated from the same nominal rule.

**Disposition: DEFERRED BY DECISION, and the deferral is now safe.**

Phase 2 reproduces this behaviour deliberately and exactly. Changing the
routing rule would change existing MJCF output, which Phase 2 explicitly
forbids. What Phase 2 *does* change is that there is now exactly one place to
make the change: `spirob.geometry._build_tendon_paths`. Before, the rule was
implemented twice (in `csv2xml.py` and, differently, in `preview.py`).

The open question is unchanged: should the routed tendon be a constant offset
from the realised staircase surface, or a smooth taper fitted through the
links? That decision belongs with the fabrication phase, because it must be
made once for both the MJCF tendon and the printed channel. Held as the single
remaining strict xfail, `test_tendon_offset_from_surface_is_constant`.

---

### F-09 · Realised tip and root widths miss their targets — B

| | Nominal | Realised | Error |
|---|---|---|---|
| Tip width | 7.1390 mm | 7.0733 mm | −0.92 % |
| Root width | 32.0449 mm | 31.0876 mm | −2.99 % |

A consequence of F-06: the realised width is scaled by the chord-to-radius
sagitta factor `dist(O, chord_i)/r_c(theta_i)`, which is < 1 and shrinks as
`Delta_theta` grows. The root is worse than the tip only because the terminal
unit's truncation happens to reduce the effect there.

A user asking for a 7.139 mm tip gets 7.073 mm. Harmless in simulation;
material for fabrication fit. Phase 6 validation must report both nominal and
realised widths, which the manifest already does.

---

### F-10 · `preview.py` forks the spiral maths — C

`preview.py` privately reimplements `_phi_from_b`, `_solve_b`, `_normalize`,
`_angle_between`, `_rotate2d`, and the whole spiral → straighten → invert
pipeline as `_build_quads`, rather than importing `helper_functions`.

The fork has already drifted and been re-synced by hand — three lines carry the
comment `# Correction made on Mar 19, 26`, mirroring a fix applied separately to
`helper_functions.py`. F-07 is the fork drifting again and not being caught.

This was the strongest argument for the canonical model.

**FIXED in Phase 2.** The fork is deleted. `_phi_from_b`, `_solve_b`,
`_normalize`, `_angle_between`, `_rotate2d` and the private `_build_quads`
pipeline are gone from `preview.py`; it now imports `spirob.geometry`. A
regression guard, `test_preview_no_longer_forks_the_spiral_maths`, fails if any
of those definitions reappears. `csv2geom_nlobe.py` and `csv2xml.py` are the
remaining consumers to migrate; both are unchanged in Phase 2 so that default
output stays byte-identical.

---

### F-11 · `--phi-deg` default mismatch — C

`csv2xml.py` defaults `--phi-deg` to `5.7`; `params.json` ships `6.3`.
`build.py` always passes it explicitly, so the pipeline is fine — but anyone
invoking `csv2xml.py` directly (as the README's project-structure section
implies is possible) silently gets a different taper. Same for
`csv2geom_nlobe.py:231`, `phi_deg=5.7`. Defaults should be `None` and required
from params.

---

### F-12 · Three vertex-naming conventions — C

| Module | Convention | Slot 0 |
|---|---|---|
| `helper_functions` (pre-invert) | `[A0, A1, B1, B0]` | centre at `theta_i` |
| `helper_functions.generate_cable_sites_csv_zrot_from_P` | `A0, A1, B1, B0` | bottom centre |
| `preview.py`, `csv2geom_nlobe.py` | `A1, A0, B0, B1` | bottom centre |

Both post-invert conventions pick the *same points*; they disagree only on
labels. That is survivable in the STL path (which only ever reads `x`/`z` of
each slot) and fatal in the tendon path (F-07), which uses a signed `dz`.

Worse, `generate_cable_sites_csv_zrot_from_P`'s own docstring contradicts its
own code: it documents `P[i] = [A1_i, A0_i, B0_i, B1_i]` and
`site1: (B1x*cosψ, B1x*sinψ, B0y)` — mixing the `x` of one vertex with the `y`
of another — while the code correctly pairs `r0` with `B0y`.

**Action.** Phase 3: replace all tuple unpacking with a frozen dataclass
carrying explicit `centerline_start` / `centerline_end` /
`inner_start` / `inner_end` fields. No positional unpacking anywhere.

---

### F-13 · CadQuery pinning — C

`requirements.txt` pins `cadquery @ git+https://github.com/CadQuery/cadquery.git@d338160`
plus `cadquery-ocp==7.8.1.1.post1`. A clean `pip install cadquery` now resolves
to **cadquery 2.8.0 / cadquery-ocp 7.9.3.1.1** from PyPI and works, which
materially simplifies Phase 5/6/10 CI. Verified in this audit environment; the
full CSV → 21 STL → MJCF pipeline ran green on it.

---

## 3. Phase 2 definitions and instrumentation

`tools/geometry_audit.py` is read-only and imports nothing from the production
path. It already implements and reports:

```
requested length · continuous arc length · discrete backbone length
relative length error · requested Delta_theta · effective Delta_theta
number of units · terminal unit truncated · nominal beta
per-unit actual scale ratios · tip width · root width · full taper angle
```

emitted as a versioned JSON build manifest (`--out`). Discrete length is defined
exactly as specified:

```
sum( norm( center_point(theta[i+1]) - center_point(theta[i]) ) )
```

Both `length_definition` modes and both implemented `terminal_unit_policy`
values are exercised by the test suite. `uniform_scale` is specified but not yet
implemented — it must not ship until it declares which parameter it moves.

---

## 4. Baseline regression state

Established before any change, on the audit commits:

| Suite | Result |
|---|---|
| `spirob-assembly/test_pipeline.py` | **17 passed, 0 failed** |
| Primary: `spirob_csv_generator.py` | 21 elements, CSV written |
| Primary: `csv2geom_nlobe.py` | 21/21 STL written (CadQuery 2.8.0) |
| Primary: `csv2xml.py` | XML written |
| MJCF load (MuJoCo 3.10.0) | `nbody=22 njnt=21 nsite=128 ntendon=3 nu=3 nmesh=21` |
| MJCF 200-step rollout | no NaN; total mass 40.329 g; tendon rest length 219.66 mm |
| New `tests/` (Phase 1) | **55 passed, 3 xfailed** |
| New `tests/` (Phase 2) | **139 passed, 1 xfailed** |

Phase 1 held three strict xfails. Their Phase 2 disposition:

| Phase 1 xfail | Disposition |
|---|---|
| F-04 `test_base_of_robot_sits_at_origin` | **Wrong expectation.** Replaced by `test_inversion_anchors_the_tip_at_the_requested_length`, which asserts the real convention. |
| F-07 `test_preview_tendon_path_matches_mjcf` | **Real defect, fixed.** Marker removed; the test now passes on merit. |
| F-08 `test_tendon_offset_from_surface_is_constant` | **Deferred by decision.** Still strict-xfail, with the reason restated. |

Default pipeline output is byte-identical to `40fb850`: CSV, XML and all 21
STL files compare equal.

---

## 5. Proposed change surface

Nothing below is implemented yet. Listed for review before Phase 3 begins.

### 5.1 New files (no compatibility risk)

```
docs/GEOMETRY_AUDIT.md          this document
docs/FABRICATION_PIPELINE.md    Phase 5
docs/PARAMETER_REFERENCE.md     Phase 11
docs/LICENSE_PROVENANCE.md      Phase 11
docs/MIGRATION_NOTES.md         Phase 11
tools/geometry_audit.py         landed
spirob/geometry.py              SpiRobGeometry canonical model (Phase 3)
spirob/tendons.py               single canonical tendon-path definition
fabrication/                    Phase 5 CAD exporter + validator + splitter
gui/                            Phase 8, lazy-imported
tests/                          landed: test_analytical.py, test_conventions.py
```

### 5.2 Modified files

| File | Change | Risk |
|---|---|---|
| `helper_functions.py` | keep every public signature; re-express bodies over `SpiRobGeometry` | **Medium** — `spirob-assembly` does not import it, but user notebooks may |
| `spirob_csv_generator.py` | build geometry via the canonical model; identical CSV bytes under defaults | **High** — the CSV is the pipeline's stable interface |
| `preview.py` | delete the private fork, import canonical model, fix F-07 | Low — visual output only |
| `csv2xml.py` | consume canonical tendon paths; `--phi-deg` default `None` | **High** — MJCF is consumed by `spirob-assembly` |
| `csv2geom_nlobe.py` | consume canonical model; STL output unchanged | Medium |
| `build.py` | accept `--nlobe` no-op; emit build manifest | Low |
| `params.json` | add `length_definition`, `terminal_unit_policy`, `fabrication` | Low — all default to current behaviour |
| `README.md` | fix F-01, F-02 | None |

### 5.3 Backward-compatibility risks

1. **CSV schema is load-bearing.** `csv2geom_nlobe.py`, `csv2xml.py` and
   `spirob-assembly` all parse it by column name. Column names and order must
   not change. Any new field appends only.
2. **Byte-identical default output.** With `length_definition = "continuous_arc"`
   and `terminal_unit_policy = "truncate"`, the refactor must reproduce today's
   CSV, STL set and XML exactly. This needs a golden-file test recorded from the
   `a6ea28d` baseline *before* Phase 3 starts.
3. **F-04 and F-07 fixes change output.** Both are genuine corrections, so
   byte-identity is impossible once they land. They must ship in their own
   commit group, after the golden files exist, with the numeric delta published.
   The base offset in particular shifts every `z` by 2.6246 mm.
4. **`spirob-assembly` mount alignment.** `test_palm_camera_position` asserts
   `palm_cam Z ≈ L = 0.22628`. If the primary generator starts reporting a
   realised length of 223.6554 mm, that assertion is the first thing to break.
   The manifest must expose both lengths so the assembly layer can choose, and
   the assembly repo must keep its current default.
5. **`post_gen.robot_pos` defaults to `[0, 0, 0.22628]`.** Users have compensated
   for F-04 by hand. Fixing it will move existing scenes by 2.6 mm.
6. **`phi_deg` alias.** Accepting `taper_angle_deg` must error, not warn, if both
   are present and disagree.
7. **CadQuery version.** Moving off the git pin changes the tessellator; STL
   byte-identity across versions is not guaranteed even when the solid is.
   Golden tests must compare geometry (volume, bbox, manifoldness), not bytes.

### 5.4 Recommended commit groups

```
1  audit + tools + tests            <- this change group
2  golden-file baseline capture
3  documentation corrections        (F-01, F-02, F-11, F-12 docstrings)
4  canonical geometry model         (Phase 3, byte-identical output)
5  geometry corrections             (F-04, F-07, F-08 — output changes)
6  fabrication CAD + validation     (Phases 5, 6)
7  splitter                         (Phase 7)
8  GUI                              (Phase 8)
9  docs, examples, migration notes  (Phase 11)
```

---

## 6. Unresolved questions

These need a decision before Phase 3, and three of them need physical
measurement rather than analysis.

1. **Which length is authoritative?** Should a user asking for `L = 226.28 mm`
   receive a robot whose backbone measures 226.28 mm when uncurled? The
   specification's default preserves the current answer (no). Recommend flipping
   the default to `discrete_backbone` in a future major version, since it also
   dissolves F-04.
2. ~~**Where should the remainder go?**~~ **RESOLVED.** The partial unit belongs
   at the base; that is what preserves large-at-base ordering. Users who need
   whole units select `terminal_unit_policy: "whole_units"`, which extends the
   spiral and reports the excess. The partial unit is never moved to the tip.
3. **What is the canonical tendon path?** Constant offset from the realised
   staircase surface, or a smooth taper fitted through the links? The former is
   simpler and matches the geometry; the latter is closer to a real routed
   cable. This choice must be made once and shared by MJCF and CAD.
4. **Does `d_tip` mean nominal or realised width?** Fabrication cares. Currently
   nominal, off by −0.92 %.
5. **Physical validation needed:** the elastic-layer percentages (5 % two-cable,
   10 % three-cable) are quoted defaults, not measurements from this lab's TPU.
   Given the cantilever free-vibration characterisation already done on the
   printed TPU specimens, the modulus from that work should be what sets the
   minimum printable thickness — not a literature default.
6. **Physical validation needed:** `tendon_hole_diameter_m` must be set from the
   actual cable and the actual printer's hole shrinkage. Phase 5 must warn rather
   than guess.
7. **Physical validation needed:** `inter_unit_gap_m` determines the achievable
   bend before self-contact. The 21-unit staircase with `Delta_theta = 30.78°`
   implies a specific wedge angle; whether a printed TPU hinge survives that
   range is an empirical question.
