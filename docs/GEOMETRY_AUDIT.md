# SpiRob Geometry Audit

**Scope:** `spirob-physics-model-development-pipeline` (primary), with cross-checks
against `spirob-assembly` and behavioural reference to `Open-Spiral-Robots`.

**Status:** Phase 1 complete. **Phase 2 complete** — see
`docs/CANONICAL_GEOMETRY.md`. Findings below carry their post-Phase-2
disposition. Three Phase 1 findings were revised on further evidence; those
revisions are marked **REVISED** and explained in place. Historical wording is
preserved and struck through rather than deleted, so the original reasoning
stays auditable.

**Phase numbering.** Two numbering schemes appear in this project's history and
they do not line up. The original eleven-part task specification numbered the
canonical geometry model "Phase 3", CAD "Phase 5", the GUI "Phase 8" and so on.
The project as executed uses its own sequence: **Phase 1 = audit**,
**Phase 2 = canonical geometry layer**, with fabrication, GUI and splitter work
still ahead and not yet numbered. Throughout this document, unqualified "Phase
1" and "Phase 2" mean the *project* phases. Where a reference to the original
specification's numbering is intended, it is written as "spec Phase N".

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
against those equations. A formal `docs/LICENSE_PROVENANCE.md` is still to be
written as part of the final deliverables; until then this note stands.

---

## 0. Length vocabulary

Five quantities are easy to conflate and are kept strictly separate everywhere
in this document and in `spirob/geometry.py`. Two of them are *lengths*; two are
*effects* that relate those lengths; one is a *structural* property.

| Term | Meaning | Field |
|---|---|---|
| **requested continuous length** | what the user asks for in `params.json` as `L`; a spiral **arc** length | `requested_continuous_length_m` |
| **effective continuous length** | the spiral **arc** actually generated, after the terminal-unit policy has been applied | `effective_continuous_length_m` |
| **discrete chord (backbone) length** | the sum of straightened **chords**, i.e. what the built link chain physically measures | `discrete_chord_length_m` |
| **partial-unit completion** | the *effect* `requested -> effective`. Zero under `exact_requested_length`; positive under `whole_units` | `completion_delta_m` / `_rel` |
| **arc-versus-chord shortening** | the *effect* `effective -> discrete`. Always negative, second order in `Delta_theta`, present under **both** policies | `chord_deficit_m` / `_rel` |

```
requested continuous length
    |
    |  partial-unit completion        (unit-structure effect)
    v
effective continuous length
    |
    |  arc-versus-chord shortening    (discretisation effect)
    v
discrete chord (backbone) length
```

The two effects have **different causes and different remedies**. Completion is
about whether the requested length contains a whole number of nominal units, and
is controlled by `terminal_unit_policy`. Arc-versus-chord shortening is about
straightening curved intervals into straight links, and is controlled only by
`Delta_theta`. Selecting `whole_units` does *not* remove the chord deficit — it
slightly increases it, because the model is longer.

The two shipped policies are:

| `terminal_unit_policy` | Behaviour |
|---|---|
| `exact_requested_length` | **Default.** Effective length equals requested length exactly. Permits one partial unit, which after inversion is the base link. |
| `whole_units` | Extends the spiral to the next complete nominal unit boundary. Never shortens. No partial unit. Reports the excess. |

Earlier drafts of this document floated a `length_definition` parameter with
values `continuous_arc` / `discrete_backbone`, and a `terminal_unit_policy` with
values `truncate` / `uniform_dtheta` / `uniform_scale`. **None of those shipped.**
Where they still appear below they are marked as superseded. Note separately that
`tools/geometry_audit.py` retains an internal, independent reference
implementation whose own helper argument is still spelled `truncate`; that is a
test-oracle detail, not a user-facing policy name, and is deliberately not
renamed so the oracle stays a genuine cross-check.

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
| F-03 | A — **reported, by design** | Requested `L` is a continuous arc length; the built model realises a chord sum 1.16 % shorter. Now measured and reported separately as arc-versus-chord shortening. |
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

### F-03 · Arc-versus-chord shortening — A, now measured and reported

`q0` is solved so the *continuous arc* length equals `L`. The pipeline then
straightens each sampled interval as a **chord**. Chords are shorter than arcs,
so the built chain measures less than the requested arc:

| Quantity | Value |
|---|---|
| Requested continuous length | 226.2800 mm |
| Effective continuous length | 226.2800 mm |
| Discrete chord (backbone) length | **223.6554 mm** |
| Arc-versus-chord shortening | −2.6246 mm |
| Relative | **−1.1599 %** |

This matches the specification's expected 223.6554 mm exactly. The effect is
second order in `Delta_theta`; `test_discrete_converges_to_continuous` confirms
`O(Delta_theta^2)` convergence across 0.5°–60°.

**This is a discretisation effect, not a unit-completion effect.** Under the
default policy the effective length equals the requested length exactly, so the
completion delta is zero while this deficit is −2.6246 mm. The two are reported
as separate fields and must not be summed into a single "error". See §0.

**Disposition (Phase 2): reported, not silently absorbed.** The superseded
proposal was a `length_definition` parameter with values
`continuous_arc` / `discrete_backbone` that would have re-solved `q0` so the
*chord sum* hit `L`. That did not ship, and deliberately so: it would change the
spiral for every existing `params.json`. What shipped instead is measurement —
`LengthReport` carries all three lengths and both effects, `to_manifest()`
serialises them, and `tools/geometry_audit.py` prints them. A user who wants a
226.28 mm backbone can now see exactly what they are getting and by how much it
differs.

The chord-solving routine still exists in `tools/geometry_audit.py` as
`solve_q0_for_discrete_length()` (deterministic bracketed bisection, explicit
`tol` and `max_iter`, raises on failure to bracket or converge) and remains
covered by `tests/test_analytical.py`. It is retained as a verified reference
implementation should a future major version choose to adopt it; it is not
reachable from the shipped pipeline. Measured: it drives the relative error
below `1e-10`, raising `q0` from 10.9632 to a value whose continuous arc length
is 228.9254 mm.

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

**The 2.6246 mm is the continuous-arc versus discrete-chord difference** — the
same quantity reported by F-03 as arc-versus-chord shortening, appearing here in
the frame because the tip is the anchored end. It is **not** an accidental
MuJoCo root displacement, and the geometry must **not** be shifted by it.

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

---

### F-05 · The partial unit is the base link — REVISED: correct by design

**Phase 1 flagged this as surprising and implicitly wrong. It is neither.**

`theta[i] = min(i*Delta_theta, q0)` clips the last spiral interval at `q0`, and
`Invert_pose()` then reverses the chain, so the partial unit becomes
`link_001`. With the committed defaults, `q0/Delta_theta = 20.4076`, so unit 21
spans 12.5452° instead of 30.78°. (This clipping is the *mechanism*. It is not a
policy name: the shipped policy that permits it is `exact_requested_length`.)

**The requested length creates a partial terminal unit. After
`Invert_pose()`, this unit becomes the base link by design, preserving the
intended large-at-base and small-at-tip ordering.**

The spiral is generated tip-first: `theta = 0` is the smallest radius. The
clipped interval therefore lands on the largest-radius end, which after
inversion is the base. The smallest element remains at the tip. Moving the
partial unit to the tip would invert the size ordering the whole design depends
on. **This is not an inversion or ordering defect and must not be "fixed".**

Confirmed numerically: `link_001` is simultaneously the partial unit
(12.5452° span) *and* the largest unit (31.0876 mm realized width). It is
partial in **angular span** while remaining at the **large-radius, wide end**.
`link_021` is the smallest unit, at the tip, and is complete.

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
sagitta factor `dist(O, chord_i)/r_c(theta_i)`, which is < 1 and shrinks as the
unit's angular span grows.

The root figure above is policy-dependent, which is worth stating precisely.
Under `exact_requested_length` the base unit spans only 12.5452°, so its sagitta
factor is 0.99966 and its realised width (31.0876 mm) is close to the nominal
width at its own start angle. Under `whole_units` the base unit spans the full
30.78°, the factor drops to 0.990794, and the realised root width *falls* to
30.8119 mm even though the spiral now extends further and the nominal root width
*rises* to 33.4726 mm. Fabrication must therefore read
`realized_root_width_m`; `nominal_root_width_m` is the width the continuous
spiral reaches at `q0`, which no straightened block ever realises.

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

**Action.** Future consumer-migration work: replace all tuple unpacking with a frozen dataclass
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

## 3. Instrumentation as shipped

`tools/geometry_audit.py` now has two clearly separated halves.

**Upper half — the reference oracle.** An independent transcription of the
published equations, importing nothing from the production path. It is retained
precisely so the test suite can check `spirob/geometry.py` against something
that was not derived from it. Its internal helpers still use the exploratory
vocabulary (`theta_samples(..., "truncate")`, `solve_q0_for_discrete_length`,
`discrete_backbone_length`). Those are oracle-internal argument names, **not**
user-facing policy values, and are intentionally left alone.

**Lower half — reporting.** Consumes the canonical model and performs no
geometry of its own. The Phase 1 statement that this file "is read-only and
imports nothing from the production path" is therefore **superseded**: the
reporting half imports `spirob.geometry`.

Reported per build, via `SpiRobGeometry.to_manifest()`:

```
requested continuous length · effective continuous length
discrete chord (backbone) length
partial-unit completion delta (absolute and relative)
arc-versus-chord shortening (absolute and relative)
requested Delta_theta · effective/actual per-unit angular spans
units total · complete units · partial unit present · partial unit span
effective q0 · nominal beta · per-unit scale ratios
nominal and realized tip width · nominal and realized root width
full taper angle · base-frame anchor · per-unit slit references
```

emitted as a versioned JSON build manifest (`--out`, `schema_version` 2.0).
Discrete chord length is defined exactly as specified:

```
sum( norm( center_point(theta[i+1]) - center_point(theta[i]) ) )
```

Both shipped `terminal_unit_policy` values — `exact_requested_length` and
`whole_units` — are exercised by the test suite, and
`--compare-policies` prints them side by side. The exploratory
`length_definition` parameter and the `uniform_dtheta` / `uniform_scale`
policies were **not** implemented and are not planned; see §0.

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

## 5. Change surface — status

This section was written in Phase 1 as a forward plan. It is now annotated with
what actually landed. Rows marked **landed** are complete on
`canonical-geometry`; the rest remain future work.

### 5.1 New files

```
docs/GEOMETRY_AUDIT.md          landed  (this document)
docs/CANONICAL_GEOMETRY.md      landed  (Phase 2)
spirob/__init__.py              landed  (Phase 2)
spirob/geometry.py              landed  (Phase 2) — SpiRobGeometry canonical model
tools/geometry_audit.py         landed  (Phase 1, rewired in Phase 2)
tests/test_analytical.py        landed  (Phase 1)
tests/test_conventions.py       landed  (Phase 1, revised in Phase 2)
tests/test_canonical_geometry.py landed (Phase 2)

docs/FABRICATION_PIPELINE.md    future  (fabrication phase)
docs/PARAMETER_REFERENCE.md     future
docs/LICENSE_PROVENANCE.md      future
docs/MIGRATION_NOTES.md         future
fabrication/                    future  — CAD exporter + validator + splitter
gui/                            future  — lazy-imported
```

The Phase 1 plan listed the canonical model as "Phase 3" and a separate
`spirob/tendons.py`. Both are superseded: the model landed as project Phase 2,
and the canonical tendon definition lives in
`spirob.geometry._build_tendon_paths` rather than a separate module, so there is
one file to read and one place to change the routing rule when F-08 is settled.

### 5.2 Modified files

| File | Change | Status / Risk |
|---|---|---|
| `preview.py` | delete the private fork, import canonical model, fix F-07 | **landed** — visual output only; no pipeline artefact changed |
| `params.json` | add `terminal_unit_policy` | **landed** — key is optional and defaults to current behaviour |
| `helper_functions.py` | keep every public signature; re-express bodies over `SpiRobGeometry` | not started. **Medium** — `spirob-assembly` does not import it, but user notebooks may |
| `spirob_csv_generator.py` | build geometry via the canonical model; identical CSV bytes under defaults | not started. **High** — the CSV is the pipeline's stable interface |
| `csv2xml.py` | consume canonical tendon paths; `--phi-deg` default `None` | not started. **High** — MJCF is consumed by `spirob-assembly` |
| `csv2geom_nlobe.py` | consume canonical model; STL output unchanged | not started. Medium |
| `build.py` | accept `--nlobe` no-op; emit build manifest | not started. Low |
| `README.md` | fix F-01, F-02 | not started. None |

The superseded `length_definition` key is not in the shipped `params.json` and
is not planned; a `fabrication` block belongs to the fabrication phase.

### 5.3 Backward-compatibility risks

1. **CSV schema is load-bearing.** `csv2geom_nlobe.py`, `csv2xml.py` and
   `spirob-assembly` all parse it by column name. Column names and order must
   not change. Any new field appends only.
2. **Byte-identical default output.** ~~With `length_definition = "continuous_arc"`
   and `terminal_unit_policy = "truncate"`~~ — superseded naming. The
   requirement, restated: with `terminal_unit_policy = "exact_requested_length"`
   (the default, and the behaviour when the key is absent), the refactor must
   reproduce the baseline CSV, STL set and XML exactly. **Verified for Phase 2**
   by building `main` in a separate worktree and comparing: CSV identical, XML
   identical, 21/21 STL identical. The same check must be repeated for each
   remaining consumer migration.
3. ~~**F-04 and F-07 fixes change output.**~~ **CORRECTED — this risk was
   misdiagnosed and does not exist.**

   The Phase 1 text claimed both findings were defects whose repair would break
   byte-identity, and specifically that "the base offset shifts every `z` by
   2.6246 mm". Both halves are wrong.

   - **F-04 is not a defect and nothing is shifted.** `Invert_pose()`
     intentionally anchors the **tip** at the requested length. The 2.6246 mm is
     the continuous-arc versus discrete-chord difference (F-03) surfacing in the
     frame — not an accidental root displacement, and not a quantity to correct
     away. The simulated root pose is established by `csv2xml.py` from
     `post_gen.robot_pos`, independently of this frame offset. No geometry shift
     is required or permitted.
   - **F-07 was a real defect, and fixing it changed no pipeline artefact.** The
     bug lived only in `preview.py`, which produces a PNG and nothing else. CSV,
     STL and XML were never affected, and are byte-identical after the fix.

   The residual risk in this item therefore reduces to F-08, which is
   deliberately unfixed precisely because settling it *would* change MJCF
   output. See item 3a.

3a. **F-08 will change MJCF output when it is settled.** The tendon routing rule
   is still undecided. Whichever rule is chosen, adopting it will move tendon
   site positions and therefore change `spirob_physics_model.xml`. That change
   must ship in its own commit group with the numeric delta published, and must
   be coordinated with the fabrication phase so the MJCF tendon and the printed
   channel agree.
4. **`spirob-assembly` mount alignment.** `test_palm_camera_position` asserts
   `palm_cam Z ≈ L = 0.22628`. If the primary generator starts reporting a
   realised length of 223.6554 mm, that assertion is the first thing to break.
   The manifest must expose both lengths so the assembly layer can choose, and
   the assembly repo must keep its current default.
5. ~~**`post_gen.robot_pos` ... users have compensated for F-04 by hand.**~~
   **CORRECTED.** `post_gen.robot_pos = [0, 0, 0.22628]` is not a hand-applied
   workaround for a defect. It is how `csv2xml.py` establishes the simulated
   root pose: `rel[0]` is overridden from `post_gen`, so the CSV-frame base
   offset never reaches the model. Verified on the built model — `link_001`
   world position is exactly `[0, 0, 0.22628]`. Nothing here needs "fixing" and
   no existing scene moves.
6. **`phi_deg` alias.** Accepting `taper_angle_deg` must error, not warn, if both
   are present and disagree.
7. **CadQuery version.** Moving off the git pin changes the tessellator; STL
   byte-identity across versions is not guaranteed even when the solid is.
   Golden tests must compare geometry (volume, bbox, manifoldness), not bytes.

### 5.4 Recommended commit groups

```
DONE  Phase 1  audit + oracle tooling + tests            40fb850
DONE  Phase 2  canonical geometry model + preview
               migration + F-07 fix                      06221eb, 9f0735a
               (byte-identical CSV / STL / XML verified)
DONE  Phase 2  documentation reconciliation              this change group

NEXT           migrate spirob_csv_generator.py           (byte-identity gate)
               migrate csv2geom_nlobe.py, csv2xml.py
               documentation corrections F-01, F-02, F-11, F-12 docstrings
               settle F-08 routing rule                  (changes MJCF output)
               fabrication CAD + validation
               splitter
               GUI
               examples, parameter reference, migration notes
```

Phase numbering here is the project's own; the original specification's
"Phase 3 = canonical model, Phase 5 = CAD, Phase 8 = GUI" numbering is not used.
F-04 no longer appears as a corrective work item, because it is not a defect.

---

## 6. Unresolved questions

Three of these need physical measurement rather than analysis. Item 3 is the
one blocking the fabrication phase.

1. **Which length is authoritative?** Should a user asking for `L = 226.28 mm`
   receive a robot whose backbone measures 226.28 mm when uncurled? Today: no —
   `L` is a continuous arc length and the chord sum is 1.16 % shorter, which is
   now measured and reported rather than hidden (F-03). ~~Recommend flipping the
   default to `discrete_backbone` ... since it also dissolves F-04.~~
   **Superseded on two counts:** `discrete_backbone` never shipped as a
   parameter value, and F-04 is not a defect for anything to dissolve. If a
   future major version wants chord-exact lengths, the verified
   `solve_q0_for_discrete_length()` reference implementation is available, but
   adopting it changes the spiral for every existing `params.json` and must be a
   deliberate breaking change. Note that this is **independent** of
   `terminal_unit_policy`: `whole_units` addresses partial units, not the chord
   deficit.
2. ~~**Where should the remainder go?**~~ **RESOLVED.** The partial unit belongs
   at the base; that is what preserves large-at-base ordering. Users who need
   whole units select `terminal_unit_policy: "whole_units"`, which extends the
   spiral and reports the excess. The partial unit is never moved to the tip.
3. **What is the canonical tendon path? (F-08 — still open, still the blocker.)**
   Constant offset from the realised staircase surface, or a smooth taper fitted
   through the links? The former is simpler and matches the geometry; the latter
   is closer to a real routed cable. Phase 2 did **not** resolve this. It
   reproduced the existing rule exactly, so MJCF output is unchanged, and
   consolidated it into a single definition
   (`spirob.geometry._build_tendon_paths`) so there is now one place to change.
   The decision must be made once and shared by MJCF and CAD, and it will change
   `spirob_physics_model.xml` when it lands. Tracked by the strict xfail
   `test_tendon_offset_from_surface_is_constant`.
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
