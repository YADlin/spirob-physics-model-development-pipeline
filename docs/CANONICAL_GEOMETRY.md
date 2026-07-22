# Canonical Geometry Layer

`spirob/geometry.py` holds the one authoritative geometric description of a
SpiRob. It is computed once and consumed by everything downstream. No consumer
may re-derive spiral maths.

The layer imports only the standard library and NumPy — never matplotlib,
MuJoCo, CadQuery, Qt or VTK — so it stays importable in a headless, CAD-free
environment. Enforced by `test_canonical_layer_has_no_heavy_dependencies`.

---

## Quick start

```python
import json
from spirob.geometry import from_params

geo = from_params(json.load(open("params.json")))

geo.units[0].link_name              # 'link_001' — base, largest
geo.units[-1].link_name             # 'link_021' — tip, smallest
geo.lengths.discrete_chord_length_m # what the built chain measures
geo.tendon_path(0).routed_polyline()
geo.to_manifest()                   # JSON-serialisable build manifest
```

Command line:

```bash
python tools/geometry_audit.py --params params.json
python tools/geometry_audit.py --params params.json --policy whole_units
python tools/geometry_audit.py --params params.json --compare-policies
python tools/geometry_audit.py --params params.json --out build/manifest.json
```

---

## Conventions of record

All of these are stated once, in the `spirob/geometry.py` module docstring, and
asserted by `test_module_documents_the_conventions`.

| Question | Answer |
|---|---|
| Units | SI throughout — metres, radians. A name ending `_deg` or `_mm` is the only exception. `params.json` keeps its degree keys and is converted at the boundary. |
| Is `phi` a half-angle? | **No.** `phi` is the FULL included taper angle. The half-width slope is `tan(phi/2)`. `taper_angle_deg` is accepted as an alias. |
| Angle direction | `theta` increases counter-clockwise from +X. |
| `theta = 0` | The **tip** — smallest radius. |
| `theta = q0` | The **base** — largest radius. |
| Link order | `units[0]` is `link_001`, the base. Matches the CSV and MJCF. |
| Spiral order | `units_tip_to_base[0]` is `theta = 0`, the tip. |
| Relation | Exact reverses: `link[k] = spiral[N-1-k]`. Both indices are stored on every `UnitRecord`; never infer ordering from a name. |
| Reported lengths | Three distinct fields — see below. Arcs and chords are never conflated. |
| Tendon points | Defined once here, in the inverted (CSV) frame, in two stages: `attachment_m` on the surface, `routed_m` after shift and correction. |

### Field naming

Every dimensional field declares its unit in its name (`_m`, `_rad`, `_deg`,
`_rel`). Enforced by
`test_every_dimensional_field_carries_its_unit_in_its_name`.

### Tolerances

`Tolerances` is a documented, overridable dataclass:

| Field | Default | Meaning |
|---|---|---|
| `angle_rad` | `1e-9` | angles closer than this are equal; decides whether a partial unit exists |
| `length_m` | `1e-12` | lengths closer than this are equal |
| `solver_rel` | `1e-12` | relative convergence target |
| `solver_max_iter` | `200` | hard iteration cap; exceeding it raises |

---

## What `Invert_pose()` does, and why it is correct

`helper_functions.Invert_pose(quads, Length)`:

1. reflects `y` about the requested continuous length: `y -> -y + Length`
2. permutes vertex slots `[0,1,2,3] -> [1,0,3,2]`
3. reverses the element list

The result is the intended physical arrangement: largest at the base, smallest
at the tip.

**When the requested length does not contain a whole number of nominal units,
the partial unit is the last spiral interval, and after `Invert_pose()` it
becomes `link_001`, the base link. This is correct by design and is preserved.**
It is not an inversion error. Moving the partial unit to the tip would invert
the size ordering the design depends on.

The partial unit is short in **angle**, not small in **radius**: with the
committed defaults `link_001` spans 12.5452° yet is still the widest unit at
31.0876 mm.

Step 1 anchors the **tip** at exactly `y = requested_length_m`. The base
therefore sits at `requested - discrete`, exposed as
`BaseFrame.origin_offset_m`. This does not reach the simulation, because
`csv2xml.py` overrides the root body position from `post_gen.robot_pos`.

---

## Terminal unit policy

```json
{ "terminal_unit_policy": "exact_requested_length" }
```

| Value | Behaviour |
|---|---|
| `exact_requested_length` | **Default.** Preserves the requested continuous length exactly. Permits one partial unit at the base. Byte-identical to pre-Phase-2 output. |
| `whole_units` | Extends the spiral forward to the next complete nominal unit boundary. Never shortens. No partial unit. Reports the excess. |

Omitting the key selects the default, so every existing `params.json` keeps
working unchanged.

### Measured comparison, committed defaults

`L = 226.28 mm`, `d_tip = 7.139 mm`, `phi = 6.3°`, `Delta_theta = 30.78°`

| Quantity | `exact_requested_length` | `whole_units` |
|---|---:|---:|
| requested continuous (mm) | 226.2800 | 226.2800 |
| effective continuous (mm) | 226.2800 | 239.2515 |
| discrete chord (mm) | 223.6554 | 236.3856 |
| completion delta (mm) | +0.0000 | **+12.9715** |
| completion delta (%) | +0.0000 | **+5.7325** |
| chord deficit (mm) | −2.6246 | −2.8658 |
| chord deficit (%) | −1.1599 | −1.1978 |
| units total | 21 | 21 |
| complete units | 20 | **21** |
| partial unit | True | **False** |
| partial span (deg) | 12.5452 | 0.0000 |
| effective `q0` (rad) | 10.9632 | 11.2815 |
| nominal root width (mm) | 32.0449 | 33.4726 |
| realized root width (mm) | 31.0876 | **30.8119** |
| realized tip width (mm) | 7.0733 | 7.0733 |

Two results here are worth reading carefully.

**Both policies produce 21 units.** `whole_units` does not add a unit; it
*completes* the one that was partial. That is why the length grows 5.73 % while
the unit count is unchanged.

**`whole_units` raises the nominal root width but lowers the realized one**
(32.0449 → 33.4726 nominal, 31.0876 → 30.8119 realized). This is not a bug. Per
F-06 each straightened unit is a constant-width block, and its realized width is
evaluated at the unit's tip-side boundary with the chord sagitta factor applied.
Under `exact_requested_length` the base unit spans only 12.5452°, so its sagitta
factor is 0.99966 and realized ≈ nominal. Under `whole_units` the base unit
spans the full 30.78°, so the factor drops to 0.9908 and the block comes out
narrower — even though the continuous spiral now extends further.

The consequence for fabrication: `nominal_root_width_m` is the width the
*continuous* spiral reaches at `q0`, which no printed block ever realizes. Use
`realized_root_width_m`.

---

## The three lengths

Unit completion and arc-versus-chord shortening are **different effects** and
are reported separately. Never add them into a single "error".

```
requested_continuous_length_m    what the user asked for
    │
    │  completion_delta_m / _rel      <- unit-completion effect
    │                                    zero under exact_requested_length
    ▼
effective_continuous_length_m    spiral arc actually generated
    │
    │  chord_deficit_m / _rel         <- arc-to-chord effect
    │                                    always negative, O(Delta_theta²)
    ▼
discrete_chord_length_m          what the built link chain measures
```

`LengthReport.requested_vs_discrete_m` gives the total the user experiences, if
a single number is genuinely wanted.

---

## Widths

`nominal` comes from `d(theta) = a*(E-1)*exp(b*theta)`. `realized` is what
straightening actually produces:

```
realized_width = d(theta_start) * dist(O, chord) / r_c(theta_start)
```

The sagitta factor is `< 1`, so realized is always smaller. Both are reported;
fabrication should use `realized`.

---

## Deferred: dovetail and slit fabrication requirements

**Not implemented in Phase 2.** No solids are built. What Phase 2 provides is
stable, named reference geometry so a later CAD phase can consume it without
re-deriving anything.

### Anchors available today

| Anchor | Type | Purpose |
|---|---|---|
| `SpiRobGeometry.base_frame.mount_plane_point_m` | `(x, y, z)` | the flat face a dovetail or mount plate attaches to |
| `SpiRobGeometry.base_frame.mount_plane_normal` | unit vector | outward normal of that face |
| `SpiRobGeometry.base_frame.root_width_m` | metres | realized outer width at the mounting face |
| `SpiRobGeometry.base_frame.origin_offset_m` | metres | where the base sits on the inversion axis |
| `UnitRecord.slit_reference_m` | `(x, y, z)` | centre of the boundary between this unit and the next one toward the tip — where the compliant slit goes |
| `UnitRecord.slit_normal` | unit vector | slit plane normal, along the unit's local axis |
| `UnitRecord.local_frame_origin_m` / `local_frame_axis` | | per-unit placement frame |
| `TendonPoint.attachment_m` / `routed_m` | `(x, y, z)` | channel centreline for printed tendon holes |

Slit references are ordered base-to-tip and strictly increasing in `z`
(`test_slit_references_are_ordered_base_to_tip`). There are `n_units` of them;
the last one is the tip face.

### Parameters a later phase must add

None of these exist yet. They belong in a `fabrication` block in
`params.json`, defaulting to disabled so existing files keep working.

**Dovetail (base)**

| Parameter | Notes |
|---|---|
| `dovetail_enabled` | default `false` |
| `dovetail_width_m`, `dovetail_depth_m`, `dovetail_height_m` | nominal envelope |
| `dovetail_flare_deg` | included flare angle of the tail |
| `dovetail_clearance_m` | per-face clearance; **must be measured, not guessed** |
| `dovetail_fillet_m` | stress relief at the root |

**Slits (inter-unit compliance)**

| Parameter | Notes |
|---|---|
| `slit_enabled` | default `false` |
| `slit_width_m` | kerf; interacts with nozzle diameter |
| `slit_depth_fraction` | depth as a fraction of local realized width |
| `slit_profile` | `straight` \| `filleted` \| `keyhole` |
| `slit_count_per_unit` | 1 for two-cable, possibly `n_cables` for radial designs |

**Tolerances**

| Parameter | Notes |
|---|---|
| `linear_tolerance_m`, `angular_tolerance_deg` | tessellation |
| `printer_build_volume_mm` | split checking |

### Open questions requiring physical measurement

1. `dovetail_clearance_m` depends on printer and material shrinkage. The TPU
   cantilever free-vibration characterisation already done in this lab should
   set it, not a literature default.
2. `slit_width_m` and `slit_depth_fraction` determine achievable bend before
   self-contact. With `Delta_theta = 30.78°` the implied wedge angle is fixed
   by the geometry; whether a printed TPU hinge survives that range across
   21 units is empirical.
3. Whether the partial base unit is acceptable as a printed part, or whether
   `whole_units` should become the fabrication default. Purely a fabrication
   decision — the simulation default stays `exact_requested_length`.

---

## Migration status

| Consumer | Status |
|---|---|
| `tools/geometry_audit.py` | migrated — reporting comes from the canonical model |
| `preview.py` | migrated — private fork deleted, tendon points shared |
| `spirob_csv_generator.py` | not yet — still calls `helper_functions` directly |
| `csv2geom_nlobe.py` | not yet — reads the CSV |
| `csv2xml.py` | not yet — reads the CSV, computes routing independently |
| future STEP/STL exporter | will consume the canonical model from the start |

`helper_functions.py` is untouched and still works. The canonical layer
reproduces its output bit-for-bit
(`test_canonical_reproduces_legacy_curled_pose_exactly` and siblings), which is
what makes the remaining migrations safe to do one at a time.
