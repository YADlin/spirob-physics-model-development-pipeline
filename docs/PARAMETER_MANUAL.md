# SpiRob generator parameter manual

This manual describes the generator's JSON parameters, build options, desktop
controls and supporting tools. Use Python 3.12 and the supplied uv requirements.
Commands below run from the repository root. JSON files cannot contain comments.

## 1. Start here: which dimension are you changing?

Each link has a local body frame. **Z** runs along the straight robot, from base
towards tip; **X** spans the two-cable link's width; **Y** spans its thickness.
The end-on view is therefore the **XY** view. The robot's world pose can rotate
these directions; a world-axis measurement is not necessarily local thickness.

For a two-cable robot, the adjustable absolute dimension is **the centre
thickness of the largest base link, `link_001`**, along local Y. It includes
both sides of the centre plane. It is not half-thickness, axial link length,
flexure width, cable-hole diameter, or thickness at a hexagon's outer edge.

Let `W_base` be the generated base-link width, `W_i` another link's width and
`T_base` the chosen centre thickness. The generator uses:

```text
T_i = T_base × W_i / W_base
T_tip = T_base × W_tip / W_base
```

Thus **base thickness is the independent setting; tip thickness is calculated**.
The thickness is not constant along the chain. This scaling lets all complete
links keep sharing a single STL template. A partial base uses its own STL.
To target a particular tip thickness, use `T_base = T_tip × W_base / W_tip`.
Changing the absolute base thickness does not change X widths or Z link lengths.

The new initial setting is **thickness equals width**: `T_i = W_i` for every
link. In a rectangular section this makes its XY envelope square. A hex section
has the same square bounding envelope but retains its six-sided outline.

| User interface | Thickness equals width | Example: 20 mm at the base |
|---|---|---|
| Build command | `--base-thickness-mm auto` | `--base-thickness-mm 20` |
| JSON parameter | `"base_thickness_m": null` | `"base_thickness_m": 0.020` |
| Desktop GUI | Enter `auto` in **Base centre thickness (mm or auto)** | Enter `20` |

For the supplied two-cable geometry (`L=0.22628`, `d_tip=0.007139`,
`phi_deg=6.3`, `Delta_theta_deg=30.78`, exact-length policy):

| Location | Actual generated width | Initial centre thickness |
|---|---:|---:|
| Base, `link_001` | 31.087574 mm | 31.087574 mm |
| Tip, `link_021` | 7.073275 mm | 7.073275 mm |

These are generated dimensions, not measurements of the photograph. The nominal
`d_tip` input is 7.139 mm; the discretised link's width is slightly different.
The authoritative per-link numbers are saved in **`section_dimensions.json`**.

**Explicit versus automatic:** a numeric `base_thickness_m` stays fixed when
other geometry inputs change. `null`/`auto` recalculates thickness to equal the
new widths. Older files with only `flat_thickness_ratio` retain that ratio.
Do not put both thickness fields in one JSON file. The CLI thickness override
removes the old ratio from the saved build parameters, so it can safely override
an older file. With neither field present, the two-cable default is ratio 1.

## 2. Quick commands

Install into your existing environment:

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

For a fresh environment, run `uv venv --python 3.12 .venv` first.

Build the initial six-sided two-cable model:

```bash
uv run python build.py \
  --params examples/params-two-cable-hex.json \
  --base-thickness-mm auto --timestep 0.0001 \
  --collision-mode mesh --collision-margin-m 0 \
  --no-preview --cad --cad-profile simulation \
  --output-dir build/two-cable-square
```

Build the same shape with a 20 mm base centre thickness:

```bash
uv run python build.py \
  --params examples/params-two-cable-hex.json \
  --base-thickness-mm 20 --timestep 0.0001 \
  --collision-mode mesh --collision-margin-m 0 \
  --no-preview --output-dir build/two-cable-20mm
```

Use `examples/params-two-cable.json` for the rectangular section. Add
`--hex-section` to select the hex outline from a rectangular parameter file.
Use `examples/params-three-cable.json` or `examples/params-four-cable.json` for
n-lobe sections; do not pass the two-cable thickness or hex flags to them.

```bash
uv run python build.py --params examples/params-three-cable.json \
  --no-preview --timestep 0.0001 --output-dir build/three-cable

uv run python tools/inspect_section.py \
  --mjcf build/two-cable-square/spirob_physics_model.xml \
  --out build/two-cable-square/sections.png --show

uv run python -m mujoco.viewer \
  --mjcf=build/two-cable-square/spirob_physics_model.xml

uv run python design_gui.py \
  --params examples/params-two-cable-hex.json \
  --output-dir build/gui-review
```

## 3. Core geometry parameters in JSON

The sample values below describe the shipped examples, not universal defaults.
These parameters are common to two-cable and n-cable models.

| Parameter | Unit; sample | Meaning and effect | Accepted values / notes |
|---|---|---|---|
| `L` | m; `0.22628` | Requested **continuous spiral arc length**. It determines the robot geometry before discretisation. | Positive finite number. It is not exactly the straight articulated chain length. |
| `d_tip` | m; `0.007139` | Nominal tip width in the continuous spiral construction. Controls the small end's geometric scale. | Positive finite number. Neither final discretised width nor an n-lobe STL's XY bounding diameter is guaranteed to equal it. |
| `phi_deg` | degrees; `6.3` | Full included taper angle. The half-width slope uses `tan(phi_deg/2)`. | Strictly between 0 and 45 degrees. Do not enter the half-angle. |
| `Delta_theta_deg` | degrees; `30.78` | Angular span of one nominal segment in the curled spiral. Larger values generally produce fewer, coarser links. | Strictly between 0 and 180 degrees. This is not a MuJoCo joint limit. |
| `n_cables` | integer; `2`, `3` or `4` | Number of cable routes and actuators; also chooses the normal section family. | Integer at least 2. Two uses flat links/hinge joints; three and above use n-lobe links/ball joints. |
| `terminal_unit_policy` | text; `exact_requested_length` | Decides how to finish a requested length that does not contain a whole number of segments. | `exact_requested_length` preserves continuous length and may make a partial base. `whole_units` extends to a complete segment boundary; never shortens. Default: exact. |
| `tendon_inward_shift` | m; `0.0015` | Moves the cable routing sites radially inward from the nominal outer edge. | Nonnegative and less than `d_tip/2`. It changes routing and moment arms, not the STL's outer silhouette. |
| `show_preview` | Boolean; `true` | Retained compatibility field, required by the parameter validator. | Currently printed/validated by the CSV stage; it does **not** control `build.py`'s preview. Use `--no-preview` to suppress that preview. |

The canonical Python geometry API also accepts `taper_angle_deg` as an alias
for `phi_deg`; if both exist they must agree. The full build's validator still
requires `phi_deg`, so use that spelling in user parameter files.

Length changes can alter segment count and the size of the partial base. Some
very short partial remainders cannot retain a valid flat mounting face and are
rejected. All reported dimensions use the canonical generated geometry.

The current cable-site rule includes an intra-link radial correction:
`r1 = max(r_outer - shift, 1e-6)` and
`r2 = max(r_outer - shift - dz*tan(phi/2), 1e-6)`, in metres.
Consequently the offset is not identical at both ends of a link. This retained
routing behavior is documented as F-08 in `GEOMETRY_AUDIT.md`; changing thickness
does not change it.

## 4. Section parameters: two cables versus n cables

| Parameter | Applies to | Meaning | Default / valid values |
|---|---|---|---|
| `base_thickness_m` | Two-cable flat section | Full centre thickness of the actual largest base link, along local Y. All other link thicknesses scale with width. | `null` for thickness=width; otherwise positive finite metres. CLI/GUI use mm instead. A numeric value is rejected for n≥3 or `--plain`. |
| `flat_section` | Two cables | Shape of the XY end-on outline. | `rectangular` or `hex`; default rectangular. |
| `hex_edge_ratio` | Two-cable hex only | Outer-edge thickness divided by centre ridge-to-ridge thickness. Edge thickness at link i is `e*T_i`. | Default `0.75`; strictly `0 < e < 1`. Smaller e gives sharper ridges. |
| `flat_thickness_ratio` | Legacy two-cable files | Centre thickness divided by link width; equivalent to `T_base/W_base`. | Legacy validated range `[0.05, 1]`. Retained only for compatibility. Cannot coexist with `base_thickness_m`. New files use the absolute/auto parameter. |
| `nlobe_t` | n≥3, without `--plain` | Changes the polygon cutter radius relative to the revolved surface; controls the balance between flat facets and remaining circular portions. | Default `0.5`; range `[0,1]`. See exact interpretation below. |
| `notch_factor` | n≥3, without `--plain` | Circular notch radius as a fraction of half the polygon side length. Larger values remove more material. | Default `0.25`; validator accepts `[0,0.4]`. Use a positive value: zero currently produces a degenerate zero-radius CAD cutter. |
| `flat_edge_ratio` | Legacy two-cable fabrication only | Edge/centre thickness ratio for the old constant-thickness manufacturing lens. | Default `0.25`; `0 < e ≤ 1`. It does not select or change the simulation hex section. Prefer `hex_edge_ratio` for the current shape. |

For a hex section of width `2R`, centre half-thickness `H=T_i/2`, and edge
ratio e, the six XY vertices are:

```text
(-R,-eH), (0,-H), (R,-eH), (R,eH), (0,H), (-R,eH)
```

This is generally not a regular hexagon. Two flat outer side walls connect
four sloping faces. The existing XZ slopes and the partial base's flat mounting
face are retained.

For n≥3, the user does not independently set Y thickness. The section is
constructed around the longitudinal axis using the spiral radius R, cable
count n, a drafted polygon cutter and circular notches. The cutter's starting
circumradius is:

```text
R_polygon = R × [1 + nlobe_t × (1/cos(pi/n) - 1)]
notch_radius = R_polygon × sin(pi/n) × notch_factor
```

At `nlobe_t=0`, polygon vertices lie on the reference circle; at `1`, polygon
sides are tangent to it before the axial draft. These geometric definitions
avoid ambiguity in the older "inscribed/circumscribed" labels. The 2D preview
is a reference section; inspect generated STL/CAD for the full drafted surface.
The new base-thickness setting does not anisotropically stretch n-cable robots.

## 5. Placement and dynamics: the `post_gen` object

These optional fields affect the generated XML. They do not rotate/translate
fabrication CAD, whose base is at z=0 with +Z towards the tip.

| Parameter within `post_gen` | Unit / order | Meaning | Example / omission behavior |
|---|---|---|---|
| `robot_pos` | m; `[x,y,z]` | Root body's location in the world. Child body offsets stay relative to their parents. | Example `[0,0,0.22628]`; omitted uses the computed geometry pose. |
| `robot_quat` | Quaternion; `[w,x,y,z]` | Rotation of the root/body frame into the world frame. Use a nonzero unit quaternion. | Example `[0,0,-1,0]` turns local +Z towards world −Z. Omitted keeps computed orientation. |
| `joint_stiffness_base` | N·m/rad | Coefficient K0 of the exponential rotational spring law. | Example `0.2`; nonnegative. Omitted uses the selected preset. |
| `joint_damping_base` | N·m·s/rad | Coefficient D0 of the exponential rotational damping law. | Example `0.01`; nonnegative. Omitted uses the selected preset. |
| `joint_beta` | Dimensionless | Controls the base-to-tip decay of stiffness and damping. | Example/default `1.03`; positive. Greater than 1 decreases gains towards the tip, 1 keeps them constant, below 1 increases them. |
| `first_joint_stiffness` | N·m/rad | Replaces the stiffness of `j_001` only. | Example `100`; nonnegative. Omitted leaves the exponential value. |
| `first_joint_damping` | N·m·s/rad | Replaces the damping of `j_001` only. | Example `50`; nonnegative. Omitted leaves the exponential value. |
| `tip_site_pos` | m; `[x,y,z]` | Optional `tip_site` position in the last link's **local body frame**. | Example `[0,0,0.005]`; omitted means no extra tip site. |
| `target_site_pos` | m; `[x,y,z]` | Optional `target` site location in the **world frame**. | Example `[0.02,0,0.04]`; omitted means no target site. A site has no collision surface or mass. |
| `timestep` | seconds | Explicit simulation step size, overriding the preset. | Use `0.0001` for the present review workflow. Positive finite number; omitted uses preset. CLI `--timestep` takes final precedence. |
| `_comment` | Text | Human-readable note in shipped JSON. | Ignored by generation. It is not a physics setting. |

For link/joint number i starting at **1**:

```text
K_i = K0 / beta^(3*(i-1))
D_i = D0 / beta^(3*(i-1))
```

The first-joint overrides are applied afterwards. Therefore, with a protected
base, `joint_stiffness_base` is a scaling coefficient rather than the literal
stiffness of `j_002`: the latter is `K0/beta^3`. A large first-joint stiffness
is still a finite joint, not a welded body. Two-cable joints hinge about local
Y; the normal n-cable model uses ball joints with rotational spring/damping.
These coefficients require physical identification when matching hardware.

Changing thickness changes mass, COM and inertia and therefore dynamics. The
build recomputes explicit link inertias from the simulation mesh. Massless
collision proxies do not add their own inertial properties. Fabrication cable
holes and the flexure are not part of the simulation's un-drilled rigid-link
inertia model.

## 6. Every `build.py` option

Run `uv run python build.py --help` for the installed version's syntax.

| Option | Meaning / default |
|---|---|
| `--params FILE` | Input JSON; default `params.json`. |
| `--output-dir DIR` | Destination for outputs; default current directory. `build/…` is ignored by Git and suitable for review builds. |
| `--no-preview` | Suppress the interactive geometry preview and its approval prompt. |
| `--noclean` | Compatibility flag. Builds always stage fresh outputs; this flag no longer changes cleaning behavior. |
| `--base-thickness-mm MM\|auto` | Two-cable base centre thickness; mm or automatic equality with width. Overrides both JSON thickness modes. |
| `--hex-section` | Select the two-cable six-sided XY outline. |
| `--hex-edge-ratio E` | Override the hex edge/centre thickness ratio. Requires the hex section. |
| `--plain` | Use an uncut solid of revolution. No flat/n-lobe shape; no thickness/hex override. The build uses ball joints in this mode. |
| `--nlobe` | Compatibility selection for normal n_cables-dependent geometry. Current default already does this. Mutually exclusive with `--plain`. |
| `--mesh-layout shared\|individual` | Default shared: one complete-link STL plus a separate partial-base STL when needed; XML uses per-link scale. Individual emits `link_NNN.stl` for each link. |
| `--collision-mode mesh\|capsule\|compound` | Contact representation. Default mesh except the safe preset. Compound supports only the **rectangular two-cable** section. Hex compound is not implemented. |
| `--collision-corner-radius-ratio R` | Compound corner radius / link half-width; default `0.04`, allowed `[0.005,0.2]`. A particular partial base can require a smaller value. Larger radii round off more corners; this does not alter CAD geometry. |
| `--collision-margin-m M` | Nonnegative contact margin. Default zero for compound; otherwise the selected preset margin. Larger margins can activate contacts earlier. |
| `--arena-memory-mib N` | Native MuJoCo arena allocation per data instance; integer 1–4096 MiB. Default 128 for compound, otherwise the compiler default. Capacity does not fix timestep instability; allocation must fit available RAM. |
| `--timestep DT` | Positive seconds; overrides `post_gen.timestep` and preset. |
| `--safe` | Select the safe preset below. |
| `--fast` | Select the fast preset below. |
| `--high` | Select the high-fidelity preset below. The three preset flags are mutually exclusive. Names do not establish a stability guarantee. |
| `--cad` | Also export whole-robot STEP/STL in millimetres. |
| `--cad-profile fabrication\|simulation` | Default fabrication. Simulation exports the rigid-link solids; fabrication adds a connecting flexure and optional holes. |
| `--fuse-cad` | Requires `--cad`; attempts to fuse simulation-profile solids. Simulation links meet at ideal hinges and need not form a printable connected body. Fabrication already fuses its flexure and links. |
| `--neck-width-mm N` | Fabrication flexure width; default 1 mm; positive and less than actual tip width. For flat links it is the X width of the central ligament; for n-cable shapes it is the cylindrical core diameter. The check also runs for simulation CAD although no flexure is added there. |
| `--cable-hole-diameter-mm D` | Fabrication drilling diameter; default 0 disables holes. Positive values cut the canonical cable paths; oversized holes can disconnect the part and are rejected. |
| `--flat-thickness-m T` | Legacy **fabrication-only** constant centre thickness, in metres. Distinct from base thickness. Requires old ratio-mode two-cable fabrication; rejected for the new scaled flat workflow and simulation CAD. |
| `--flat-edge-ratio E` | Legacy fabrication lens edge/centre ratio; `0<E≤1`. Distinct from `--hex-edge-ratio`; rejected for the new scaled flat workflow and simulation CAD. |
| `--align-geom-frames` | Optional legacy export of `spirob_aligned.mjb` for MuJoCo 3.3.5. Does not make XML retain aligned mesh geom axes. Not needed for thickness or ordinary XML usage. |
| `--help` | Print options and exit. |

Use `--base-thickness-mm` to choose one physical dimension consistently for
simulation and CAD. Do not use the older fabrication-only thickness option
for that purpose. Old ratio-mode fabrication retains its constant-thickness
lens behavior; new absolute/auto mode uses the selected rectangular/hex section
and scales its thickness along the chain. Fabrication then adds its flexure and
any requested holes.

### Physics preset values

These are code defaults **before** explicit JSON/CLI overrides. The shipped
`post_gen` gains override preset gains. `--collision-mode` can override the
preset's contact shape, and `--timestep` overrides the preset's timestep.

| Setting | Default | `--safe` | `--fast` | `--high` |
|---|---:|---:|---:|---:|
| Timestep (s) | 0.002 | 0.001 | 0.005 | 0.0005 |
| Integrator | implicit | implicit | implicit | RK4 |
| Contact shape | mesh | capsule | mesh | mesh |
| K0 (N·m/rad) | 0.2 | 0.2 | 0.01 | 0.1 |
| D0 (N·m·s/rad) | 0.01 | 0.2 | 0.001 | 0.02 |
| Hinge angular range | ±180° | ±15° | ±180° | ±180° |
| Contact margin (m) | 0.001 | 0.0002 | 0.002 | 0.0005 |
| Actuator control range | [−10,0] | [−5,5] | [−10,10] | [−5,5] |
| Rendered tendon width (m) | 0.0006 | 0.0004 | 0.0008 | 0.0005 |

The hinge range applies to hinge models; these presets do not add a ball-joint
angular limit. Tendon display width is visual, not cable-hole diameter. Motors
use tendon force actuation with gear 1 in this generator; controls are force
commands in N under that mapping, not desired tendon lengths. Negative values
in the normal [−10,0] range provide pulling force. Positive control is
mathematically permitted by some presets; it does not make a real cable able
to push. A sign convention
is not a general model of real motor/transmission limits.

Other current `MJCFConfig` defaults are density 1200 kg/m³, gravity
`[0,0,-9.81]` m/s², friction `[0.6,0.01,0.001]` (sliding, torsional, rolling),
and site size 0.001 m. They are Python configuration values, **not implemented
JSON parameter keys**. Adding `density`, `gravity`, `friction` or a custom
control range to params.json does not configure them. Use the Python writer's
configuration or deliberately edit XML and verify the result. Editing geom
density alone does not recompute already explicit body inertias. Torsional
and rolling friction entries are effective only with corresponding contact
dimensions; the normal generated contact dimension is 3.

## 7. Desktop GUI

The desktop requires Tk and a graphical display. On Ubuntu the OS Tk package
is `python3-tk`; the selected Python installation must also include its Tkinter
module. The command-line build and PNG inspection do not require a desktop.

The GUI exposes the core dimensions, cable count, inward shift, n-lobe t,
notch factor, base centre thickness, hex edge ratio and flat-section selection.
Thickness/hex controls are disabled for n≥3; n-lobe controls are disabled for
two cables. With `auto`, the preview reports the actual base and tip thicknesses.
When loading an old ratio-only file, its actual base thickness is displayed
numerically. Saving that GUI configuration converts it to absolute-base mode.
Thereafter geometry changes keep that numeric thickness fixed; enter `auto`
if you want thickness to follow width instead.

The GUI's geometry preview shows the two-cable XY outline and its width/centre
thickness in mm. The left pane shows the axial profile. **Preview exported
CAD** opens the last exported fabrication/simulation geometry, including holes
if present. Changing fields does not retroactively update an already-built XML;
save and rebuild it.

The other GUI controls map directly to CLI settings: output folder, CAD profile,
flexure width, cable-hole diameter, array count, and array radius. Timestep,
post_gen gains/pose, terminal-unit policy, contact mode and arena allocation are
configured through JSON/CLI, not independent GUI widgets. The frame-inspection
button opens its existing optional alignment workflow; it is not required to
inspect physical dimensions.

## 8. Output files and audit procedure

| Output | What it contains / unit |
|---|---|
| `build_params.json` | Input geometry settings plus selected CLI section/thickness/timestep overrides. Auto remains `null`, so future geometry edits retain auto behavior. |
| `section_dimensions.json` | Two-cable width, centre thickness and edge thickness for every link; explicit base/tip summaries and resolved ratio. **Metres**. Absent for n-lobe/plain models. |
| `Geom_Data_CSV/Spirob_geom_data.csv` | Canonical link profiles and cable-site input geometry, metres. Transverse thickness is supplied by params, not extra CSV columns. |
| `meshes/link_template.stl` | Simulation complete-link template, metres. XML applies each link's scale. |
| `meshes/link_001.stl` | Separate partial-base STL when present, metres. |
| `spirob_physics_model.xml` | Generated and compiled standalone MJCF with explicit body inertias. Keep its referenced meshes with it. |
| `cad/spirob.step` and `cad/spirob.stl` | Whole-robot CAD export; numeric dimensions **mm**. STL has no inherent unit declaration, so import as mm. |
| `cad/spirob_cad_report.json` | CAD options, units, bounds, topology, volume and hashes. |
| `spirob_aligned.mjb` | Optional version-specific compiled model; generated only with the alignment option. |

The saved parameter file does not capture every CLI choice, such as collision
mode or CAD profile. Keep the build command with your experiment records. A
build uses temporary staging, verifies the meshes, compiles MJCF and publishes
after success. Failure retains previous published outputs. Rebuilding without
`--cad` leaves an existing CAD folder in place; it may describe an earlier
build. Rebuild CAD explicitly before treating it as current.

To check dimensions and inertia:

```bash
uv run python tools/inspect_section.py \
  --mjcf build/two-cable-square/spirob_physics_model.xml \
  --links link_001 link_002 link_021 \
  --out build/two-cable-square/sections.png

uv run python tools/audit_inertia.py \
  --mjcf build/two-cable-square/spirob_physics_model.xml \
  --params build/two-cable-square/build_params.json \
  --json build/two-cable-square/inertia.json
```

The audit compares tensors in axes parallel to each link body, about the COM.
Its CAD reference assumes a uniform un-drilled simulation solid at the chosen
density. It is not a measurement of a printed part. Use the matching saved
parameters: auditing a changed-thickness XML against an older parameter file
will compare different shapes.

## 9. Supporting command-line tools

### Standalone generation and preview

Normally use `build.py`, which passes matching parameters between stages.

| Tool | Additional options and meaning |
|---|---|
| `spirob_csv_generator.py` | `--params` input; `--out` output CSV; `--yes` skip confirmation. It does not itself generate a transverse STL section. |
| `csv2geom_nlobe.py` | `--in` CSV; `--outdir` meshes; `--params` matching geometry; `--axis` revolution axis (default y workplane convention); `--angle` revolution sweep degrees (default 360); `--plain`, `--mesh-layout`, `--hex-section`, `--hex-edge-ratio`, `--base-thickness-mm` as above. Partial revolution options need a separately reviewed model. |
| `csv2xml.py` | `--in`, `--out`, `--meshdir`, `--params`; section/collision/preset/timestep/arena flags as above; `--hinge` selects hinge joints; `--digits` zero-padding (default 3); `--tendon-shift` in m; `--phi-deg` full taper in degrees. Supply matching CSV/STLs. Section/thickness flags do not regenerate meshes at this stage. |
| `cad_export.py` | `--in` CSV; `--params`; `--outdir` (default cad); `--prefix` filenames (default spirob); `--profile`, `--fuse`, `--plain`, section/thickness flags and manufacturing dimensions as above. Here `--profile` corresponds to build's `--cad-profile`. |
| `preview.py` | `--params`; `--out` PNG; `--nlobe` also shows the n-lobe or flat-section preview; `--proceed` skips its yes/no prompt. |
| `tools/geometry_audit.py` | `--params`; `--out` JSON; `--policy` overrides terminal-unit policy; `--compare-policies` compares both length policies. |

`csv2xml.py` has historical standalone defaults `--phi-deg 5.7` and
`--tendon-shift 0.0015`; its `--params` option does not replace those CLI
geometry values. `build.py` passes the actual values from JSON. Prefer the
build driver to avoid a mismatched standalone command. Changing `--digits`
can also change the external naming contract; keep 3 for the existing mjlab
consumer.

### Inspection and joint gains

| Tool | Parameters |
|---|---|
| `tools/inspect_section.py` | `--mjcf` XML; `--links` body names (default base, first complete, tip); `--out` PNG (default cross_sections.png); `--json` optional coordinates/report; `--show` plot window. Views are actual compiled-mesh projections in body axes, not slices at a chosen Z. |
| `tools/audit_inertia.py` | `--mjcf`; `--params` enables CAD comparison; `--density` kg/m³ (default 1200); `--links`; `--plain`; matching section/thickness override flags; `--reference-mjcf` an older XML; `--json` report. This reads but does not modify inertia. |
| `tools/inspect_collision.py` | `--mjcf`; `--view` frozen CAD/proxy viewer or `--stress` bounded stepping; `--controls` one value per actuator; `--seconds` duration (default 10); `--ramp-seconds` linear ramp (default 2, between 0 and duration); `--json` report. Stress commands must lie within that XML's control limits. |
| `tools/set_joint_gains.py` | `--mjcf` source; `--out` different destination; `--stiffness`, `--damping`, `--beta` required; `--anchor generator` keeps the original exponent indexing, `first-flexible` applies the supplied values to the first unprotected joint; `--exclude-joints` protects additional joints; `--arena-memory-mib` optional; `--json` report; `--overwrite` permits an existing output. `j_001` is always protected. |
| `tools/inspect_model.py` | `--mjcf` (alias `--model`) XML/MJB; `--view`; `--frames geom\|body`; optional `--align-geom-frames`; `--save-mjb` compiled output; `--json` prints a frame report. This is frame inspection, not thickness tuning. |

### Fabrication splitting and robot arrays

| Tool / parameter | Meaning |
|---|---|
| `fabrication/part_splitter.py INPUT` | Split STEP/STP or STL. Cuts create separate pieces, not automatic interlocking joints or fasteners. |
| `--axis x\|y\|z` | Direction along which to split; default z. |
| `--max-span-mm` | Maximum span per part; produces evenly divided pieces. |
| `--cut-positions-mm` | Explicit comma-separated cut coordinates along that axis, in the imported model frame. Do not combine with max-span mode. |
| `--build-volume-mm X,Y,Z` | Printer envelope used for fit checks, with candidate axis permutations. It does not establish print orientation/support quality. |
| `--file-units m\|mm` | Input numeric units; default mm for fabricated outputs. Use m for simulation STLs. |
| `--out-dir` | Destination, otherwise derived from the input filename. |
| `tools/multi_array.py --in FILE` | Source single-robot XML; default spirob_physics_model.xml. |
| `--count` | Number of robots; integer at least 1, default 6. |
| `--radius-m` | Circle radius for robot base placement, metres; default 0.12. |
| `--base-rot-deg` | Angular offset of the whole array about Z, degrees; default 0. |
| `--tilt-deg` | Robot tilt about each base's local radial axis, degrees; default 0. |
| `--out` | Array XML destination; default spirob_array.xml next to source. |

## 10. Common sources of confusion

- **Thickness versus width:** the new dimension changes Y for two-cable links;
  `d_tip`, length, taper and segment angle determine canonical X widths.
- **Base versus tip:** the input refers to actual `link_001`, including a
  partial base. Tip thickness is a derived value, not an independent control.
- **Hex centre versus edge:** centre thickness is ridge-to-ridge. Edge thickness
  is smaller by `hex_edge_ratio`. Setting thickness=width does not remove the
  hexagonal corners.
- **Geometry versus contact approximation:** mesh contacts use convex hulls;
  capsules are an approximation; existing compound proxies fit rectangular
  two-cable links only. Their thickness follows the new base setting. Hex
  compound refitting and primitive reduction remain pending.
- **Geometry versus dynamics:** a thicker link has different inertia. The gain
  coefficients are not automatically identified or retuned by a CAD change.
- **Timestep versus arena:** a smaller timestep changes numerical integration;
  more arena memory provides room for contacts/constraints. Neither substitutes
  for the other.
- **SI versus printing:** JSON lengths and simulation STLs use metres; CLI
  thickness with the `-mm` suffix, CAD exports and printer dimensions use mm.

For the underlying MJCF semantics, see the official
[MuJoCo XML reference](https://mujoco.readthedocs.io/en/3.3.5/XMLreference.html)
and [mesh/contact documentation](https://mujoco.readthedocs.io/en/3.3.5/XMLreference.html#asset-mesh).
Repository-specific behavior in this manual follows the current source code.
The existing detailed geometry, collision and dynamics documents supplement it.

## 11. Updating this manual

The Markdown file is the editable source. The HTML copy is standalone and
opens locally in a browser; it needs no internet connection or Python server.
If Pandoc is installed, regenerate it with:

```bash
pandoc docs/PARAMETER_MANUAL.md --standalone --toc --toc-depth=2 \
  --metadata pagetitle="SpiRob generator parameter manual" \
  --css=docs/parameter-manual.css --embed-resources \
  -o docs/PARAMETER_MANUAL.html
```

Pandoc is only a documentation-build tool; it is not required to generate or
simulate SpiRob models.
