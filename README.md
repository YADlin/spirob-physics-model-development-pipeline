# SpiRob Physics Model Pipeline

Generate canonical spiral geometry, shared simulation meshes, MuJoCo MJCF,
and whole-robot STEP/STL fabrication models. Two cables use a flat section;
three or more cables use the existing n-lobe construction.

## Install

Use a dedicated Python 3.12 environment with uv (repairs tested on 3.12.13/14):

```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements-fabrication.txt
```

On Windows use `.venv\Scripts\activate` instead of `source`. The desktop GUI
requires Tk: on Debian/Ubuntu install the OS package `python3-tk`; standard
Windows CPython normally includes it. WSL requires a working graphical display.
`requirements.txt` installs the simulation pipeline; `requirements-fabrication.txt`
adds CAD mesh validation, STL splitting and CAD preview dependencies.

## First build

```bash
python build.py --params examples/params-two-cable.json --no-preview --output-dir build/two-cable --cad
```

Output:

| File | Meaning | Units |
|---|---|---|
| `Geom_Data_CSV/Spirob_geom_data.csv` | Canonical link geometry | m |
| `meshes/link_template.stl` | Shared complete-link mesh; MJCF scales it per link | m |
| `meshes/link_001.stl` | Separate partial-base mesh, when present | m |
| `spirob_physics_model.xml` | Compiled and checked MuJoCo model | m |
| `cad/spirob.step` | Full CAD solid, base at z=0, +Z toward tip | mm |
| `cad/spirob.stl` | Full fabrication mesh; import into slicer as mm | mm |
| `cad/spirob_cad_report.json` | Dimensions, topology checks, parameters and hashes | Explicit per field |

The build uses a fresh staging directory, checks the complete mesh set, compiles
MJCF, and publishes outputs after successful validation. A failed generation
leaves existing outputs in place. Paths containing spaces are supported. All
subprocesses use the interpreter that launched the build.

```bash
python -m mujoco.viewer --mjcf=build/two-cable/spirob_physics_model.xml
python build.py --params examples/params-four-cable.json --no-preview --output-dir build/four-cable --cad
```

## Collision shapes and body inertia

Every generated link has an explicit `<inertial>` element derived from its mesh.
Collision proxies have zero mass. The final compiler uses `inertiafromgeom="auto"`,
which respects those explicit inertias while allowing new task objects to infer
their own. Regenerate after changing geometry or material density; changing a
collider alone does not change a link's mass, centre of mass or inertia tensor.

For the **two-cable flat section**, use boxes along the sloping faces and cylinders
at the corners:

```bash
python build.py --params examples/params-two-cable.json --no-preview --collision-mode compound --output-dir build/two-cable
python tools/inspect_collision.py --mjcf build/two-cable/spirob_physics_model.xml --view
```

The inspection view shows translucent blue CAD and orange collision primitives
at a frozen pose. Toggle geom groups 1 and 3 to compare. The standard MuJoCo
viewer also opens this XML directly; group 3 is hidden by default.

`--collision-mode capsule` retains the capsule approximation with corrected
inertia handling; `--collision-mode mesh` remains the default. Capsule/mesh modes
and shared STLs support other cable counts. Compound mode currently requires
two cables without `--plain`. `--safe` still chooses capsules unless explicitly
overridden by `--collision-mode`.

`--collision-corner-radius-ratio` defaults to `0.04` of each link's half-width.
Smaller values fit sharp corners more closely but need more boxes. Rounding is
inward and does not change CAD. Compound mode defaults to zero contact margin;
use `--collision-margin-m` to set a deliberate contact buffer. Other modes retain
their preset margins unless this flag is supplied.

The default `--mesh-layout shared` needs only a complete-link STL and, when
present, a partial-base STL. Separate named MJCF assets apply each link's scale.
Use `--mesh-layout individual` for consumers that directly open every
`link_NNN.stl`. Link/body/site/tendon/actuator names remain unchanged. Copy the
XML **and its meshes folder** when moving a generated robot.

See [collision and shared-mesh validation](docs/COLLISION_AND_SHARED_MESHES.md)
for the measured fit, contact checks and limitations.

Compound builds now allocate 128 MiB of native MuJoCo arena memory to accommodate
large contact sets. `--arena-memory-mib` overrides it. This addresses allocation
capacity; extreme actuation can still become numerically unstable.

The new [dynamics tools](docs/DYNAMICS_TOOLS.md) compare full inertia tensors
against mesh/CAD references and edit the exponential joint gains while preserving
the protected base. `tools/inspect_collision.py --stress` runs a bounded actuator
test and reports contacts, arena usage and the first warning.

## Simulation and fabrication geometry

The simulation generator retains its existing link/site/tendon/actuator names,
routing convention, stiffness, damping and collision settings. The partial base
now has a flat mounting face and the same joint-facing slope as complete links.
Its surface, mass/inertia and cable attachment heights intentionally change.
Its width, backbone endpoints and the complete links retain their geometry.
See [base and frame repair](docs/LINK_ORIENTATION_REPAIR.md) for validation.

To inspect **aligned Geom axes** at a frozen pose:

```bash
python tools/inspect_model.py --mjcf build/two-cable/spirob_physics_model.xml --align-geom-frames --view
```

The viewer now starts with **Geom** frames. The opt-in alignment rebases actual
compiled mesh coordinates while keeping their world surfaces and compiled body
inertias. Without `--align-geom-frames`, the report and viewer show MuJoCo's
original principal-axis frames. `--frames body` explicitly selects Body frames.
The report checks actual Geom axes separately; its exit status is 1 if either
the authored/body rest frames or the actual Geom axes fail alignment.

**XML compilation always restores MuJoCo's principal-axis choice.** To retain
aligned Geom axes in the standard viewer, export a compiled MJB:

```bash
python build.py --params examples/params-two-cable.json --no-preview --output-dir build/two-cable --align-geom-frames
python tools/inspect_model.py --model build/two-cable/spirob_aligned.mjb --view
# Run dynamics in the standard MuJoCo viewer (select Rendering > Frame > Geom):
python -m mujoco.viewer --mjcf=build/two-cable/spirob_aligned.mjb
```

The optional MJB is for the pinned **MuJoCo 3.3.5**. Regenerate it after changing
the robot; rebuilding without this option removes any older aligned MJB.
Existing XML/STL generation and downstream naming are preserved. Consumers
using geom-local axes must account for their changed convention; body/site
conventions retain their existing meaning. See the
[actual Geom-frame repair](docs/GEOM_FRAME_ALIGNMENT.md) for tests, limitations,
and a Python loader example.

The fabrication profile preserves the sloped segment faces and gaps. It adds a
finite central flexure so the physical robot is one connected solid. The
fabrication two-cable section is a piecewise planar lens, thick at its centre
and thinner at its edges. These are manufacturing dimensions; they do not
establish mechanical equivalence to the simulated joints.

```bash
# 1 mm central flexure, 1 mm cable channels following the canonical routed paths.
python build.py --params examples/params-two-cable.json --no-preview --output-dir build/holed --cad --neck-width-mm 1 --cable-hole-diameter-mm 1

# Assemble the existing simulation link geometry for inspection (may be disconnected).
python build.py --no-preview --output-dir build/assembly --cad --cad-profile simulation
```

| CAD setting | Default | Meaning |
|---|---|---|
| `--cad-profile` | `fabrication` | Connected fabrication solid or simulation assembly |
| `--neck-width-mm` | 1 | Central ligament width for two cables; core diameter for n-lobe |
| `--cable-hole-diameter-mm` | 0 | 0 leaves the model undrilled; positive drills routed cable channels |
| `--flat-thickness-m` | derived | Centre thickness; default is `flat_thickness_ratio` × realized base width |
| `--flat-edge-ratio` | 0.25 | Edge/centre thickness ratio; in (0,1] |
| `--fuse-cad` | off | Union simulation assembly; fabrication is always fused |

Example two-cable default centre thickness is **9.326 mm**, edge thickness
**2.332 mm**, flexure width **1 mm**. These defaults are design starting points,
not measured TPU stiffness or damping inputs. Hole clearance, wall thickness,
base attachment, cable anchoring and print settings still need design review.
The exporter checks valid CAD, connectivity, STEP re-import and a closed oriented
fabrication STL; it cannot establish print quality or structural performance.

The standalone exporter takes an existing CSV and checks it against the supplied
parameters. Use the full build after editing geometry to regenerate the CSV.

```bash
python cad_export.py --params examples/params-two-cable.json --in build/two-cable/Geom_Data_CSV/Spirob_geom_data.csv --outdir build/cad-only
```

## Fabrication splitting

```bash
python fabrication/part_splitter.py build/two-cable/cad/spirob.step --axis z --max-span-mm 100 --build-volume-mm 256,256,256
python fabrication/part_splitter.py build/two-cable/cad/spirob.stl --axis z --cut-positions-mm 80,160 --build-volume-mm 256,256,256
```

Cuts use coordinates in the file frame, expressed in mm. STEP/STL fabrication
files default to mm; use `--file-units m` only for numeric metre meshes or legacy
exports whose numeric coordinates were metres. Explicit cuts must be distinct,
finite and inside the model. Choose cut positions or a maximum span, not both.

Each format gets its own `<stem>_<format>_split` directory and JSON report.
`--out-dir` selects another directory. Fit checks allow axis permutations and
check bounding-box dimensions; they do not find arbitrary optimal rotations or
allow for supports, clearance or assembly joints. The CLI fails if a requested
build-volume check fails. Splitting adds no joining features or adhesive joints.

## Multi-robot arrays

```bash
python tools/multi_array.py --in build/two-cable/spirob_physics_model.xml --count 6 --radius-m 0.12 --tilt-deg -20 --out build/arrays/six.xml
```

The array centre is the input robot base XY position. Height is retained. Each
copy rotates about world Z and can tilt about its outward radial axis. Names and
references in bodies, tendons, actuators, sensors, contacts and equality
constraints are renamed with `_r0`, `_r1`, etc. World objects remain shared.
Assets are shared through absolute paths so the output can move directories on
the same machine; the array is not a self-contained transferable package.
The result is compiled before being written. Unsupported includes, keyframes,
plugins and structural expansion tags are rejected explicitly. Start from the
single-robot generator output, not a whole mjlab task scene.

The existing mjlab repository has its own asset adaptation and actuator settings.
Keeping generator names stable does not mean copying a generated scene directly
over a trained mjlab asset is an approved model migration.

## Desktop GUI

```bash
python design_gui.py --params examples/params-two-cable.json --output-dir build/gui
```

Edit parameters, update the geometry preview, then build. **Build + export
STEP/STL** saves validated parameters and regenerates every stage. **Preview
exported CAD** shows the actual last exported mesh and supports mouse rotation.
The canonical preview is explicitly labelled as simulation geometry.

The GUI also offers file splitting, adjustable array count/radius, an output
folder selector and **Inspect link frames**, which opens the frozen viewer with
actual Geom axes aligned in memory. Work runs in one background job at a time;
messages reach Tk through a main-thread queue. Invalid parameters stop the job.
Desktop window execution remains unverified in the repair environment because
it has no display server. Run the command above on the workstation as the GUI
acceptance check.

## Geometry parameters and physics

| Parameter | Meaning |
|---|---|
| `L` | Requested continuous uncoiled centreline length, m |
| `d_tip` | Nominal tip width, m |
| `phi_deg` | **Full included** taper angle, degrees; half-width slope = tan(phi/2) |
| `Delta_theta_deg` | Angular span per segment, degrees |
| `terminal_unit_policy` | `exact_requested_length` or `whole_units` |
| `n_cables` | Integer >=2 |
| `tendon_inward_shift` | Cable routing inward offset, m |
| `nlobe_t`, `notch_factor` | Existing n-lobe section controls |
| `flat_thickness_ratio` | Existing simulation thickness control; fabrication interpretation above |
| `post_gen` | Existing pose, tip/target site and joint coefficient overrides |

For the example parameters, the continuous length is 226.280 mm and the sum of
straight segment chords is 223.655405 mm. The **2.624595 mm deficit is arc-to-chord
shortening**, not an omitted terminal tip segment. Under `exact_requested_length`
the effective continuous length equals the requested length; the partial unit
is the **base** link. Fabrication coordinates translate the base to z=0 without
stretching the chain. See [canonical geometry](docs/CANONICAL_GEOMETRY.md).
If a partial base is too short to retain its width and nominal joint-facing
slope, generation fails with an explanation; choose `whole_units` or adjust the
length/angular span. It does not silently create an inverted surface.

The legacy stiffness/damping law remains `k_i=k_base/beta^(3*i)` and
`d_i=d_base/beta^(3*i)`, with first-joint overrides. At beta=1.03, index 20 has
about **5.89 times** lower coefficients than index 0 before overrides. This is an
assigned index law, not a calibrated beam law. `--safe`, `--fast`, and `--high`
select the existing mutually exclusive presets; `--plain` retains the full
revolution diagnostic mode. `--nlobe` remains a compatibility alias for the
cross-section chosen by the cable count.

## Verification and provenance

```bash
uv pip install -r requirements-dev.txt
python -m pytest -q
```

See [repair findings and validation](docs/CONSOLIDATION_REPAIR.md), including the
recovered bundle/commit discrepancy and CAD comparison. The primary calibration,
training and evaluator work is outside this change.

## Attribution

This repository is MIT licensed; see [LICENSE](LICENSE). The added CAD,
fabrication, array and GUI capabilities are inspired by the authors'
[OpenSpiRobs toolkit](https://github.com/ZhanchiWang/Open-Spiral-Robots), which has
its own PolyForm Noncommercial license. This repair uses this repository's own
canonical geometry and independently written code; it does not import or copy
the authors' implementation. The inherited feature commits describe themselves
as clean-room implementations; that historical claim is not independently
certified by these tests. Retain the appropriate SpiRobs publication citation in
academic work.
