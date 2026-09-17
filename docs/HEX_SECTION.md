# Optional six-sided two-cable section

The photographed outline is transverse to the robot's longitudinal Z axis.
The new `--hex-section` makes that **XY** outline six-sided. It does not merely
change the existing sloping XZ outline. This is an independent construction
using this repository's canonical geometry and its existing MIT fabrication
lens helper. No code from Open-Spiral-Robots was copied.

## Dimensions and selection

For each link, let R be half its width, H be half its centre thickness, and
e = `hex_edge_ratio`. The end-on vertices, in counterclockwise order, are:

`(-R, -eH), (0, -H), (R, -eH), (R, eH), (0, H), (-R, eH)`.

Both centre ridges run along the link. The two outer side walls remain flat.
The default e is **0.75**, an adjustable design choice, not a dimensional
measurement from the photographs. Centre thickness remains the existing
base setting: `--base-thickness-mm auto` (the new default) makes centre
thickness equal width. `--base-thickness-mm 20` sets the largest base link to
20 mm; the tip scales by its width/base-width ratio. The equivalent JSON
parameter is `base_thickness_m`, in metres, with `null` meaning auto.
Older ratio-only parameter files keep their specified ratio. A smaller
e makes the ridges sharper; e must be strictly between zero and one.

Use either the command-line flag or these JSON fields:

```json
{
  "flat_section": "hex",
  "hex_edge_ratio": 0.75
}
```

These fields supplement the usual complete parameter file. The flag is only
valid for two cables without `--plain`. CLI edge ratios override JSON values.
To return to the rectangular section, omit the flag and remove both JSON
fields (or set `flat_section` to `rectangular` and remove `hex_edge_ratio`).

## Build and inspect the XML

From the repository root, using the existing uv environment:

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
uv run python build.py \
  --params examples/params-two-cable.json \
  --hex-section --hex-edge-ratio 0.75 --base-thickness-mm auto \
  --timestep 0.0001 \
  --collision-mode mesh --collision-margin-m 0 \
  --no-preview --cad --cad-profile simulation \
  --output-dir build/hex-review

uv run python tools/inspect_section.py \
  --mjcf build/hex-review/spirob_physics_model.xml \
  --out build/hex-review/cross_sections.png \
  --json build/hex-review/cross_sections.json

uv run python -m mujoco.viewer \
  --mjcf=build/hex-review/spirob_physics_model.xml
```

The inspection PNG shows the base, first complete link and tip in their own
body axes. Its left column is the end-on XY projection; middle is XZ; right
is a perspective surface view. It uses actual compiled mesh vertices and
undoes MuJoCo's mesh-frame rotation. It does not alter geom frames or advance
the simulation. Choose links with `--links link_001 link_010`, and use `--show`
to open the plot window. The flat-section/GUI preview now shows the selected XY outline and physical
thickness; this tool additionally checks the actual compiled surface.

`--timestep` overrides both the preset and `post_gen.timestep`. Without an
override, the existing preset defaults remain. Builds save `build_params.json`
with resolved section parameters and an explicit timestep override, if given.
This file is the geometry input for subsequent audits; other CLI physics and
CAD options still need to be repeated when rebuilding.

The standard shared layout still writes only `link_template.stl` and the
partial base's `link_001.stl`. Each named XML mesh asset applies its own scale.
With the whole-units policy, only the template is needed. Move the XML together
with its `meshes` directory. The STL coordinates used by XML are metres.

## CAD and inertia

`--cad-profile simulation` exports the same link solids as the simulation,
in millimetres, as a STEP assembly. It has no physical flexure or drilled holes.
For the connected fabrication version with example 1 mm cable holes:

```bash
uv run python cad_export.py \
  --in build/hex-review/Geom_Data_CSV/Spirob_geom_data.csv \
  --params build/hex-review/build_params.json \
  --profile fabrication --neck-width-mm 1 \
  --cable-hole-diameter-mm 1 \
  --outdir build/hex-review/fabrication
```

The fabrication version uses the same tapered link solids, connects them with
a narrow flexure that follows the ridges, and optionally drills the existing
cable routes. It validates a single CAD solid and a closed, oriented STL.
Flexure/hole dimensions are manufacturing choices. Simulation mass properties
describe its un-drilled rigid-link solids; they do not include the fabrication
flexure or model the drilled version's material removal. CAD dimensions and
fabrication STL import units are **mm**.

For this hex option, use `--base-thickness-mm` and `--hex-edge-ratio` to choose
dimensions for both simulation and CAD. The older fabrication-only
`--flat-thickness-m`/`--flat-edge-ratio` overrides are rejected to avoid two
different shapes under one build. Legacy ratio-mode fabrication without the hex option keeps its existing
constant-thickness lens behavior. New absolute/auto mode also gives rectangular
fabrication links the selected tapered thickness.

```bash
uv run python tools/audit_inertia.py \
  --mjcf build/hex-review/spirob_physics_model.xml \
  --params build/hex-review/build_params.json \
  --json build/hex-review/inertia.json
```

Use the generated `build_params.json`, rather than the original rectangular
parameters. Alternatively pass the matching `--hex-section --hex-edge-ratio`
flags to the audit with the original parameters. Changed material geometry
correctly changes mass, COM and inertia. Each body still receives explicit
mesh-derived inertial values, so collision proxies cannot add mass. Body,
joint, site, tendon and actuator names, their placements, and the gain law
are preserved, including the protected base and its flat mounting face.

## Collision scope

This change is for shape review. `--collision-mode mesh` uses the new shape;
MuJoCo's built-in mesh contacts use convex hulls, as described in its
[mesh documentation](https://mujoco.readthedocs.io/en/3.3.5/XMLreference.html#asset-mesh).
Capsules remain an explicitly approximate alternative. The old compound
boxes/cylinders fit a rectangular extrusion, so `--hex-section` combined with
`--collision-mode compound` fails before publishing a build. Redesigning and
reducing those primitives remains a separate next step after shape approval.

The timestep used for the included review model is 0.0001 s. This is the user's
working value; this feature does not claim stability at every actuator force
or contact configuration.

## Checks

```bash
uv run python -m pytest tests/test_hex_section.py -q
uv run python -m pytest -q
```

Regression checks cover every compiled link's six-sided XY outline, retained
XZ shape and flat base mount, unchanged names/routing/gains, mass and inertia
against exact CAD, shared-mesh scaling, custom edge ratios, rejected mismatched
compound colliders, timestep precedence, and connected fabrication exports
with and without holes.

See [the parameter manual](PARAMETER_MANUAL.md) for every user setting and the
base-to-tip thickness convention.
