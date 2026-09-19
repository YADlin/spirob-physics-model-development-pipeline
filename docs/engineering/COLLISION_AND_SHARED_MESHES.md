> Engineering history. For current commands, parameters and supported workflows, use the [main README](../../README.md).

# Collision geometry and shared source meshes

This change continues `fix/consolidation-cad-workflows` from `9b1513b`.
It preserves the accepted partial-base orientation and does not rebase geom axes.

## Inertia handling

Previously the capsule mode added a density-bearing capsule while only disabling
contact on the visual mesh. Both geoms therefore contributed to the compiler's
inferred body mass, centre of mass and inertia. Contact masks do not exclude
geoms from inertia calculations.

Generation now first compiles using only the group-1 mesh for each link's inertia.
It writes that body's mass (kg), centre of mass (m), inertial quaternion and
principal moments (kg m²) into an explicit `<inertial>` element. Every collision
proxy has `mass="0" density="0"` and belongs to group 3. The delivered XML uses
`inertiafromgeom="auto"`; it respects explicit link inertias while newly added
task objects can still infer their own. This follows MuJoCo's documented
[compiler inertia rules](https://mujoco.readthedocs.io/en/3.3.5/XMLreference.html#compiler-inertiafromgeom).

The inertias retain the existing mesh/density model (default 1200 kg/m³), not
measured printed-robot properties. Regenerate after changing geometry or density.
Do not force `inertiafromgeom="true"` in a containing task XML, since that would
override explicit inertias. `auto` also works when integrating the robot into a
larger scene. Tests add an ordinary 50 g task object and verify both its inferred
mass and the unchanged robot masses.

## Two-cable compound collision

The flat simulation link is an extrusion of its canonical XZ polygon along Y.
Its top and bottom slopes mean a single rounded rectangle would be a poor fit.
The new collider insets that polygon by the selected radius, then covers the
rounded outline with oriented edge boxes, corner cylinders along Y and interior
boxes. All primitives belong directly to the existing link body. No extra bodies,
joints or constraints are introduced.

The union stays inside the original CAD, retains tangent flat faces, and rounds
off convex corners. Front and back remain flat at the existing extrusion
thickness. It has no internal holes. This is a contact approximation of the
**simulation mesh**; the separate fabrication profile can have a different lens
section and flexures.

For `examples/params-two-cable.json` at the default radius ratio 0.04:

| Link | Primitives | Corner radius (mm) | Maximum inward profile gap (mm) | Profile area covered |
|---|---:|---:|---:|---:|
| Partial base `link_001` | 17 | 0.62175 | 0.25754 | 99.8776% |
| First complete `link_002` | 33 | 0.57252 | 0.18735 | 99.9646% |
| Tip `link_021` | 33 | 0.14147 | 0.04629 | 99.9646% |

The complete model has 677 collision primitives. MuJoCo processes overlapping
geoms individually: this is not a Boolean-fused collision solid, and overlapping
faces can generate multiple contact points. Contact count and solver cost should
be assessed in the actual grasping scene. Native MuJoCo 3.3.5 was tested here;
the user's mjlab training environment and its GPU backend were not tested.

Compound contact margins default to zero so the measured surface fit also
describes contact onset against zero-margin objects. A positive margin on the
robot or contacted object can cause contact before visible surfaces touch.

Rounded compound fitting is currently limited to the two-cable flat section.
Three or more cables retain mesh or massless-capsule contact options; a fitted
compound approximation of their lobes requires a separate design.

## Shared STLs

All complete canonical profiles are uniformly scaled copies of the first
complete link. Generation validates that similarity, exports one
`meshes/link_template.stl`, and preserves `link_NNN` asset names using individual
MJCF `scale` values. Partial units get their own STL; whole-unit designs need
only the template. The largest complete link is the reference, so scaling down
does not magnify its tessellation error. The separate base mounting geometry is
preserved.

MuJoCo still compiles a separate mesh asset for each scale. The saving is in
export work and source files, not a guarantee of reduced runtime mesh memory.
The legacy layout remains available with `--mesh-layout individual`.
Fabrication STEP/STL generation continues to build its own solids.

Compared with the previous individual-STL exports on the two supplied examples:

| Model | Previous mesh-derived total mass (g) | Shared-source total mass (g) | Largest per-link relative mass change |
|---|---:|---:|---:|
| Two cables | 25.84535955 | 25.84535991 | 0.00000995% |
| Three cables | 40.82149599 | 40.82395433 | 0.01937% |

Small differences are from sharing one tessellation instead of independently
tessellating each size. The three-cable total difference is about 0.0060%.
For a fixed mesh layout, changing between mesh/capsule/compound contact preserves
all body inertias to numerical precision. Tests compare inertia tensors in body
coordinates, since MuJoCo may reorder principal axes between scaled assets.

## Build and inspect

From the repository root, with the existing uv virtual environment:

```bash
source .venv/bin/activate
uv pip install -r requirements-dev.txt
python build.py --params examples/params-two-cable.json --no-preview --collision-mode compound --output-dir build/two-cable
python tools/inspect_collision.py --mjcf build/two-cable/spirob_physics_model.xml --view
```

Blue shows CAD; orange shows colliders. Geom groups 1 and 3 toggle them separately.
The inspection view does not integrate time; use joint sliders to examine poses.
To run dynamics, open the same XML directly:

```bash
python -m mujoco.viewer --mjcf=build/two-cable/spirob_physics_model.xml
```

To halve corner rounding, pass `--collision-corner-radius-ratio 0.02` to the build.
To set a 0.1 mm contact buffer, pass `--collision-margin-m 0.0001`. Rebuild into a
separate directory when comparing variants. The GUI's existing build action uses
the default mesh contact mode; select compound mode with the command above.

## Validation

The full suite passed 249 tests, with 16 inapplicable two-cable-only cases skipped
and the existing F08 tendon-routing expected failure retained. Focused collision
checks were repeated after the final compiler integration adjustment. They cover:

- Equal mass, centre of mass and inertia across collision modes, including after
  deliberately assigning a large mass to a collider in the XML.
- Unchanged body, joint, site, tendon and actuator interfaces and tendon lengths.
- Solid, contained profile coverage for every link, both terminal-unit policies,
  and corner ratios 0.02, 0.04 and 0.08.
- Native sphere and box contacts against flat faces and curved corners of links
  1, 2, 11 and 21, with separation checks and no mesh contacts in compound mode.
- Correct shared-source scaling, including mass cubed and inertia fifth-power
  scaling, and multi-robot array mass/asset preservation.
- Successful compilation of final two-, three- and four-cable example XML files.

These checks establish geometry and compiler behavior. They do not establish
grasp success or tune friction, tendon actuation, contact stiffness or material
properties for a particular experiment.
