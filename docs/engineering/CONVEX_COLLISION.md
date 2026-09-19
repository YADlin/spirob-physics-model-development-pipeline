> Engineering history. For current commands, parameters and supported workflows, use the [main README](../../README.md).

# Convex collision

Select `--collision-mode convex` for one massless convex mesh collider per link.
This supports two-cable rectangular/hex sections with linear/stepped thickness,
and n-lobe sections for three or more cables. It excludes `--plain` models.
Existing defaults and collision modes remain available. For n-cable examples,
notch approximation measurements and the target-marker change, see
[NCABLE_CONVEX.md](NCABLE_CONVEX.md).

The old compound model overlaps many boxes/cylinders on each rigid link.
MuJoCo solves contacts from their individual geom pairs; it does not Boolean
union their exterior surfaces. Those geoms do not add separate rigid bodies,
but their overlapping contact patches can add many redundant constraints.

The new mode takes the convex hull of the actual generated simulation STL,
without vertex decimation, and embeds its hull vertices in XML. Collision
assets use the same scale and body frame as the visual CAD. No extra STL files
are needed: the complete links still share one STL and the partial base another.
Each `collision_NNN_hull` is group 3, mass/density zero; `gmesh_NNN` remains
group 1 and visual-only. Explicit CAD-derived body mass, COM and inertia,
joint gains, body/joint/site/tendon/actuator names and cable paths are preserved.

MuJoCo's previous `mesh` mode also collided with a convex hull. This update is
not a new triangle-by-triangle collision engine. It makes a separate, visible
hull asset, selects zero contact margin by default and enables native CCD plus
`multiccd` for surface contact support. One collider can still produce several
contact points. Friction, `condim`, cone, solver and body collision filters are
not changed. No artificial contact cap is introduced.

Reference: [MuJoCo 3.3.5 collision detection](https://mujoco.readthedocs.io/en/3.3.5/computation/index.html#collision-detection).
The native multi-point path requires zero margins on both contacting objects;
positive margins trigger MuJoCo's slower fallback. The flag is a global option
and therefore also applies to other compatible object pairs in the scene.

## Two-cable surface fit

Rectangular sections (including the continuous axial taper) and stepped hex
sections are convex and retain their CAD envelope to mesh precision. The old
compound's inward-rounded corners are replaced by the CAD's sharp corners.
This is deliberate; the new mode has no corner-radius parameter.

The linear hex section has slightly nonconvex ruled surfaces. A single convex
geom cannot represent those exactly. Its hull fills the shallow concavity.
For the supplied geometry with automatic base thickness (31.087574 mm):

| Link | Largest sampled surface gap, either direction | Extra volume |
|---|---:|---:|
| Partial base, link_001 | 0.02113 mm | 0.05711% |
| Largest full link, link_002 | 0.05844 mm | 0.17999% |
| Tip, link_021 | 0.01444 mm | 0.17999% |

These are sampled comparisons to the simulation STL, not certified maximum
errors or measurements of printed parts. Other thickness/shape settings need
their own fit check. A future exact nonconvex representation would need several
convex pieces or another collision method. Hulls also do not preserve cable
holes, recesses or spaces in a fused multi-link fabrication STL; this mode uses
the undrilled simulation solid of each individual link.

## Build and inspect

From the repository root:

```bash
uv pip install -r requirements-dev.txt

uv run python build.py --params examples/params-two-cable-hex.json \
  --base-thickness-mm auto --thickness-profile linear \
  --collision-mode convex --timestep 0.0001 \
  --no-preview --output-dir build/convex-hex

uv run python -m mujoco.viewer --mjcf=build/convex-hex/spirob_physics_model.xml

uv run python tools/inspect_collision.py \
  --mjcf build/convex-hex/spirob_physics_model.xml --view

uv run python tools/inspect_collision_surface.py \
  --mjcf build/convex-hex/spirob_physics_model.xml \
  --out build/convex-hex/surface.png --json build/convex-hex/surface.json
```

Use `examples/params-two-cable.json` for rectangular links. Replace `auto` with
your base thickness in mm. `collision_summary.json` is generated automatically.
The inspection viewer is frozen: blue is CAD, orange is the actual hull, and
groups 1/3 toggle them. Use the normal MuJoCo viewer to simulate actuators.

`inspect_collision_surface.py` samples each triangle on a barycentric grid,
measures nearest-surface distances in both directions, checks containment and
compares volume. It plots transverse cuts, not a misleading end-on projection.
The default selected links are the base, largest full link and tip. Use
`--links link_001 link_005` to inspect others.

## Contact and constraint measurements

Controlled comparison on the same 21-link **stepped rectangular** CAD and
inertias; largest complete link; prescribed 0.05 mm face overlap:

| Metric | Legacy compound | Convex |
|---|---:|---:|
| Robot contact geoms | 677 | 21 |
| Link against plane: contact points | 126 | 3 |
| Link against plane: scalar constraints | 504 | 12 |
| Link against box: contact points | 114 | 4 |
| Link against box: scalar constraints | 456 | 16 |
| Link against link: contact points | 740 | 4 |
| Link against link: scalar constraints | 2960 | 16 |

These are specific static patches, not counts promised for every pose.
`condim=3` with the existing pyramidal friction cone gives four scalar rows per
active contact here. Counts elsewhere can also include joint/tendon limits.

To reproduce the comparison without changing the CAD between the two modes:

```bash
uv run python build.py --params examples/params-two-cable.json \
  --thickness-profile stepped --collision-mode compound --timestep 0.0001 \
  --no-preview --output-dir build/compound-baseline
uv run python build.py --params examples/params-two-cable.json \
  --thickness-profile stepped --collision-mode convex --timestep 0.0001 \
  --no-preview --output-dir build/convex-comparison
uv run python tools/benchmark_collision.py \
  --baseline build/compound-baseline/spirob_physics_model.xml \
  --candidate build/convex-comparison/spirob_physics_model.xml \
  --json build/collision-benchmark.json
```

The benchmark rejects different CAD/inertias. It includes plane, box and
link-pair probes. Bounded dynamic checks are separate:

```bash
uv run python tools/inspect_collision.py \
  --mjcf build/convex-hex/spirob_physics_model.xml --stress \
  --controls -10 0 --seconds 5 --ramp-seconds 1 \
  --json build/convex-hex/stress-left.json
```

For the supplied linear hex model, five seconds each at targets `[-10,0]`,
`[0,-10]`, and `[-10,-10]`, with a one-second ramp and 0.0001 s timestep,
completed without MuJoCo warnings. The unilateral runs peaked at 22 contacts,
88 scalar constraints and 157,504 bytes of arena use. The symmetric run had
no contacts. The normal 14 MiB arena was sufficient; it was not increased.
These tests do not cover every object, impact or actuator history.

Multi-point contacts preserve surface support but do not guarantee perfectly
static resting contact: isolated flat mesh tests exhibited small rocking and
drift as the contact manifold changed. The regression checks bounded support,
not zero residual velocity. Sharp corners and slightly filled hex concavities
can also change contact normals compared with the old rounded compound.

## Locate forces without adding colliders

```bash
uv run python tools/inspect_contact_forces.py \
  --mjcf build/convex-hex/spirob_physics_model.xml \
  --seconds 2 --ramp-seconds 1 --controls -10 0 \
  --json build/convex-hex/contact-forces.json
```

The JSON lists solved contacts, their world and body-local positions, the
force **on** each link in both frames, and the summed force and moment about
its body origin. Cable loads and joint reactions are separate quantities.
Self-contact appears with opposite forces for the two participating links.

To sum contacts within 2 mm of a chosen point, add for example
`--body link_002 --point-local-m 0 0 0.01 --radius-m 0.002`. Coordinates are
metres in that link's body frame, with the joint origin at zero. The example
point is illustrative; choose coordinates from your model. Moments then use
that point as the reference. An empty region correctly reports zero contacts.

This reports discrete solver contacts, not a continuous pressure distribution
or a unique force at an arbitrary mathematical point. Contact indices change
between timesteps. Choose a body-local region rather than tracking an index.

For your existing simulation, import
`from tools.inspect_contact_forces import contact_report` and call it with the
live `model, data`. Ensure frames and forces describe the same pipeline state;
the CLI calls `mj_forward` after the final step. You can also save `qpos`,
`qvel`, `ctrl` with `numpy.savez` and pass `--state-npz state.npz` (this restores
those arrays, not every possible simulation state field). Force-frame/sign
tests check a supported sphere against its weight and equal/opposite forces
on a contacting body pair.

All inspection tools, tests and these instructions are included in the commit.
The full parameter manual and broader README rewrite remain deferred.
