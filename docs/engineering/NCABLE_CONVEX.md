> Engineering history. For current commands, parameters and supported workflows, use the [main README](../../README.md).

# Convex collision for n-cable SpiRobs

`--collision-mode convex` now supports the existing n-lobe CAD geometry for
`n_cables >= 3`, as well as the previously supported two-cable sections.
The CLI/API extension uses the same hull extraction and contact settings:
one massless collision mesh per link, native CCD, multi-point contact support
and zero default contact margin. The number of cables does not increase the
number of collision geoms per link.

Here **convex** describes the collision envelope. `--hex-section` describes
the special six-sided two-cable CAD section and remains a two-cable-only flag.
For n-cable robots, the existing n-lobe CAD geometry is retained; choosing
six cables does not turn the entire section into a regular hexagon.

The extension is generic in cable count. Validation covers 3, 4 and 6 cables.
The ready examples all have 21 links and therefore 21 contact geoms. The
n-cable models retain ball joints, n cable tendons/motors, existing names,
site positions and routing. Body mass, COM and inertia continue to come from
the original CAD mesh, not from the filled collision hull. There is still only
one complete-link STL plus the separate partial-base STL.

## The n-lobe notches are filled for collision

A single convex shape cannot contain an inward notch. It spans that notch,
so an external object can contact the hull before it reaches the visible CAD
surface there. This approximation is much larger than for the two-cable hex
section. It must be considered when interpreting contacts near a notch.

Measured on the supplied examples (`notch_factor=0.25`, `nlobe_t=0.5`):

| Cables | Link | Largest sampled surface difference | Extra collision volume |
|---|---|---:|---:|
| 3 | Partial base, link_001 | 4.9788 mm | 24.2985% |
| 3 | Largest full link, link_002 | 4.4132 mm | 22.8716% |
| 4 | Partial base, link_001 | 3.1982 mm | 10.3946% |
| 4 | Largest full link, link_002 | 2.6392 mm | 8.4847% |
| 6 | Partial base, link_001 | 1.8884 mm | 4.5869% |
| 6 | Largest full link, link_002 | 1.1975 mm | 2.1355% |

These are sampled distances in both directions, not certified maximum errors.
The added *collision volume* does not add mass. Contact on the convex outer
envelope is represented; contact inside the recesses is not. If notch-level
contact matters for a task, one whole-link hull is unsuitable for that task;
it would need a separate decomposition or collision representation.

The existing `mesh` collision mode also used MuJoCo's convex-hull semantics.
Explicit convex mode separates the visible collision asset from the CAD and
uses the contact options described above. It does not promise exact concave
mesh contact. See [MuJoCo collision detection](https://mujoco.readthedocs.io/en/3.3.5/computation/index.html#collision-detection).

The surface-inspection tool now handles collapsed seam triangles in compiled
n-lobe meshes. It omits only triangles with exactly zero area when checking
the inspection copy's topology, and records the count in its JSON report.
It does not modify the simulation mesh, fill holes or change inertia.

## Generate and inspect

From the repository root, using your uv environment:

```bash
uv pip install -r requirements-dev.txt

uv run python build.py --params examples/params-three-cable.json \
  --collision-mode convex --timestep 0.0001 --no-preview \
  --output-dir build/convex-three

uv run python -m mujoco.viewer \
  --mjcf=build/convex-three/spirob_physics_model.xml

uv run python tools/inspect_collision.py \
  --mjcf build/convex-three/spirob_physics_model.xml --view

uv run python tools/inspect_collision_surface.py \
  --mjcf build/convex-three/spirob_physics_model.xml \
  --out build/convex-three/surface.png --json build/convex-three/surface.json
```

Substitute `params-four-cable.json` or `params-six-cable.json` and a separate
output directory for those models. For other counts, set `n_cables` in your
own parameter file. Use valid existing n-lobe parameters. Do not pass the
two-cable-only `--hex-section`, `--base-thickness-mm` or `--thickness-profile`
flags to an n-cable build.

The inspection viewer shows CAD in blue (group 1) and the actual collision
hull in orange (group 3). Toggling these makes the filled notches apparent.
Use the normal viewer for actuator-driven simulation. The build prints a
short note about notch filling and saves `collision_summary.json`.

## Red target sphere

The red target marker is hidden by default in both `build.py` and direct
`csv2xml.py` output, for all cable counts and collision modes. This is a
rendering change: the named `target` site and its position are retained when
`post_gen.target_site_pos` is supplied, so downstream references remain valid.
Its alpha is zero. It never generated collision forces or contributed inertia.

If you explicitly want it visible for debugging, add `--show-target-marker`.
Without a configured `target_site_pos`, no target site is created. Existing
XMLs are not edited automatically; regenerate them or use the new examples.

## Validation and force inspection

`tests/test_ncable_convex.py` checks actual generated 3-, 4- and 6-cable
models. It verifies one collider per link, unchanged physical properties and
names, ball joints, cable paths at rest and in a rotated pose, hull containment
and directional support against CAD, contact and separation around all lobe
sectors and end caps, multi-robot shared assets, and target-marker visibility.

To run a bounded dynamic check on the three-cable model:

```bash
uv run python tools/inspect_collision.py \
  --mjcf build/convex-three/spirob_physics_model.xml --stress \
  --controls -10 0 0 --seconds 5 --ramp-seconds 1 \
  --json build/convex-three/stress.json

uv run python tools/inspect_contact_forces.py \
  --mjcf build/convex-three/spirob_physics_model.xml \
  --controls -10 0 0 --seconds 2 --ramp-seconds 1 \
  --json build/convex-three/contact-forces.json
```

Supply one control per actuator, in XML order. The force tool also supports
body-local region selection; see [CONVEX_COLLISION.md](CONVEX_COLLISION.md).
The shipped package includes measured surface reports and bounded stress
reports for its examples. These runs do not prove stability for every object,
impact or control history.

For the skipped tests and known expected failure, see
[TEST_STATUS.md](TEST_STATUS.md). Tendon-routing changes, overall optimization,
redundancy review and the web GUI are deferred from this update.
