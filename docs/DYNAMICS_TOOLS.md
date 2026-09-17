# Inertia audit, joint gains and contact-memory diagnostics

The tools read standalone MJCF XML. `audit_inertia.py` is read-only.
`set_joint_gains.py` writes and compiles a separate XML, leaving the input intact.
`inspect_collision.py` provides a frozen viewer and a bounded headless test.

## What the reported crash means

The supplied log has 4,643 contacts and 18,572 scalar constraints. The native
MuJoCo arena stores dynamic contact/constraint arrays and temporary solver
storage. Running out of the latter causes the fatal stack-allocation error.
This is distinct from a QVEL, QPOS or QACC numerical-instability warning.
MuJoCo documents this behavior and the `maxuse_arena` diagnostic in its
[memory-allocation guide](https://mujoco.readthedocs.io/en/3.3.5/modeling.html#memory-allocation).

The earlier compound approximation has 677 collision primitives. Deep folding
can create many contacts between primitives on different links. Increasing
arena memory addresses allocation capacity; it does not change the contact
geometry or establish stable dynamics.

The historical tests below used the two-cable example with thickness/width
ratio 0.3 (before the new thickness=width defaults), unchanged gains and
inertias, a 2 ms timestep, and controls `[-10, 0]` from the initial pose:

| Run | Result |
|---|---|
| Existing 14 MiB arena, immediate control step | Fatal allocation error at about 0.666 s; 4,410 contacts and 17,640 constraints |
| 128 MiB arena, same immediate step | No allocation error before a QACC warning at about 5.994 s; peak 23,027 contacts, about 73.45 MiB arena use |
| 128 MiB arena, two-second ramp to the same target | QACC warning at about 5.802 s; peak 25,557 contacts, about 81.52 MiB arena use |
| 128 MiB arena, two-second ramp to `[-1, 0]` | Completed 12 s without warnings; this particular trajectory produced no contacts |

These are specific test sequences, not force limits or a stability envelope.
The regenerated model's inertias, gains, collision poses/sizes and actuator
ranges were also checked to be bit-for-bit equal to the earlier model.
The larger-arena run confirms that the memory fault and subsequent instability
can occur with the same inertia and gains. It does not identify a unique cause
of the instability. Contact overlap, timestep, actuation and joint gains all
remain relevant.

New compound builds allocate 128 MiB explicitly. Override using
`--arena-memory-mib N`. This is native MuJoCo memory per `MjData`; GPU backends
have their own allocation behavior. The report includes peak use, so allocation
can be chosen from measured contact-heavy tests. No contacts are disabled to
make the test pass.

## 1. Compare inertia in a consistent frame

MuJoCo stores a body's three principal moments in `body_inertia`, together with
the principal-frame orientation in `body_iquat`. Comparing just the three
numbers can be misleading when principal axes are permuted or rotated.

The audit resolves every tensor along axes parallel to the containing link's
body axes. The origin is at the centre of mass. If `R` maps the inertial axes
into body axes, the compiled tensor is

```text
I_COM_body = R @ diag(body_inertia) @ R.T
```

The report includes mass in kg, COM position in body coordinates in metres,
and the full symmetric 3 × 3 tensor in kg m². Each geometric reference has its
own COM, which is reported separately. For a common-origin comparison the tool
also shifts its tensor to the MuJoCo COM using the parallel-axis theorem:

```text
I_reference_at_MJ_COM = I_reference_at_own_COM
                       + mass * ((r @ r) * identity - outer(r, r))
```

Here `r` is the displacement between those COM locations, expressed in body
axes. The printed tensor differences use this common origin and orientation.
The JSON also retains every tensor about its own COM.

There are two independent geometric references:

- **Mesh:** signed triangle-volume integration using trimesh, excluding all
  collision proxies. Compiled mesh vertices are placed in body coordinates
  using the compiled geom pose, compensating MuJoCo's mesh centring/rotation.
  This checks the compiled body properties against the geometry it displays.
- **CAD:** OpenCascade volume integration of the simulation-profile solid
  reconstructed from the matching parameters. This avoids STL tessellation.
  CAD is integrated in mm, then mass and inertia are converted with factors
  `density × 10^-9` and `density × 10^-15`, respectively.

Both assume a homogeneous solid at the stated density. They are not measurements
of an infilled printed part. The CAD reference is the simulation profile, not
the fabrication assembly with flexures and optional cable holes.

Install the comparison dependencies in the existing uv environment:

```bash
source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

Inspect selected links, using the parameters that produced the XML:

```bash
python tools/audit_inertia.py \
  --mjcf build/two-cable/spirob_physics_model.xml \
  --params examples/params-two-cable.json \
  --density 1200 \
  --links link_001 link_002 link_021 \
  --json build/two-cable/inertia_audit.json
```

Omit `--links` to audit every link. Omit `--params` for mesh-only comparison.
Add `--plain` only when the XML was generated with the plain revolved profile.
Add `--reference-mjcf path/to/older.xml` for a before/after compiled-inertia
comparison; both XMLs must use the same link-body coordinate convention.

For the supplied two-cable example, all 21 CAD tensors agree with MuJoCo to a
maximum relative Frobenius error of approximately `4.1 × 10^-8` at density
1200 kg/m³. This does not support a significant geometric inertia error in
that example.

The three-cable example shows a separate discrepancy: selected complete links
have approximately 3.04% higher compiled mass and 2.57% tensor difference from
the CAD reference. Its compiled triangle surfaces also fail the mesh reference's
closed/winding-consistent checks, so that reference is explicitly marked
unavailable rather than silently repaired. The existing MuJoCo 3.3.5 `legacy`
mesh-inertia algorithm can overcount non-convex geometry; see its
[mesh inertia documentation](https://mujoco.readthedocs.io/en/3.3.5/XMLreference.html#asset-mesh-inertia).
The audit reveals this discrepancy but does not change the model's inertias.
With any unavailable mesh reference, the tool writes the available report and
returns exit code 2. Parameters, density and geometry must be checked before
treating a CAD difference as a MuJoCo error.

## 2. Edit the flexible-joint gains

The existing generator law uses the numerical suffix `n` in `j_NNN`:

```text
k_n = K0 / beta**(3*(n-1))
d_n = D0 / beta**(3*(n-1))
```

The tool preserves that indexing by default and skips `j_001`. For `K0=0.2`,
`D0=0.01` and `beta=1.03`, `j_002` therefore receives `k=0.18302833` N m/rad
and `d=0.0091514166` N m s/rad. The protected base remains at its existing
values, 100 and 50 in the example.

To reproduce the current gain values while changing only arena capacity:

```bash
python tools/set_joint_gains.py \
  --mjcf build/two-cable/spirob_physics_model.xml \
  --out build/two-cable/spirob_arena128.xml \
  --stiffness 0.2 --damping 0.01 --beta 1.03 \
  --arena-memory-mib 128 \
  --json build/two-cable/gain_changes.json
```

For your next gain experiment, supply your chosen stiffness and damping and use
a new output filename. There is no automatically selected stable gain pair.
`--anchor first-flexible` instead assigns the entered coefficients directly to
`j_002`, then decays toward the tip. The printed formula states which convention
was used. `--exclude-joints` can name additional protected joints.

Hinge and ball joints are supported; a ball joint gets the same damping on all
three rotational DOFs. The tool checks the compiled output, verifies inertias,
geometry and the protected gains, and publishes only after verification. Asset
paths remain usable when writing into another directory. `--overwrite` permits
replacing an earlier output; overwriting the input is intentionally rejected.

This edits the XML, not `params.json`. To retain a chosen generator-indexed fit
across future builds, update these fields in `post_gen`:

```json
"joint_stiffness_base": 0.2,
"joint_damping_base": 0.01,
"joint_beta": 1.03
```

Keep `first_joint_stiffness` and `first_joint_damping` unchanged. For the
first-flexible convention, multiply both entered coefficients by `beta**3`
when storing them as the generator's base coefficients.

The protected base is a high-gain joint in the current XML, not a true welded
connection. Joint dynamics also involve downstream links through the coupled
mass matrix; an isolated link's COM tensor alone is insufficient to calculate
a guaranteed stable gain or damping ratio for the whole robot.

## 3. Inspect and check a candidate

Frozen comparison of CAD and colliders:

```bash
python tools/inspect_collision.py --mjcf build/two-cable/spirob_arena128.xml --view
```

Native simulation of that exact file:

```bash
python -m mujoco.viewer --mjcf=build/two-cable/spirob_arena128.xml
```

A bounded headless test, using controls in actuator order:

```bash
python tools/inspect_collision.py \
  --mjcf build/two-cable/spirob_arena128.xml \
  --stress --controls -1 0 --seconds 12 --ramp-seconds 2 \
  --json build/two-cable/stress_report.json
```

Use another target to test another actuation condition. The tool does not alter
gains, force limits, timestep or contacts. It stops at the first MuJoCo warning
or fatal error, saves the report and returns exit code 2 on failure. Reports
include peak contacts, constraints and arena use. An instability can reset
`data.time`; therefore the report separately records the attempted step's start
time, nominal completed-step duration and final data time.

## Checks included with this update

Regression tests check a rotated, offset box against its analytic full tensor;
CAD mm-to-SI conversion; detection of a deliberately doubled inertia; parallel
axis shifting; unchanged protected-base gains, mass and geometry; both indexing
conventions for hinge and ball joints; relocated asset paths; invalid-input
rejection; and allocation of a reproduced pose with over 4,000 contacts.

The high-force instability remains open. The new tools support controlled
comparisons before selecting gains or changing the contact approximation.
