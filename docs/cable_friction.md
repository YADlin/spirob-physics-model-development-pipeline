# Cable friction for SpiRob tendons (`spirob/cable_friction.py`)

MuJoCo transmits a tendon force to every joint the tendon crosses as if the
cable slid through its holes without friction. A printed SpiRob is different:
the cable rubs on the edge of every hole, so the tension that reaches a distal
unit depends on how much the cable is bent on the way and on which way it last
slid. Wang & Freris (RoboSoft 2024) use exactly this to make a two-cable SpiRob
reach in many directions.

## Model (stick-slip capstan)

The routed cable is the chain of straight segments between the sites of the
spatial tendon, ordered from the motor (base) to the tip. At every site the
cable turns by an angle α_k (measured each step from the site positions).
Coulomb friction on a curved contact bounds the tension ratio across it:

    T_{k-1}·e^{-μα_k}  ≤  T_k  ≤  T_{k-1}·e^{+μα_k}

Inside the bounds the cable sticks and keeps its tension; at a bound it slips.
Updating the segments in order from the motor gives a rate-independent law with
memory:

    T_0 = motor tension
    T_k = clip(T_k(previous step), T_{k-1}·e^{-μα_k}, T_{k-1}·e^{+μα_k})

| What the motor does | Result |
| --- | --- |
| pulls from rest | T_k = T_0·e^{-μΣα}: tension falls toward the tip |
| holds | nothing changes, the shape is held |
| releases | the bound flips: distal tension can stay above the motor tension |
| μ = 0 | every segment carries T_0: identical to a MuJoCo tendon |

Joint forces: a segment between sites on two adjacent units only depends on the
hinge between them, and its length gradient equals the tendon's moment arm
`ten_J` at that hinge, so `qfrc_j = -T_span(j) · ten_J[c, j]`.

Assumptions: one friction coefficient for every hole; no friction where the cable
is straight (α = 0); the base guide is lumped into the motor tension; the cable is
inextensible (tension changes travel instantly).

## Use

MuJoCo, force-controlled cables (the study's set-up):

```python
from spirob.cable_friction import CableFriction
cf = CableFriction(m, tendons=("cable_0", "cable_1"), mu=0.2)
mujoco.mj_forward(m, d)
for k in range(n):
    d.ctrl[:] = 0                                   # tendon motors idle
    cf.apply(d, tension=[T0_cable0, T0_cable1])     # writes d.qfrc_applied
    mujoco.mj_step(m, d)
```

MuJoCo with actuators already on the tendons (spirob-mjlab's position servos):
`cf.apply(d, correction=True)` reads each tendon's actuator force from the last
step and adds only the difference between the frictional and the uniform
transmission. Call `cf.reset()` at every episode reset.

mjlab / MJWarp (GPU): MuJoCo C plugins cannot run there, so the model is array
code. `TorchCableFriction` is the batched version (torch, any device):

```python
tcf = TorchCableFriction(m_cpu, mu=0.2)            # once, from the CPU model
site_ids = tcf.site_ids()                          # (C, S) gather indices
# every physics substep, per environment batch:
f = tcf(site_xpos[:, site_ids], tension, ten_J_joints, ideal_tension=actuator_tension)
qfrc_applied[:, tcf.joint_dofs] += f
tcf.reset(env_ids)                                 # on reset
```

`tests/test_cable_friction.py` checks: μ = 0 equals MuJoCo's tendon exactly;
pulling from rest gives the capstan profile; holding keeps the tensions;
releasing obeys the bounds; correction mode equals the direct mode; the torch
batch equals the numpy model step by step (runs when torch is installed).

Cost: about 50 µs per step for two 40-site cables (pure Python chain), against
about 20 µs for the MuJoCo step of the 22.6 cm robot.

## What it changes (taper/length study, part 2)

See `studies/taper_selection/README.md`. In short: friction does not make the
body curl from the base when one cable is pulled from straight (the tip still
locks first), but it gives the body memory (large load-unload hysteresis), it
stops the body packing fully at μ ≥ 0.2, and it is what lets the longer bodies
wrap a 45 mm object in the reach-then-wrap programme.
