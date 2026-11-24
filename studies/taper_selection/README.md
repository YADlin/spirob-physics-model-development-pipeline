# Body study for the two-cable SpiRob: taper, length, cable friction and wrapping

> **Correction (24 Nov 2026).** Part 1 judged a grasp by the angle of the object's circumference
> that the body touched anywhere ("enclosure"). An egg pinched between the coil and the base scored
> 180° without being wrapped, and the part-1 egg animation shows exactly that. Part 1's grasp
> results (size windows, egg map, and the egg argument for printing 8°) are withdrawn. Part 2
> replaces them with a strict wrap test. Part 1's curling, printing, validation and joint-limit
> results are unaffected.

## Part 2 — cable friction, length and wrapping (24 Nov 2026)

**Questions.** (1) What does cable friction do to the SpiRob, and how can it be put in MuJoCo and
mjlab? (2) How does body length (22.6–45 cm) change what the body can wrap? (3) Can any body wrap
a 45 mm egg — egg in the middle of the curl, at least half wrapped — and then hold it and move it?

**Definitions (agreed with NYK).** *Wrap*: one unbroken stretch of the body touches the object over
at least 180° of its circumference (`wrap.wrap_angle`; touches by two separate parts of the body do
not add up). *Grasp*: after the wrap, the cables move the object by at least one diameter while it
stays wrapped and the contact can carry its weight out of plane (μ_c·ΣN ≥ m·g, μ_c = 0.5, m = 60 g).

**Answer.**

In MuJoCo, with the cable programmes tested here, **no body wraps the 45 mm egg when it simply
sits on the spirob-mjlab pedestal**. The body pushes on the egg while it closes (1.4 N for the current
robot, 5–33 N for the 45 cm bodies) and the pedestal holds it with only 0.24 N, so the egg is pushed
off or never wrapped beyond 66°. When the egg is **held in place while the body closes** and released
afterwards, the picture splits by length:

| Body | Egg inside the curl (positions near success) | Moved ≥ 45 mm while held | Push the holder must resist | Packing force |
|---|---|---|---|---|
| 22.6 cm 6.3° (current) | 16 of 21 | 0 (≤ 13 mm) | 1.4–1.6 N | 6 N |
| 22.6 cm 8° / 10° | 0 | 0 | – | 9 / 14 N |
| 30 cm 6.3° | 10 of 15 | 0 (≤ 24 mm) | 3–4 N | 10 N |
| 37.5 cm 6.3° | 17 of 21 | 1 (45 mm) | 5–12 N | 14 N |
| **45 cm 6.3°** | **21 of 21** | **14 (≤ 75 mm)** | **5–15 N** | **18 N** |
| 45 cm 8° | 21 of 26 | 6 (≤ 84 mm) | 9–16 N | 26 N |
| 45 cm 10° | 15 of 20 | 5 (≤ 71 mm) | 13–33 N | 37 N |
| authors' 45 cm (10°, 24 units) | 21 of 25 | 15 (≤ 68 mm) | 16–23 N | 42 N |

* **Length matters more than taper.** Only the 45 cm bodies (and one 37.5 cm case) could hold the egg
  and move it by its own diameter. At 45 cm, 6.3° did it at the most positions for the least force.
* **Cable friction is what makes the wraps happen.** Without it no 45 mm wrap passed 70° on any body
  longer than 22.6 cm; every strict wrap of the egg (one stretch ≥ 180°) needed μ ≥ 0.2 (small
  objects, 15–25 mm, can be wrapped without friction). The tip then curls round
  the egg instead of coiling up beside it (Animation 4). Friction does not make the body curl from the
  base under one cable (the bending centre moves ≤ 0.08 L), but it gives the body memory: released at
  the same force, the tip is 10–85 mm from where it was when pulled, and settled shapes leave the
  frictionless one-parameter curve by 5–15 mm on average. At μ = 0.3 the body no longer packs at
  1.25 × the frictionless packing force.
* **The authors' reach-then-wrap sequence beats a plain pull** on a held egg (235° against 105°,
  Animation 5).
* **Object size vs contact point.** Held objects wrap only on or just below the paper's Eq. 15 line
  (largest diameter for the free length L − s): 15–25 mm at s/L 0.5–0.8 on 22.6 cm 8°, 25–45 mm at
  s/L 0.7–0.8 on 45 cm 8°. Free objects of every size are pushed off.
* **Release is violent.** When the holder lets go at full cable force, the body pulls the egg 70–300 mm
  into its coil within about 0.2 s, where it usually stays held (Animation 7); a few 45 cm cases
  fling it out.
* **The task egg position** (5 cm across, 7.6 cm from the base) is never wrapped by any body (best 89°).

**Recommendation, revised.** Part 1's "print 8°" was based on the faulty grasp metric and is withdrawn.
If the egg must be wrapped and moved, a **45 cm body** is needed in this model; at that length
**φ = 6.3°** (tip 7.1 mm, base 57 mm, 28 units) did best for the least force, and the authors' own
45 cm file also works. Whatever the body, the egg must be **held against a lateral push** while the
body closes (≈ 2 N for the current robot, ≈ 15 N for 45 cm 6.3°) and let go once it is wrapped, or a
faster, dynamic grasp must be found (not tested here). For RL: add the cable friction model, keep
`±Δθ` joint limits and a width-dependent stiffness prior, and check the egg/pedestal friction for
creep (see below).

Figures: `fig10_friction_curl`, `fig11_friction_workspace`, `fig12_wrap_maps`, `fig13_wrap_summary`,
`fig14_size_position`, `fig15_grasp_summary`. Animations 4–7 (`anim4_friction_wrap`,
`anim5_pull_vs_reach`, `anim6_task_egg`, `anim7_wrap_and_carry`) show the wrap angle and the curl angle
measured on every frame.

Two success measures are reported. *Wrap* (strict): one unbroken stretch touches ≥ 180°.
*Inside the curl*: the whole body winds ≥ 270° round the egg centre and the contact carries its weight
(an egg pinched between the base and the outside of the coil, part 1's failure, scores about 220°).
The strict measure was run on every trial; the curl measure on every held-egg case that reached at
least 90° of contact (286 cases, `results/wrap/carry.json`).

### What was added

| Module | What it does |
|---|---|
| `spirob/cable_friction.py` (repo package) | Stick-slip capstan friction for site-routed tendons: numpy for MuJoCo, batched torch for mjlab. See `docs/cable_friction.md`; tests in `tests/test_cable_friction.py`. |
| `designs.py`, `analytic.osr_full_design` | Length grid 22.6 / 30 / 37.5 / 45 cm × 6.3 / 8 / 10°, tip 7.14 mm; the authors' 45 cm robot (their print file at scale 1: 24 units, φ = 10°, tip 10 mm, the robot of the paper's Fig. 3) |
| `exp_friction.py` | F1 load–unload with μ = 0–0.3; F2 settled workspace under random two-cable forces |
| `wrap.py` | Wrap metric, cable programmes ("pull"; "reach" = the authors' curl-away-then-reach-and-wrap), egg on a pedestal or in a holder, carry test |
| `exp_wrap.py` | `main` (egg on a grid, 2052 trials), `pull` (baseline), `size` (diameter × contact point), `refine` (around the best positions, μ up to 0.3), `recheck` (after the two model fixes below) |
| `figures2.py`, `media2.py`, `media_runs.py` | Figures 10–14, animations 4–7 (every reported wrap has a video with the wrap angle measured on each frame) |

Two model issues were found and fixed during the work: (a) MuJoCo's dof friction (`frictionloss`)
is a soft constraint that creeps under any force (an egg pushed with 0.1 N against 0.24 N of
"friction" drifts 3.5 mm in 2 s; under a 200 N limit it still drifts 157 mm at 4.5 N). The pedestal
friction now uses a 2-timestep time constant (drift 0.07 mm). (b) A static object welded to the
world cannot touch the robot's first moving link (MuJoCo's parent filter) and cannot be released,
so the "holder" is now a stiff weld equality that is switched off for the carry test. `recheck`
re-ran every affected case.

# Part 1 — taper sweep at 22.6 cm (23 Nov 2026)

**Question.** Which two-cable SpiRob body should be printed for the egg-to-bucket RL task?
The body is fixed by four inputs of this generator: length `L`, tip width `d_tip`,
taper angle `φ` and step `Δθ`. This study keeps the current robot's length and tip width
(`L = 226.28 mm`, `d_tip = 7.14 mm`), uses the paper's `Δθ = 30°`, and changes only the taper:
`φ = 4, 5, 6.3, 8, 10, 12, 15°`. Everything is simulated in MuJoCo with models built by this
repository's `build.py`.

**Answer (23 Nov, partly withdrawn 24 Nov).** ~~Print φ = 8°~~ — withdrawn, see Part 2. The curling,
printing (≤ 0.12 mm layers), validation and joint-limit findings below still hold. The full write-up, with
figures and animations, is the Notion page *"Choosing the SpiRob body — a taper-angle study"*.

![Summary of the sweep](figures/fig8_summary.png)

## What was simulated

| # | Experiment | What it measures | Module |
|---|---|---|---|
| 1 | One cable pulled slowly | where curling starts, how gradually it spreads, the force to pack | `exp_curl.py` |
| 2 | Tip workspace | the paper's kinematic envelope, the slow-pull path, random dynamic forcing | `exp_workspace.py` |
| 3a | Free cylinder beside the body (**withdrawn**) | superseded by Part 2 | `exp_grasp.py` |
| 3b | 40 mm egg on a position grid (**withdrawn**) | superseded by Part 2 | `exp_grasp.py` |
| 4 | Validation | geometry vs Open-Spiral-Robots, packed shape vs analytic spiral, lock forces vs closed form, joint limits | `exp_validation.py` |
| 5 | Printing and friction | flexure thickness rounded to whole layers; capstan friction on the cable | `exp_print.py` |

Model changes made for the study (in `model.py`, applied to a copy of each generated XML):
gravity off (the rig bends in a horizontal plane), base welded, joint limits `±Δθ`, joint stiffness
`K_i = κ W_i³` with one material constant for all designs, damping `0.05 s × K_i`, and force-controlled
cables.

## Main results

| φ | Packing force | Mass | Wrapped and held (direct pull) | 40 mm egg positions | 45 mm, contact span | Distal share of bending (μ = 0 / 0.3) |
|---|---|---|---|---|---|---|
| 4° | 3.6 N | 52 g | 40–50 mm | 13 | not run | 0.72 / 0.59 |
| 5° | 4.5 N | 65 g | 30–60 mm | 17 | 0.4–0.8 | 0.75 / 0.62 |
| 6.3° | 6.1 N | 86 g | 20–60 mm | 17 | 0.5–0.7 | 0.82 / 0.71 |
| **8°** | **9.0 N** | **118 g** | **20–50 mm** | **13** | **0.3–0.7** | **0.85 / 0.76** |
| 10° | 14.3 N | 160 g | 10–40 mm | 11 | 0.5–0.6 | 0.88 / 0.80 |
| 12° | 15.6 N | 207 g | 10–30 mm | 0 | not run | 0.90 / 0.84 |
| 15° | 29.6 N | 282 g | 10–20 mm | 0 | not run | 0.92 / 0.89 |

* A larger taper curls more strongly from the tip, and keeps doing so under cable friction and
  printing. It also costs force and mass, and shrinks the largest object the body can wrap by a
  direct pull. At this length, 12° and 15° cannot hold the egg at all.
* ~~8° is the largest taper that still holds a real egg over a useful stretch of the body.~~ Withdrawn:
  the grasp columns of this table used the faulty enclosure metric.
* Validation: this repository's geometry matches the authors' design tool to 1e-14 (relative) unit by unit.
  Simulated lock forces follow the closed form `F_i = K_i Δθ / h_i` within a few percent. The packed
  body matches the analytic spiral to sub-millimetre accuracy. One defect was found: the generator
  writes hinge ranges of `±π`, and because MuJoCo ignores parent–child contacts, adjacent units pass
  through each other (up to 1.62 Δθ).

## What this changes for the RL model

1. **Joint limits.** Set each flexible hinge to `±Δθ` (as the paper and the authors' generator do).
   Do not leave it at `±π`.
2. **Stiffness profile.** The spirob-mjlab law `k_i = k₂·0.915^(i−2)` makes the tip about 12× stiffer than
   geometric similarity predicts. The robot then bends almost uniformly (distal share 0.49, against 0.79
   for `K ∝ W³`), and its grasp map changes (Fig. 9). Use `ρ_k = β_g⁻³` as the prior and identify it on
   the new print.
3. **Cables.** With frictionless MuJoCo tendons, the quasi-static shape depends almost only on
   `F₁ − F₀`. The two cables act as one input plus co-contraction. Real cables have friction.
   Measure it; `sim.ramp_single_cable(mu=…)` shows its effect.

## Reproduce

```bash
uv sync --locked
git clone https://github.com/ZhanchiWang/Open-Spiral-Robots ../Open-Spiral-Robots   # validation only
uv run python studies/taper_selection/run_all.py --osr ../Open-Spiral-Robots
```

All results are written to `results/` (JSON is tracked; the large `.npz` logs are not) and figures to
`figures/`. Generated models go to `_build/` (ignored). Rendering uses software EGL; on a desktop it
also works with `MUJOCO_GL=glfw`.

## Limitations

* The joint model is a rigid-link chain with linear springs. TPU hyperelasticity, flexure shear and
  unit deformation are not modelled.
* Absolute stiffness is anchored to spirob-mjlab's `k₂ = 0.183 N·m/rad`, so forces are relative.
  The campaign fit (0.28 N·m/rad) would scale every force by about 1.5×.
* Grasps use a single-cable pull (superseded by Part 2, which adds cable friction and the reach–wrap
  strategy).
* The 40 mm egg is the spirob-mjlab egg cap, a cylinder in plane. Out-of-plane holding is reported only
  as the normal grip force.

## Licence note

The validation loads two pure geometry functions from a **local clone** of Open-Spiral-Robots at run
time (PolyForm-Noncommercial). It uses the same `ast` method as the authors' own parity tests. No
code from that repository is copied here.
