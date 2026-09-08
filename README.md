# SpiRob Physics Model Pipeline

Converts a small set of physical parameters into a fully simulation-ready MuJoCo model in three automated steps: CSV → STL → XML.

---

## Requirements

Python 3.9+ with dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Quick Start

**1. Edit `params.json`** — set your physical dimensions and cross-section choice.

**2. Run the pipeline:**

```bash
python build.py --nlobe
```

This validates parameters, shows the interactive geometry preview, then generates `Geom_Data_CSV/`, `meshes/`, and `spirob_physics_model.xml`.

**3. Load in MuJoCo:**

```bash
python -m mujoco.viewer --mjcf=spirob_physics_model.xml
```

---

## Build Flags

| Flag | Effect |
|---|---|
| `--nlobe` | n-lobe or flat cross-section driven by `n_cables`. Use for all real builds. |
| `--plain` | Full solid of revolution — no cut. Useful for debugging. |
| `--safe` | MuJoCo safe-mode preset (capsule collision, high damping, small timestep) |
| `--fast` | Lightweight preset (mesh physics, large timestep) |
| `--high` | High-fidelity preset (RK4 integrator, fine timestep) |
| `--noclean` | Keep previous CSV/STL/XML outputs |
| `--no-preview` | Skip interactive preview (batch / CI use) |
| `--cad` | Also export a whole-robot **STEP + solid STL** to `cad/` (for 3-D printing / CAD) |
| `--fuse-cad` | With `--cad`, boolean-union the elements into one solid (slower, cleaner) |
| `--params FILE` | Use a different params JSON file (default: `params.json`) |

Flags can be combined: `python build.py --nlobe --safe --no-preview --cad`

---

## Project Structure

```
├── build.py                  # Pipeline driver
├── params.json               # All user-facing parameters — edit this
├── spirob_csv_generator.py   # Step 1: spiral maths → geometry CSV
├── csv2geom_nlobe.py         # Step 2: CSV → per-link STL meshes (for MuJoCo)
├── csv2xml.py                # Step 3: CSV + STL → MuJoCo MJCF XML
├── cad_export.py             # Optional: whole-robot STEP + solid STL (printing/CAD)
├── design_gui.py             # Optional: desktop GUI wrapping the pipeline
├── preview.py                # Interactive 2-D geometry preview
├── helper_functions.py       # Shared spiral maths
├── fabrication/
│   └── part_splitter.py      # Split oversized STEP/STL into printable parts
├── tools/
│   └── multi_array.py        # Replicate the model into a circular robot array
├── requirements.txt
├── README.md
└── .gitignore
```
Secondary Helper functions:
add_touch_sensor.py
  - Helps to add sensors on the spirob surface between the tendon sites. Needs the spirob_physics_model.xml to be already generated.
  - The touch sensors are can also be placed at user defined location on the links of choice.
2_cable_touch_debug.py
  - Can be used for monitoring touch detection for the 2 cable model, shows a live graph of when the touch is dectected at sensor of desire.
  - 
Generated outputs (not committed):

```
Geom_Data_CSV/   meshes/   spirob_physics_model.xml
```

---

## Parameters Reference

### Core geometry

| Parameter | Type | Valid range | Description |
|---|---|---|---|
| `L` | float (m) | > 0 | Total uncoiled length |
| `d_tip` | float (m) | > 0 | Outer diameter at the tip |
| `phi_deg` | float (°) | (0, 45) | Taper half-angle of the logarithmic spiral |
| `Delta_theta_deg` | float (°) | (0, 180) | Angular span per element — smaller = more elements |
| `n_cables` | int | ≥ 2 | Number of cables; also selects cross-section |
| `tendon_inward_shift` | float (m) | [0, d_tip/2) | Cable site inward offset from outer surface |
| `show_preview` | bool | — | Launch preview in standalone mode |

### Cross-section — n-lobe (`n_cables >= 3`)

| Parameter | Valid range | Description |
|---|---|---|
| `nlobe_t` | [0, 1] | 0 = circumscribed polygon, 1 = inscribed, 0.5 = balanced default |
| `notch_factor` | [0, 0.4] | Notch radius as fraction of polygon half-side |

### Cross-section — flat slab (`n_cables <= 2`)

| Parameter | Valid range | Description |
|---|---|---|
| `flat_thickness_ratio` | [0.05, 1.0] | Element thickness as fraction of outer radius |

### `post_gen` block — post-generation overrides

All fields optional. Applied by `csv2xml.py` during XML generation — no separate script needed.

| Field | Type | Description |
|---|---|---|
| `robot_pos` | [x, y, z] | World-frame position of the base link |
| `robot_quat` | [w, x, y, z] | World-frame orientation of the base link |
| `joint_stiffness_base` | float | Stiffness of joint 0; decays as `k_base / β³ⁱ` |
| `joint_damping_base` | float | Damping of joint 0; same decay |
| `joint_beta` | float > 0 | Decay rate β (default 1.03) |
| `first_joint_stiffness` | float | Independent override for joint 0 stiffness |
| `first_joint_damping` | float | Independent override for joint 0 damping |
| `tip_site_pos` | [x, y, z] | Green sphere site on the tip body |
| `target_site_pos` | [x, y, z] | Red sphere site in the world frame |

---

## Cross-Section Modes

| Condition | Joint type | Description |
|---|---|---|
| `--plain` | ball | Full solid of revolution, no cut |
| `n_cables <= 2` | hinge | Flat tapered slab (trapezoidal quad profile, extruded ±Y) |
| `n_cables >= 3` | ball | Revolved cylinder ∩ regular n-gon with notch cuts |

Joint type (hinge vs ball) is set automatically — no flag needed.

---

## Joint Stiffness & Damping Decay

```
k_i = k_base / β³ⁱ      d_i = d_base / β³ⁱ
```

With the default β = 1.03, joint 20 has ~17× less stiffness than joint 0, matching the increasing compliance from base to tip of a real soft robot.

Use `first_joint_stiffness` / `first_joint_damping` to independently override joint 0 when the base attachment is mechanically much stiffer.

---

## Fabrication & CAD

The per-link STL meshes in `meshes/` are what MuJoCo loads — each is in its own
link frame. For **3-D printing or CAD**, you want the whole robot as one solid,
which `cad_export.py` produces.

### Whole-robot STEP + solid STL

```bash
python cad_export.py                 # writes cad/spirob_<timestamp>.step and .stl
python cad_export.py --fuse          # boolean-union into a single manifold solid
python build.py --nlobe --cad        # or export as part of the normal build
```

It assembles every element in its world position (reusing the exact geometry the
simulation meshes are built from) and writes a STEP (authoritative solid for CAD
/ slicers) plus a solid STL. The default combines elements as a compound (fast);
`--fuse` produces one boolean-fused solid (slower).

- **`n_cables >= 3`** → the n-lobe solid (revolved cross-section with notch cuts).
- **`n_cables <= 2`** → a flat leaf with a **lens (biconvex) cross-section**: the
  thickness is highest along the centre-line and tapers toward both lateral edges
  (the cable holes), like the physical SpiRob. Centre thickness defaults to
  `flat_thickness_ratio × base outer diameter` (override with `--flat-thickness-m`);
  the edge thickness is `flat_edge_ratio × centre` (default 0.25, override with
  `--flat-edge-ratio`; use `0` for a knife edge).

### Splitting oversized parts for printing

If the robot is larger than your printer's build volume, split it along one axis:

```bash
# STEP (true solid splitting via CadQuery)
python fabrication/part_splitter.py cad/spirob_<ts>.step --axis z --max-span-mm 180

# STL (mesh splitting) with explicit cut planes and build-volume fit checks
python fabrication/part_splitter.py cad/spirob_<ts>.stl  --axis z \
       --cut-positions-mm 80,160 --build-volume-mm 256,256,256
```

Output goes to a sibling `<input>_split/` folder with numbered parts and a
`split_report.json` (each part's bounds, span, and build-volume fit). Sizes are
given in **mm**; the input files are numerically in metres, which is the default
(`--file-units m`). The STL path needs `trimesh`, `shapely`, and `mapbox-earcut`
(see `requirements.txt`). This first version does geometric splitting only — no
keyed joints or pins yet.

---

## Multi-robot array

Replicate a generated model into a circular array of robots sharing one world:

```bash
python tools/multi_array.py --in spirob_physics_model.xml --count 6 \
       --radius-m 0.12 --tilt-deg -20 --out spirob_array.xml
```

Each copy's bodies, joints, geoms, sites, tendons and actuators are renamed with
a `_r{k}` suffix (mesh assets are shared), so the result loads directly in MuJoCo.

---

## Desktop GUI

A lightweight params-driven GUI wraps the whole pipeline — edit parameters with a
live 2-D preview (side profile + cross-section), then build, export STEP/STL,
generate an array, or open the MuJoCo viewer:

```bash
python design_gui.py                 # loads params.json
```

Built on Tkinter (ships with standard CPython; on Debian/Ubuntu install
`python3-tk`) and Matplotlib. The preview reuses this repo's own geometry code,
so it always matches what the pipeline builds.

---

## Troubleshooting

**Parameter error on startup** — `validate_params()` lists every problem. Fix `params.json` and re-run.

**STL fails with `BRepAdaptor_Curve::No geometry`** — degenerate element geometry. Try reducing `Delta_theta_deg` by ~20%, or adjust `nlobe_t` away from exactly 0 or 1.

**Bodies all appear at the same location in MuJoCo** — set `post_gen.robot_pos` to `[0, 0, L]` so the base clears the ground plane.

**Tendons miss the link geometry** — increase `tendon_inward_shift`. The preview shows the corrected tendon path; verify visually before generating STL.

**Simulation unstable** — start with `--safe`. If stable, move to `--fast` then default. If still unstable, increase `joint_damping_base` in `post_gen`.

---

## Credits & attribution

This project is MIT-licensed (see `LICENSE`). The core spiral geometry follows
the logarithmic-spiral formulation of **SpiRobs** (Wang et al., 2024, *SpiRobs:
Logarithmic spiral-shaped robots for versatile grasping across scales*).

The fabrication and CAD features added here — whole-robot STEP/STL export, the
build-volume-aware part splitter, the multi-robot array, and the design GUI —
were **inspired by** the authors' [OpenSpiRobs toolkit](https://github.com/ZhanchiWang/Open-Spiral-Robots)
(Zhanchi Wang et al.), which is released under the PolyForm Noncommercial
license. To keep this repository MIT-licensed and commercial-friendly, those
features are **clean-room reimplementations** built on this repo's own geometry
stack — no OpenSpiRobs source code is used or included. Unlike that toolkit,
they support the full n-cable (`> 3`) cross-section this pipeline generates.
