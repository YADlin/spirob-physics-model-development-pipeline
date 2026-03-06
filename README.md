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
| `--params FILE` | Use a different params JSON file (default: `params.json`) |

Flags can be combined: `python build.py --nlobe --safe --no-preview`

---

## Project Structure

```
├── build.py                  # Pipeline driver
├── params.json               # All user-facing parameters — edit this
├── spirob_csv_generator.py   # Step 1: spiral maths → geometry CSV
├── csv2geom_nlobe.py         # Step 2: CSV → STL meshes
├── csv2xml.py                # Step 3: CSV + STL → MuJoCo MJCF XML
├── preview.py                # Interactive 2-D geometry preview
├── helper_functions.py       # Shared spiral maths
├── requirements.txt
├── README.md
└── .gitignore
```

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

## Troubleshooting

**Parameter error on startup** — `validate_params()` lists every problem. Fix `params.json` and re-run.

**STL fails with `BRepAdaptor_Curve::No geometry`** — degenerate element geometry. Try reducing `Delta_theta_deg` by ~20%, or adjust `nlobe_t` away from exactly 0 or 1.

**Bodies all appear at the same location in MuJoCo** — set `post_gen.robot_pos` to `[0, 0, L]` so the base clears the ground plane.

**Tendons miss the link geometry** — increase `tendon_inward_shift`. The preview shows the corrected tendon path; verify visually before generating STL.

**Simulation unstable** — start with `--safe`. If stable, move to `--fast` then default. If still unstable, increase `joint_damping_base` in `post_gen`.
