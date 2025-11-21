# MJCF writer with "safe preset" to avoid NaNs during bring-up

import csv, math, os
from typing import List, Dict, Tuple
import numpy as np
import argparse
from dataclasses import dataclass
from typing import Tuple
import json

# Added for fixing the tendon distance from body center
# Load params.json ONCE
with open("params.json") as f:
    params = json.load(f)
TENDON_INWARD_SHIFT = 0.0015     # shift towards the center (meters)
PHI_DEG = float(params["phi_deg"])   # read from params.json
HALF_PHI = math.radians(PHI_DEG / 2)


@dataclass
class MJCFConfig:
    safe: bool = False
    physics_mode: str = "mesh"
    joint_type: str = "ball"
    disable_contacts: bool = False
    hinge_axis_local: Tuple[float, float, float] = (0, 1, 0)
    hinge_range_deg: float = 180.0

    # Defaults
    geom_density: float = 1200.0
    geom_margin: float = 0.001
    geom_rgba: Tuple[float, float, float, float] = (0.20, 0.35, 0.65, 1.0)
    friction: Tuple[float, float, float] = (0.6, 0.01, 0.001)
    site_size: float = 0.001
    joint_damping: float = 0.01
    joint_stiffness: float = 0.2
    beta: float = 1.03   # beta > 1 → decreasing stiffness/damping along chain
    motor_gear: float = 1.0
    ctrl_range: Tuple[float, float] = (-5.0, 0.0)
    tendon_width: float = 0.0006
    integrator: str = "implicit"
    timestep: float = 0.002
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    capsule_radius_scale: float = 0.25
    capsule_radius_min: float = 8e-4
    mesh_scale_attr: str = None

    # =========================
    # Presets
    # =========================
    @classmethod
    def safe_mode(cls):
        """Stable simulation: small timestep, capsules, high damping."""
        return cls(
            safe=True,
            physics_mode="capsule",
            hinge_range_deg=15.0,
            geom_margin=0.0002,
            joint_damping=0.2,
            joint_stiffness=0.2,
            ctrl_range=(-5.0, 5.0),
            tendon_width=0.0004,
            timestep=0.001,
            integrator="implicit"
        )

    @classmethod
    def fast_mode(cls):
        """Lightweight simulation: larger timestep, low damping."""
        return cls(
            safe=False,
            physics_mode="mesh",
            hinge_range_deg=180.0,
            geom_margin=0.002,          # looser margin
            joint_damping=0.001,        # almost no damping
            joint_stiffness=0.01,       # soft
            ctrl_range=(-10.0, 10.0),   # wide control
            tendon_width=0.0008,
            timestep=0.005,             # big timestep
            integrator="implicit"       # implicit is still robust
        )

    @classmethod
    def high_fidelity_mode(cls):
        """Accurate simulation: fine timestep, RK4 integrator, full mesh physics."""
        return cls(
            safe=False,
            physics_mode="mesh",        # meshes for contact realism
            hinge_range_deg=180.0,
            geom_margin=0.0005,         # tighter margin
            joint_damping=0.02,         # realistic damping
            joint_stiffness=0.1,        # stiffer response
            ctrl_range=(-5.0, 5.0),
            tendon_width=0.0005,
            timestep=0.0005,            # very fine timestep
            integrator="RK4"            # high accuracy
        )

# ============================================================
# HELPERS
# ============================================================
def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def _frame_from_segment(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    z = _unit(p1 - p0)
    fallback = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = _unit(np.cross(fallback, z))
    if np.linalg.norm(x) < 1e-12:
        fallback = np.array([0.0, 1.0, 0.0])
        x = _unit(np.cross(fallback, z))
    y = np.cross(z, x)
    return np.column_stack([x, y, z])

def _mat_to_quat_wxyz(R: np.ndarray) -> Tuple[float, float, float, float]:
    m = R; t = m[0,0]+m[1,1]+m[2,2]
    if t > 0:
        s = math.sqrt(t+1.0)*2.0; w=0.25*s
        x=(m[2,1]-m[1,2])/s; y=(m[0,2]-m[2,0])/s; z=(m[1,0]-m[0,1])/s
    else:
        if m[0,0] > m[1,1] and m[0,0] > m[2,2]:
            s = math.sqrt(1.0+m[0,0]-m[1,1]-m[2,2])*2.0
            w=(m[2,1]-m[1,2])/s; x=0.25*s; y=(m[0,1]+m[1,0])/s; z=(m[0,2]+m[2,0])/s
        elif m[1,1] > m[2,2]:
            s = math.sqrt(1.0+m[1,1]-m[0,0]-m[2,2])*2.0
            w=(m[0,2]-m[2,0])/s; x=(m[0,1]+m[1,0])/s; y=0.25*s; z=(m[1,2]+m[2,1])/s
        else:
            s = math.sqrt(1.0+m[2,2]-m[0,0]-m[1,1])*2.0
            w=(m[1,0]-m[0,1])/s; x=(m[0,2]+m[2,0])/s; y=(m[1,2]+m[2,1])/s; z=0.25*s
    q = np.array([w,x,y,z], float); q /= (np.linalg.norm(q)+1e-15); return tuple(q.tolist())

def _world_to_local(Rp: np.ndarray, pp: np.ndarray, pw: np.ndarray) -> np.ndarray:
    return Rp.T @ (pw - pp)

def _relative_pose(Rp: np.ndarray, pp: np.ndarray, Rc: np.ndarray, pc: np.ndarray):
    pos_rel = Rp.T @ (pc - pp); R_rel = Rp.T @ Rc; return pos_rel, _mat_to_quat_wxyz(R_rel)

def _parse_sites_csv(csv_path: str):
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f); header = next(reader); rows = list(reader)
    # infer n_cables
    cable_ids = []
    for name in header:
        if name.startswith("c") and "_s1_x" in name:
            try: cable_ids.append(int(name[1:name.index("_")]))
            except: pass
    n_cables = (max(cable_ids)+1) if cable_ids else 1
    def idx(n): return header.index(n)
    elements = []
    for r in rows:
        i = int(float(r[idx("elem")]))
        A0 = np.array([float(r[idx("joint_s1_x")]), float(r[idx("joint_s1_y")]), float(r[idx("joint_s1_z")])])
        A1 = np.array([float(r[idx("joint_s2_x")]), float(r[idx("joint_s2_y")]), float(r[idx("joint_s2_z")])])
        sites = {}
        for c in range(n_cables):
            s1 = np.array([float(r[idx(f"c{c}_s1_x")]), float(r[idx(f"c{c}_s1_y")]), float(r[idx(f"c{c}_s1_z")])])
            s2 = np.array([float(r[idx(f"c{c}_s2_x")]), float(r[idx(f"c{c}_s2_y")]), float(r[idx(f"c{c}_s2_z")])])
            sites[c] = {"s1": s1, "s2": s2}
        elements.append(dict(idx=i, p0=A0, p1=A1, sites=sites))
    elements.sort(key=lambda d: d["idx"]); return elements, n_cables

# ============================================================
# MAIN WRITER FUNCTION
# ============================================================
def write_mjcf_from_sites_csv(
    csv_path: str,
    out_xml_path: str = "spirob_mujoco.xml",
    mesh_dir: str = "meshes",
    digits: int = 3,
    config: MJCFConfig = MJCFConfig()
) -> str:
    elements, n_cables = _parse_sites_csv(csv_path)
    if not elements: raise ValueError("CSV has no elements.")

    # frames & local site coords
    frames = []
    for e in elements:
        p0, p1 = e["p0"], e["p1"]
        R = _frame_from_segment(p0, p1); q = _mat_to_quat_wxyz(R)
        L = float(np.linalg.norm(p1 - p0))
        if L <= 1e-9: raise ValueError(f"Element {e['idx']} has ~zero length; check joint_s1/s2.")
        # check that the two tendon sites are not identical (zero segment)
        for c, ss in e["sites"].items():
            if np.linalg.norm(ss["s1"] - ss["s2"]) < 1e-9:
                raise ValueError(f"Tendon site s1==s2 for element {e['idx']}, cable {c}.")
        # sites_local = {c: {"s1": _world_to_local(R, p0, ss["s1"]),
        #                    "s2": _world_to_local(R, p0, ss["s2"])}
        #                for c, ss in e["sites"].items()}
        # --- BEGIN: tendon radius corrected placement ---
        sites_local = {}

        for c, ss in e["sites"].items():

            # world → local
            s1_local = _world_to_local(R, p0, ss["s1"]).astype(float)
            s2_local = _world_to_local(R, p0, ss["s2"]).astype(float)

            # r_old = radius of s1 (your chosen rule)
            r1_old = float(np.linalg.norm(s1_local[:2]))

            # safe fallback if s1 is on the axis
            if r1_old < 1e-12:
                unit_r = np.array([1.0, 0.0])
                r1_old = 1e-3
            else:
                unit_r = s1_local[:2] / r1_old

            # axial separation
            dz = float(s2_local[2] - s1_local[2])

            # new radii
            r1_new = max(r1_old - TENDON_INWARD_SHIFT, 1e-6)
            r2_new = max(r1_old - TENDON_INWARD_SHIFT - dz * math.tan(HALF_PHI), 1e-6)

            # apply new positions
            s1_local[:2] = unit_r * r1_new
            s2_local[:2] = unit_r * r2_new

            # store
            sites_local[c] = {"s1": s1_local, "s2": s2_local}
        # --- END: tendon radius corrected placement ---
        frames.append(dict(p0=p0, p1=p1, R=R, quat=q, L=L, sites_local=sites_local))
    # relative poses for nesting
    rel = []
    for i in range(len(frames)):
        if i == 0: pos, qt = frames[i]["p0"], frames[i]["quat"]
        else: pos, qt = _relative_pose(frames[i-1]["R"], frames[i-1]["p0"], frames[i]["R"], frames[i]["p0"])
        rel.append((pos, qt))

    # asset meshes
    mesh_assets = []
    for i in range(len(frames)):
        mesh_name = f"link_{i+1:0{digits}d}"
        mesh_file = os.path.join(mesh_dir, f"{mesh_name}.stl")
        if not os.path.exists(mesh_file):
            print(f"[WARN] Mesh not found: {mesh_file} (model will still load, but mesh won't render)")
        scale_attr = f' scale="{config.mesh_scale_attr}"' if config.mesh_scale_attr else ""
        mesh_assets.append(f'    <mesh name="{mesh_name}" file="{mesh_file}"{scale_attr}/>')

    # header & defaults
    header = f'''<mujoco model="SpiRob">
  <compiler angle="radian" inertiafromgeom="true" autolimits="true"/>
  <option timestep="{config.timestep}" gravity="{config.gravity[0]} {config.gravity[1]} {config.gravity[2]}" integrator="{config.integrator}"/>
  <visual>
    <quality shadowsize="4096"/>
    <map znear="0.01" zfar="20"/>
    <scale connect="0"/>
  </visual>

  <!-- ===== DEFAULTS: edit these to tune global behavior ===== -->
  <default>
    <geom density="{config.geom_density}" margin="{config.geom_margin}"
          rgba="{config.geom_rgba[0]} {config.geom_rgba[1]} {config.geom_rgba[2]} {config.geom_rgba[3]}"
          friction="{config.friction[0]} {config.friction[1]} {config.friction[2]}"/>
    <site size="{config.site_size}"/>
    <joint damping="{config.joint_damping}" stiffness="{config.joint_stiffness}" limited="{str(config.joint_type=='hinge').lower()}"/>
    <default class="cable_motor">
        <motor ctrllimited="true" ctrlrange="{config.ctrl_range[0]} {config.ctrl_range[1]}" gear="{config.motor_gear}"/>
    </default>
  </default>

  <asset>
{os.linesep.join(mesh_assets)}
  </asset>

  <worldbody>
    <light name="light1" diffuse="1 1 1" active="true" pos="0 -0.5 0.01" dir="0 1 0"/>
    <light name="light2" diffuse="1 1 1" active="true" pos="-0.01 -0.5 0.5" dir="0 0 -1"/>
    <light name="light3" diffuse="1 1 1" active="true" pos="0 -0.1 1.5" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.05" rgba="0.9 0.9 0.9 1"/>
'''
        # ---- Joint stiffness/damping geometric decay (β > 1 gives decreasing values) ----
    base_k = config.joint_stiffness
    base_d = config.joint_damping

    beta3 = config.beta ** 3   # β^3

    # For joint i: k_i = k0 / (β^3)^i
    joint_stiffness = [ base_k / (beta3 ** i) for i in range(len(frames)) ]
    joint_damping   = [ base_d / (beta3 ** i) for i in range(len(frames)) ]

    # bodies chain
    body_xml = []
    for i, fr in enumerate(frames):
        name   = f"link_{i+1:0{digits}d}"
        L      = fr["L"]
        pos,qt = rel[i]
        pos_str = f'{pos[0]} {pos[1]} {pos[2]}'
        qt_str  = f'{qt[0]} {qt[1]} {qt[2]} {qt[3]}'

        # joint
        if config.joint_type == "hinge":
            jtag = (
                f'<joint name="j_{i+1:0{digits}d}" type="hinge" '
                f'axis="{config.hinge_axis_local[0]} {config.hinge_axis_local[1]} {config.hinge_axis_local[2]}" '
                f'range="-{math.radians(config.hinge_range_deg)} {math.radians(config.hinge_range_deg)}" '
                f'damping="{joint_damping[i]}" stiffness="{joint_stiffness[i]}" />'
            )
        else:
            jtag = (
                f'<joint name="j_{i+1:0{digits}d}" type="ball" '
                f'damping="{joint_damping[i]}" stiffness="{joint_stiffness[i]}" />'
            )

        # geoms
        geoms = []
        if config.physics_mode == "capsule":
            # physics by capsule; mesh visual-only
            s0 = fr["sites_local"][0]["s1"]; s1 = fr["sites_local"][0]["s2"]
            r0 = float(np.linalg.norm(s0[:2])); r1 = float(np.linalg.norm(s1[:2]))
            cap_r = max(config.capsule_radius_min, config.capsule_radius_scale*max(r0, r1))
            if config.disable_contacts:
                geoms.append(f'<geom type="capsule" fromto="0 0 0  0 0 {L}" size="{cap_r}" contype="0" conaffinity="0"/>')
            else:
                geoms.append(f'<geom type="capsule" fromto="0 0 0  0 0 {L}" size="{cap_r}"/>')
            # mesh visual-only
            geoms.append(f'<geom type="mesh" mesh="{name}" contype="0" conaffinity="0" group="1"/>')
        else:
            # physics by mesh
            if config.disable_contacts:
                geoms.append(f'<geom type="mesh" mesh="{name}" contype="0" conaffinity="0" group="1"/>')
            else:
                geoms.append(f'<geom name="gmesh_{i+1:0{digits}d}" type="mesh" mesh="{name}" group="1"/>')

        # sites
        sites = []
        for c in range(len(fr["sites_local"])):
            s1loc = fr["sites_local"][c]["s1"]; s2loc = fr["sites_local"][c]["s2"]
            sites.append(f'<site name="c{c}_{i+1:0{digits}d}_s1" pos="{s1loc[0]} {s1loc[1]} {s1loc[2]}"/>')
            sites.append(f'<site name="c{c}_{i+1:0{digits}d}_s2" pos="{s2loc[0]} {s2loc[1]} {s2loc[2]}"/>')

        block = f'''    <body name="{name}" pos="{pos_str}" quat="{qt_str}">
      {jtag}
      {' '.join(geoms)}
      {' '.join(sites)}
    </body>'''
        body_xml.append(block)

    # nest parent→child
    nested = body_xml[-1]
    for k in range(len(body_xml)-2, -1, -1):
        insert_at = body_xml[k].rfind("    </body>")
        body_xml[k] = body_xml[k][:insert_at] + "\n" + nested + "\n" + body_xml[k][insert_at:]
        nested = body_xml[k]

    # tendons
    tendon_xml = []
    n_cables = len(frames[0]["sites_local"])
    for c in range(n_cables):
        path = []
        path.append(f'      <site site="c{c}_{len(frames):03d}_s2"/>')
        path.append(f'      <site site="c{c}_{len(frames):03d}_s1"/>')
        for i in range(len(frames)-1, 0, -1):
            path.append(f'      <site site="c{c}_{i:03d}_s2"/>')
            path.append(f'      <site site="c{c}_{i:03d}_s1"/>')
        tendon_xml.append(f'''    <spatial name="cable_{c}" width="{config.tendon_width}">
{os.linesep.join(path)}
    </spatial>''')

    # actuators
    actuator_xml = []
    for c in range(n_cables):
        actuator_xml.append(f'    <motor class="cable_motor" name="motor_c{c}" tendon="cable_{c}"/>')

    # assemble
    xml = header + nested + "\n  </worldbody>\n  <tendon>\n" + \
          (os.linesep.join(tendon_xml)) + "\n  </tendon>\n  <actuator>\n" + \
          (os.linesep.join(actuator_xml)) + "\n  </actuator>\n</mujoco>\n"

    os.makedirs(os.path.dirname(out_xml_path) or ".", exist_ok=True)
    with open(out_xml_path, "w") as f: f.write(xml)
    return out_xml_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Spirob CSV to MJCF XML")
    parser.add_argument("--in", dest="input", required=True, help="Input CSV file")
    parser.add_argument("--out", dest="output", default="spirob_physics_model.xml", help="Output XML file")
    parser.add_argument("--meshdir", default="meshes", help="Directory containing STL meshes")
    parser.add_argument("--safe", action="store_true", help="Enable safe preset mode")
    parser.add_argument("--fast", action="store_true", help="Enable fast preset mode")
    parser.add_argument("--high", action="store_true", help="Enable high-fidelity preset mode")
    parser.add_argument("--digits", type=int, default=3, help="Zero-padding for link names")
    args = parser.parse_args()

    # config selection
    if args.safe:
        config = MJCFConfig.safe_mode()
    elif args.fast:
        config = MJCFConfig.fast_mode()
    elif args.high:
        config = MJCFConfig.high_fidelity_mode()
    else:
        config = MJCFConfig()  # default balanced

    out_xml = write_mjcf_from_sites_csv(
        csv_path=args.input,
        out_xml_path=args.output,
        mesh_dir=args.meshdir,
        digits=args.digits,
        config=config
    )
    print(f"Wrote {out_xml}")
