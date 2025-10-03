# MJCF writer with "safe preset" to avoid NaNs during bring-up

import csv, math, os
from typing import List, Dict, Tuple
import numpy as np

# ------------ SAFETY PRESET ------------
SAFE_PRESET = False
# If True:
#  - PHYSICS_MODE="capsule" (fast, robust)
#  - INTEGRATOR="implicit", TIMESTEP=0.001
#  - JOINT_DAMPING=0.2
#  - CTRL_RANGE=(0, 5)
#  - DISABLE_CONTACTS=False (set True to isolate tendons/joints—no collisions)
#  - TENDON_WIDTH=0.0004

# ------------ MAIN KNOBS ---------------
PHYSICS_MODE = "mesh" if not SAFE_PRESET else "capsule"   # "mesh" or "capsule"
JOINT_TYPE   = "ball"                                      # "ball" or "hinge"
DISABLE_CONTACTS = False if not SAFE_PRESET else False     # True = no collisions (all geoms visual-only)

# Hinge (if you pick JOINT_TYPE="hinge")
HINGE_AXIS_LOCAL = (0, 1, 0)
HINGE_RANGE_DEG  = 15.0 if SAFE_PRESET else 180.0
JOINT_LIMITED    = (JOINT_TYPE == "hinge")                 # hinges limited; balls free unless you add range

# Physics & rendering defaults (go into <default>)
GEOM_DENSITY     = 1200.0
GEOM_MARGIN      = 0.0002 if SAFE_PRESET else 0.001
GEOM_RGBA        = (0.20, 0.35, 0.65, 1.0)
FRICTION         = (0.6, 0.01, 0.001)
SITE_SIZE        = 0.001
JOINT_DAMPING    = 0.2 if SAFE_PRESET else 0.05
MOTOR_GEAR       = 1.0
CTRL_RANGE       = (-5.0, 5.0) if SAFE_PRESET else (-50.0, 50.0)
TENDON_WIDTH     = 0.0004 if SAFE_PRESET else 0.0006
INTEGRATOR       = "implicit" if SAFE_PRESET else "RK4"
TIMESTEP         = 0.001 if SAFE_PRESET else 0.002
GRAVITY          = (0.0, 0.0, -9.81)

# Capsules (used only if PHYSICS_MODE=="capsule")
CAPSULE_RADIUS_SCALE = 0.25
CAPSULE_RADIUS_MIN   = 8e-4

# Optional mesh global scale (use "0.001 0.001 0.001" if your STL is **mm**)
MESH_SCALE_ATTR = None  # e.g., "0.001 0.001 0.001"  -> adds scale="..." on <mesh>

# ------------- Helpers ----------------
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

def write_mjcf_from_sites_csv(
    csv_path: str,
    out_xml_path: str = "spirob_mujoco.xml",
    mesh_dir: str = "meshes",
    digits: int = 3
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
        sites_local = {c: {"s1": _world_to_local(R, p0, ss["s1"]),
                           "s2": _world_to_local(R, p0, ss["s2"])}
                       for c, ss in e["sites"].items()}
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
        scale_attr = f' scale="{MESH_SCALE_ATTR}"' if MESH_SCALE_ATTR else ""
        mesh_assets.append(f'    <mesh name="{mesh_name}" file="{mesh_file}"{scale_attr}/>')

    # header & defaults
    header = f'''<mujoco model="SpiRob">
  <compiler angle="radian" inertiafromgeom="true" autolimits="true"/>
  <option timestep="{TIMESTEP}" gravity="{GRAVITY[0]} {GRAVITY[1]} {GRAVITY[2]}" integrator="{INTEGRATOR}"/>
  <visual>
    <quality shadowsize="4096"/>
    <map znear="0.01" zfar="20"/>
    <scale connect="0"/>
  </visual>

  <!-- ===== DEFAULTS: edit these to tune global behavior ===== -->
  <default>
    <geom density="{GEOM_DENSITY}" margin="{GEOM_MARGIN}"
          rgba="{GEOM_RGBA[0]} {GEOM_RGBA[1]} {GEOM_RGBA[2]} {GEOM_RGBA[3]}"
          friction="{FRICTION[0]} {FRICTION[1]} {FRICTION[2]}"/>
    <site size="{SITE_SIZE}"/>
    <joint damping="{JOINT_DAMPING}" limited="{str(JOINT_LIMITED).lower()}"/>
    <motor ctrllimited="true" ctrlrange="{CTRL_RANGE[0]} {CTRL_RANGE[1]}" gear="{MOTOR_GEAR}"/>
  </default>

  <asset>
{os.linesep.join(mesh_assets)}
  </asset>

  <worldbody>
    <light diffuse="1 1 1" pos="0 0 1.5" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.05" rgba="0.9 0.9 0.9 1"/>
'''

    # bodies chain
    body_xml = []
    for i, fr in enumerate(frames):
        name   = f"link_{i+1:0{digits}d}"
        L      = fr["L"]
        pos,qt = rel[i]
        pos_str = f'{pos[0]} {pos[1]} {pos[2]}'
        qt_str  = f'{qt[0]} {qt[1]} {qt[2]} {qt[3]}'

        # joint
        if JOINT_TYPE == "hinge":
            jtag = f'<joint name="j_{i+1:0{digits}d}" type="hinge" axis="{HINGE_AXIS_LOCAL[0]} {HINGE_AXIS_LOCAL[1]} {HINGE_AXIS_LOCAL[2]}" range="-{math.radians(HINGE_RANGE_DEG)} {math.radians(HINGE_RANGE_DEG)}"/>'
        else:
            jtag = f'<joint name="j_{i+1:0{digits}d}" type="ball"/>'

        # geoms
        geoms = []
        if PHYSICS_MODE == "capsule":
            # physics by capsule; mesh visual-only
            s0 = fr["sites_local"][0]["s1"]; s1 = fr["sites_local"][0]["s2"]
            r0 = float(np.linalg.norm(s0[:2])); r1 = float(np.linalg.norm(s1[:2]))
            cap_r = max(CAPSULE_RADIUS_MIN, CAPSULE_RADIUS_SCALE*max(r0, r1))
            if DISABLE_CONTACTS:
                geoms.append(f'<geom type="capsule" fromto="0 0 0  0 0 {L}" size="{cap_r}" contype="0" conaffinity="0"/>')
            else:
                geoms.append(f'<geom type="capsule" fromto="0 0 0  0 0 {L}" size="{cap_r}"/>')
            # mesh visual-only
            geoms.append(f'<geom type="mesh" mesh="{name}" contype="0" conaffinity="0" group="1"/>')
        else:
            # physics by mesh
            if DISABLE_CONTACTS:
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
            path.append(f'      <site site="c{c}_{i:03d}_s1"/>')
        tendon_xml.append(f'''    <spatial name="cable_{c}" width="{TENDON_WIDTH}">
{os.linesep.join(path)}
    </spatial>''')

    # actuators
    actuator_xml = []
    for c in range(n_cables):
        actuator_xml.append(f'    <motor name="motor_c{c}" tendon="cable_{c}"/>')

    # assemble
    xml = header + nested + "\n  </worldbody>\n  <tendon>\n" + \
          (os.linesep.join(tendon_xml)) + "\n  </tendon>\n  <actuator>\n" + \
          (os.linesep.join(actuator_xml)) + "\n  </actuator>\n</mujoco>\n"

    os.makedirs(os.path.dirname(out_xml_path) or ".", exist_ok=True)
    with open(out_xml_path, "w") as f: f.write(xml)
    return out_xml_path


out_xml = write_mjcf_from_sites_csv(
    csv_path= ".\Geom_Data_CSV\Spirob_geom_data.csv",
    out_xml_path = "spirob_mujoco_inverted.xml",
    mesh_dir = "meshes",
    digits = 3
)
print("Wrote", out_xml)