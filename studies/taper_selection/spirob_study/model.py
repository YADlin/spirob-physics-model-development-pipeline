"""Turn a pipeline MJCF into the planar study model.

The generated model is left untouched on disk; a separate XML is written
next to it so that its relative mesh paths still resolve. Changes made for
the study, and why:

* gravity off: the physical rig bends in a horizontal plane, so gravity is
  perpendicular to the bending plane and does no work on the hinge joints;
* base welded and placed at the origin, robot pointing along +Y, bending in
  the XY plane (the overhead-camera view of the rig);
* joint limits +/- Delta_theta: adjacent units touch at this angle
  (paper, Note S3). The pipeline itself emits +/- pi and relies on contacts,
  but MuJoCo filters parent-child contacts, so adjacent units would
  interpenetrate (see the validation section of the study);
* joint stiffness from one material law for every design (below);
* cable motors force-controlled, 0..F_max, as in the paper's experiments.

Stiffness law
-------------
Each joint is the tapered elastic core between two units. For geometrically
similar cores, bending stiffness K = E I / l with I ~ W^4 and l ~ W, so
K_i = kappa * W_i^3, where W_i is the local continuous reference width at the
joint (the same width this repository uses to taper the core). kappa is a
single material constant shared by all designs. It is anchored so that the
current robot's j_002 equals the value used in spirob-mjlab (0.18303 N m/rad);
the anchor only sets the force scale, all comparisons are relative.
"""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .designs import Design

K2_ANCHOR = 0.18302833187063192      # N m/rad, spirob-mjlab j_002
W2_ANCHOR = None                       # set lazily from the current robot
DAMPING_TIME = 0.05                    # s, d_i = DAMPING_TIME * k_i (as in the pipeline's 0.0092/0.183)
BASE_QUAT = (math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0)   # local +Z -> world +Y, hinge axis -> world -Z


@dataclass
class StudyModel:
    design: Design
    xml_path: Path
    joint_names: list[str]
    joint_width_m: np.ndarray        # reference width at each flexible joint, base -> tip
    joint_s_m: np.ndarray            # arc length of each joint from the base
    stiffness: np.ndarray
    length_m: float                  # assembled (chord) length, base face -> tip face
    mass_kg: float
    link_names: list[str] = field(default_factory=list)

    def load(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        m = mujoco.MjModel.from_xml_path(str(self.xml_path))
        return m, mujoco.MjData(m)


def _width_law(build_dir: Path):
    law = json.loads((build_dir / "elastic_core_dimensions.json").read_text())["width_law"]
    wb, zb, za = law["base_width_m"], law["z_base_m"], law["virtual_apex_z_m"]
    return lambda z_from_base: wb * (za - (zb + z_from_base)) / (za - zb)


def anchor_kappa() -> float:
    """kappa such that the current robot's j_002 equals K2_ANCHOR."""
    from .designs import CURRENT
    law = _width_law(CURRENT.build_dir)
    root = ET.parse(CURRENT.build_dir / "spirob_physics_model.xml").getroot()
    l2 = root.find(".//body[@name='link_002']")
    z2 = float(l2.get("pos").split()[2])
    return K2_ANCHOR / law(z2) ** 3


def make(design: Design, *, law: str = "cubic", limits: str = "dtheta",
         f_max: float = 50.0, timestep: float = 2e-4, objects: list[dict] | None = None,
         tag: str = "study", k_factor=None) -> StudyModel:
    """Write ``<build>/<tag>.xml`` and return its description.

    law:    "cubic" (K ~ W^3, default), "quadratic" (K ~ W^2, the
            Open-Spiral-Robots generator) or "mjlab" (k2 * 0.9151^(i-2), the
            current spirob-mjlab model). All are anchored at the same j_002.
    limits: "dtheta" (+/- Delta_theta) or "pipeline" (keep +/- pi).
    objects: list of {"name", "pos": (x, y), "radius", "free": bool}.
    k_factor: optional per-joint multiplier (array, or callable of the joint
            widths), applied after the law.
    """
    bdir = design.build_dir
    tree = ET.parse(bdir / "spirob_physics_model.xml")
    root = tree.getroot()
    kappa = anchor_kappa()
    width = _width_law(bdir)

    opt = root.find("option")
    opt.set("gravity", "0 0 0")
    opt.set("timestep", f"{timestep}")
    opt.set("integrator", "implicitfast")
    opt.set("cone", "elliptic")

    vis = root.find("visual")
    if vis is not None:
        vis.find("map").set("znear", "0.001")
        ET.SubElement(vis, "headlight", ambient="0.55 0.55 0.55", diffuse="0.45 0.45 0.45", specular="0 0 0")
    asset = root.find("asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="flat", rgb1="1 1 1", rgb2="1 1 1",
                  width="16", height="16")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", width="64",
                  height="64", rgb1="0.97 0.97 0.97", rgb2="0.91 0.91 0.91")
    ET.SubElement(asset, "material", name="floor", texture="grid", texrepeat="60 60", specular="0",
                  shininess="0", reflectance="0", rgba="1 1 1 1")

    wb = root.find("worldbody")
    for gplane in [g for g in wb.findall("geom") if g.get("type") == "plane"]:
        wb.remove(gplane)       # the pipeline's ground would cut the planar robot
    for lt in wb.findall("light"):
        wb.remove(lt)           # lighting is set for the overhead renders below
    ET.SubElement(wb, "light", pos="0 0.1 1.0", dir="0 0 -1", directional="true", diffuse="0.35 0.35 0.35",
                  specular="0 0 0", castshadow="false")
    dflt = root.find("default/geom")
    if dflt is not None:
        dflt.set("rgba", "0.11 0.36 0.67 1")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="0.6 0.6 0.01", pos="0 0 -0.08",
                  material="floor", rgba="1 1 1 1", contype="0", conaffinity="0")
    ET.SubElement(wb, "camera", name="top", pos="0 0.08 0.75", xyaxes="1 0 0 0 1 0", fovy="45")

    base = wb.find("body[@name='link_001']")
    base.set("pos", "0 0 0")
    base.set("quat", " ".join(f"{q:.12g}" for q in BASE_QUAT))
    j1 = base.find("joint")
    if j1 is not None:
        base.remove(j1)

    # walk the chain base -> tip
    names, widths, svals, ks = [], [], [], []
    body, s = base, 0.0
    links = ["link_001"]
    dth = math.radians(design.dtheta_deg)
    i = 1
    while True:
        child = body.find("body")
        if child is None:
            break
        i += 1
        links.append(child.get("name"))
        s += float(child.get("pos").split()[2])
        jnt = child.find("joint")
        w = width(s)
        if law == "cubic":
            k = kappa * w ** 3
        elif law == "quadratic":
            k = None  # filled below, needs j_002
        elif law == "mjlab":
            k = None
        else:
            raise ValueError(law)
        names.append(jnt.get("name")); widths.append(w); svals.append(s); ks.append(k)
        body = child
    # last link length from its collision hull
    tip_body = body
    tip_len = _hull_zmax(root, tip_body)
    length = s + tip_len

    widths = np.array(widths); svals = np.array(svals)
    k2 = kappa * widths[0] ** 3
    if law == "cubic":
        ks = kappa * widths ** 3
    elif law == "quadratic":
        ks = k2 * (widths / widths[0]) ** 2
    elif law == "mjlab":
        ks = k2 * 0.9151416593531595 ** np.arange(len(widths))
    ks = np.asarray(ks, float)
    if k_factor is not None:                 # e.g. layer-quantised flexures
        ks = ks * np.asarray(k_factor(widths) if callable(k_factor) else k_factor, float)

    for jn, k in zip(names, ks):
        jnt = root.find(f".//joint[@name='{jn}']")
        jnt.set("stiffness", f"{k:.10g}")
        jnt.set("damping", f"{DAMPING_TIME * k:.10g}")
        if limits == "dtheta":
            jnt.set("range", f"{-dth:.10g} {dth:.10g}")
            # stiff unit-to-unit stop: units are rigid where they touch
            jnt.set("solreflimit", f"{max(4 * timestep, 1e-3):.6g} 1")
            jnt.set("solimplimit", "0.95 0.99 0.0005")
        jnt.set("limited", "true")
        jnt.set("armature", f"{1e-3 * DAMPING_TIME * k:.6g}")

    ET.SubElement(tip_body, "site", name="tip", pos=f"0 0 {tip_len:.9g}", size="0.0015",
                  rgba="0.9 0.1 0.1 1")

    for obj in objects or []:
        x, y = obj["pos"]
        ob = ET.SubElement(wb, "body", name=obj["name"], pos=f"{x:.9g} {y:.9g} 0")
        if obj.get("free"):
            ET.SubElement(ob, "joint", name=f"{obj['name']}_x", type="slide", axis="1 0 0", damping="0.05", limited="false")
            ET.SubElement(ob, "joint", name=f"{obj['name']}_y", type="slide", axis="0 1 0", damping="0.05", limited="false")
            ET.SubElement(ob, "joint", name=f"{obj['name']}_r", type="hinge", axis="0 0 1", damping="1e-4", limited="false")
        if obj.get("weld"):          # a holder: a stiff weld to the world, switched off to release
            eq = root.find("equality")
            if eq is None:
                eq = ET.SubElement(root, "equality")
            ET.SubElement(eq, "weld", name=f"{obj['name']}_holder", body1=obj["name"],
                          solref=f"{max(2 * timestep, 4e-4):.6g} 1")
        ET.SubElement(ob, "geom", name=f"{obj['name']}_geom", type="cylinder",
                      size=f"{obj['radius']:.9g} 0.06", rgba="0.95 0.75 0.25 0.9",
                      friction="0.6 0.01 0.001", density="300", condim="3")

    act = root.find("actuator")
    for mtr in list(act):
        act.remove(mtr)
    for c in (0, 1):
        ET.SubElement(act, "motor", name=f"motor_c{c}", tendon=f"cable_{c}", gear="1",
                      ctrllimited="true", ctrlrange=f"{-f_max:.6g} 0")
    sens = root.find("sensor")
    if sens is None:
        sens = ET.SubElement(root, "sensor")
    for c in (0, 1):
        ET.SubElement(sens, "tendonpos", name=f"len_c{c}", tendon=f"cable_{c}")

    out = bdir / f"{tag}.xml"
    tree.write(out)
    m = mujoco.MjModel.from_xml_path(str(out))
    mass = float(sum(m.body_mass[m.body(n).id] for n in links))
    return StudyModel(design, out, names, widths, svals, ks, length, mass, links)


def _hull_zmax(root, body) -> float:
    geom = next(g for g in body.findall("geom") if g.get("name", "").startswith("collision_"))
    mesh = root.find(f".//mesh[@name='{geom.get('mesh')}']")
    v = np.array(mesh.get("vertex").split(), float).reshape(-1, 3)
    sc = np.array(mesh.get("scale", "1 1 1").split(), float)
    return float((v * sc)[:, 2].max())
