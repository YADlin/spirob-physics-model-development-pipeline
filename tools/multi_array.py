"""
multi_array.py  —  Replicate a single SpiRob MuJoCo model into a circular array.

Takes a single-robot MJCF (as produced by ``csv2xml.py``) and writes a new MJCF
containing ``count`` copies arranged evenly around a circle, sharing one ground
plane, lighting, and mesh assets. Useful for multi-robot / swarm experiments.

What it does
------------
* Clones the robot's root body subtree ``count`` times.
* Renames every body / joint / geom / site (and each robot's tendons and
  actuators) with a ``_r{k}`` suffix so MuJoCo's uniqueness rules hold.
* Keeps mesh ``<asset>`` entries **shared** (all copies reference the same
  meshes) — efficient, and geoms keep their original ``mesh=`` references.
* Places copy ``k`` at angle ``base_rot + k·360/count`` on a circle of the
  given radius, composing the rotation with the robot's own base orientation,
  with an optional tilt about the radial axis.

Because mesh files are referenced relatively, the output XML is written next to
the source XML by default so ``meshes/`` still resolves.

Design note
-----------
Clean-room reimplementation for this pipeline; the multi-robot-array idea is
inspired by the OpenSpiRobs extension (Zhanchi Wang et al.;
https://github.com/ZhanchiWang/Open-Spiral-Robots, PolyForm-Noncommercial).
No code from that project is used here.

CLI
---
  python tools/multi_array.py --in spirob_physics_model.xml --count 6
  python tools/multi_array.py --in spirob_physics_model.xml --count 8 \
         --radius-m 0.12 --base-rot-deg 0 --tilt-deg -30 --out array.xml
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple

Quat = Tuple[float, float, float, float]   # (w, x, y, z), MuJoCo convention


# ── quaternion helpers (w, x, y, z) ───────────────────────────────────────────

def quat_mul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_axis_angle(axis: Tuple[float, float, float], angle_rad: float) -> Quat:
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / n, y / n, z / n
    h = angle_rad / 2.0
    s = math.sin(h)
    return (math.cos(h), x * s, y * s, z * s)


def quat_rotate_vec(q: Quat, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    w, x, y, z = q
    vx, vy, vz = v
    # t = 2 * cross(q.xyz, v)
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _parse_floats(s: str, n: int, default) -> list:
    if not s:
        return list(default)
    vals = [float(v) for v in s.split()]
    return vals if len(vals) == n else list(default)


def _fmt(vals) -> str:
    return " ".join(f"{v:.9g}" for v in vals)


# ── renaming ──────────────────────────────────────────────────────────────────

_NAMED_TAGS = {"body", "joint", "geom", "site"}


def _suffix_names(elem: ET.Element, suffix: str) -> None:
    """Append ``suffix`` to name attributes of bodies/joints/geoms/sites."""
    for e in elem.iter():
        if e.tag in _NAMED_TAGS and "name" in e.attrib:
            e.set("name", e.attrib["name"] + suffix)


# ── main build ─────────────────────────────────────────────────────────────────

def build_array(src_xml: str, count: int, *, radius_m: float,
                base_rot_deg: float = 0.0, tilt_deg: float = 0.0,
                out_path: str = None) -> str:
    tree = ET.parse(src_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("No <worldbody> in source XML.")

    # Identify the robot root body: the worldbody <body> child.
    robot_bodies = [c for c in worldbody if c.tag == "body"]
    if len(robot_bodies) != 1:
        raise ValueError(f"Expected exactly one robot root body, found {len(robot_bodies)}.")
    robot_root = robot_bodies[0]

    base_pos = _parse_floats(robot_root.attrib.get("pos", ""), 3, (0, 0, 0))
    base_quat: Quat = tuple(_parse_floats(robot_root.attrib.get("quat", ""), 4, (1, 0, 0, 0)))  # type: ignore

    src_tendon = root.find("tendon")
    src_actuator = root.find("actuator")

    # Remove the template robot + its tendons/actuators; we re-add per copy.
    worldbody.remove(robot_root)
    if src_tendon is not None:
        root.remove(src_tendon)
    if src_actuator is not None:
        root.remove(src_actuator)

    new_tendon = ET.SubElement(root, "tendon") if src_tendon is not None else None
    new_actuator = ET.SubElement(root, "actuator") if src_actuator is not None else None

    z0 = base_pos[2]
    for k in range(count):
        suffix = f"_r{k}"
        theta = math.radians(base_rot_deg) + 2.0 * math.pi * k / count

        # Placement: on a circle in the XY plane, then rotate the robot about
        # world Z by theta and tilt about the radial (outward) axis.
        cx, cy = radius_m * math.cos(theta), radius_m * math.sin(theta)
        q_spin = quat_axis_angle((0, 0, 1), theta)
        radial = (math.cos(theta), math.sin(theta), 0.0)
        q_tilt = quat_axis_angle(radial, math.radians(tilt_deg))
        world_quat = quat_mul(q_tilt, quat_mul(q_spin, base_quat))

        body = copy.deepcopy(robot_root)
        _suffix_names(body, suffix)
        body.set("pos", _fmt((cx, cy, z0)))
        body.set("quat", _fmt(world_quat))
        worldbody.append(body)

        # Clone tendons, renaming tendon names + their site references.
        if src_tendon is not None:
            for sp in src_tendon:
                sp2 = copy.deepcopy(sp)
                if "name" in sp2.attrib:
                    sp2.set("name", sp2.attrib["name"] + suffix)
                for site_ref in sp2.iter("site"):
                    if "site" in site_ref.attrib:
                        site_ref.set("site", site_ref.attrib["site"] + suffix)
                new_tendon.append(sp2)

        # Clone actuators, renaming actuator names + their tendon references.
        if src_actuator is not None:
            for act in src_actuator:
                act2 = copy.deepcopy(act)
                if "name" in act2.attrib:
                    act2.set("name", act2.attrib["name"] + suffix)
                if "tendon" in act2.attrib:
                    act2.set("tendon", act2.attrib["tendon"] + suffix)
                new_actuator.append(act2)

    out_path = out_path or os.path.join(os.path.dirname(os.path.abspath(src_xml)),
                                        "spirob_array.xml")
    try:
        ET.indent(tree, space="  ")   # py3.9+
    except Exception:
        pass
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"  ✓ array XML ({count} robots, radius={radius_m} m) → {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Build a circular array of SpiRob robots.")
    p.add_argument("--in", dest="input", default="spirob_physics_model.xml",
                   help="Source single-robot MJCF (default: spirob_physics_model.xml)")
    p.add_argument("--count", type=int, default=6, help="Number of robots (default: 6)")
    p.add_argument("--radius-m", type=float, default=0.12,
                   help="Array circle radius in metres (default: 0.12)")
    p.add_argument("--base-rot-deg", type=float, default=0.0,
                   help="Rotation offset of the whole array about Z (default: 0)")
    p.add_argument("--tilt-deg", type=float, default=0.0,
                   help="Tilt of each robot about its radial axis (default: 0)")
    p.add_argument("--out", default=None,
                   help="Output XML path (default: spirob_array.xml next to source)")
    args = p.parse_args()

    if args.count < 1:
        p.error("--count must be >= 1")
    build_array(args.input, args.count, radius_m=args.radius_m,
                base_rot_deg=args.base_rot_deg, tilt_deg=args.tilt_deg,
                out_path=args.out)


if __name__ == "__main__":
    main()
