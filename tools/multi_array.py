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

_REF_TYPES = {"body":"body", "body1":"body", "body2":"body", "target":"body",
              "joint":"joint", "joint1":"joint", "joint2":"joint",
              "geom":"geom", "geom1":"geom", "geom2":"geom", "sidesite":"site",
              "site":"site", "site1":"site", "site2":"site", "refsite":"site",
              "cranksite":"site", "slidersite":"site", "tendon":"tendon",
              "tendon1":"tendon", "tendon2":"tendon", "actuator":"actuator"}
_BODY_TYPES = {"body","joint","freejoint","geom","site","camera","light"}


def build_array(src_xml: str, count: int, *, radius_m: float,
                base_rot_deg: float = 0.0, tilt_deg: float = 0.0,
                out_path: str = None) -> str:
    from pathlib import Path
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be an integer >= 1")
    if not all(math.isfinite(v) for v in (radius_m, base_rot_deg, tilt_deg)) or radius_m < 0:
        raise ValueError("radius must be nonnegative and angles/radius must be finite")
    source=Path(src_xml).resolve()
    output=Path(out_path).resolve() if out_path else source.with_name("spirob_array.xml")
    if source == output: raise ValueError("Array output must differ from source XML")
    tree=ET.parse(source); root=tree.getroot()
    # Keyframe state vectors require dimension-aware expansion. Includes must be
    # resolved before cloning; rejecting prevents a plausible but broken scene.
    for tag in ("include","keyframe","deformable","extension","attach","replicate"):
        if root.find(".//"+tag) is not None:
            raise ValueError(f"Unsupported <{tag}>; provide an expanded single-robot MJCF without keyframes/plugins")
    world=root.find("worldbody")
    bodies=[] if world is None else world.findall("body")
    if len(bodies)!=1: raise ValueError(f"Expected exactly one robot root body, found {len(bodies)}")
    template=bodies[0]
    if any(k in template.attrib for k in ("euler","axisangle","xyaxes","zaxis")):
        raise ValueError("Convert robot root orientation to quat before array generation")
    pos=_parse_floats(template.get("pos",""),3,(0,0,0))
    quat=tuple(_parse_floats(template.get("quat",""),4,(1,0,0,0)))
    norm=math.sqrt(sum(v*v for v in quat))
    if not norm or not all(math.isfinite(v) for v in (*pos,*quat)):
        raise ValueError("Invalid robot pose")
    quat=tuple(v/norm for v in quat)
    names={}
    for e in template.iter():
        if e.tag in _BODY_TYPES and e.get("name"):
            kind="joint" if e.tag=="freejoint" else e.tag
            names.setdefault(kind,set()).add(e.get("name"))
    for section in ("tendon","actuator","sensor"):
        for e in root.findall(section+"/*"):
            if e.get("name"): names.setdefault(section,set()).add(e.get("name"))

    def references(e):
        for child in e.iter():
            for attr,value in child.attrib.items():
                kind=_REF_TYPES.get(attr)
                if attr in ("objname","refname"):
                    kind=child.get("objtype" if attr=="objname" else "reftype")
                if kind in names and value in names[kind]: yield child,attr,value

    def clone(e,suffix,section=None):
        c=copy.deepcopy(e)
        for child,attr,value in references(c): child.set(attr,value+suffix)
        for child in c.iter():
            if child.get("name"):
                child.set("name",child.get("name")+suffix)
        return c

    # Resolve assets before relocation. Keep one shared copy of each asset.
    compiler=root.find("compiler")
    settings={} if compiler is None else dict(compiler.attrib)
    for asset in root.findall("asset/*"):
        file=asset.get("file")
        if not file: continue
        folder=settings.get("meshdir" if asset.tag=="mesh" else "texturedir",settings.get("assetdir",""))
        asset_path=Path(file)
        if settings.get("strippath","false")=="true": asset_path=Path(asset_path.name)
        if not asset_path.is_absolute(): asset_path=source.parent/folder/asset_path
        asset.set("file",str(asset_path.resolve()))
    if compiler is not None:
        for key in ("meshdir","texturedir","assetdir","strippath"): compiler.attrib.pop(key,None)
    sections={}
    for name in ("tendon","actuator","sensor","contact","equality"):
        section=root.find(name)
        if section is not None:
            sections[name]=[]
            for e in list(section):
                if name in ("tendon","actuator") or list(references(e)):
                    sections[name].append(e); section.remove(e)
    world.remove(template)
    for k in range(count):
        suffix=f"_r{k}"; theta=math.radians(base_rot_deg)+2*math.pi*k/count
        q_spin=quat_axis_angle((0,0,1),theta)
        q_tilt=quat_axis_angle((math.cos(theta),math.sin(theta),0),math.radians(tilt_deg))
        body=clone(template,suffix)
        body.set("pos",_fmt((pos[0]+radius_m*math.cos(theta),pos[1]+radius_m*math.sin(theta),pos[2])))
        body.set("quat",_fmt(quat_mul(q_tilt,quat_mul(q_spin,quat))))
        world.append(body)
        for name,elements in sections.items():
            for e in elements: root.find(name).append(clone(e,suffix,name))
    output.parent.mkdir(parents=True,exist_ok=True)
    ET.indent(tree,space="  ")
    # Compile first: catches dangling references and malformed assets before write.
    import mujoco
    mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
    tree.write(output,encoding="utf-8",xml_declaration=True)
    print(f"Array compiled: {count} robots -> {output}")
    return str(output)


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
