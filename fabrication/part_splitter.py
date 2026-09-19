"""
part_splitter.py  —  Split an oversized robot solid/mesh into printable parts.

When the whole-robot export (``cad_export.py``) is larger than your printer's
build volume, this tool slices it along one axis into several parts, checks each
against a build volume, and writes a JSON fit report.

Formats
-------
  * STEP / STP  — sliced as true solids via CadQuery box intersection.
  * STL         — sliced as meshes via trimesh plane clipping (with caps).

Units
-----
Fabrication CAD exports are in millimetres. Simulation link STL files use metres. Build-volume and span options are
given in millimetres for convenience; ``--file-units`` says what the file's
numbers mean (default ``mm``) so the mm thresholds are converted correctly.

Design note
-----------
Clean-room reimplementation for this pipeline. The idea of a build-volume-aware
splitter with a JSON report is inspired by the OpenSpiRobs fabrication tool
(Zhanchi Wang et al.; https://github.com/ZhanchiWang/Open-Spiral-Robots,
PolyForm-Noncommercial). No code from that project is used here — the STEP path
uses CadQuery and the STL path uses trimesh, both MIT-compatible. Like that
tool, this first version does geometric splitting only: it does not yet add
keyed joints, dovetails, pins, or snap-fit connectors (see "Next step").

CLI
---
  python fabrication/part_splitter.py robot.step --axis z --max-span-mm 180
  python fabrication/part_splitter.py robot.stl  --axis z --cut-positions-mm 80,160
  python fabrication/part_splitter.py robot.step --axis z --max-span-mm 150 \
         --build-volume-mm 256,256,256
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import List, Optional, Tuple

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


# ──────────────────────────────────────────────────────────────────────────────
#  Cut-plane planning (format-independent)
# ──────────────────────────────────────────────────────────────────────────────

def plan_cuts(lo: float, hi: float, *, max_span: Optional[float] = None,
              cut_positions: Optional[List[float]] = None) -> List[float]:
    """Return interior cut positions between ``lo`` and ``hi`` (exclusive).

    If ``cut_positions`` is given it is used verbatim (filtered to the interval).
    Otherwise the span is divided into the fewest equal parts each ≤ ``max_span``.
    """
    if not all(math.isfinite(v) for v in (lo, hi)) or hi <= lo:
        raise ValueError("Invalid input bounds")
    if cut_positions is not None:
        if max_span is not None: raise ValueError("Choose cut positions OR max span")
        if not cut_positions or any(not math.isfinite(v) or not lo < v < hi for v in cut_positions):
            raise ValueError("Cut positions must be finite and strictly inside the bounds")
        if len(set(cut_positions)) != len(cut_positions): raise ValueError("Duplicate cut positions")
        return sorted(cut_positions)
    if max_span is None or not math.isfinite(max_span) or max_span <= 0:
        raise ValueError("max_span must be finite and positive")
    total = hi - lo
    n_parts = max(1, math.ceil(total / max_span - 1e-9))
    if n_parts == 1:
        return []
    step = total / n_parts
    return [lo + step * k for k in range(1, n_parts)]


def _fits(bounds_span: Tuple[float, float, float],
          build_volume: Tuple[float, float, float]) -> bool:
    """True if a part (any axis permutation) fits inside the build volume."""
    return all(s <= b + 1e-9 for s, b in zip(sorted(bounds_span), sorted(build_volume)))


# ──────────────────────────────────────────────────────────────────────────────
#  STEP path (CadQuery solids)
# ──────────────────────────────────────────────────────────────────────────────

def split_step(path: str, axis: str, cut_positions: List[float], out_dir: str,
               prefix: str) -> List[dict]:
    import cadquery as cq

    solid = cq.importers.importStep(path)
    bb = solid.val().BoundingBox()
    lows = (bb.xmin, bb.ymin, bb.zmin)
    highs = (bb.xmax, bb.ymax, bb.zmax)
    ai = _AXIS_INDEX[axis]

    edges = [lows[ai]] + list(cut_positions) + [highs[ai]]
    # Generous cross-section so the intersecting slab fully spans the other axes.
    pad = 10.0 * max(highs[j] - lows[j] for j in range(3)) + 1.0

    reports = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        centre = [(lows[j]+highs[j])/2 for j in range(3)]
        length = [2 * pad, 2 * pad, 2 * pad]
        centre[ai] = 0.5 * (a + b)
        length[ai] = (b - a)
        # Box centred at `centre`, spanning [a, b] on the cut axis.
        slab = (cq.Workplane("XY")
                .transformed(offset=(centre[0], centre[1], centre[2]))
                .box(length[0], length[1], length[2]))
        part = solid.intersect(slab)
        if not part.val().isValid() or not part.val().Solids() or part.val().Volume() <= 0:
            raise ValueError(f"Part {i+1} is empty or invalid")
        part_path = os.path.join(out_dir, f"{prefix}_part{i+1:02d}.step")
        cq.exporters.export(part, part_path)
        pbb = part.val().BoundingBox()
        reports.append({
            "part": os.path.basename(part_path),
            "path": part_path,
            "bounds_min": [pbb.xmin, pbb.ymin, pbb.zmin],
            "bounds_max": [pbb.xmax, pbb.ymax, pbb.zmax],
            "span": [pbb.xlen, pbb.ylen, pbb.zlen],
            "volume": part.val().Volume(), "valid": True, "solid_count": len(part.val().Solids()),
        })
    return reports


# ──────────────────────────────────────────────────────────────────────────────
#  STL path (trimesh meshes)
# ──────────────────────────────────────────────────────────────────────────────

def split_stl(path: str, axis: str, cut_positions: List[float], out_dir: str,
              prefix: str) -> List[dict]:
    import numpy as np
    import trimesh

    mesh = trimesh.load(path, force="mesh")
    if not mesh.is_volume:
        raise ValueError("STL must enclose a consistently oriented watertight volume before splitting")
    ai = _AXIS_INDEX[axis]
    lo = float(mesh.bounds[0][ai])
    hi = float(mesh.bounds[1][ai])
    edges = [lo] + list(cut_positions) + [hi]

    normal = np.zeros(3); normal[ai] = 1.0

    reports = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        # Keep the material between planes a and b: clip below b, then above a.
        part = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal=-normal, plane_origin=normal * b, cap=True)
        if part is not None and len(part.faces):
            part = trimesh.intersections.slice_mesh_plane(
                part, plane_normal=normal, plane_origin=normal * a, cap=True)
        part_path = os.path.join(out_dir, f"{prefix}_part{i+1:02d}.stl")
        if part is None or not len(part.faces):
            reports.append({"part": os.path.basename(part_path), "path": part_path,
                            "empty": True})
            continue
        if not part.is_volume:
            raise ValueError(f"Part {i+1} is not a closed oriented volume")
        part.export(part_path)
        pb = part.bounds
        reports.append({
            "part": os.path.basename(part_path),
            "path": part_path,
            "bounds_min": [float(v) for v in pb[0]],
            "bounds_max": [float(v) for v in pb[1]],
            "span": [float(pb[1][k] - pb[0][k]) for k in range(3)],
            "watertight": bool(part.is_watertight), "volume": float(part.volume),
        })
    return reports


# ──────────────────────────────────────────────────────────────────────────────
#  Orchestration
# ──────────────────────────────────────────────────────────────────────────────

def split_file(path: str, *, axis: str = "z", max_span_mm: Optional[float] = None,
               cut_positions_mm: Optional[List[float]] = None,
               build_volume_mm: Optional[Tuple[float, float, float]] = None,
               file_units: str = "mm", out_dir: Optional[str] = None) -> dict:
    if file_units not in ("m", "mm"): raise ValueError("file_units must be m or mm")
    if build_volume_mm is not None and (len(build_volume_mm)!=3 or
            any(not math.isfinite(v) or v<=0 for v in build_volume_mm)):
        raise ValueError("Build volume must have three finite positive dimensions")
    axis = axis.lower().strip()
    if axis not in _AXIS_INDEX:
        raise ValueError(f"axis must be x, y or z (got {axis!r})")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    stem = os.path.splitext(os.path.basename(path))[0]
    out_dir = out_dir or (os.path.splitext(path)[0] + "_" + ext.lstrip(".") + "_split")
    os.makedirs(out_dir, exist_ok=True)

    # mm thresholds → file units
    to_file = 0.001 if file_units == "m" else 1.0
    max_span = max_span_mm * to_file if max_span_mm is not None else None
    cut_positions = [c * to_file for c in cut_positions_mm] if cut_positions_mm is not None else None
    build_volume = tuple(v * to_file for v in build_volume_mm) if build_volume_mm else None

    # Bounds on the cut axis
    if ext in (".step", ".stp"):
        import cadquery as cq
        bb = cq.importers.importStep(path).val().BoundingBox()
        lo, hi = (bb.xmin, bb.ymin, bb.zmin)[_AXIS_INDEX[axis]], \
                 (bb.xmax, bb.ymax, bb.zmax)[_AXIS_INDEX[axis]]
    elif ext == ".stl":
        import trimesh
        b = trimesh.load(path, force="mesh").bounds
        lo, hi = float(b[0][_AXIS_INDEX[axis]]), float(b[1][_AXIS_INDEX[axis]])
    else:
        raise ValueError(f"Unsupported format {ext!r}; use STEP/STP or STL.")

    cuts = plan_cuts(lo, hi, max_span=max_span, cut_positions=cut_positions)
    print(f"Splitting {os.path.basename(path)} along {axis}: "
          f"span={hi-lo:.4f} {file_units}, {len(cuts)} cut(s) → {len(cuts)+1} part(s)")

    if ext in (".step", ".stp"):
        parts = split_step(path, axis, cuts, out_dir, stem)
    else:
        parts = split_stl(path, axis, cuts, out_dir, stem)

    if build_volume:
        for p in parts:
            if p.get("empty"):
                p["fits_build_volume"] = False
                continue
            p["fits_build_volume"] = _fits(tuple(p["span"]), build_volume)

    report = {
        "input": path,
        "axis": axis,
        "file_units": file_units,
        "n_parts": len(parts),
        "cut_positions": cuts,
        "build_volume": list(build_volume) if build_volume else None,
        "parts": parts,
        "all_fit": all(p.get("fits_build_volume", True) for p in parts) if build_volume else None,
        "fit_scope": "Axis permutations only; excludes supports, clearance and assembly joints",
        "assembly": "Geometric cuts only; joining method must be designed separately",
    }
    report_path = os.path.join(out_dir, "split_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for p in parts:
        span = p.get("span")
        fit = p.get("fits_build_volume")
        fit_str = "" if fit is None else ("  ✓ fits" if fit else "  ✗ too big")
        span_str = ("[%.3f %.3f %.3f]" % tuple(span)) if span else "(empty)"
        print(f"  {p['part']}  span={span_str} {file_units}{fit_str}")
    print(f"  report → {report_path}")
    return report


def _parse_triple(s: str) -> Tuple[float, float, float]:
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers")
    return tuple(vals)  # type: ignore[return-value]


def main():
    p = argparse.ArgumentParser(
        description="Split an oversized SpiRob solid/mesh into printable parts.")
    p.add_argument("input", help="Input STEP/STP or STL file")
    p.add_argument("--axis", default="z", choices=list(_AXIS_INDEX),
                   help="Split axis (default: z)")
    p.add_argument("--max-span-mm", type=float, default=None,
                   help="Max part span on the axis; auto-divides into equal parts")
    p.add_argument("--cut-positions-mm", type=lambda s: [float(x) for x in s.split(",")],
                   default=None, help="Explicit comma-separated cut positions (mm)")
    p.add_argument("--build-volume-mm", type=_parse_triple, default=None,
                   help="Printer build volume 'x,y,z' (mm) for fit checks")
    p.add_argument("--file-units", default="mm", choices=["m", "mm"],
                   help="Numeric units in the input file (default: mm for fabrication; use m for simulation STL)")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: <stem>_<format>_split)")
    args = p.parse_args()

    if not args.max_span_mm and not args.cut_positions_mm:
        p.error("give --max-span-mm or --cut-positions-mm")

    report = split_file(args.input, axis=args.axis, max_span_mm=args.max_span_mm,
               cut_positions_mm=args.cut_positions_mm,
               build_volume_mm=args.build_volume_mm, file_units=args.file_units,
               out_dir=args.out_dir)
    if report["all_fit"] is False:
        raise SystemExit("One or more parts exceed the build volume; see split_report.json")


if __name__ == "__main__":
    main()
