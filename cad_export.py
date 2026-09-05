"""
cad_export.py  —  Whole-robot solid CAD export (STEP + solid STL).

Where the per-element ``csv2geom_nlobe.py`` writes one STL *per link* in each
link's own local frame (what MuJoCo needs), this module assembles every element
in its **world position** into a single solid model and writes it as a STEP
file (for CAD / slicers) and a fused solid STL. That is the file you hand to a
slicer to 3-D print the physical SpiRob, or open in a CAD package to modify.

Design notes
------------
* This is a *clean-room* reimplementation, written for this pipeline. The idea
  of a solid STEP/STL export for a spiral robot is inspired by the OpenSpiRobs
  design tool by Zhanchi Wang et al. (SpiRobs, Wang et al. 2024,
  https://github.com/ZhanchiWang/Open-Spiral-Robots, PolyForm-Noncommercial).
  No code from that project is used or copied here; this module builds on this
  repo's own MIT-licensed geometry stack. Unlike that tool, it supports the
  full n-cable (>3) n-lobe cross-section this repo generates.
* It reuses the exact element-construction helpers from ``csv2geom_nlobe.py``,
  so the printed solid matches the simulated meshes vertex-for-vertex — the
  only difference is that elements are kept in world coordinates and combined,
  instead of being recentred to each link's local origin.
* The per-element STL path in ``csv2geom_nlobe.py`` is left untouched, so the
  project's byte-identical STL guarantee and its tests are unaffected.

CLI
---
  python cad_export.py                          # uses params.json + default CSV
  python cad_export.py --fuse                   # boolean-union into one solid
  python cad_export.py --in Geom_Data_CSV/Spirob_geom_data.csv --outdir cad
  python cad_export.py --plain                  # no n-lobe cut (solid of rev.)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence

import cadquery as cq
import pandas as pd

from spirob.geometry import SpiRobGeometry, from_params
from csv2geom_nlobe import (
    UnitMeshInputs,
    build_unit_inputs,
    make_profile_from_points,
    revolve_profile,
    add_nlobe_cut,
)


@dataclass(frozen=True)
class CadExportResult:
    step_path: str
    stl_path: str
    n_elements: int
    fused: bool


# ──────────────────────────────────────────────────────────────────────────────
#  World-coordinate element solids
#
#  These mirror the two construction paths in csv2geom_nlobe.process_csv, but
#  WITHOUT the local-frame recentring (revolve path) / z-origin shift (flat
#  path), so each element sits where it physically belongs along the robot.
# ──────────────────────────────────────────────────────────────────────────────

def _flat_element_world(row, thickness_ratio: float = 0.3):
    """Flat tapered element extruded in ±Y, kept at its world Z position.

    This is the world-frame twin of ``csv2geom_nlobe.build_flat_element``: same
    trapezoidal quad profile and the same mirror-and-fuse construction, but the
    z-origin shift is omitted so consecutive elements stack correctly.
    """
    A1x = float(row["joint_s1_x"]); A1z = float(row["joint_s1_z"])
    A0x = float(row["joint_s2_x"]); A0z = float(row["joint_s2_z"])
    B0x = float(row["c0_s2_x"]);    B0z = float(row["c0_s2_z"])
    B1x = float(row["c0_s1_x"]);    B1z = float(row["c0_s1_z"])

    # Hinge points lie on the central axis (x == 0 up to float noise).
    A0x = 0.0; A1x = 0.0

    pts = [
        (A1x, A1z),   # inner, base end
        (B1x, B1z),   # outer, base end
        (B0x, B0z),   # outer, tip end
        (A0x, A0z),   # inner, tip end
    ]
    half_t = abs(B0x) * thickness_ratio / 2.0
    pts_right = [(-x, z) for x, z in pts]

    left = cq.Workplane("XZ").polyline(pts).close().extrude(half_t * 2.0, both=True)
    right = cq.Workplane("XZ").polyline(pts_right).close().extrude(half_t * 2.0, both=True)
    return left.union(right)


def build_world_solid(unit: UnitMeshInputs, *, n_cables: int, draft_angle_deg: float,
                      nlobe_t: float, notch_factor: float,
                      plain: bool, flat_mode: bool, flat_thickness_ratio: float):
    """Build one element as a CadQuery solid in world coordinates."""
    if flat_mode:
        return _flat_element_world(unit.row, flat_thickness_ratio)

    points = list(unit.profile_xyz)
    if len(points) < 2:
        return None
    profile = make_profile_from_points(points)
    solid = revolve_profile(profile, 360, "y")          # already in world coords
    if not plain:
        solid = add_nlobe_cut(
            solid, n_cables, unit.outer_radius_m, unit.height_z_m,
            draft_angle_deg, nlobe_t=nlobe_t, notch_factor=notch_factor,
        )
    return solid


def build_assembled_solid(units: Sequence[UnitMeshInputs], *, n_cables: int,
                          draft_angle_deg: float, nlobe_t: float, notch_factor: float,
                          plain: bool, flat_mode: bool, flat_thickness_ratio: float,
                          fuse: bool = False):
    """Assemble all elements into one CadQuery object in world coordinates.

    ``fuse=False`` (default) returns a compound of the element solids — fast and
    robust, and printable/CAD-openable as-is because the elements meet at their
    shared interfaces. ``fuse=True`` boolean-unions them into a single manifold
    solid, which is cleaner but far slower and occasionally fragile in OCC.
    """
    solids = []
    for unit in units:
        s = build_world_solid(
            unit, n_cables=n_cables, draft_angle_deg=draft_angle_deg,
            nlobe_t=nlobe_t, notch_factor=notch_factor, plain=plain,
            flat_mode=flat_mode, flat_thickness_ratio=flat_thickness_ratio,
        )
        if s is None:
            continue
        # Unwrap Workplane -> Solid(s)
        solids.extend(s.vals() if isinstance(s, cq.Workplane) else [s])

    if not solids:
        raise ValueError("No element solids were produced for CAD export.")

    if fuse:
        fused = solids[0]
        for s in solids[1:]:
            fused = fused.fuse(s)
        return fused, len(solids)

    return cq.Compound.makeCompound(solids), len(solids)


# ──────────────────────────────────────────────────────────────────────────────
#  Orchestration
# ──────────────────────────────────────────────────────────────────────────────

def process_cad(csv_file: str, params: dict, *, outdir: str = "cad",
                prefix: str = "spirob", fuse: bool = False,
                plain: bool = False, stl_tolerance: float = 1e-4,
                geometry: Optional[SpiRobGeometry] = None) -> CadExportResult:
    """CSV (+ params) → assembled STEP and solid STL of the whole robot."""
    if geometry is None:
        geometry = from_params(params)

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Input CSV not found: {csv_file}")

    df = pd.read_csv(csv_file)
    units = build_unit_inputs(df, geometry)

    n_cables = geometry.inputs.n_cables
    phi_deg = geometry.inputs.phi_deg_full_included
    draft_angle_deg = phi_deg / 2.0
    flat_mode = (not plain) and (n_cables <= 2)

    mode = ("plain revolve" if plain
            else f"{n_cables}-cable flat" if flat_mode
            else f"{n_cables}-lobe")
    print("CAD export settings:")
    print(f"  mode       = {mode}")
    print(f"  elements   = {len(units)}")
    print(f"  combine    = {'boolean union (fused)' if fuse else 'compound'}")

    combined, n_elem = build_assembled_solid(
        units, n_cables=n_cables, draft_angle_deg=draft_angle_deg,
        nlobe_t=float(params.get("nlobe_t", 0.5)),
        notch_factor=float(params.get("notch_factor", 0.25)),
        plain=plain, flat_mode=flat_mode,
        flat_thickness_ratio=float(params.get("flat_thickness_ratio", 0.3)),
        fuse=fuse,
    )

    os.makedirs(outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    step_path = os.path.join(outdir, f"{prefix}_{ts}.step")
    stl_path = os.path.join(outdir, f"{prefix}_{ts}.stl")

    cq.exporters.export(combined, step_path)
    cq.exporters.export(combined, stl_path, tolerance=stl_tolerance)

    print(f"  ✓ STEP  →  {step_path}")
    print(f"  ✓ STL   →  {stl_path}")
    return CadExportResult(step_path=step_path, stl_path=stl_path,
                           n_elements=n_elem, fused=fuse)


def main():
    parser = argparse.ArgumentParser(
        description="Whole-robot solid CAD export (STEP + solid STL) for SpiRob."
    )
    parser.add_argument("--in", dest="input",
                        default="Geom_Data_CSV/Spirob_geom_data.csv",
                        help="Input geometry CSV (default: Geom_Data_CSV/Spirob_geom_data.csv)")
    parser.add_argument("--params", default="params.json",
                        help="Path to params.json (default: params.json)")
    parser.add_argument("--outdir", default="cad",
                        help="Output directory (default: cad)")
    parser.add_argument("--prefix", default="spirob",
                        help="Output filename prefix (default: spirob)")
    parser.add_argument("--fuse", action="store_true",
                        help="Boolean-union elements into a single solid (slow, cleaner)")
    parser.add_argument("--plain", action="store_true",
                        help="Plain revolve — no n-lobe cut")
    args = parser.parse_args()

    with open(args.params, encoding="utf-8") as f:
        params = json.load(f)

    process_cad(args.input, params, outdir=args.outdir, prefix=args.prefix,
                fuse=args.fuse, plain=args.plain)


if __name__ == "__main__":
    main()
