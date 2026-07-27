"""
csv2geom_nlobe.py  —  CSV → STL meshes with n-lobe or flat cross-section,
                      driven by n_cables from params.json.

Cross-section rules
-------------------
n_cables = 1 or 2  →  Flat tapered extrusion (lofted rectangular slab).
                       Each element has a bottom and top rectangle whose
                       widths follow the outer radius taper, so consecutive
                       elements mate seamlessly.
                       Joints in the XML should be hinge (build.py handles
                       this automatically when n_cables <= 2).

n_cables >= 3      →  Revolved cylinder intersected with a regular n-gon
                       (n = n_cables sides), with a circular notch cut at
                       each edge midpoint for mass/inertia reduction.
                       Circumradius of the polygon = outer_radius * fill_ratio.
                       Notch radius = (side_length / 2) * notch_factor.

params.json fields used
-----------------------
  n_cables              : drives the cross-section type
  phi_deg               : taper angle — used for loft draft on n-lobe cutter
  notch_factor          : notch radius fraction     (n >= 3)
  flat_thickness_ratio  : thickness / width ratio for flat extrusion (n <= 2)
                          default 0.3
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cadquery as cq
import pandas as pd

from spirob.geometry import SpiRobGeometry, from_params


# ══════════════════════════════════════════════════════════════════════════════
#  Canonical geometry adapter
#
#  This module does not derive spiral geometry and never did — it reads the CSV
#  produced by spirob_csv_generator.py. What it *did* duplicate was knowledge
#  ABOUT that geometry: which column means which quad slot, that the CAD draft
#  angle is phi/2 because phi is the full included taper angle, how many units
#  there are, and in what order. All of that now comes from spirob/geometry.py.
#
#  NUMERIC SOURCE — read this before "improving" it.
#  ------------------------------------------------
#  The 2-D profile fed to CadQuery is still taken from the pandas-parsed CSV
#  row, not from SpiRobGeometry.inverted_quads(), and that is deliberate.
#
#    * The CSV text is lossless: stdlib csv + float() reproduces the canonical
#      doubles bit-for-bit.
#    * pandas.read_csv does NOT: it loses up to ~1 ULP (measured 9.7e-17).
#    * Substituting the exact canonical doubles changes the STL files: the
#      byte stream and the order in which triangles are written both change.
#
#  What that difference is, measured rather than assumed
#  -----------------------------------------------------
#  A same-index comparison of the two files shows float words differing by up
#  to ~5e-3. That number is an artefact of comparing reordered files position
#  by position; it is NOT surface displacement. Equally, an equal triangle
#  count on its own would not have established equal topology.
#
#  An order-independent spatial comparison was run over all 21 meshes:
#
#    * exact float32 triangle multisets are identical for every mesh
#      (each triangle canonicalised by sorting its three vertices)
#    * vertex Hausdorff distance is exactly 0.0 m in both directions
#    * vertex and triangle counts are equal
#
#  So the two meshings describe the same surface, vertex for vertex, and the
#  only difference is the order triangles appear in the file. No coordinate
#  moves.
#
#  The pandas parse is nevertheless retained here because byte-identical
#  historical STL output is a project requirement, and file ordering is part
#  of the bytes. The canonical model is used for structure, naming,
#  conventions and validation instead. Switching the numeric source would be
#  geometrically a no-op but would break byte comparisons against every STL
#  the project has shipped. See tests/test_stl_generator_migration.py.
# ══════════════════════════════════════════════════════════════════════════════

#: CSV column prefixes in canonical inverted-quad slot order.
#: quad[0] centreline base end · quad[1] centreline tip end
#: quad[2] inner edge tip end  · quad[3] inner edge base end
_SLOT_COLUMNS = ("joint_s1", "joint_s2", "c0_s2", "c0_s1")

#: Largest permitted disagreement, in degrees, between an explicitly passed
#: phi_deg and the canonical value before it is treated as a conflict. Sized to
#: absorb float round-tripping through degrees/radians while still catching a
#: genuinely wrong value such as the historical 5.7-vs-6.3 mismatch (F-11).
_PHI_AGREEMENT_TOL_DEG = 1e-9

#: Largest permitted disagreement between a CSV row and the canonical model, in
#: metres. Sized to absorb pandas' float-parse error (~1e-16) with headroom,
#: while still catching a genuinely mismatched CSV (which differs by millimetres).
CSV_CANONICAL_TOLERANCE_M = 1e-9


@dataclass(frozen=True)
class UnitMeshInputs:
    """Everything the CAD stage needs for one unit, named rather than indexed.

    ``profile_xyz`` is the pandas-parsed CSV profile; see the NUMERIC SOURCE
    note above for why it is not taken from the canonical model.
    """
    element_id: int
    link_name: str
    is_partial: bool
    profile_xyz: Tuple[Tuple[float, float, float], ...]
    outer_radius_m: float
    height_z_m: float
    origin_m: Tuple[float, float, float]
    row: object                      # the raw CSV row, for the flat-mode path


def _row_slot(row, prefix):
    return (float(row[f"{prefix}_x"]),
            float(row[f"{prefix}_y"]),
            float(row[f"{prefix}_z"]))


def build_unit_inputs(df, geometry: SpiRobGeometry) -> Sequence[UnitMeshInputs]:
    """Pair each CSV row with its canonical unit, validating as we go.

    Raises ValueError if the CSV and the canonical model disagree — which is
    what happens when a stale CSV is passed alongside a params.json carrying a
    different terminal_unit_policy. Previously that mismatch was silent.
    """
    if len(df) != geometry.n_units:
        raise ValueError(
            f"CSV has {len(df)} elements but params.json "
            f"(terminal_unit_policy="
            f"{geometry.inputs.terminal_unit_policy.value}) describes "
            f"{geometry.n_units} units. Regenerate the CSV with "
            f"spirob_csv_generator.py before generating meshes.")

    units = []
    for i, (unit, (_, row)) in enumerate(zip(geometry.units_base_to_tip,
                                             df.iterrows())):
        element_id = int(row["elem"])
        expected_id = i + 1
        if element_id != expected_id:
            raise ValueError(
                f"CSV row {i} has elem={element_id}, expected {expected_id}; "
                f"element numbering must be contiguous and base-to-tip.")

        profile = tuple(_row_slot(row, p) for p in _SLOT_COLUMNS)
        quad = geometry.inverted_quads()[i]
        for slot, (cx, _cy, cz) in enumerate(profile):
            if (abs(cx - float(quad[slot][0])) > CSV_CANONICAL_TOLERANCE_M or
                    abs(cz - float(quad[slot][1])) > CSV_CANONICAL_TOLERANCE_M):
                raise ValueError(
                    f"{unit.link_name}: CSV slot '{_SLOT_COLUMNS[slot]}' "
                    f"({cx:.9g}, {cz:.9g}) disagrees with the canonical model "
                    f"({float(quad[slot][0]):.9g}, {float(quad[slot][1]):.9g}) "
                    f"by more than {CSV_CANONICAL_TOLERANCE_M} m. The CSV was "
                    f"generated from different parameters.")

        units.append(UnitMeshInputs(
            element_id=element_id,
            link_name=unit.link_name,
            is_partial=unit.is_partial,
            profile_xyz=profile,
            outer_radius_m=abs(profile[3][0]),      # inner edge, base end
            height_z_m=abs(profile[1][2] - profile[0][2]),
            origin_m=(profile[0][0], profile[0][1], profile[0][2]),
            row=row,
        ))
    return units


# ══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def extract_points(row):
    """Extract joint + cable-0 site points from a CSV row as 3-D tuples.

    Column order is the canonical inverted-quad slot order (see
    ``_SLOT_COLUMNS``), so the returned list is quad[0..3]. Retained for
    backward compatibility; the pipeline now goes through
    :func:`build_unit_inputs`.
    """
    points = []
    for p in _SLOT_COLUMNS:
        x, y, z = row[f"{p}_x"], row[f"{p}_y"], row[f"{p}_z"]
        if pd.notna(x) and pd.notna(y) and pd.notna(z):
            points.append((float(x), float(y), float(z)))
    return points


def make_profile_from_points(points):
    """Closed CadQuery workplane profile from the (x, z) projection."""
    return cq.Workplane("XZ").polyline([(x, z) for x, y, z in points]).close()


def revolve_profile(profile, angle=360, axis="y"):
    """Revolve a 2-D profile into a solid.
    CadQuery revolves around Y which produces rotation about Z in practice.
    """
    axis_map = {
        "x": ((0,0,0),(1,0,0)),
        "y": ((0,0,0),(0,1,0)),
        "z": ((0,0,0),(0,0,1)),
    }
    start, end = axis_map.get(str(axis).lower(), axis_map["y"])
    return profile.revolve(angle, axisStart=start, axisEnd=end)


# ══════════════════════════════════════════════════════════════════════════════
#  Flat tapered extrusion  (n_cables = 1 or 2)
# ══════════════════════════════════════════════════════════════════════════════

def build_flat_element(row, thickness_ratio=0.3):
    """
    Build the flat element by extruding the actual trapezoidal quad profile
    in ±Y (perpendicular to the XZ plane the profile lives in).

    Profile: the 4 quad points in XZ (joint_s1, joint_s2, c0_s2, c0_s1)
    form a trapezoid — the same profile that is revolved for n>=3 cables.
    Here it is extruded symmetrically in ±Y using CadQuery's both=True.

    The profile is mirrored about x=0 to give a symmetric cross-section:
    the original left half (x<=0) plus its mirror (x>=0) form a diamond/
    arrowhead shape when viewed from above, matching the physical element.

    Thickness = |c0_s1_x| * thickness_ratio, scales with element size.
    All coordinates shifted so joint_s1 is at z=0.
    """
    # Four corners of the quad in XZ
    A1x = float(row["joint_s1_x"]);  A1z = float(row["joint_s1_z"])
    A0x = float(row["joint_s2_x"]);  A0z = float(row["joint_s2_z"])
    B0x = float(row["c0_s2_x"]);     B0z = float(row["c0_s2_z"])
    B1x = float(row["c0_s1_x"]);     B1z = float(row["c0_s1_z"])

    # Snap hinge points to exactly x=0 (they are ~1e-17 due to float math)
    A0x = 0.0;  A1x = 0.0

    # Shift so joint_s1 is at z=0
    oz = A1z
    pts = [
        (A1x, 0.0),           # inner bottom
        (B1x, B1z - oz),      # outer bottom
        (B0x, B0z - oz),      # outer top
        (A0x, A0z - oz),      # inner top
    ]

    # Half-thickness
    half_t = abs(B0x) * thickness_ratio / 2.0

    # Build left half (x<=0) and right half (x>=0) as two separate extrusions,
    # then fuse them. This avoids the degenerate zero-length edge at x=0
    # that OCC produces when a mirrored 8-point polygon is closed.
    pts_right = [(-x, z) for x, z in pts]   # mirror: flip x sign

    left = (
        cq.Workplane("XZ")
        .polyline(pts).close()
        .extrude(half_t * 2.0, both=True)
    )
    right = (
        cq.Workplane("XZ")
        .polyline(pts_right).close()
        .extrude(half_t * 2.0, both=True)
    )

    solid = left.union(right)
    return solid


# ══════════════════════════════════════════════════════════════════════════════
#  N-lobe cutter  (n_cables >= 3)
# ══════════════════════════════════════════════════════════════════════════════

def _regular_polygon_verts(n, circumradius, angle_offset=0.0):
    """
    Return vertices of a regular n-gon centred at origin.

    angle_offset rotates all vertices — use to control orientation:
      0.0       : first vertex at (circumradius, 0)  — right
      π/n       : flat edge on top (standard for even n)
      π/2       : first vertex pointing up
    """
    return [
        (circumradius * math.cos(2*math.pi*k/n + angle_offset),
         circumradius * math.sin(2*math.pi*k/n + angle_offset))
        for k in range(n)
    ]


def add_nlobe_cut(solid, n, outer_radius, height_z, draft_angle_deg,
                  nlobe_t=0.5, notch_factor=0.25):
    """
    Intersect *solid* with a drafted regular n-gon prism and cut circular
    notches at each edge midpoint.

    Parameterisation
    ----------------
    nlobe_t = 0  →  circumscribed: polygon vertices ON the circle wall.
                    Flat facets cut well inside, circular arcs not visible.
    nlobe_t = 1  →  inscribed: polygon edges tangent to the circle.
                    Vertices outside, circular arcs fully visible between flats.
    nlobe_t = 0.5 → midpoint: both arcs and flat facets clearly present (default).

    Circumradius interpolation:
        circ_R = outer_radius * (1 + t * (1/cos(π/n) - 1))

    Orientation: one vertex always at angle π (pointing left), aligning each
    lobe with a cable plane (cable-0 is on the negative-x side).

    Notch radius = (side_length / 2) * notch_factor
    """
    if height_z <= 1e-9 or outer_radius <= 1e-9:
        return solid

    # Interpolate circumradius between circumscribed (t=0) and inscribed (t=1)
    circ_R = outer_radius * (1.0 + nlobe_t * (1.0 / math.cos(math.pi / n) - 1.0))
    side   = 2.0 * circ_R * math.sin(math.pi / n)

    # One vertex always points left — aligns lobes with cable planes
    angle_offset = math.pi

    verts_bot = _regular_polygon_verts(n, circ_R, angle_offset)

    # Cutter height — just enough to fully span the solid
    cutter_h = height_z * 2.0

    # Draft: scale top polygon
    draft_rad   = math.radians(draft_angle_deg)
    taper_scale = 1.0 - math.tan(draft_rad) * (cutter_h / side)
    taper_scale = max(taper_scale, 1e-3)
    verts_top   = [(x * taper_scale, y * taper_scale) for x, y in verts_bot]

    # Notch radius and centres (edge midpoints)
    notch_r   = (side / 2.0) * notch_factor
    def _mid(a, b): return ((a[0]+b[0])/2, (a[1]+b[1])/2)
    midpoints = [_mid(verts_bot[k], verts_bot[(k+1) % n]) for k in range(n)]

    # Lofted n-gon prism
    prism = (
        cq.Workplane("XY")
        .polyline(verts_bot).close()
        .workplane(offset=cutter_h)
        .polyline(verts_top).close()
        .loft()
    )

    # Notch cutters
    notch_cutters = (
        cq.Workplane("XY")
        .pushPoints(midpoints)
        .circle(notch_r)
        .extrude(cutter_h * 1.1)
    )
    prism = prism.cut(notch_cutters)

    # Align cutter centre to solid bounding-box centre
    bbox  = solid.val().BoundingBox()
    cx    = 0.5 * (bbox.xmin + bbox.xmax)
    cy    = 0.5 * (bbox.ymin + bbox.ymax)
    cz    = 0.5 * (bbox.zmin + bbox.zmax)
    prism = prism.translate((cx, cy, cz - cutter_h / 2.0))

    return solid.intersect(prism)


# ══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def process_csv(csv_file, outdir="meshes", revolve_axis="y", angle=360,
                n_cables=None, phi_deg=None,
                nlobe_t=0.5, notch_factor=0.25,
                flat_thickness_ratio=0.3, plain=False,
                geometry: Optional[SpiRobGeometry] = None,
                params: Optional[dict] = None):
    """
    CSV → STL meshes.

    plain=True       →  full solid of revolution, no cut  (replaces csv2geom.py)
    n_cables <= 2    →  flat extrusion of actual quad profile  (hinge joints)
    n_cables >= 3    →  revolved cylinder with n-lobe cut  (replaces csv2geom_trilob.py)

    ``geometry`` is the canonical model. Supply it, or ``params`` from which it
    is built. When present it is AUTHORITATIVE: the unit count, the base-to-tip
    ordering, link naming, ``n_cables`` and the draft-angle convention all come
    from it, and it validates the CSV.

    ``n_cables`` and ``phi_deg`` remain accepted for backward compatibility with
    direct callers that supply no canonical model. If they are passed *alongside*
    a canonical model and disagree with it, that is a configuration error and
    raises ValueError rather than silently preferring one of the two. Their old
    literal defaults (3 and 5.7) are gone: 5.7 silently disagreed with the 6.3
    that params.json ships — audit finding F-11.
    """
    if geometry is None and params is not None:
        geometry = from_params(params)

    if geometry is not None:
        # φ is the FULL included taper angle; the CAD draft is half of it.
        # Sourcing it here keeps that convention in exactly one place.
        canonical_phi_deg = geometry.inputs.phi_deg_full_included
        canonical_n_cables = geometry.inputs.n_cables

        if phi_deg is not None and abs(float(phi_deg) - canonical_phi_deg) > _PHI_AGREEMENT_TOL_DEG:
            raise ValueError(
                f"phi_deg={phi_deg} was passed explicitly but the canonical "
                f"geometry says {canonical_phi_deg}. The canonical model is "
                f"authoritative; drop the explicit argument or fix params.json.")
        if n_cables is not None and int(n_cables) != canonical_n_cables:
            raise ValueError(
                f"n_cables={n_cables} was passed explicitly but the canonical "
                f"geometry says {canonical_n_cables}. The canonical model is "
                f"authoritative; drop the explicit argument or fix params.json.")

        # Canonical wins unconditionally, not merely when the argument is None.
        phi_deg = canonical_phi_deg
        n_cables = canonical_n_cables

    if phi_deg is None or n_cables is None:
        raise ValueError(
            "process_csv needs either `geometry`/`params` (preferred) or "
            "explicit `n_cables` and `phi_deg`. There is deliberately no "
            "default for phi_deg; see audit finding F-11.")

    draft_angle_deg = phi_deg / 2.0
    flat_mode       = (not plain) and (n_cables <= 2)

    print("Mesh generation settings:")
    if plain:
        print(f"  mode                 = plain revolve (no cut)")
    elif flat_mode:
        print(f"  n_cables             = {n_cables}  →  flat extrusion")
        print(f"  flat_thickness_ratio = {flat_thickness_ratio}")
    else:
        print(f"  n_cables             = {n_cables}  →  {n_cables}-lobe cross-section")
        print(f"  phi_deg              = {phi_deg}  →  draft = {draft_angle_deg:.3f} deg")
        print(f"  nlobe_t              = {nlobe_t}  (0=circumscribed … 1=inscribed)")
        print(f"  notch_factor         = {notch_factor}")

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Input CSV not found: {csv_file}")

    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_file)

    if geometry is not None:
        units = build_unit_inputs(df, geometry)
        lr = geometry.lengths
        print(f"  terminal_unit_policy = "
              f"{geometry.inputs.terminal_unit_policy.value}")
        print(f"  units                = {lr.n_units_total} "
              f"({lr.n_complete_units} complete, "
              f"{1 if lr.has_partial_unit else 0} partial)")
    else:
        # Legacy path for direct callers that pass neither geometry nor params.
        units = [
            UnitMeshInputs(
                element_id=int(row["elem"]),
                link_name=f"link_{int(row['elem']):03d}",
                is_partial=False,
                profile_xyz=tuple(_row_slot(row, p) for p in _SLOT_COLUMNS),
                outer_radius_m=abs(float(row["c0_s1_x"])),
                height_z_m=abs(float(row["joint_s2_z"])
                               - float(row["joint_s1_z"])),
                origin_m=(float(row["joint_s1_x"]),
                          float(row["joint_s1_y"]),
                          float(row["joint_s1_z"])),
                row=row,
            )
            for _, row in df.iterrows()
        ]

    for unit in units:
        element_id = unit.element_id
        try:
            if flat_mode:
                # ── Flat extrusion of actual trapezoidal quad profile ─────
                solid = build_flat_element(unit.row, flat_thickness_ratio)

            else:
                # ── Revolved cylinder (+ optional n-lobe cut) ─────────────
                points = list(unit.profile_xyz)
                if len(points) < 2:
                    print(f"  Skipping element {element_id}: not enough points")
                    continue

                profile = make_profile_from_points(points)
                solid   = revolve_profile(profile, angle, revolve_axis)

                dx, dy, dz = unit.origin_m
                solid = solid.translate((-dx, -dy, -dz))

                if not plain:
                    solid = add_nlobe_cut(
                        solid, n_cables, unit.outer_radius_m, unit.height_z_m,
                        draft_angle_deg,
                        nlobe_t=nlobe_t,
                        notch_factor=notch_factor,
                    )

            output_path = os.path.join(outdir, f"{unit.link_name}.stl")
            cq.exporters.export(solid, output_path, tolerance=1e-4)
            print(f"  ✓ {output_path}")

        except Exception as e:
            print(f"  ❌ Failed on element {element_id}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(
        description="Spirob CSV → STL mesh generator  "
                    "(plain revolve / flat extrusion / n-lobe, driven by params.json)"
    )
    parser.add_argument("--in",     dest="input",  required=True,
                        help="Input CSV file")
    parser.add_argument("--outdir", default="meshes",
                        help="Output directory for STL files (default: meshes)")
    parser.add_argument("--axis",   default="y",
                        help="Revolution axis (default: y)")
    parser.add_argument("--angle",  type=float, default=360,
                        help="Revolution angle in degrees (default: 360)")
    parser.add_argument("--params", default="params.json",
                        help="Path to params.json (default: params.json)")
    parser.add_argument("--plain",  action="store_true",
                        help="Plain revolve — skip n-lobe cut (full solid of revolution)")
    args = parser.parse_args()

    with open(args.params, encoding="utf-8") as f:
        params = json.load(f)

    # n_cables and phi_deg are deliberately NOT passed: they come from the
    # canonical model, so the CLI exercises the same path every other caller
    # should use.
    process_csv(
        csv_file             = args.input,
        outdir               = args.outdir,
        revolve_axis         = args.axis,
        angle                = args.angle,
        nlobe_t              = float(params.get("nlobe_t", 0.5)),
        notch_factor         = float(params.get("notch_factor", 0.25)),
        flat_thickness_ratio = float(params.get("flat_thickness_ratio", 0.3)),
        plain                = args.plain,
        geometry             = from_params(params),
    )
