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

import cadquery as cq
import pandas as pd
import os
import argparse
import math


# ══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def extract_points(row):
    """Extract joint + cable-0 site points from a CSV row as 3-D tuples."""
    point_sets = ["joint_s1", "joint_s2", "c0_s2", "c0_s1"]
    points = []
    for p in point_sets:
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
    A0x = float(row["joint_s1_x"]);  A0z = float(row["joint_s1_z"])
    A1x = float(row["joint_s2_x"]);  A1z = float(row["joint_s2_z"])
    B1x = float(row["c0_s2_x"]);     B1z = float(row["c0_s2_z"])
    B0x = float(row["c0_s1_x"]);     B0z = float(row["c0_s1_z"])

    # Snap hinge points to exactly x=0 (they are ~1e-17 due to float math)
    A0x = 0.0;  A1x = 0.0

    # Shift so joint_s1 is at z=0
    oz = A0z
    pts = [
        (A0x, 0.0),           # inner bottom
        (B0x, B0z - oz),      # outer bottom
        (B1x, B1z - oz),      # outer top
        (A1x, A1z - oz),      # inner top
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
                n_cables=3, phi_deg=5.7,
                nlobe_t=0.5, notch_factor=0.25,
                flat_thickness_ratio=0.3, plain=False):
    """
    CSV → STL meshes.

    plain=True       →  full solid of revolution, no cut  (replaces csv2geom.py)
    n_cables <= 2    →  flat extrusion of actual quad profile  (hinge joints)
    n_cables >= 3    →  revolved cylinder with n-lobe cut  (replaces csv2geom_trilob.py)
    """
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

    for _, row in df.iterrows():
        element_id = int(row["elem"])
        try:
            if flat_mode:
                # ── Flat extrusion of actual trapezoidal quad profile ─────
                solid = build_flat_element(row, flat_thickness_ratio)

            else:
                # ── Revolved cylinder (+ optional n-lobe cut) ─────────────
                points = extract_points(row)
                if len(points) < 2:
                    print(f"  Skipping element {element_id}: not enough points")
                    continue

                outer_radius = abs(float(row["c0_s1_x"]))
                height_z     = abs(float(row["joint_s2_z"]) -
                                   float(row["joint_s1_z"]))

                profile = make_profile_from_points(points)
                solid   = revolve_profile(profile, angle, revolve_axis)

                dx = float(row["joint_s1_x"])
                dy = float(row["joint_s1_y"])
                dz = float(row["joint_s1_z"])
                solid = solid.translate((-dx, -dy, -dz))

                if not plain:
                    solid = add_nlobe_cut(
                        solid, n_cables, outer_radius, height_z,
                        draft_angle_deg,
                        nlobe_t=nlobe_t,
                        notch_factor=notch_factor,
                    )

            filename    = f"link_{element_id:03d}.stl"
            output_path = os.path.join(outdir, filename)
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

    with open(args.params) as f:
        params = json.load(f)

    process_csv(
        csv_file             = args.input,
        outdir               = args.outdir,
        revolve_axis         = args.axis,
        angle                = args.angle,
        n_cables             = int(params["n_cables"]),
        phi_deg              = float(params["phi_deg"]),
        nlobe_t              = float(params.get("nlobe_t", 0.5)),
        notch_factor         = float(params.get("notch_factor", 0.25)),
        flat_thickness_ratio = float(params.get("flat_thickness_ratio", 0.3)),
        plain                = args.plain,
    )
