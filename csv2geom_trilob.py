import cadquery as cq
import pandas as pd
import os
import argparse
import math
import json

csv_file = os.path.join("Geom_Data_CSV", "Spirob_data_points.csv")

output_dir = "meshes"
os.makedirs(output_dir, exist_ok=True)


def extract_points(row):
    """Extracts joint/connector points from a CSV row into a list of 3D tuples."""
    point_sets = ["joint_s1", "joint_s2", "c0_s2", "c0_s1"]
    points = []
    for p in point_sets:
        x, y, z = row[f"{p}_x"], row[f"{p}_y"], row[f"{p}_z"]
        if pd.notna(x) and pd.notna(y) and pd.notna(z):
            points.append((float(x), float(y), float(z)))
    return points


def make_profile_from_points(points):
    """Make a CadQuery workplane profile from (x,z) projection of points."""
    wp = cq.Workplane("XZ").polyline([(x, z) for x, y, z in points]).close()
    return wp


def revolve_profile(profile, angle=360, axis="z"):
    """Revolve a profile around a given axis by a certain angle."""
    axis_map = {
        "x": ((0, 0, 0), (1, 0, 0)),
        "y": ((0, 0, 0), (0, 1, 0)),
        "z": ((0, 0, 0), (0, 0, 1)),
    }
    start, end = axis_map.get(str(axis).lower(), axis_map["z"])
    return profile.revolve(angle, axisStart=start, axisEnd=end)


def add_triangle_notch_cut(solid, points, draft_angle_deg):
    """
    Apply the drafted equilateral triangle cut with semicircle notches.
    Cutter is centered on the cylinder's bounding-box center.
    """

    # height of the profile in Z (from input points)
    z_vals = [p[2] for p in points]
    if len(z_vals) < 2:
        return solid

    height_z = max(z_vals) - min(z_vals)
    if height_z <= 1e-6:
        return solid

    # triangle side length = 1.5 * Z-height
    side = 2.2 * height_z

    # cutter height (loft height)
    cutter_height = 3.0 * height_z

    # Perfect equilateral triangle
    V1 = (-side / math.sqrt(3), 0)
    V2 = (side / (2 * math.sqrt(3)), +side / 2)
    V3 = (side / (2 * math.sqrt(3)), -side / 2)
    triangle_pts = [V1, V2, V3]

    # Draft scaling (top profile)
    draft_rad = math.radians(draft_angle_deg)
    taper_scale = 1 - math.tan(draft_rad) * (cutter_height / side)
    if taper_scale <= 0:
        taper_scale = 1e-3

    triangle_pts_top = [(x * taper_scale, y * taper_scale) for (x, y) in triangle_pts]

    # Notch radius
    notch_radius = side / 10.0

    # Midpoints
    (x1, y1), (x2, y2), (x3, y3) = triangle_pts
    m12 = ((x1 + x2) / 2, (y1 + y2) / 2)
    m23 = ((x2 + x3) / 2, (y2 + y3) / 2)
    m31 = ((x3 + x1) / 2, (y3 + y1) / 2)

    # Cylindrical cutters for semicircle notches
    circle_cutters = (
        cq.Workplane("XY")
        .pushPoints([m12, m23, m31])
        .circle(notch_radius)
        .extrude(cutter_height * 1.2)
    )

    # Lofted triangular prism with draft
    drafted_prism = (
        cq.Workplane("XY")
        .polyline(triangle_pts).close()
        .workplane(offset=cutter_height)
        .polyline(triangle_pts_top).close()
        .loft()
    )

    # Apply notch cuts
    drafted_prism = drafted_prism.cut(circle_cutters)

    # Center cutter on cylinder bbox
    bbox = solid.val().BoundingBox()
    cx = 0.5 * (bbox.xmin + bbox.xmax)
    cy = 0.5 * (bbox.ymin + bbox.ymax)
    cz = 0.5 * (bbox.zmin + bbox.zmax)

    # Move prism so its center aligns with solid center
    drafted_prism = drafted_prism.translate((cx, cy, cz - cutter_height / 2))

    # Final boolean
    return solid.intersect(drafted_prism)


def process_csv(csv_file, outdir="meshes", revolve_axis="y", angle=360):
    """
    Main geometry processing pipeline: CSV → STL meshes.
    Draft angle is read from params.json (phi_deg/2).
    """

    # Load params.json ONCE
    with open("params.json") as f:
        params = json.load(f)

    phi_deg = params.get("phi_deg", 0.0)
    draft_angle_deg = phi_deg / 2.0  # <-- required behavior

    print(f"Using phi_deg={phi_deg}, draft={draft_angle_deg}")

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Input CSV not found: {csv_file}")

    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        element_id = row["elem"]
        try:
            points = extract_points(row)

            if len(points) < 2:
                print(f"Skipping {element_id}: not enough points")
                continue

            # Build revolved solid
            profile = make_profile_from_points(points)
            solid = revolve_profile(profile, angle, revolve_axis)

            # Translate solid so joint_s1 is at origin
            dx, dy, dz = row["joint_s1_x"], row["joint_s1_y"], row["joint_s1_z"]
            solid = solid.translate((-dx, -dy, -dz))

            # Apply triangle + notch cutter
            if abs(draft_angle_deg) > 1e-6:
                solid = add_triangle_notch_cut(solid, points, draft_angle_deg)

            # Save STL
            filename = f"link_{int(element_id):03d}.stl"
            output_path = os.path.join(outdir, filename)
            cq.exporters.export(solid, output_path, tolerance=1e-4)

            print(f"Generated {output_path}")

        except Exception as e:
            print(f"❌ Failed on element {element_id}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Spirob CSV to STL meshes")
    parser.add_argument("--in", dest="input", required=True, help="Input CSV file")
    parser.add_argument("--outdir", default="meshes", help="Output directory for STL files")
    parser.add_argument("--axis", default="y", help="Axis for revolution (x, y, z)")
    parser.add_argument("--angle", type=float, default=360, help="Revolve angle in degrees")
    args = parser.parse_args()

    process_csv(args.input, args.outdir, args.axis, args.angle)
