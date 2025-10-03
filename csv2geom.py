import cadquery as cq
import pandas as pd
import os
import argparse

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
    wp = cq.Workplane("XZ").polyline([(x,z) for x,y,z in points]).close()
    return wp

def revolve_profile(profile, angle=360, axis="z"):
    """Revolve a profile around a given axis by a certain angle."""
    axis_map = {
        "x": ((0,0,0),(1,0,0)),
        "y": ((0,0,0),(0,1,0)),
        "z": ((0,0,0),(0,0,1)),
        }
    start, end = axis_map.get(str(axis).lower(), axis_map["z"])
    return profile.revolve(angle, axisStart=start, axisEnd=end)

def process_csv(csv_file, outdir="meshes", revolve_axis="y", angle=360): 
    """Main geometry processing pipeline: CSV → STL meshes.
    ! For some weird reason, revolving around y axis gives the solid revolving around z axis"""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Input CSV not found: {csv_file}")
    
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_file)
    
    for _,row in df.iterrows():
        element_id = row['elem']
        try:
            points = extract_points(row)
            
            
            if len(points)<2:
                print(f"Skipping {element_id}: not enough points")
                continue
            
            # Build solid
            profile = make_profile_from_points(points)
            solid = revolve_profile(profile, angle, revolve_axis)

            # Translating the solid to origin
            dx, dy, dz = row['joint_s1_x'], row['joint_s1_y'], row['joint_s1_z']
            solid = solid.translate((-dx, -dy, -dz))
            
            #save stl
            filename = f"link_{int(element_id):03d}.stl"
            output_path = os.path.join(output_dir, filename)
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