import subprocess
import sys
import os
import shutil
import argparse

def run_step(cmd, desc, check=True):
    print(f"\n=== {desc} ===")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"❌ {desc} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result.returncode

def clean_outputs():
    print("\n=== Cleaning old outputs ===")
    targets = ["Geom_Data_CSV", "meshes", "spirob_physics_model.xml"]
    for t in targets:
        if os.path.isdir(t):
            shutil.rmtree(t)
            print(f"🗑️ Removed directory: {t}")
        elif os.path.isfile(t):
            os.remove(t)
            print(f"🗑️ Removed file: {t}")

def main():
    parser = argparse.ArgumentParser(description="Spirob pipeline build script")
    parser.add_argument("--noclean", action="store_true",
                        help="Skip cleaning old outputs before running")
    args = parser.parse_args()

    if not args.noclean:
        clean_outputs()
    else:
        print("\n=== Skipping clean step (using existing outputs) ===")

    # Step 1: CSV
    run_step("python spirob_csv_generator.py --params params.json --yes", "Generating CSV")

    # Step 2: STL (ignore crash if files exist)
    stl_exit = run_step("python csv2geom.py --in Geom_Data_CSV/Spirob_geom_data.csv",
                        "Generating STL meshes", check=False)
    if not os.path.isdir("meshes") or len(os.listdir("meshes")) == 0:
        print("❌ No STL files generated — treating as failure.")
        sys.exit(stl_exit)
    elif stl_exit != 0:
        print(f"⚠️ STL generator exited with code {stl_exit}, but meshes exist — continuing...")

    # Step 3: XML
    run_step("python csv2xml.py --in Geom_Data_CSV/Spirob_geom_data.csv --out spirob_physics_model.xml",
             "Generating XML model")

    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
