"""
build.py  —  SpiRob pipeline driver.

Usage
-----
  python build.py                        # interactive preview, then full pipeline
  python build.py --plain                # plain revolve (no n-lobe cut)
  python build.py --no-preview           # skip preview gate (CI / batch)
  python build.py --noclean              # keep previous outputs
  python build.py --safe / --fast / --high   # physics preset passed to csv2xml

Cross-section is driven by n_cables in params.json:
  n_cables <= 2   →  flat extrusion  (hinge joints)
  n_cables >= 3   →  n-lobe cut
  --plain         →  full solid of revolution regardless of n_cables
"""

import subprocess
import sys
import os
import shutil
import argparse
import json


def run_step(cmd, desc, check=True):
    print(f"\n=== {desc} ===")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"❌  {desc} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result.returncode


def clean_outputs():
    print("\n=== Cleaning old outputs ===")
    for t in ["Geom_Data_CSV", "meshes", "spirob_physics_model.xml"]:
        if os.path.isdir(t):
            shutil.rmtree(t);  print(f"🗑️   Removed directory: {t}")
        elif os.path.isfile(t):
            os.remove(t);      print(f"🗑️   Removed file: {t}")


def run_preview_gate(params_path, nlobe=False):
    print("\n=== Launching geometry preview ===")
    cmd = f"python preview.py --params {params_path}"
    if nlobe:
        cmd += " --nlobe"
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("\n❌  Preview rejected or closed. Pipeline aborted.")
        print("    Adjust params.json and re-run.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="SpiRob pipeline build script")
    parser.add_argument("--params",     default="params.json",
                        help="Path to params.json (default: params.json)")
    parser.add_argument("--noclean",    action="store_true",
                        help="Skip cleaning old outputs before running")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip the interactive preview gate (CI / batch use)")
    parser.add_argument("--plain",      action="store_true",
                        help="Plain revolve — no n-lobe cut (full solid of revolution)")
    parser.add_argument("--safe",       action="store_true", help="MuJoCo safe-mode preset")
    parser.add_argument("--fast",       action="store_true", help="MuJoCo fast-mode preset")
    parser.add_argument("--high",       action="store_true", help="MuJoCo high-fidelity preset")
    args = parser.parse_args()

    # ── Load & validate params ────────────────────────────────────────────────
    with open(args.params) as f:
        params = json.load(f)

    from spirob_csv_generator import validate_params
    try:
        validate_params(params)
    except ValueError as e:
        print(f"\n❌ Parameter error:\n{e}\n")
        print("Please fix params.json and re-run.")
        sys.exit(1)

    n_cables     = int(params["n_cables"])
    tendon_shift = float(params["tendon_inward_shift"])
    phi_deg      = float(params["phi_deg"])
    use_nlobe    = (not args.plain) and (n_cables >= 3)

    # ── Step 0: Preview ───────────────────────────────────────────────────────
    if not args.no_preview:
        approved = run_preview_gate(args.params, nlobe=use_nlobe)
        if not approved:
            sys.exit(1)
    else:
        print("\n=== Skipping preview (--no-preview) ===")

    # ── Clean ─────────────────────────────────────────────────────────────────
    if not args.noclean:
        clean_outputs()
    else:
        print("\n=== Skipping clean (--noclean) ===")

    # ── Step 1: CSV ───────────────────────────────────────────────────────────
    run_step(
        f"python spirob_csv_generator.py --params {args.params} --yes",
        "Generating CSV"
    )

    # ── Step 2: STL ───────────────────────────────────────────────────────────
    stl_cmd = (f"python csv2geom_nlobe.py "
               f"--in Geom_Data_CSV/Spirob_geom_data.csv "
               f"--params {args.params}")
    if args.plain:
        stl_cmd += " --plain"
        stl_desc = "Generating STL meshes (plain revolve)"
    elif n_cables <= 2:
        stl_desc = "Generating STL meshes (flat extrusion)"
    else:
        stl_desc = f"Generating STL meshes ({n_cables}-lobe)"

    stl_exit = run_step(stl_cmd, stl_desc, check=False)
    if not os.path.isdir("meshes") or not os.listdir("meshes"):
        print("❌  No STL files generated — treating as failure.")
        sys.exit(stl_exit or 1)
    elif stl_exit != 0:
        print(f"⚠️   STL generator exited {stl_exit} but meshes exist — continuing…")

    # ── Step 3: XML ───────────────────────────────────────────────────────────
    xml_cmd = (f"python csv2xml.py "
               f"--in Geom_Data_CSV/Spirob_geom_data.csv "
               f"--out spirob_physics_model.xml "
               f"--tendon-shift {tendon_shift} "
               f"--phi-deg {phi_deg} "
               f"--params {args.params}")

    if n_cables <= 2 and not args.plain:
        xml_cmd += " --hinge"
    if args.safe:
        xml_cmd += " --safe"
    elif args.fast:
        xml_cmd += " --fast"
    elif args.high:
        xml_cmd += " --high"

    run_step(xml_cmd, "Generating XML model")

    print("\n✅  Pipeline completed successfully!")


if __name__ == "__main__":
    main()
