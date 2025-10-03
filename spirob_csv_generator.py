import json
import argparse
from helper_functions import *
import os
import sys

# -----------------------------
# Load parameters from JSON
# -----------------------------
def load_params(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spirob CSV Generator")
    parser.add_argument("--params", required=True, help="Path to params.json")
    parser.add_argument("--out", help="Output CSV file (defaults to Geom_Data_CSV/Spirob_geom_data.csv)")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation and auto-accept params")
    args = parser.parse_args()

    # Load JSON
    params = load_params(args.params)

    # Assign variables
    L = params["L"]                     # [m] total length
    d_tip = params["d_tip"]             # [m] tip width d(0)
    phi_deg = params["phi_deg"]         # [deg] taper angle
    Delta_theta_deg = params["Delta_theta_deg"]  # [deg] per-element bend
    n_cables = params["n_cables"]       # number of cables

    #1] ---MANUAL USER INPUTS ---
    # L = 0.21             # [m] total length
    # d_tip = 0.008       # [m] tip width d(0)
    # phi_deg = 5.7       # [deg] taper angle
    # Delta_theta_deg = 35.34  # [deg] per-element bend (target)
    # n_cables = 2         # 2, 3, or general n

    # Print to confirm
    print(f"""Inputs:
        L        =   {L}\tm
        d(theta) =   {d_tip}\tm
        phi      =   {phi_deg}\tdeg
        dtheta   =   {Delta_theta_deg}\tdeg
        n        =   {n_cables}\tnos.""")

    if not args.yes:
        choice = input("\nIs this correct? (y = yes, n = no/exit): ").strip().lower()
        if choice == "y":
            print("\nData confirmed. Moving to the next part...")
        elif choice == "n":
            print("\nExiting program. Please restart if needed.")
            sys.exit(1)
        else:
            print("Invalid choice, please enter 'y' or 'n'.")
            sys.exit(1)
    else:
        print("\nData confirmed (auto-yes). Moving to the next part...")



    # -----------------------------
    # Parameter calculations
    # -----------------------------
    phi = math.radians(phi_deg)
    Delta_theta = math.radians(Delta_theta_deg)
    b = solve_b_for_phi(phi)                # Spiral curve growth parameter
    print("Solved growth parameter b =", b)
    print("max bend angle delta_theta (rad) =", Delta_theta)

    e2pb = math.exp(2*math.pi*b)
    a = d_tip/(e2pb - 1.0)                  # Spiral curve length parameter
    A = (a/b)*math.sqrt(b**2 +1)
    q0 = (1.0/b) * math.log(1.0 + L/A)      # Total spiral angle upon curling
    print(f"""Spiral length parameter a={a:.9g}
        Total spiral angle upon curling q0={q0:.9g} rad  ({q0*180/math.pi:.3f} deg)""")

    # -----------------------------
    # Pose calculations
    # -----------------------------
    spiral_pose = generate_spiral_pose(a, b, Length=L, delta_theta=Delta_theta)
    straightened_pose = straighten_pose(spiral_pose)
    Spirob_final_pose = Invert_pose(straightened_pose, L)
    
    # optional visualization
    plot_quad_chain(Spirob_final_pose, "Red half-Spirob", 'red')

    # -----------------------------
    # CSV File Output
    # -----------------------------
    if args.out is None:
        # Default path: ./Geom_Data_CSV/Spirob_geom_data.csv    
        new_folder = os.path.join(os.getcwd(), "Geom_Data_CSV") # Create a new folder inside the current directory
        os.makedirs(new_folder, exist_ok=True)  # creates the folder if not already present
        out_file = os.path.join(new_folder, "Spirob_geom_data.csv") # Define file path inside the new folder
    else:
        out_file = args.out
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    csv_file = generate_cable_sites_csv_zrot_from_P(
        Spirob_final_pose, 
        n_cables=n_cables, 
        csv_path=out_file, 
        radial_scale=1.0
        )
    
    print("Wrote:", csv_file)