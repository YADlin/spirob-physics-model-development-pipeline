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


# -----------------------------
# Parameter validation
# -----------------------------
def validate_params(p: dict):
    """
    Validate every parameter in params.json before any computation.
    Collects ALL errors and raises a single ValueError listing them all.
    """
    errors = []

    # ── Required keys ─────────────────────────────────────────────────────────
    required = ["L", "d_tip", "phi_deg", "Delta_theta_deg", "n_cables",
                "tendon_inward_shift", "show_preview"]
    for key in required:
        if key not in p:
            errors.append(f"  • '{key}' is missing from params.json")

    if errors:                          # can't continue without core keys
        raise ValueError("Missing required parameters:\n" + "\n".join(errors))

    L               = p["L"]
    d_tip           = p["d_tip"]
    phi_deg         = p["phi_deg"]
    Delta_theta_deg = p["Delta_theta_deg"]
    n_cables        = p["n_cables"]
    tendon_shift    = p["tendon_inward_shift"]
    show_preview    = p["show_preview"]

    # ── Type checks ───────────────────────────────────────────────────────────
    if not isinstance(show_preview, bool):
        errors.append("  • 'show_preview' must be true or false (boolean)")
    if not isinstance(n_cables, int) or isinstance(n_cables, bool):
        errors.append("  • 'n_cables' must be an integer (e.g. 2, 3, 4)")

    # ── Core geometry ─────────────────────────────────────────────────────────
    if isinstance(L, (int, float)):
        if L <= 0:
            errors.append(f"  • 'L' must be > 0  (got {L})")
    if isinstance(d_tip, (int, float)):
        if d_tip <= 0:
            errors.append(f"  • 'd_tip' must be > 0  (got {d_tip})")
    if isinstance(phi_deg, (int, float)):
        if not (0 < phi_deg < 45):
            errors.append(f"  • 'phi_deg' must be in (0, 45) deg  (got {phi_deg})")
    if isinstance(Delta_theta_deg, (int, float)):
        if not (0 < Delta_theta_deg < 180):
            errors.append(
                f"  • 'Delta_theta_deg' must be in (0, 180) deg  (got {Delta_theta_deg}). "
                f"Values ≥ 180 produce degenerate elements."
            )
    if isinstance(n_cables, int) and not isinstance(n_cables, bool):
        if n_cables < 2:
            errors.append(f"  • 'n_cables' must be ≥ 2  (got {n_cables})")
    if isinstance(tendon_shift, (int, float)):
        if tendon_shift < 0:
            errors.append(f"  • 'tendon_inward_shift' must be ≥ 0  (got {tendon_shift})")
        if isinstance(d_tip, (int, float)) and d_tip > 0 and tendon_shift >= d_tip / 2:
            errors.append(
                f"  • 'tendon_inward_shift' ({tendon_shift*1e3:.2f} mm) must be "
                f"< d_tip/2 = {d_tip/2*1e3:.2f} mm"
            )

    # ── N-lobe / trilobe (optional, validated when present) ───────────────────
    notch_factor = p.get("notch_factor")
    nlobe_t      = p.get("nlobe_t")
    flat_t       = p.get("flat_thickness_ratio")

    if notch_factor is not None:
        if not isinstance(notch_factor, (int, float)) or not (0 <= notch_factor <= 0.4):
            errors.append(f"  • 'notch_factor' must be in [0, 0.4]  (got {notch_factor})")
    if nlobe_t is not None:
        if not isinstance(nlobe_t, (int, float)) or not (0.0 <= nlobe_t <= 1.0):
            errors.append(
                f"  • 'nlobe_t' must be in [0, 1]  "
                f"(0=circumscribed, 1=inscribed)  (got {nlobe_t})"
            )
    if flat_t is not None:
        if not isinstance(flat_t, (int, float)) or not (0.05 <= flat_t <= 1.0):
            errors.append(f"  • 'flat_thickness_ratio' must be in [0.05, 1.0]  (got {flat_t})")

    # ── post_gen block ─────────────────────────────────────────────────────────
    post = p.get("post_gen", {})
    if not isinstance(post, dict):
        errors.append("  • 'post_gen' must be a JSON object  (got non-object)")
    else:
        def _check_pos(key, lo=0, inclusive=True):
            v = post.get(key)
            if v is None:
                return
            if not isinstance(v, (int, float)):
                errors.append(f"  • 'post_gen.{key}' must be a number  (got {v!r})")
                return
            ok = (v >= lo) if inclusive else (v > lo)
            if not ok:
                sym = "≥" if inclusive else ">"
                errors.append(f"  • 'post_gen.{key}' must be {sym} {lo}  (got {v})")

        _check_pos("joint_stiffness_base", lo=0)
        _check_pos("joint_damping_base",   lo=0)
        _check_pos("first_joint_stiffness",lo=0)
        _check_pos("first_joint_damping",  lo=0)
        _check_pos("joint_beta",           lo=0, inclusive=False)  # beta=0 → division by zero

        pos = post.get("robot_pos")
        if pos is not None:
            if not (isinstance(pos, list) and len(pos) == 3 and
                    all(isinstance(v, (int, float)) for v in pos)):
                errors.append("  • 'post_gen.robot_pos' must be [x, y, z] list of numbers")

        quat = post.get("robot_quat")
        if quat is not None:
            if not (isinstance(quat, list) and len(quat) == 4 and
                    all(isinstance(v, (int, float)) for v in quat)):
                errors.append("  • 'post_gen.robot_quat' must be [w, x, y, z] list of 4 numbers")

        tip_site = post.get("tip_site_pos")
        if tip_site is not None:
            if not (isinstance(tip_site, list) and len(tip_site) == 3 and
                    all(isinstance(v, (int, float)) for v in tip_site)):
                errors.append("  • 'post_gen.tip_site_pos' must be [x, y, z] list of numbers")

        target_site = post.get("target_site_pos")
        if target_site is not None:
            if not (isinstance(target_site, list) and len(target_site) == 3 and
                    all(isinstance(v, (int, float)) for v in target_site)):
                errors.append("  • 'post_gen.target_site_pos' must be [x, y, z] list of numbers")

    if errors:
        raise ValueError("Invalid parameter values:\n" + "\n".join(errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spirob CSV Generator")
    parser.add_argument("--params", required=True, help="Path to params.json")
    parser.add_argument("--out", help="Output CSV file (defaults to Geom_Data_CSV/Spirob_geom_data.csv)")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt and auto-accept params")
    args = parser.parse_args()

    # Load & validate
    params = load_params(args.params)

    try:
        validate_params(params)
    except ValueError as e:
        print(f"\n❌ Parameter error:\n{e}\n")
        print("Please fix params.json and re-run.")
        sys.exit(1)

    # Assign variables
    L                   = params["L"]                      # [m] total length
    d_tip               = params["d_tip"]                  # [m] tip outer diameter
    phi_deg             = params["phi_deg"]                # [deg] taper angle
    Delta_theta_deg     = params["Delta_theta_deg"]        # [deg] per-element bend angle
    n_cables            = params["n_cables"]               # number of cables
    tendon_inward_shift = params["tendon_inward_shift"]    # [m] shift tendons inward
    show_preview        = params["show_preview"]           # bool

    # Print summary
    print(f"""
Inputs:
    L                   = {L} m
    d_tip               = {d_tip} m
    phi                 = {phi_deg} deg
    Delta_theta         = {Delta_theta_deg} deg
    n_cables            = {n_cables}
    tendon_inward_shift = {tendon_inward_shift} m
    show_preview        = {show_preview}""")

    if not args.yes:
        choice = input("\nIs this correct? (y = yes, n = no/exit): ").strip().lower()
        if choice == "y":
            print("\nData confirmed. Moving to the next part...")
        elif choice == "n":
            print("\nExiting program. Please update params.json and restart.")
            sys.exit(1)
        else:
            print("Invalid choice. Please enter 'y' or 'n'.")
            sys.exit(1)
    else:
        print("\nData confirmed (auto-yes). Moving to the next part...")

    # Parameter calculations
    phi         = math.radians(phi_deg)
    Delta_theta = math.radians(Delta_theta_deg)
    b           = solve_b_for_phi(phi)
    e2pb        = math.exp(2 * math.pi * b)
    a           = d_tip / (e2pb - 1.0)
    A           = (a / b) * math.sqrt(b ** 2 + 1)  * (e2pb+1)/2 
    q0          = (1.0 / b) * math.log(1.0 + L / A)

    print(f"""
Derived:
    growth parameter b  = {b:.6g}
    spiral scale a      = {a:.9g}
    total curl angle q0 = {q0:.6g} rad  ({q0 * 180 / math.pi:.3f} deg)""")

    # Pose calculations
    spiral_pose       = generate_spiral_pose(a, b, Length=L, delta_theta=Delta_theta)
    straightened_pose = straighten_pose(spiral_pose)
    Spirob_final_pose = Invert_pose(straightened_pose, L)

    n_elements = len(Spirob_final_pose)
    print(f"\n    Number of elements generated: {n_elements}")

    # CSV output
    if args.out is None:
        new_folder = os.path.join(os.getcwd(), "Geom_Data_CSV")
        os.makedirs(new_folder, exist_ok=True)
        out_file = os.path.join(new_folder, "Spirob_geom_data.csv")
    else:
        out_file = args.out
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

    csv_file = generate_cable_sites_csv_zrot_from_P(
        Spirob_final_pose,
        n_cables=n_cables,
        csv_path=out_file,
        radial_scale=1.0,
    )

    print("Wrote:", csv_file)
