"""Established CSV serializer; geometry is supplied by spirob.geometry."""
import csv
import math
from typing import Sequence

def generate_cable_sites_csv_zrot_from_P(
    P: Sequence[Sequence[Sequence[float]]],
    n_cables: int,
    csv_path: str = "spirob_sites_zrot.csv",
    radial_scale: float = 1.0
):
    """
    P[i] = [A1_i, A0_i, B0_i, B1_i], each [x, y] in straight pose (2D).
    World mapping: (x, y) -> (X=x, Y=0, Z=y).

    Cable-0 (psi=0):
      site1: (B1x, 0, B1y),  site2: (B0x, 0, B0y).
    Cable-c (psi = 2π c / n):
      site1: (B1x*cosψ, B1x*sinψ, B0y),
      site2: (B0x*cosψ, B0x*sinψ, B1y).

    radial_scale lets you shrink/grow radius using Bx -> radial_scale*Bx.
    """
    if n_cables <= 0:
        raise ValueError("n_cables must be > 0")

    # Build header
    cols = ["elem", "joint_s1_x", "joint_s1_y", "joint_s1_z", "joint_s2_x", "joint_s2_y", "joint_s2_z"]
    for c in range(n_cables):
        cols += [f"c{c}_s1_x", f"c{c}_s1_y", f"c{c}_s1_z",
                 f"c{c}_s2_x", f"c{c}_s2_y", f"c{c}_s2_z"]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)

        for i, quad in enumerate(P, start=1):

            # Correct ordering
            A0, A1, B1, B0 = quad

            A0x, A0y = A0
            A1x, A1y = A1
            B0x, B0y = B0
            B1x, B1y = B1

            row = [i, A0x, 0.0, A0y, A1x, 0.0, A1y]

            r0 = radial_scale * B0x
            r1 = radial_scale * B1x

            for c in range(n_cables):
                psi = 2.0 * math.pi * c / n_cables
                cosp, sinp = math.cos(psi), math.sin(psi)

                row.extend([
                    r0 * cosp, r0 * sinp, B0y,
                    r1 * cosp, r1 * sinp, B1y
                ])

            w.writerow(row)

    return csv_path
