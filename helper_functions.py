import math

''' 
phi is a  trig function with b as a parameter, we solve for b using bisection method using the
two functions below.
'''
def phi_from_b(b: float) -> float:
    e = math.exp(2*math.pi*b)
    num = b*(e-1.0)
    den = math.sqrt(1.0+b*b)*(e+1.0)
    return 2.0*math.atan2(num, den)

def solve_b_for_phi(phi_target: float, lo: float=1e-6, hi: float=2.0) -> float:
    # Expand hi if needed
    def f(b): return phi_from_b(b) - phi_target
    flo, fhi = f(lo), f(hi)
    tries = 0
    while flo*fhi > 0 and tries < 50:
        hi *= 1.5
        fhi = f(hi)
        tries += 1
    if flo*fhi > 0:
        # robust fallback for small angles
        return max(1e-6, math.tan(phi_target/2.0))
    for _ in range(120):
        mid = 0.5*(lo+hi)
        fmid = f(mid)
        if abs(fmid) < 1e-14 or (hi-lo) < 1e-14:
            return mid
        if flo*fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5*(lo+hi)

'''
Some transformation helper function
'''
import numpy as np

def normalize(v):
    return v / np.linalg.norm(v)

def angle_between(v1, v2):
    # Returns angle in radians between vectors v1 and v2
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.arccos(dot)

def rotate(points, angle):
    # Rotate a set of 2D points around the origin
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    return points @ R.T

'''
Plotting helper
'''
import matplotlib.pyplot as plt

def plot_quad_chain(quads, title="", color='blue'):
    for q in quads:
        q = np.array(q)
        closed = np.vstack([q, q[0]])
        plt.plot(closed[:, 0], closed[:, 1], '-', color=color)
    plt.axis('equal')
    plt.title(title)
    plt.show()

## Function to generate Spiral pose from Spiral parameters
'''
Spiral pose generator that will need the general spiral parameter a, b, Δθ, L to generate the 
x,y points of the spirob element points A is for the centerline and B is for the innerline of the spirob
and the points are stored in a form that makes a closed loop when plotted.
[A0, A1, B1, B0] = [[x,y value of A0], [x,y value of A1], [x,y value of B1], [x,y value of B0]]
the return value will have a list of lists of [A0, A1, B1, B0] with each list having the information of one element
from the smallest to the largest element sizewise.
'''
def generate_spiral_pose(a=1.0, b=0.1, Length=0.23, delta_theta=np.pi/6):
    
    A = (a/b)*math.sqrt(b**2 +1)
    Q0 = (1.0/b) * math.log(1.0 + Length/A)
    N = int(math.ceil(Q0/delta_theta))
    theta = np.array([min(k*delta_theta, Q0) for k in range(N+1)])

    # Outer spiral (edge A)
    r1 = a * np.exp(b * theta)
    x1 = r1 * np.cos(theta)
    y1 = r1 * np.sin(theta)
    side_A = np.column_stack((x1, y1))

    # Inner spiral (edge B), with growth shifted one full turn earlier
    theta_shifted = theta - 2 * np.pi
    r2 = a * np.exp(b * theta_shifted)
    x2 = r2 * np.cos(theta)
    y2 = r2 * np.sin(theta)
    side_B = np.column_stack((x2, y2))

    # Form quads
    quads = []
    for i in range(len(theta) - 1):
        A0 = side_A[i]
        A1 = side_A[i + 1]
        B1 = side_B[i + 1]
        B0 = side_B[i]
        quads.append(np.array([A0, A1, B1, B0]))  # [A0, A1, B1, B0]
    return quads

## Function to generate straighten the Spiral pose.
'''
This takes the list of lists of [A0, A1, B1, B0] as input and outputs the same after carrying out
rotation and position transformation. The straightened Spirob will still be inverted, as in, the smallest
will be at the bottom and the biggest one at the top.
'''

def straighten_pose(quads):
    straight_quads = []

    # Place first quad with hinge vertical
    quad0 = quads[0].copy()
    hinge_vec = quad0[1] - quad0[0]  # A to B
    v = np.array([hinge_vec[0], hinge_vec[1], 0.0], dtype=float) # converting into three element 
    ref = np.array([0.0, 1.0, 0.0], dtype=float)  # +Y
    angle = angle_between(v, ref)  # vertical
    z = np.cross(v, ref)[2]  # scalar z-component

    # Correct direction (sign)
    if z < 0:
        angle = -angle

    quad0_rot = rotate(quad0 - quad0[0], angle)  # rotate about A0
    quad0_rot += np.array([0, 0])  # start at origin
    straight_quads.append(quad0_rot)

    current_origin = quad0_rot[1]  # top of hinge
    target_dir = np.array([0, 1, 0])  # vertical

    for i in range(1, len(quads)):
        q = quads[i].copy()
        hinge = q[0], q[1]
        hinge_vec = hinge[1] - hinge[0]
        h_v = np.array([hinge_vec[0], hinge_vec[1], 0.0], dtype=float) # converting into three element

        angle = angle_between(h_v, target_dir)
        if np.cross(h_v, target_dir)[2] < 0:
            angle = -angle

        q_shifted = q - hinge[0]  # move A to origin
        q_rotated = rotate(q_shifted, angle)
        q_placed = q_rotated + current_origin  # place B at current top

        straight_quads.append(q_placed)

        # Update origin for next hinge
        current_origin = q_placed[1]  # B (top of new hinge)

    return straight_quads

# Mirror across centerline if needed
'''
This was created in the .ipynb to visualize the spirob 2d form with the entire spirob visible instead
of just the half one.
'''
def Mirror_pose(straight_quads):
    mirrored_quads = []
    q = straight_quads.copy()
    for i in range(len(straight_quads)):
        A0, A1, B1, B0 = q[i]
        Bi0 = [-q[i][3][0],q[i][3][1]]
        Bi1 = [-q[i][2][0],q[i][2][1]]
        polygon = np.array([A1, A0, B0, B1, A1, Bi1, Bi0, A0])
        mirrored_quads.append(polygon)
    return mirrored_quads

# Making the quads (list of np.arrays of [A0, A1, B1, B0]) Inverted 
'''
This makes the biggest element at the bottom and the smallest at the top. With the element list reversed
so that the element index is such that the biggest on at the bottom is the 1st item with index zero of the
list.
'''
def Invert_pose(quads, Length):
    invert_quads = []
    q = quads.copy()
    for i in range(len(quads)):
        for j in range(len(q[i])):
            q[i][j][1] = - q[i][j][1] + Length # This makes the first element at the top
        invert_quads.append(q[i])
    invert_quads.reverse()
    return invert_quads

# To save data to a csv file with allong with the point for cables to go through.
import csv
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
      site1: (B0x, 0, B0y),  site2: (B1x, 0, B1y).
    Cable-c (psi = 2π c / n):
      site1: (B0x*cosψ, B0x*sinψ, B0y),
      site2: (B1x*cosψ, B1x*sinψ, B1y).

    radial_scale lets you shrink/grow radius using Bx -> radial_scale*Bx.
    """

    # Build header
    cols = ["elem", "joint_s1_x", "joint_s1_y", "joint_s1_z", "joint_s2_x", "joint_s2_y", "joint_s2_z"]
    for c in range(n_cables):
        cols += [f"c{c}_s1_x", f"c{c}_s1_y", f"c{c}_s1_z",
                 f"c{c}_s2_x", f"c{c}_s2_y", f"c{c}_s2_z"]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)

        for i, quad in enumerate(P, start=1):
            # Unpack: [A1, A0, B0, B1] -> each [x, y]
            A1, A0, B0, B1 = quad
            A1x, A1y = float(A1[0]), float(A1[1])
            A0x, A0y = float(A0[0]), float(A0[1])
            B0x, B0y = float(B0[0]), float(B0[1])
            B1x, B1y = float(B1[0]), float(B1[1])

            # Joint (start) position at A0: (x, 0, y)
            row = [i, A0x, 0.0, A0y, A1x, 0.0, A1y]

            # Radii (allow scaling)
            r0 = radial_scale * B0x
            r1 = radial_scale * B1x

            for c in range(n_cables):
                psi = 2.0 * math.pi * c / n_cables
                cosp, sinp = math.cos(psi), math.sin(psi)

                # site1 at B0
                x1 = r0 * cosp
                y1 = r0 * sinp
                z1 = B0y

                # site2 at B1
                x2 = r1 * cosp
                y2 = r1 * sinp
                z2 = B1y

                row += [x1, y1, z1, x2, y2, z2]

            w.writerow(row)
    return csv_path