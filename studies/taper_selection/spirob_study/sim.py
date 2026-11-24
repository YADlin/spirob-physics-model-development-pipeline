"""Shared simulation helpers: force ramps, contact metrics and rendering."""
from __future__ import annotations

import math
import os

import mujoco
import numpy as np

from .model import StudyModel

LAT_SHIFT = 0.0015   # m, tendon_inward_shift of every design in this study


def f_pack_estimate(sm: StudyModel) -> float:
    """Force at which the stiffest (base) joint reaches its limit.

    Joint i locks when F h_i >= k_i * Delta_theta, with moment arm
    h_i = W_i/2 - shift. The base joint is the last to lock because
    k_i / h_i grows like W_i^2.
    """
    dth = math.radians(sm.design.dtheta_deg)
    return float(np.max(sm.stiffness * dth / (sm.joint_width_m / 2 - LAT_SHIFT)))


def lock_forces_analytic(sm: StudyModel) -> np.ndarray:
    dth = math.radians(sm.design.dtheta_deg)
    return sm.stiffness * dth / (sm.joint_width_m / 2 - LAT_SHIFT)


def qpos_index(m: mujoco.MjModel, sm: StudyModel) -> np.ndarray:
    return np.array([m.joint(n).qposadr[0] for n in sm.joint_names])


def body_ids(m: mujoco.MjModel, sm: StudyModel) -> np.ndarray:
    return np.array([m.body(n).id for n in sm.link_names])


def centerline(m, d, sm) -> np.ndarray:
    """XY of every joint (link origin) plus the tip, base -> tip."""
    pts = d.xpos[body_ids(m, sm), :2]
    return np.vstack([pts, d.site("tip").xpos[:2]])


def ramp_single_cable(sm: StudyModel, cable: int, f_end: float, t_ramp: float = 6.0,
                      t_hold: float = 0.5, log_hz: float = 250.0, mu: float = 0.0) -> dict:
    """Quasi-static single-cable ramp 0 -> f_end; returns a time log.

    mu > 0 replaces the ideal tendon actuator by a capstan-friction cable:
    the tension reaching joint i is F exp(-mu * sum_{j<i} |q_j|), i.e. it
    drops by exp(-mu * angle) at every bend between the motor and that joint
    (sliding friction while the cable is being pulled in). The torque on
    joint i is that local tension times the tendon's moment arm dL/dq_i,
    computed by MuJoCo from the routed path.
    """
    m, d = sm.load()
    dt = m.opt.timestep
    qi = qpos_index(m, sm)
    dof = np.array([m.joint(n).dofadr[0] for n in sm.joint_names])
    n_r, n_h = int(t_ramp / dt), int(t_hold / dt)
    every = max(1, int(round(1.0 / (log_hz * dt))))
    log = {k: [] for k in ("t", "F", "q", "tip", "ten", "line")}
    for k in range(n_r + n_h):
        F = f_end * min(1.0, k / n_r)
        d.ctrl[:] = 0.0
        if mu <= 0.0:
            d.ctrl[cable] = -F
        else:
            q = np.abs(d.qpos[qi])
            wrap = np.concatenate([[0.0], np.cumsum(q)[:-1]])     # bends proximal to joint i
            J = np.asarray(d.ten_J).reshape(m.ntendon, m.nv)[cable, dof]
            d.qfrc_applied[:] = 0.0
            d.qfrc_applied[dof] = -F * np.exp(-mu * wrap) * J
        mujoco.mj_step(m, d)
        if k % every == 0 or k == n_r + n_h - 1:
            log["t"].append(d.time); log["F"].append(F)
            log["q"].append(d.qpos[qi].copy())
            log["tip"].append(d.site("tip").xpos[:2].copy())
            log["ten"].append(d.ten_length[:2].copy())
            log["line"].append(centerline(m, d, sm))
    return {k: np.asarray(v) for k, v in log.items()}


def hull_points_world(m, d, geom_prefix: str = "collision_") -> np.ndarray:
    pts = []
    for i in range(m.ngeom):
        if not m.geom(i).name.startswith(geom_prefix):
            continue
        mid = m.geom_dataid[i]
        adr, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = m.mesh_vert[adr:adr + nv]
        pts.append(d.geom_xpos[i] + v @ d.geom_xmat[i].reshape(3, 3).T)
    return np.vstack(pts)


def enclosure_angle(m, d, center: np.ndarray, radius: float, tol: float = 0.002) -> float:
    """Angle (deg) of the object's circumference covered by the robot body.

    Robot surface points within ``tol`` of the cylinder are projected on its
    circumference; the covered angle is 360 deg minus the largest uncovered
    gap. >= 180 deg means the body surrounds at least half of the object,
    the paper's minimum for a wrapping grasp (Note S2).
    """
    p = hull_points_world(m, d)[:, :2] - center
    r = np.linalg.norm(p, axis=1)
    sel = p[r < radius + tol]
    if len(sel) < 2:
        return 0.0
    a = np.sort(np.arctan2(sel[:, 1], sel[:, 0]))
    gaps = np.diff(np.r_[a, a[0] + 2 * np.pi])
    return math.degrees(2 * np.pi - gaps.max())


def normal_force_on(m, d, geom_id: int) -> float:
    """Sum of normal contact forces acting on one geom (N)."""
    f6 = np.zeros(6)
    tot = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if geom_id in (c.geom1, c.geom2):
            mujoco.mj_contactForce(m, d, i, f6)
            tot += abs(f6[0])
    return tot


# ---------------------------------------------------------------- rendering
def gl_env() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")


class TopView:
    """Overhead render of the planar robot (matches the rig's camera)."""

    def __init__(self, m, width=480, height=480, center=(0.0, 0.09), extent=0.36):
        self.r = mujoco.Renderer(m, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [center[0], center[1], 0.0]
        self.cam.distance = extent / (2 * math.tan(math.radians(22.5)))
        self.cam.azimuth = 90.0
        self.cam.elevation = -89.99
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[1] = 1   # visual meshes
        self.opt.geomgroup[0] = 1   # objects, floor
        self.opt.sitegroup[:] = 0
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 1

    def __call__(self, d) -> np.ndarray:
        self.r.update_scene(d, camera=self.cam, scene_option=self.opt)
        return self.r.render().copy()
