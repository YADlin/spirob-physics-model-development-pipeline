"""Cable-to-body friction for site-routed MuJoCo tendons (stick-slip capstan).

Why this exists
---------------
MuJoCo transmits a tendon force to every joint the tendon crosses as if the
cable slid through the holes without friction: the tension is the same along
the whole cable. In a printed SpiRob the cable rubs on the edge of every hole
it passes, so the tension that reaches a distal unit depends on how much the
cable is bent between it and the motor and on the direction the cable last
slid. Wang & Freris (RoboSoft 2024, "Exploiting frictional effects to
reproduce octopus-like reaching movements with a cable-driven spiral robot")
show that this is what lets a two-cable SpiRob bend from the base, reach in
many directions and hold a curl while the other cable moves.

Model
-----
The routed cable is a chain of straight segments between the sites of the
spatial tendon, ordered from the motor (base) to the tip. Every interior site
is a contact where the cable changes direction by an angle alpha_k. Coulomb
friction on a curved contact (the capstan relation) bounds the tension ratio
across it:

    T_{k-1} exp(-mu alpha_k)  <=  T_k  <=  T_{k-1} exp(+mu alpha_k)

Inside these bounds the cable sticks and the segment keeps the tension it had;
at a bound it slips. Updating the segments in order from the motor gives a
rate-independent stick-slip law with memory:

    T_0 = motor tension
    T_k = clip(T_k_previous, T_{k-1} e^{-mu alpha_k}, T_{k-1} e^{+mu alpha_k})

Pulling from rest gives the classic capstan profile T_k = T_0 e^{-mu sum alpha}
(tension falls toward the tip). Releasing reverses the inequality that is
active, so the distal tension can stay higher than the motor tension. Holding
the motor still changes nothing, so a curl can be held while the other cable
is used. With mu = 0 every segment carries T_0 and the model is exactly
MuJoCo's ideal tendon.

The force on the joints is the sum over segments of tension times the
segment's length gradient. A segment between two sites on adjacent bodies
depends only on the joint between those bodies, and that derivative equals the
whole tendon's moment arm ten_J at that joint, so

    qfrc_j = -T_{k(j)} * ten_J[c, dof_j]

where k(j) is the segment spanning joint j. Segments inside one body carry no
moment.

Use
---
MuJoCo (CPU), force-controlled cables::

    cf = CableFriction(m, tendons=("cable_0", "cable_1"), mu=0.2)
    for each step:
        d.ctrl[:] = 0                            # motors on the cables stay idle
        cf.apply(d, tension=[T0_cable0, T0_cable1])   # writes d.qfrc_applied
        mujoco.mj_step(m, d)

MuJoCo with any actuator already on the tendon (e.g. the position servos of
spirob-mjlab): ``cf.apply(d, correction=True)`` reads each tendon's actuator
force from the previous step and adds only the difference between the frictional
and the ideal (uniform) transmission.

mjlab / MJWarp (GPU, batched): MJWarp cannot load MuJoCo C plugins, so the
model is written as array code. :func:`stick_slip_chain` and
:func:`contact_angles_from_sites` run unchanged on torch tensors of shape
(num_envs, ...); see ``TorchCableFriction`` for the batched wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CableRoute", "CableFriction", "stick_slip_chain", "capstan_profile",
           "contact_angles_from_sites", "TorchCableFriction"]


# --------------------------------------------------------------------------- math
def _xp(a):
    """numpy, or torch for torch tensors (torch is never imported here)."""
    if type(a).__module__.startswith("torch"):
        import torch
        return torch
    return np


def stick_slip_chain(t_prev, t0, alpha, mu):
    """One update of the stick-slip capstan chain.

    t_prev : (..., K) segment tensions from the previous step (segment 1..K)
    t0     : (...,)   motor tension (segment 0)
    alpha  : (..., K) direction change at the contact before each segment, rad
    mu     : friction coefficient (scalar or broadcastable)
    returns (..., K) new segment tensions.  Works for numpy and torch.
    """
    xp = _xp(t_prev)
    lo_f = xp.exp(-mu * alpha)
    hi_f = xp.exp(mu * alpha)
    out = []
    prev = t0
    for k in range(t_prev.shape[-1]):
        lo = prev * lo_f[..., k]
        hi = prev * hi_f[..., k]
        tk = xp.minimum(xp.maximum(t_prev[..., k], lo), hi)
        out.append(tk)
        prev = tk
    return xp.stack(out, -1) if xp is not np else np.stack(out, -1)


def _chain_scalar(t_prev: np.ndarray, t0: float, alpha: np.ndarray, mu: float) -> np.ndarray:
    """stick_slip_chain for one cable, in plain Python floats (about 30x faster
    than numpy for the ~40 contacts of a SpiRob cable)."""
    lo = np.exp(-mu * alpha).tolist()
    tp = t_prev.tolist()
    out = [0.0] * len(tp)
    prev = t0
    for k in range(len(tp)):
        a = prev * lo[k]
        b = prev / lo[k]
        v = tp[k]
        v = a if v < a else (b if v > b else v)
        out[k] = v
        prev = v
    return np.array(out)


def capstan_profile(t0: float, alpha: np.ndarray, mu: float, direction: int = +1) -> np.ndarray:
    """Memoryless sliding profile: direction +1 = cable pulled in (tension
    falls toward the tip), -1 = cable paid out (tension rises toward the tip)."""
    return t0 * np.exp(-direction * mu * np.cumsum(alpha, -1))


def contact_angles_from_sites(p):
    """Direction change at every interior site of a routed cable.

    p : (..., S, 3) site positions in path order (base -> tip)
    returns (..., S-2) angles in rad. numpy or torch.
    """
    xp = _xp(p)
    v = p[..., 1:, :] - p[..., :-1, :]
    if xp is np:
        v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
        c = np.clip((v[..., :-1, :] * v[..., 1:, :]).sum(-1), -1.0, 1.0)
        return np.arccos(c)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    c = (v[..., :-1, :] * v[..., 1:, :]).sum(-1).clamp(-1.0, 1.0)
    return xp.arccos(c)


# --------------------------------------------------------------------- MuJoCo
@dataclass
class CableRoute:
    tendon_id: int
    site_ids: np.ndarray        # base -> tip
    seg_dof: np.ndarray         # (K,) dof spanned by segment k (k = 1..K), -1 if none
    rest_alpha: np.ndarray      # (K,) contact angle before segment k at q = 0


class CableFriction:
    """Stick-slip capstan friction for the spatial cable tendons of a model."""

    def __init__(self, m, tendons=("cable_0", "cable_1"), mu: float = 0.2,
                 mode: str = "stick_slip"):
        import mujoco
        if mode not in ("stick_slip", "pull"):
            raise ValueError(mode)
        self.m, self.mu, self.mode = m, float(mu), mode
        self.routes: list[CableRoute] = []
        depth = _body_depth(m)
        for name in tendons:
            tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, name)
            if tid < 0:
                raise KeyError(name)
            adr, num = m.tendon_adr[tid], m.tendon_num[tid]
            sites = []
            for w in range(adr, adr + num):
                if m.wrap_type[w] != mujoco.mjtWrap.mjWRAP_SITE:
                    raise ValueError(f"{name}: only site-routed tendons are supported")
                sites.append(int(m.wrap_objid[w]))
            bodies = [int(m.site_bodyid[s]) for s in sites]
            if depth[bodies[0]] > depth[bodies[-1]]:      # make it base -> tip
                sites, bodies = sites[::-1], bodies[::-1]
            seg_dof = []
            for a, b in zip(bodies[:-1], bodies[1:]):
                if a == b:
                    seg_dof.append(-1)
                    continue
                child = b if m.body_parentid[b] == a else a
                if m.body_jntnum[child] != 1:
                    raise ValueError(f"{name}: body {child} must have exactly one hinge")
                seg_dof.append(int(m.jnt_dofadr[m.body_jntadr[child]]))
            self.routes.append(CableRoute(tid, np.array(sites), np.array(seg_dof[1:]), np.zeros(0)))
            if seg_dof[0] != -1:
                raise ValueError(f"{name}: the first segment must lie inside the base body")
        d0 = mujoco.MjData(m)
        mujoco.mj_forward(m, d0)
        for r in self.routes:
            r.rest_alpha = self.contact_angles(d0, r)
        self._dofs = np.unique(np.concatenate([r.seg_dof[r.seg_dof >= 0] for r in self.routes]))
        self.reset()

    # -- state
    def reset(self):
        self.t_seg = [np.zeros(len(r.seg_dof)) for r in self.routes]

    # -- geometry
    @staticmethod
    def contact_angles(d, route: CableRoute) -> np.ndarray:
        return contact_angles_from_sites(d.site_xpos[route.site_ids])   # (K,)

    # -- tension
    def update(self, d, tension) -> list[np.ndarray]:
        out = []
        for i, (r, t0) in enumerate(zip(self.routes, tension)):
            t0 = max(float(t0), 0.0)                 # cables only pull
            alpha = self.contact_angles(d, r)
            if self.mode == "pull" or self.mu == 0.0:
                t = t0 * np.exp(-self.mu * np.cumsum(alpha))
            else:
                t = _chain_scalar(self.t_seg[i], t0, alpha, self.mu)
            self.t_seg[i] = t
            out.append(t)
        return out

    def generalized_force(self, d, tension) -> np.ndarray:
        tens = self.update(d, tension)
        nv = self.m.nv
        J = np.asarray(d.ten_J).reshape(self.m.ntendon, nv)
        f = np.zeros(nv)
        for r, t in zip(self.routes, tens):
            sel = r.seg_dof >= 0
            dof = r.seg_dof[sel]
            f[dof] += -t[sel] * J[r.tendon_id, dof]
        return f

    def apply(self, d, tension=None, correction: bool = False):
        """Call after mj_forward/mj_step so that site positions and ten_J are
        current (after mj_step they lag by one step, which is negligible).

        Write the cable forces into d.qfrc_applied (overwrites the joint
        dofs the cables span, leaves every other dof untouched).

        tension     : motor tension of each cable (N, >= 0). Required unless
                      correction=True.
        correction  : the model has actuators on the tendons that MuJoCo will
                      apply uniformly; read their forces and add only the
                      friction correction.
        """
        dofs = self._dofs
        if correction:
            act_f = _tendon_actuator_force(self.m, d)
            tension = [-act_f[r.tendon_id] for r in self.routes]
            f = self.generalized_force(d, tension)
            J = np.asarray(d.ten_J).reshape(self.m.ntendon, self.m.nv)
            ideal = np.zeros(self.m.nv)
            for r, t0 in zip(self.routes, tension):
                ideal += -max(t0, 0.0) * J[r.tendon_id]
            d.qfrc_applied[dofs] = (f - ideal)[dofs]
        else:
            if tension is None:
                raise ValueError("tension is required when correction=False")
            f = self.generalized_force(d, tension)
            d.qfrc_applied[dofs] = f[dofs]
        return f

    def joint_tension(self, route_index: int) -> np.ndarray:
        """Tension of the segment spanning each joint, base -> tip."""
        r, t = self.routes[route_index], self.t_seg[route_index]
        return t[r.seg_dof >= 0]


def _body_depth(m) -> np.ndarray:
    depth = np.zeros(m.nbody, int)
    for b in range(1, m.nbody):
        depth[b] = depth[m.body_parentid[b]] + 1
    return depth


def _tendon_actuator_force(m, d) -> np.ndarray:
    import mujoco
    f = np.zeros(m.ntendon)
    for a in range(m.nu):
        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_TENDON:
            f[m.actuator_trnid[a, 0]] += d.actuator_force[a] * m.actuator_gear[a, 0]
    return f


# ---------------------------------------------------------------------- mjlab
class TorchCableFriction:
    """Batched version for mjlab / MJWarp (torch tensors, any device).

    Built once from a CPU ``mujoco.MjModel`` of the robot. Per step it needs,
    for every environment, the world positions of the cable sites (MJWarp
    computes ``site_xpos``), the motor tension of each cable and the tendon
    moment arms at the cable joints. It returns the generalised joint forces
    to add to ``qfrc_applied``.

        site_ids()                      -> (C, S) site indices to gather, base -> tip
        __call__(site_xpos, tension, ten_J_joints, ideal_tension=None)
            site_xpos     (n, C, S, 3)
            tension       (n, C)         motor tension (N, >= 0)
            ten_J_joints  (n, C, J)      ten_J at the joints spanned, base -> tip
            ideal_tension (n, C)         if the actuator already applies a uniform
                                         tension, pass it to get the correction only
        returns (n, J) joint forces, base -> tip (joint order = ``joint_dofs``)
    """

    def __init__(self, m, tendons=("cable_0", "cable_1"), mu: float = 0.2):
        cf = CableFriction(m, tendons, mu)
        self.mu = mu
        self._sites = np.stack([r.site_ids for r in cf.routes])
        spans = [r.seg_dof[r.seg_dof >= 0] for r in cf.routes]
        if any(not np.array_equal(spans[0], s) for s in spans):
            raise ValueError("all cables must span the same joints")
        self.joint_dofs = spans[0]
        self._span = np.flatnonzero(cf.routes[0].seg_dof >= 0)
        self.t_seg = None

    def site_ids(self) -> np.ndarray:
        return self._sites

    def reset(self, env_ids=None):
        if self.t_seg is not None:
            if env_ids is None:
                self.t_seg.zero_()
            else:
                self.t_seg[env_ids] = 0.0

    def __call__(self, site_xpos, tension, ten_J_joints, ideal_tension=None):
        import torch
        alpha = contact_angles_from_sites(site_xpos)                      # (n, C, K)
        if self.t_seg is None or self.t_seg.shape != alpha.shape:
            self.t_seg = torch.zeros_like(alpha)
        t = stick_slip_chain(self.t_seg, tension.clamp_min(0.0), alpha, self.mu)
        self.t_seg = t
        span = torch.as_tensor(self._span, device=t.device)
        f = -(t.index_select(-1, span) * ten_J_joints).sum(1)
        if ideal_tension is not None:
            f = f + (ideal_tension.clamp_min(0.0)[..., None] * ten_J_joints).sum(1)
        return f
