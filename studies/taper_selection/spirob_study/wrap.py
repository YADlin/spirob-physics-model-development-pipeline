"""Wrapping and grasping, version 2.

Definitions (agreed with NYK, 24 Nov 2026)
------------------------------------------
* Wrap: the object sits in the middle of the curl and one unbroken stretch of
  the body is in contact with it over at least half of its circumference
  (``wrap_deg >= 180``). Two separate touches (the object pinched between the
  coil and the base, which the old "enclosure" counted as 180 deg) do not count.
* Grasp: after the wrap, the cables hold it and move it: the object must travel
  at least one diameter while it stays wrapped and the contact can carry its
  weight (out of plane, mu_c * sum N >= m g).

Sequences (cable force programmes)
----------------------------------
"pull"   start straight, pull the cable on the object's side. The simplest
         grasp; the paper's Note S4 reports that it only catches objects near
         the tip, because the body curls from the tip.
"reach"  the authors' strategy (paper Fig. 3A, Note S4): start packed AWAY
         from the object, then shift the force antagonistically from that
         cable to the other. The body uncurls from the base (reaching), the
         outer face of its coil rolls onto the object, and the tip curls the
         other way round it ("climbing": the body rolls on the surface
         instead of sweeping into it).

Cable friction (spirob.cable_friction) can be switched on for any sequence.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

from spirob.cable_friction import CableFriction

from . import designs as D, model as M, sim

EGG_MASS = 0.060           # kg, hen's egg
MU_CONTACT = 0.5           # TPU on egg shell (assumed; dry rubber-like on smooth ceramic 0.4-0.6)
PEDESTAL_MU = 0.4          # egg on the pedestal cup
PEDESTAL_HALF = 0.014      # m, half size of the pedestal cup in spirob-mjlab
CONTACT_TRIGGER = 0.02     # N, robot-object normal force that counts as "touched"
TOL = 0.0015               # m, a body point within TOL of the surface is in contact


@dataclass
class Trial:
    design: D.Design
    pos: tuple[float, float]
    diam: float
    strategy: str = "reach"
    mu: float = 0.0
    anchored: bool = False           # True: egg held in a holder during the wrap; False: egg on pedestal
    f_pack: float = 10.0             # N, packing force of this body
    side: int = +1                   # +1 object on +X (cable_1), -1 on -X (cable_0)
    t_ramp: float = 4.0              # s, "pull": ramp time (4 s quasi-static, 0.1-0.3 s dynamic)
    transport: bool = True
    keep_frames: bool = False
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------- metrics
def _link_geoms(m, sm) -> list[np.ndarray]:
    out = []
    for name in sm.link_names:
        bid = m.body(name).id
        out.append(np.array([g for g in range(m.ngeom)
                             if m.geom_bodyid[g] == bid and m.geom(g).name.startswith("collision_")]))
    return out


def _geom_points(m, d, g):
    mid = m.geom_dataid[g]
    adr, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    v = m.mesh_vert[adr:adr + nv]
    return (d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)[:, :2]


def wrap_angle(m, d, link_geoms, center, R, tol: float = TOL, max_gap: int = 1) -> dict:
    """Largest contact sweep (deg) made by one unbroken stretch of the body.

    Every link whose surface comes within ``tol`` of the cylinder counts as
    touching. Touching links separated by at most ``max_gap`` non-touching
    links form one stretch. The sweep of a stretch is the angle, seen from the
    object centre, between its first and last contact points, unwrapped along
    the body so that a stretch going round more than once is counted fully.
    """
    per_link = []
    for gs in link_geoms:
        if len(gs) == 0:
            per_link.append(None)
            continue
        p = np.vstack([_geom_points(m, d, g) for g in gs]) - center
        r = np.linalg.norm(p, axis=1)
        sel = p[r < R + tol]
        per_link.append(np.arctan2(sel[:, 1], sel[:, 0]) if len(sel) else None)
    touching = [a is not None for a in per_link]
    runs, cur, gap = [], [], 0
    for i, t in enumerate(touching):
        if t:
            cur.append(i); gap = 0
        elif cur:
            gap += 1
            if gap > max_gap:
                runs.append(cur); cur, gap = [], 0
    if cur:
        runs.append(cur)
    best, best_run = 0.0, []
    for run in runs:
        ref, lo, hi = None, np.inf, -np.inf
        for i in run:
            a = per_link[i]
            mean = math.atan2(np.sin(a).mean(), np.cos(a).mean())
            if ref is None:
                ref = mean
            else:  # continue from the previous link
                mean = ref + (mean - ref + math.pi) % (2 * math.pi) - math.pi
                ref = mean
            span = (a - mean + math.pi) % (2 * math.pi) - math.pi + mean
            lo, hi = min(lo, span.min()), max(hi, span.max())
        sweep = math.degrees(hi - lo)
        if sweep > best:
            best, best_run = sweep, run
    return dict(wrap_deg=best, n_runs=len(runs), run_links=best_run,
                n_touching=int(sum(touching)))


def curl_angle(m, d, center) -> float:
    """How far (deg) the whole body winds round the object centre, base to tip.

    >= 270 deg means the spiral closes round the object: it sits inside the
    curl, held by one or more turns. An object pinched between the base and
    the outside of the coil (part 1's failure) scores about 220 deg; a wrap
    with the object in the curl scores 330-400 deg."""
    names = sorted(m.body(i).name for i in range(m.nbody) if m.body(i).name.startswith("link_"))
    pts = np.vstack([d.xpos[[m.body(n).id for n in names], :2], d.site("tip").xpos[:2]]) - center
    a = np.unwrap(np.arctan2(pts[:, 1], pts[:, 0]))
    return float(abs(np.degrees(a[-1] - a[0])))


CURL_MIN = 270.0


def contact_normal_sum(m, d, geom_id) -> float:
    return sim.normal_force_on(m, d, geom_id)


def contact_push(m, d, geom_id) -> float:
    """Magnitude of the net in-plane contact force the robot applies to the
    object (N): what a holder or the pedestal must resist to keep it in place."""
    f6, tot = np.zeros(6), np.zeros(3)
    for i in range(d.ncon):
        c = d.contact[i]
        if geom_id in (c.geom1, c.geom2):
            mujoco.mj_contactForce(m, d, i, f6)
            fw = c.frame.reshape(3, 3).T @ f6[:3]          # contact frame -> world
            tot += fw if c.geom2 == geom_id else -fw
    return float(np.linalg.norm(tot[:2]))


# ---------------------------------------------------------------- programmes
class Programme:
    """Cable force programme; returns (F_near, F_far) for the current time.

    Event-driven: the "reach" programme watches the robot-object contact.
    """

    def __init__(self, strategy: str, f_pack: float, dt: float, t_ramp: float = 4.0):
        self.s, self.fp, self.dt, self.t_ramp = strategy, f_pack, dt, t_ramp
        self.phase, self.t_phase, self.f_at = "pack", 0.0, 0.0
        self.t_contact = None
        self.done_time = None
        if strategy == "pull":
            self.phase = "wrap"
        self.log = []

    T_PACK, T_REACH, T_HOLD = 1.0, 4.0, 1.0

    def __call__(self, t: float, touching: bool) -> tuple[float, float]:
        fp = self.fp
        ph, tp = self.phase, t - self.t_phase
        f = 0.0
        if self.s == "pull":
            f = 1.25 * fp * min(1.0, t / self.t_ramp)
            if t >= self.t_ramp + 1.0:
                self.done_time = t
            return f, 0.0
        if ph == "pack":                          # curl AWAY from the object
            ff = 1.1 * fp * min(1.0, tp / self.T_PACK)
            if tp >= self.T_PACK + 0.3:
                self._go("reach", t)
            self.log.append((t, ph))
            return 0.0, ff
        # reach + wrap: one antagonistic cross-fade. The far cable is released
        # while the near cable is pulled; the body uncurls from the base, its
        # coil rolls onto the object and the tip curls the other way round it.
        u = min(1.0, tp / self.T_REACH)
        ff = 1.1 * fp * max(0.0, 1.0 - 2.0 * u)            # far cable off by mid-way
        f = 1.25 * fp * u
        if touching and self.t_contact is None:
            self.t_contact = t
        if tp >= self.T_REACH + self.T_HOLD:
            self.done_time = t
        self.log.append((t, ph))
        return f, ff

    def _go(self, phase, t):
        self.phase, self.t_phase = phase, t


# ---------------------------------------------------------------- one trial
def run(tr: Trial) -> dict:
    R = tr.diam / 2
    x, y = tr.pos
    sm = M.make(tr.design, f_max=1e3, tag=f"wrap_{os.getpid()}",
                objects=[dict(name="obj", pos=(x, y), radius=R, free=True, weld=tr.anchored)])
    m, d = sm.load()
    dt = m.opt.timestep
    # "anchored": a holder (stiff weld) keeps the egg in place and lets go when
    # the carry starts; otherwise the egg sits on its pedestal with dry
    # friction mu_p m g. MuJoCo's dof friction is a soft constraint that
    # creeps under any force; its time constant is cut to 2 timesteps so that
    # the egg creeps < 0.1 mm/s below the friction limit (default: ~2 mm/s).
    if not tr.anchored:
        for j in ("obj_x", "obj_y"):
            a = m.joint(j).dofadr[0]
            m.dof_frictionloss[a] = PEDESTAL_MU * EGG_MASS * 9.81
            m.dof_solref[a] = (2 * m.opt.timestep, 1.0)
    bid = m.body("obj").id                      # the real object's mass, whatever its diameter
    scale = EGG_MASS / m.body_mass[bid]
    m.body_mass[bid] = EGG_MASS
    m.body_inertia[bid] *= scale
    mujoco.mj_forward(m, d)
    gid, bid = m.geom("obj_geom").id, m.body("obj").id
    links = _link_geoms(m, sm)
    near, far = (1, 0) if tr.side > 0 else (0, 1)
    cf = CableFriction(m, mu=tr.mu)
    prog = Programme(tr.strategy, tr.f_pack, dt, tr.t_ramp)
    # "reach" starts from the packed body, as in the paper: the object is not
    # in the scene while the body packs, and the position is infeasible if
    # the packed body already overlaps it.
    ct, ca = int(m.geom_contype[gid]), int(m.geom_conaffinity[gid])
    if tr.strategy == "reach":
        m.geom_contype[gid] = m.geom_conaffinity[gid] = 0
    infeasible = False
    p0 = np.array(tr.pos)
    frames, every = [], int(round(0.02 / dt))
    dropped_t = None
    peak_push = 0.0
    k, t_end = 0, 20.0
    while True:
        t = k * dt
        touching = contact_normal_sum(m, d, gid) > CONTACT_TRIGGER if (k % 10 == 0) else prog.t_contact is not None
        was = prog.phase
        fn, ff = prog(t, touching)
        if tr.strategy == "reach" and was == "pack" and prog.phase == "reach":
            m.geom_contype[gid], m.geom_conaffinity[gid] = ct, ca
            mujoco.mj_forward(m, d)
            if any(d.contact[i].dist < -0.001 and gid in (d.contact[i].geom1, d.contact[i].geom2)
                   for i in range(d.ncon)):
                infeasible = True
                break
        ten = [0.0, 0.0]
        ten[near], ten[far] = fn, ff
        d.ctrl[:] = 0.0
        cf.apply(d, tension=ten)
        mujoco.mj_step(m, d)
        if tr.keep_frames and k % every == 0:
            frames.append((t, d.qpos.copy(), tuple(ten), prog.phase))
        if k % 50 == 0:
            peak_push = max(peak_push, contact_push(m, d, gid))
        if not tr.anchored and dropped_t is None and k % 50 == 0:
            if np.linalg.norm(d.xpos[bid, :2] - p0) > PEDESTAL_HALF:
                nsum = contact_normal_sum(m, d, gid)
                if MU_CONTACT * nsum < EGG_MASS * 9.81:
                    dropped_t = t
        k += 1
        if prog.done_time is not None or t > t_end:
            break
    if infeasible:
        return dict(design=tr.design.name, phi_deg=tr.design.phi_deg, L=tr.design.L, x=x, y=y,
                    d_mm=tr.diam * 1e3, strategy=tr.strategy, t_ramp=tr.t_ramp, mu=tr.mu,
                    anchored=tr.anchored, f_pack=tr.f_pack, infeasible=True, wrap=False, grasp=False,
                    wrap_deg=0.0, carry_mm=0.0, dropped=False)
    c = d.xpos[bid, :2].copy()
    w = wrap_angle(m, d, links, c, R)
    w["curl_deg"] = curl_angle(m, d, c)
    nsum = contact_normal_sum(m, d, gid)
    out = dict(design=tr.design.name, phi_deg=tr.design.phi_deg, L=tr.design.L, x=x, y=y,
               d_mm=tr.diam * 1e3, strategy=tr.strategy, t_ramp=tr.t_ramp, mu=tr.mu,
               anchored=tr.anchored, infeasible=False, f_pack=tr.f_pack, t_contact=prog.t_contact, **w, grip_N=nsum,
               weight_ok=bool(MU_CONTACT * nsum >= EGG_MASS * 9.81),
               dropped=dropped_t is not None, obj_travel_mm=float(np.linalg.norm(c - p0) * 1e3),
               old_enclosure_deg=sim.enclosure_angle(m, d, c, R), peak_push_N=peak_push,
               final_xy=c.tolist())
    out["wrap"] = bool(w["wrap_deg"] >= 180.0 and not out["dropped"])
    out["in_curl"] = bool(w["curl_deg"] >= CURL_MIN and out["weight_ok"] and not out["dropped"])
    if tr.transport and (out["wrap"] or out["in_curl"]):
        out.update(_transport(m, d, cf, tr, links, gid, bid, R, near, far, frames if tr.keep_frames else None,
                              t0=k * dt, every=every))
    else:
        out.update(grasp=False, carry_mm=0.0, carry_best=None)
    if tr.keep_frames:      # the per-process XML is reused, so keep its text
        out["frames"] = frames
        out["xml_text"] = sm.xml_path.read_text()
        out["build_dir"] = str(tr.design.build_dir)
    return out


def _transport(m, d, cf, tr, links, gid, bid, R, near, far, frames, t0, every) -> dict:
    """Release the egg, then hold it and move it with the cables.

    1. Release: the holder lets go (or the egg is lifted off its pedestal) and
       the cables keep their forces for 0.5 s. The body, which was pressing on
       a fixed egg, settles; usually it snaps back and pulls the egg into its
       curl. The egg must still be wrapped (>= 180 deg, one stretch) and the
       contact must carry its weight (mu_c sum N >= m g): "secured".
    2. Move: from the secured state three slow cable programmes (3 s) are
       tried; the egg must stay wrapped and carried at every check (20 Hz).
       tighten: near cable to 2x the wrap force (rolls the curl up)
       swing:   keep the near cable, pull the far cable to 0.8x
       relax:   lower the near cable to 0.6x (unrolls the base)
    grasp = secured and moved by at least one object diameter.
    """
    for j in ("obj_x", "obj_y", "obj_r"):         # lifted: nothing but the robot holds it in plane
        m.dof_frictionloss[m.joint(j).dofadr[0]] = 0.0
    if m.neq:
        d.eq_active[:] = 0                         # the holder lets go
    f_w = 1.25 * tr.f_pack
    dt = m.opt.timestep
    ten = [0.0, 0.0]
    ten[near] = f_w
    c_rel = d.xpos[bid, :2].copy()
    fr_rel = []
    for k in range(int(0.5 / dt)):
        cf.apply(d, tension=ten)
        mujoco.mj_step(m, d)
        if frames is not None and k % every == 0:
            fr_rel.append((t0 + k * dt, d.qpos.copy(), tuple(ten), "release"))
    if frames is not None:
        frames.extend(fr_rel)
    t0 += 0.5
    c_set = d.xpos[bid, :2].copy()
    w_set = wrap_angle(m, d, links, c_set, R)["wrap_deg"]
    k_set = curl_angle(m, d, c_set)
    n_set = contact_normal_sum(m, d, gid)
    secured = bool((w_set >= 180.0 or k_set >= CURL_MIN) and MU_CONTACT * n_set >= EGG_MASS * 9.81)
    out = dict(snap_mm=float(np.linalg.norm(c_set - c_rel) * 1e3), wrap_after_release=w_set,
               curl_after_release=k_set, grip_after_release_N=n_set, secured=secured)
    if not secured:
        return dict(out, carry_mm=0.0, carry_best=None, carry={}, grasp=False)
    state = (d.qpos.copy(), d.qvel.copy(), [t.copy() for t in cf.t_seg])
    moves = {"tighten": (lambda u: (f_w * (1 + u), 0.0)),
             "swing": (lambda u: (f_w, 0.8 * f_w * u)),
             "relax": (lambda u: (f_w * (1 - 0.4 * u), 0.0))}
    best, best_name, best_frames, held_all = 0.0, None, None, {}
    for name, fn in moves.items():
        d.qpos[:], d.qvel[:] = state[0], state[1]
        cf.t_seg = [t.copy() for t in state[2]]
        mujoco.mj_forward(m, d)
        n = int(3.0 / dt)
        held_dist, ok, fr = 0.0, True, []
        for k in range(n):
            fnr, ffr = fn(k / n)
            ten = [0.0, 0.0]
            ten[near], ten[far] = fnr, ffr
            cf.apply(d, tension=ten)
            mujoco.mj_step(m, d)
            if frames is not None and k % every == 0:
                fr.append((t0 + k * dt, d.qpos.copy(), tuple(ten), f"carry:{name}"))
            if k % 250 == 0 or k == n - 1:
                c = d.xpos[bid, :2]
                held_in = (wrap_angle(m, d, links, c, R)["wrap_deg"] >= 180.0 or curl_angle(m, d, c) >= CURL_MIN)
                carry = MU_CONTACT * contact_normal_sum(m, d, gid) >= EGG_MASS * 9.81
                if not held_in or not carry:
                    ok = False
                    break
                held_dist = max(held_dist, float(np.linalg.norm(c - c_set)))
        held_all[name] = dict(dist_mm=held_dist * 1e3, held_to_end=ok)
        if held_dist > best or best_frames is None:
            best, best_name, best_frames = held_dist, name, fr
    if frames is not None and best_frames:
        frames.extend(best_frames)
    return dict(out, carry_mm=best * 1e3, carry_best=best_name if best > 0 else None, carry=held_all,
                grasp=bool(best >= tr.diam))
