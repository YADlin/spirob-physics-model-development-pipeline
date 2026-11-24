"""Part 3, experiment P1 - curling a SpiRob by reeling in cable (length control).

P1a  stroke:  for every body, how much cable must be reeled in to curl it
              fully (beyond that, reeling in more only raises the tension).
P1b  ramp:    reel the cable in slowly (length servo, 0.02 m/s, well under the
              spirob-mjlab 0.1 m/s limit) and record, every 10 ms, commanded
              length, actual cable length in the body, tension at the base and
              every joint angle. Two servo stiffnesses (spirob-mjlab 2.5 N/mm
              and a stiff 100 N/mm spool), with and without cable friction.
              Gives the length analogue of Fig. 2b: how much cable is reeled
              in when each joint reaches its stop.
P1c  memory:  reel in to full curl and back out, with friction, under length
              control and under force control; compare the tip position at the
              same command on the way in and the way out.
"""
from __future__ import annotations

import json
import math
import os
from multiprocessing import Pool

import mujoco
import numpy as np

from spirob.cable_friction import CableFriction

from . import actuation as A, designs as D, model as M, sim
from .analytic import osr_full_design

OUT = A.OUT


def bodies() -> list[D.Design]:
    b = list(D.sweep_by_phi().values())
    b += [D.length_design(L, 6.3) for L in (0.30, 0.375, 0.45)]
    b += [osr_full_design()]
    return b


KEY_BODIES = lambda: [D.sweep_by_phi()[6.3], D.sweep_by_phi()[10.0], D.length_design(0.45, 6.3), osr_full_design()]


def _passive(st: A.Stroke, l_near: float) -> float:
    return st.L_passive(l_near) + A.SLACK_FRAC * st.stroke


def ramp(design, kp: float, mu: float, speed: float = 0.02, back: bool = False, hold: float = 0.5,
         over: float = 0.0) -> dict:
    """Length ramp on cable_1 (cable_0 follows its free length + slack).

    over: reel in this fraction of the stroke beyond full curl (to show that
    extra cable only raises the tension)."""
    st = A.stroke(design)
    lo, hi = st.ctrlrange()
    lo -= over * st.stroke
    sm = M.make(design, f_max=A.F_MAX, tag=f"p1_{os.getpid()}", actuation="position", kp=kp,
                kv=A.kv_for(kp), ctrlrange=(lo, hi))
    m, d = sm.load()
    dt = m.opt.timestep
    d.ctrl[:] = (st.rest + A.SLACK_FRAC * st.stroke, st.rest)
    mujoco.mj_forward(m, d)
    cf = CableFriction(m, mu=mu) if mu > 0 else None
    qi = sim.qpos_index(m, sm)
    t_in = st.stroke * (1 + over) / speed
    t_tot = t_in + hold + (t_in + hold if back else 0.0)
    n, every = int(t_tot / dt), int(round(0.01 / dt))
    log = {k: [] for k in ("t", "cmd", "len", "T", "T_far", "q", "tip", "line", "len_far")}
    acc, na = np.zeros(2), 0
    for k in range(n):
        t = k * dt
        if t < t_in:
            u = t / t_in
        elif t < t_in + hold:
            u = 1.0
        elif back:
            u = max(0.0, 1.0 - (t - t_in - hold) / t_in)
        else:
            u = 1.0
        l1 = st.rest - u * st.stroke * (1 + over)
        d.ctrl[1] = l1
        d.ctrl[0] = _passive(st, max(l1, st.full))
        if cf is not None:
            cf.apply(d, correction=True)
        mujoco.mj_step(m, d)
        acc += np.maximum(-d.actuator_force[:2], 0.0); na += 1
        if k % every == 0:
            T = acc / na; acc[:] = 0; na = 0
            log["t"].append(t); log["cmd"].append(l1); log["len"].append(float(d.ten_length[1]))
            log["len_far"].append(float(d.ten_length[0]))
            log["T"].append(float(T[1])); log["T_far"].append(float(T[0]))
            log["q"].append(d.qpos[qi].copy()); log["tip"].append(d.site("tip").xpos[:2].copy())
            log["line"].append(sim.centerline(m, d, sm))
    out = {k: np.asarray(v) for k, v in log.items()}
    out["t_in"], out["hold"] = t_in, hold
    return out


def force_cycle(design, mu: float, t_ramp: float = 6.0, hold: float = 0.5) -> dict:
    """Force analogue of ramp(back=True): tension 0 -> 1.25 f_pack -> 0."""
    st = A.stroke(design)
    sm = M.make(design, f_max=1e3, tag=f"p1f_{os.getpid()}")
    m, d = sm.load()
    dt = m.opt.timestep
    cf = CableFriction(m, mu=mu)
    qi = sim.qpos_index(m, sm)
    n, every = int((2 * t_ramp + hold) / dt), int(round(0.01 / dt))
    log = {k: [] for k in ("t", "F", "len", "tip", "q", "line")}
    for k in range(n):
        t = k * dt
        u = min(1.0, t / t_ramp) if t < t_ramp + hold else max(0.0, 1.0 - (t - t_ramp - hold) / t_ramp)
        F = st.f_grip * u
        cf.apply(d, tension=[0.0, F])
        mujoco.mj_step(m, d)
        if k % every == 0:
            log["t"].append(t); log["F"].append(F); log["len"].append(float(d.ten_length[1]))
            log["tip"].append(d.site("tip").xpos[:2].copy()); log["q"].append(d.qpos[qi].copy())
            log["line"].append(sim.centerline(m, d, sm))
    out = {k: np.asarray(v) for k, v in log.items()}
    out["t_ramp"], out["hold"] = t_ramp, hold
    return out


def lock_lengths(design, log) -> dict:
    """Cable reeled in (from rest) and tension when each joint reaches its stop."""
    st = A.stroke(design)
    dth = math.radians(design.dtheta_deg)
    q = np.abs(log["q"])
    reel, T = [], []
    for j in range(q.shape[1]):
        hit = np.nonzero(q[:, j] >= 0.95 * dth)[0]
        k = hit[0] if len(hit) else None
        reel.append(float(st.rest - log["len"][k]) if k is not None else float("nan"))
        T.append(float(log["T"][k]) if k is not None else float("nan"))
    return dict(reel_mm=(np.array(reel) * 1e3).tolist(), T_N=T)


# ----------------------------------------------------------------- runners
def _p1a(design):
    st = A.stroke(design)
    sm = M.make(design, tag=f"p1a_{os.getpid()}")
    return dict(design=design.name, L=sm.length_m, phi=design.phi_deg, n_joints=len(sm.joint_names),
                rest_mm=st.rest * 1e3, full_mm=st.full * 1e3, stroke_mm=st.stroke * 1e3,
                far_payout_mm=(st.far_full - st.rest) * 1e3, f_pack=st.f_pack, f_grip=st.f_grip,
                kp_1pct=st.kp_inextensible(), stretch_mjlab_mm=st.f_grip / A.KP_MJLAB * 1e3,
                stretch_stiff_mm=st.f_grip / A.KP_STIFF * 1e3,
                lock_reel_mm=((st.rest - np.array(st.lock_L)) * 1e3).tolist(), lock_F=st.lock_F)


def _p1b(args):
    design, kp, mu = args
    log = ramp(design, kp, mu)
    st = A.stroke(design)
    ll = lock_lengths(design, log)
    sm = M.make(design, tag=f"p1b_{os.getpid()}")
    return dict(design=design.name, kp=kp, mu=mu, joint_s_mm=(sm.joint_s_m * 1e3).tolist(), **ll,
                end_stretch_mm=float((log["len"][-1] - log["cmd"][-1]) * 1e3),
                end_T=float(log["T"][-1]), max_T=float(log["T"].max()),
                end_locked=float((np.abs(log["q"][-1]) >= 0.95 * math.radians(design.dtheta_deg)).mean()),
                end_tip=log["tip"][-1].tolist(), stroke_mm=st.stroke * 1e3,
                trace=dict(reel_mm=((st.rest - log["cmd"]) * 1e3)[::5].tolist(),
                           act_reel_mm=((st.rest - log["len"]) * 1e3)[::5].tolist(),
                           T=log["T"][::5].tolist(), T_far=log["T_far"][::5].tolist()))


def _p1c(args):
    design, mu = args
    st = A.stroke(design)
    pos = ramp(design, A.KP_STIFF, mu, back=True)
    frc = force_cycle(design, mu)

    def gap(cmd, tip, n_in):
        """tip distance (mm) between loading and unloading at the same command."""
        c_in, t_in = cmd[:n_in], tip[:n_in]
        c_out, t_out = cmd[n_in:][::-1], tip[n_in:][::-1]
        out = []
        for c, p in zip(c_out, t_out):
            i = int(np.argmin(np.abs(c_in - c)))
            out.append(float(np.linalg.norm(p - t_in[i]) * 1e3))
        return out

    n_in_p = int(np.searchsorted(pos["t"], pos["t_in"] + pos["hold"]))
    n_in_f = int(np.searchsorted(frc["t"], frc["t_ramp"] + frc["hold"]))
    gp = gap(pos["cmd"], pos["tip"], n_in_p)
    gf = gap(frc["F"], frc["tip"], n_in_f)
    return dict(design=design.name, mu=mu, pos_gap_max_mm=max(gp), force_gap_max_mm=max(gf),
                pos=dict(reel_mm=((st.rest - pos["cmd"]) * 1e3)[::5].tolist(), tip=pos["tip"][::5].tolist(),
                         T=pos["T"][::5].tolist(), act_reel_mm=((st.rest - pos["len"]) * 1e3)[::5].tolist()),
                force=dict(F=frc["F"][::5].tolist(), tip=frc["tip"][::5].tolist(),
                           reel_mm=((st.rest - frc["len"]) * 1e3)[::5].tolist()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bs = bodies()
    for b in bs:                         # strokes first (cached, sequential)
        A.stroke(b)
    with Pool(2) as pool:
        p1a = pool.map(_p1a, bs)
        (OUT / "p1a_strokes.json").write_text(json.dumps(p1a, indent=1))
        jobs = [(b, kp, mu) for b in bs for kp in (A.KP_MJLAB, A.KP_STIFF) for mu in (0.0, 0.2)]
        p1b = pool.map(_p1b, jobs)
        (OUT / "p1b_ramps.json").write_text(json.dumps(p1b))
        p1c = pool.map(_p1c, [(b, mu) for b in KEY_BODIES() for mu in (0.0, 0.2)])
        (OUT / "p1c_memory.json").write_text(json.dumps(p1c))
    print("P1 done")


if __name__ == "__main__":
    main()


def equivalence(design, fracs=(0.1, 0.25, 0.5, 0.75, 1.0), t_settle: float = 1.5) -> dict:
    """Check of the translation: in free space without friction, tension F
    and cable length L(F) must give the same shape. Settle each mode at each
    level and compare tip positions."""
    st = A.stroke(design)
    out = []
    for fr in fracs:
        F = fr * st.f_grip
        tips = {}
        for mode in ("force", "position"):
            if mode == "force":
                sm = M.make(design, tag=f"eq_{os.getpid()}")
            else:
                lo, hi = st.ctrlrange()
                sm = M.make(design, tag=f"eq_{os.getpid()}", actuation="position", f_max=A.F_MAX,
                            kp=A.KP_STIFF, kv=A.kv_for(A.KP_STIFF), ctrlrange=(lo, hi))
            m, d = sm.load()
            n = int(t_settle / m.opt.timestep)
            for k in range(n):
                u = min(1.0, k / (0.6 * n))
                if mode == "force":
                    d.ctrl[:] = (0.0, -F * u)
                else:
                    l1 = st.rest + (st.L_pull(F) - st.rest) * u
                    d.ctrl[:] = (_passive(st, l1), l1)
                mujoco.mj_step(m, d)
            tips[mode] = d.site("tip").xpos[:2].copy()
            if mode == "position":
                tips["T"] = float(-d.actuator_force[1])
        out.append(dict(F=F, L_mm=(st.rest - st.L_pull(F)) * 1e3, tip_force=tips["force"].tolist(),
                        tip_length=tips["position"].tolist(), T_length=tips["T"],
                        diff_mm=float(np.linalg.norm(tips["force"] - tips["position"]) * 1e3)))
    return dict(design=design.name, levels=out)


def main_equivalence():
    res = [equivalence(b) for b in (D.sweep_by_phi()[6.3], D.length_design(0.45, 6.3))]
    (OUT / "p1d_equivalence.json").write_text(json.dumps(res, indent=1))
    for r in res:
        for l in r["levels"]:
            print(r["design"], f"F {l['F']:.1f} N  L {l['L_mm']:.1f} mm  tip diff {l['diff_mm']:.2f} mm  T_length {l['T_length']:.2f}")
