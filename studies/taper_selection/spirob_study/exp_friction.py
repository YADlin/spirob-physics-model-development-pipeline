"""Experiment F - what cable friction does to curling, for every length and taper.

F1  Load-unload. One cable is pulled slowly from 0 to 1.25 x the frictionless
    packing force (6 s), held, and released to 0 (6 s), with the stick-slip
    capstan friction of spirob.cable_friction at mu = 0, 0.1, 0.2, 0.3.
    Recorded: the force at which every joint locks, where along the body the
    bending is concentrated when the body has turned 90 and 180 deg (the
    "bend centroid", 0 = base, 1 = tip), whether the body packs, and how far
    the tip is from its loading position at the same force on the way back
    (hysteresis).

F2  Workspace. Both cables follow the same slow random force programme
    (a new target pair every 0.5 s, 40 s) with and without friction; the tip
    positions visited are counted on a 5 mm grid. Friction lets the two-cable
    robot reach places it cannot reach without it (Wang & Freris, RoboSoft 2024).
"""
from __future__ import annotations

import json
import math

import mujoco
import numpy as np

from spirob.cable_friction import CableFriction

from . import designs as D, model as M, sim
from .analytic import osr_full_design
from .designs import STUDY

OUT = STUDY / "results" / "friction"
MUS = (0.0, 0.1, 0.2, 0.3)


def bodies() -> list[D.Design]:
    return [*D.length_grid(), osr_full_design()]


def bend_centroid(q: np.ndarray, s_frac: np.ndarray) -> float:
    w = np.abs(q)
    return float((w * s_frac).sum() / max(w.sum(), 1e-12))


def load_unload(design: D.Design, mu: float, t_ramp: float = 6.0, log_hz: float = 100.0) -> dict:
    sm = M.make(design, f_max=1e3, tag=f"fric_{mu}")
    m, d = sm.load()
    mujoco.mj_forward(m, d)
    f0 = sim.f_pack_estimate(sm)
    f_end = 1.25 * f0
    cf = CableFriction(m, mu=mu)
    qi = sim.qpos_index(m, sm)
    dt = m.opt.timestep
    n_r, n_h = int(t_ramp / dt), int(0.5 / dt)
    every = int(round(1 / (log_hz * dt)))
    lim = math.radians(design.dtheta_deg)
    log = {k: [] for k in ("t", "F", "q", "tip")}
    for k in range(2 * n_r + n_h):
        if k < n_r:
            F = f_end * k / n_r
        elif k < n_r + n_h:
            F = f_end
        else:
            F = f_end * (1 - (k - n_r - n_h) / n_r)
        d.ctrl[:] = 0.0
        cf.apply(d, tension=[0.0, F])
        mujoco.mj_step(m, d)
        if k % every == 0:
            log["t"].append(k * dt); log["F"].append(F)
            log["q"].append(d.qpos[qi].copy()); log["tip"].append(d.site("tip").xpos[:2].copy())
    F = np.array(log["F"]); q = np.array(log["q"]); tip = np.array(log["tip"])
    n_load = int(n_r / every)
    s_frac = sm.joint_s_m / sm.length_m
    lock = np.full(q.shape[1], np.nan)
    for i in range(q.shape[1]):
        idx = np.nonzero(np.abs(q[:n_load + 1, i]) >= 0.9 * lim)[0]
        if len(idx):
            lock[i] = F[idx[0]]
    turn = np.degrees(np.abs(q[:n_load + 1].sum(1)))
    cent = {}
    for th in (90, 180, 360):
        idx = np.nonzero(turn >= th)[0]
        cent[th] = bend_centroid(q[idx[0]], s_frac) if len(idx) else None
    # hysteresis: tip distance between loading and unloading at the same force
    hyst = []
    for frac in (0.25, 0.5, 0.75):
        i_up = int(frac * n_load)
        i_dn = len(F) - 1 - int(frac * n_load)
        hyst.append(float(np.linalg.norm(tip[i_up] - tip[i_dn]) * 1e3))
    peak = n_load + int(0.5 / dt / every) - 1
    locked_at_peak = float(np.mean(np.abs(q[peak]) >= 0.9 * lim))
    res = dict(design=design.name, L=design.L, phi_deg=design.phi_deg, mu=mu, n_joints=len(lock),
               length_m=sm.length_m, base_width_mm=float(sm.joint_width_m[0] * 1e3),
               f_pack0=f0, f_end=f_end, lock_F=lock.tolist(), s_frac=s_frac.tolist(),
               centroid_90=cent[90], centroid_180=cent[180], centroid_360=cent[360],
               locked_frac_peak=locked_at_peak, hysteresis_mm=hyst,
               residual_turn_deg=float(abs(np.degrees(q[-1].sum()))))
    np.savez_compressed(OUT / f"{design.name}__mu{mu}.npz", F=F, q=q, tip=tip)
    return res


def workspace_settled(design: D.Design, mu: float, n_targets: int = 120, seed: int = 3) -> dict:
    """Quasi-static reachable set: ramp to a random force pair in 0.4 s, hold
    0.4 s, record where the tip settled. Without friction the settled shape
    depends (almost) only on the force difference, so the points fall on one
    curve; friction makes the shape depend on the path, which spreads them."""
    sm = M.make(design, f_max=1e3, tag=f"wss_{mu}")
    m, d = sm.load()
    mujoco.mj_forward(m, d)
    f0 = sim.f_pack_estimate(sm)
    cf = CableFriction(m, mu=mu)
    rng = np.random.default_rng(seed)
    dt = m.opt.timestep
    n_ramp, n_hold = int(0.4 / dt), int(0.4 / dt)
    targets = rng.uniform(0, 1.25 * f0, (n_targets, 2))
    prev = np.zeros(2)
    tips, forces = [], []
    for tgt in targets:
        for k in range(n_ramp + n_hold):
            ten = prev + (tgt - prev) * min(1.0, k / n_ramp)
            cf.apply(d, tension=list(ten))
            mujoco.mj_step(m, d)
        tips.append(d.site("tip").xpos[:2].copy()); forces.append(tgt.copy())
        prev = tgt
    tips = np.array(tips)
    np.savez_compressed(OUT / f"wss_{design.name}__mu{mu}.npz", tips=tips, forces=np.array(forces))
    # spread of settled tips around the frictionless one-parameter curve:
    # nearest distance from each settled tip to the mu = 0 set is computed in figures
    return dict(design=design.name, L=design.L, phi_deg=design.phi_deg, mu=mu, n=len(tips), f_pack0=f0)


def workspace(design: D.Design, mu: float, t_total: float = 40.0, seed: int = 3) -> dict:
    sm = M.make(design, f_max=1e3, tag=f"ws_{mu}")
    m, d = sm.load()
    mujoco.mj_forward(m, d)
    f0 = sim.f_pack_estimate(sm)
    cf = CableFriction(m, mu=mu)
    rng = np.random.default_rng(seed)
    dt = m.opt.timestep
    seg = int(0.5 / dt)
    n_seg = int(t_total / 0.5)
    targets = rng.uniform(0, 1.25 * f0, (n_seg + 1, 2))
    targets[0] = 0
    tips = []
    for j in range(n_seg):
        a, b = targets[j], targets[j + 1]
        for k in range(seg):
            u = k / seg
            ten = (1 - u) * a + u * b
            cf.apply(d, tension=list(ten))
            mujoco.mj_step(m, d)
            if k % 50 == 0:
                tips.append(d.site("tip").xpos[:2].copy())
    tips = np.array(tips)
    cells = {tuple(c) for c in np.floor(tips / 0.005).astype(int)}
    np.savez_compressed(OUT / f"ws_{design.name}__mu{mu}.npz", tips=tips)
    return dict(design=design.name, L=design.L, phi_deg=design.phi_deg, mu=mu,
                cells=len(cells), area_cm2=len(cells) * 0.25, f_pack0=f0)


def _lu(a):
    return load_unload(*a)


def _ws(a):
    return workspace(*a)


def _wss(a):
    return workspace_settled(*a)


def main() -> None:
    from multiprocessing import Pool
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(b, mu) for b in bodies() for mu in MUS]
    with Pool(2) as p:
        res = p.map(_lu, jobs, chunksize=1)
    (OUT / "load_unload.json").write_text(json.dumps(res, indent=1))


def main_ws() -> None:
    from multiprocessing import Pool
    OUT.mkdir(parents=True, exist_ok=True)
    sel = [D.length_design(L, p) for L in (0.22628, 0.45) for p in (6.3, 8.0, 10.0)] + [osr_full_design()]
    jobs = [(b, mu) for b in sel for mu in (0.0, 0.2)]
    with Pool(2) as p:
        res = p.map(_ws, jobs, chunksize=1)
    (OUT / "workspace.json").write_text(json.dumps(res, indent=1))


def main_wss() -> None:
    from multiprocessing import Pool
    sel = [D.length_design(L, p) for L in (0.22628, 0.45) for p in (6.3, 8.0, 10.0)] + [osr_full_design()]
    jobs = [(b, mu) for b in sel for mu in (0.0, 0.1, 0.2, 0.3)]
    with Pool(2) as p:
        res = p.map(_wss, jobs, chunksize=1)
    (OUT / "workspace_settled.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    {"ws": main_ws, "wss": main_wss}.get(arg, main)()
