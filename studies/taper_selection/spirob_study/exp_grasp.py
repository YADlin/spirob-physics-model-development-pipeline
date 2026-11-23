"""Experiment 3 - what each body can wrap, and where.

3a  Size-position map. A free cylinder (diameter d) rests beside the straight
    body, touching it at arc length s from the base. Its planar motion is
    resisted by a small "pedestal" friction. The cable on the object side is
    pulled to 1.25 x the packing force (the simplest grasp: no strategy). The
    trial succeeds when
      * the body covers >= 180 deg of the object's circumference (paper's
        wrapping criterion, Note S2), and
      * the object resists an in-plane pull of HOLD_TEST_N in each of four
        directions without moving more than a quarter of its diameter.
    The paper's position-dependent bound (Eq. 15 with the free length L - s)
    is the theoretical reference.

3b  Egg map. The same trial with d = 40 mm (the egg cap in spirob-mjlab) on a
    grid of positions around the robot, giving the region in which the egg
    can be picked by a single-cable pull (cf. the paper's Fig. 3C).
"""
from __future__ import annotations

import json
import math
import os

import mujoco
import numpy as np

from . import analytic, designs as D, model as M, sim
from .designs import STUDY

OUT = STUDY / "results" / "grasp"
CABLE = 1                    # +X cable, curls toward the object
PEDESTAL_FRICTION = 0.15     # N, ~0.25 x the weight of a 60 g egg
HOLD_TEST_N = 1.0            # N, in-plane pull used to test the grasp
S_FRACS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DIAMS_MM = (10, 20, 30, 40, 50, 60, 70, 80)
EGG_D = 0.040


def width_at(sm: M.StudyModel, s: float) -> float:
    return float(np.interp(s, sm.joint_s_m, sm.joint_width_m))


def trial(design: D.Design, pos: tuple[float, float], diam: float, f_cable: float,
          t_ramp: float = 4.0, t_hold: float = 0.5, keep_frames: bool = False, law: str = "cubic") -> dict:
    R = diam / 2
    sm = M.make(design, law=law, f_max=2 * f_cable, tag=f"grasp_{os.getpid()}",
                objects=[dict(name="obj", pos=pos, radius=R, free=True)])
    m, d = sm.load()
    for j in ("obj_x", "obj_y"):
        m.dof_frictionloss[m.joint(j).dofadr[0]] = PEDESTAL_FRICTION
    dt = m.opt.timestep
    gid, bid = m.geom("obj_geom").id, m.body("obj").id
    n_r, n_h = int(t_ramp / dt), int(t_hold / dt)
    frames = []
    for k in range(n_r + n_h):
        d.ctrl[:] = 0.0
        d.ctrl[CABLE] = -f_cable * min(1.0, k / n_r)
        mujoco.mj_step(m, d)
        if keep_frames and k % int(0.02 / dt) == 0:
            frames.append((d.qpos.copy(), d.ctrl.copy()))
    c = d.xpos[bid, :2].copy()
    enc = sim.enclosure_angle(m, d, c, R)
    grip = sim.normal_force_on(m, d, gid)
    # in-plane hold test: constant pull, four directions
    qpos0, ctrl0 = d.qpos.copy(), d.ctrl.copy()
    worst_disp = 0.0
    for ang in (0.0, 90.0, 180.0, 270.0):
        d2 = mujoco.MjData(m)
        d2.qpos[:] = qpos0
        d2.ctrl[:] = ctrl0
        mujoco.mj_forward(m, d2)
        u = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
        n_t = int(0.6 / dt)
        for k in range(n_t):
            d2.xfrc_applied[bid, :2] = HOLD_TEST_N * min(1.0, k / (0.3 / dt)) * u
            mujoco.mj_step(m, d2)
        worst_disp = max(worst_disp, float(np.linalg.norm(d2.xpos[bid, :2] - c)))
    held = worst_disp < diam / 4
    out = dict(design=design.name, phi_deg=design.phi_deg, law=law, x=pos[0], y=pos[1], d_mm=diam * 1e3,
               f_cable=f_cable, enclosure_deg=enc, grip_N=grip, hold_disp_mm=worst_disp * 1e3,
               obj_travel_mm=float(np.linalg.norm(c - np.array(pos)) * 1e3),
               final_xy=c.tolist(), success=bool(enc >= 180.0 and held))
    if keep_frames:
        out["frames"] = frames
        out["xml"] = str(sm.xml_path)
    return out


def size_position_jobs(design: D.Design, f_cable: float, diams=DIAMS_MM) -> list[tuple]:
    base = M.make(design, tag="grasp_geom")
    jobs = []
    for frac in S_FRACS:
        s = frac * base.length_m
        for dmm in diams:
            R = dmm / 2000
            x = width_at(base, s) / 2 + R + 0.001
            jobs.append((design, (x, s), dmm / 1000, f_cable, frac))
    return jobs


def _run_sp(args):
    design, pos, diam, f_cable, frac, *rest = args
    r = trial(design, pos, diam, f_cable, law=rest[0] if rest else "cubic")
    r["s_frac"] = frac
    r["s_m"] = pos[1]
    return r


def egg_jobs(design: D.Design, f_cable: float) -> list[tuple]:
    base = M.make(design, tag="grasp_geom")
    jobs = []
    for x in np.arange(0.02, 0.141, 0.02):
        for y in np.arange(-0.02, 0.241, 0.02):
            # skip positions overlapping the straight body or its base mount
            if 0 <= y <= base.length_m and x - EGG_D / 2 < width_at(base, y) / 2 + 0.001:
                continue
            if y < 0 and x - EGG_D / 2 < 0.02:
                continue
            jobs.append((design, (float(x), float(y)), EGG_D, f_cable, None))
    jobs.append((design, (0.05, 0.076), EGG_D, f_cable, "task"))
    return jobs


def main(which: str = "both") -> None:
    from multiprocessing import Pool
    OUT.mkdir(parents=True, exist_ok=True)
    curl = {r["design"]: r for r in json.loads((STUDY / "results" / "curl" / "summary.json").read_text())
            if r["law"] == "cubic"}
    designs = [*D.SWEEP]
    if which in ("both", "size"):
        jobs = []
        for d in designs:
            jobs += size_position_jobs(d, 1.25 * curl[d.name]["f_est"])
        with Pool(2) as p:
            res = p.map(_run_sp, jobs, chunksize=4)
        (OUT / "size_position.json").write_text(json.dumps(res, indent=1))
    if which in ("both", "egg"):
        jobs = []
        for d in designs:
            jobs += egg_jobs(d, 1.25 * curl[d.name]["f_est"])
        with Pool(2) as p:
            res = p.map(_run_sp, jobs, chunksize=4)
        (OUT / "egg_map.json").write_text(json.dumps(res, indent=1))


def extra() -> None:
    """The authors' print (the body NYK holds) and the current robot under the
    stiffness law used in spirob-mjlab."""
    from multiprocessing import Pool
    from .analytic import osr_print_design
    curl = {r["design"]: r for r in json.loads((STUDY / "results" / "curl" / "summary.json").read_text())
            if r["law"] == "cubic"}
    osr = osr_print_design()
    jobs = size_position_jobs(osr, 1.25 * curl[osr.name]["f_est"]) + egg_jobs(osr, 1.25 * curl[osr.name]["f_est"])
    mj = [j + ("mjlab",) for j in size_position_jobs(D.CURRENT, 1.25 * curl[D.CURRENT.name]["f_est"])]
    with Pool(2) as p:
        res = p.map(_run_sp, jobs + mj, chunksize=4)
    (OUT / "extra.json").write_text(json.dumps(res, indent=1))


def refine() -> None:
    """A real hen's egg is 42-45 mm across: test 45 mm on the candidate bodies."""
    from multiprocessing import Pool
    from .analytic import osr_print_design
    curl = {r["design"]: r for r in json.loads((STUDY / "results" / "curl" / "summary.json").read_text())
            if r["law"] == "cubic"}
    jobs = []
    for d in (*[D.sweep_by_phi()[p] for p in (5.0, 6.3, 8.0, 10.0)], osr_print_design()):
        jobs += size_position_jobs(d, 1.25 * curl[d.name]["f_est"], diams=(35, 45))
    with Pool(2) as p:
        res = p.map(_run_sp, jobs, chunksize=4)
    (OUT / "refine_35_45.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "extra":
        extra()
    elif len(sys.argv) > 1 and sys.argv[1] == "refine":
        refine()
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "both")
