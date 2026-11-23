"""Experiment 1 - how each body curls under one cable.

A single cable is pulled quasi-statically from 0 to 1.25 x the analytic
packing force. For every joint the force at which it reaches 90 % of its
limit (its "lock force") is recorded. The lock forces describe the whole
curling sequence: their order says where curling starts, their spread says
how gradually it proceeds, their maximum is the force needed to pack the body.
"""
from __future__ import annotations

import json
import math

import numpy as np

from . import analytic, designs as D, model as M, sim
from .designs import STUDY

OUT = STUDY / "results" / "curl"
CABLE = 1          # +X side; the body curls toward +X


def lock_forces(log: dict, lim: float, frac: float = 0.9) -> np.ndarray:
    q = np.abs(log["q"]) / lim
    F = log["F"]
    out = np.full(q.shape[1], np.nan)
    for i in range(q.shape[1]):
        idx = np.nonzero(q[:, i] >= frac)[0]
        if len(idx):
            out[i] = F[idx[0]]
    return out


def heading_change(line: np.ndarray) -> float:
    """Total turning angle of the centreline (deg), base -> tip."""
    v = np.diff(line, axis=0)
    ang = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    return math.degrees(abs(ang[-1] - ang[0]))


def run_one(design: D.Design, law: str = "cubic", tag: str | None = None) -> dict:
    sm = M.make(design, law=law, f_max=200.0, tag=f"curl_{law}")
    f_est = sim.f_pack_estimate(sm)
    log = sim.ramp_single_cable(sm, CABLE, 1.25 * f_est)
    lim = math.radians(design.dtheta_deg)
    lf = lock_forces(log, lim)
    i_pack = np.nonzero(np.all(np.abs(log["q"]) >= 0.9 * lim, axis=1))[0]
    f_pack = float(log["F"][i_pack[0]]) if len(i_pack) else float("nan")
    f_first = float(np.nanmin(lf))
    stroke = float(log["ten"][0, CABLE] - log["ten"][-1, CABLE])
    # shape uniformity when the tip has turned through 180 deg
    turn = np.array([heading_change(l) for l in log["line"]])
    k180 = int(np.argmax(turn >= 180.0)) if np.any(turn >= 180.0) else len(turn) - 1
    q180 = np.abs(log["q"][k180]) / lim
    s_mid = sm.joint_s_m
    res = dict(
        design=design.name, phi_deg=design.phi_deg, law=law, group=design.group,
        n_joints=len(sm.joint_names), length_m=sm.length_m, mass_kg=sm.mass_kg,
        k_base=float(sm.stiffness[0]), k_tip=float(sm.stiffness[-1]),
        f_est=f_est, f_first=f_first, f_pack=f_pack,
        force_range=f_pack / f_first, stroke_m=stroke,
        lock_force_sim=lf.tolist(), lock_force_analytic=sim.lock_forces_analytic(sm).tolist(),
        joint_s_m=s_mid.tolist(), joint_width_m=sm.joint_width_m.tolist(),
        stiffness=sm.stiffness.tolist(),
        f_at_180=float(log["F"][k180]),
        locked_fraction_at_180=float(np.mean(q180 >= 0.9)),
        q_at_180=q180.tolist(),
        packed_turn_deg=float(turn[-1]),
        spiral=analytic.spiral(design).__dict__,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    name = tag or f"{design.name}__{law}"
    np.savez_compressed(OUT / f"{name}.npz", **log)
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
    return res


def all_designs() -> list[D.Design]:
    return [D.CURRENT, *D.SWEEP, *D.PAPER, analytic.osr_print_design()]


def main() -> list[dict]:
    from multiprocessing import Pool
    jobs = [(d, "cubic") for d in all_designs()]
    # stiffness-law sensitivity on the current robot and the paper's 10 deg body
    for name in ("current_6p3", "phi10p0"):
        d = next(x for x in all_designs() if x.name == name)
        jobs += [(d, "quadratic"), (d, "mjlab")]
    with Pool(2) as p:
        res = p.starmap(run_one, jobs)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    for r in main():
        print(f"{r['design']:16s} {r['law']:9s} F_first {r['f_first']:6.2f}  F_pack {r['f_pack']:6.2f} "
              f"(est {r['f_est']:6.2f})  range {r['force_range']:5.1f}  stroke {r['stroke_m']*1e3:5.1f} mm  "
              f"mass {r['mass_kg']*1e3:5.0f} g  locked@180 {r['locked_fraction_at_180']:.2f}")
