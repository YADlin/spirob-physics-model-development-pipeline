"""Experiment 5 - does the intended curling survive 3D printing?

The flexure (elastic core) between units is 5 % of the local width. In the
printed part its bending thickness is built from whole layers, so the
designed thickness t = 0.05 W is rounded to n * h (layer height h, at least
two layers). Because joint stiffness goes as t^3, rounding changes the
stiffness profile that makes the body curl tip-first. The project's slicer
audit (Notion, Physics Equivalence section 14) found exactly this on the
current print: 1.40 -> 0.40 mm with an eight-joint 0.60 mm plateau.

For every taper and layer height the quantised profile is simulated with the
same single-cable ramp as Experiment 1.
"""
from __future__ import annotations

import json
import math

import numpy as np

from . import designs as D, model as M, sim
from .designs import STUDY
from .exp_curl import CABLE, lock_forces

OUT = STUDY / "results" / "print"
CORE = 0.05
MIN_LAYERS = 2
LAYERS = (None, 0.20, 0.12)


def quantised(t: np.ndarray, h_mm: float | None) -> np.ndarray:
    """Round thickness t (m) to whole layers of h_mm (mm), at least MIN_LAYERS."""
    if h_mm is None:
        return t
    h = h_mm * 1e-3
    return np.maximum(MIN_LAYERS, np.round(t / h)) * h


def distal_share_at(s_joint: np.ndarray, q: np.ndarray, frac: float = 1 / 3) -> float:
    """Share of the total bending carried by the distal ``frac`` of the length."""
    L = s_joint[-1]
    distal = s_joint >= (1 - frac) * L
    tot = np.sum(np.abs(q))
    return float(np.sum(np.abs(q[distal])) / tot) if tot > 0 else float("nan")


def curl_metrics(log: dict, sm: M.StudyModel) -> dict:
    lim = math.radians(sm.design.dtheta_deg)
    lf = lock_forces(log, lim)
    # total turning of the body = sum of joint angles (planar chain)
    turn = np.degrees(np.abs(log["q"]).sum(axis=1))
    k90 = int(np.argmax(turn >= 90.0))
    k180 = int(np.argmax(turn >= 180.0))
    return dict(
        lock_force=lf.tolist(),
        f_first=float(np.nanmin(lf)), f_last=float(np.nanmax(lf)),
        # lock forces should fall from base to tip; a "reversal" is a joint
        # needing >= 1 % more force than its proximal neighbour (base joint
        # excluded: the packed coil presses on the mount)
        order_reversals=int(np.sum(lf[2:] > 1.01 * lf[1:-1])),
        distal_share_at_90=distal_share_at(sm.joint_s_m, log["q"][k90]),
        distal_share_at_180=distal_share_at(sm.joint_s_m, log["q"][k180]),
        f_at_90=float(log["F"][k90]), f_at_180=float(log["F"][k180]),
    )


def run_one(design: D.Design, h: float | None) -> dict:
    fac = lambda w: (quantised(CORE * w, h) / (CORE * w)) ** 3
    sm = M.make(design, f_max=200.0, tag=f"print_{h}", k_factor=fac)
    f_est = sim.f_pack_estimate(sm)
    log = sim.ramp_single_cable(sm, CABLE, 1.25 * f_est, t_ramp=6.0)
    t = CORE * sm.joint_width_m
    tq = quantised(t, h)
    res = dict(design=design.name, phi_deg=design.phi_deg, layer_mm=h,
               neck_ideal_mm=(t * 1e3).tolist(), neck_printed_mm=(tq * 1e3).tolist(),
               distinct_thicknesses=int(len(np.unique(np.round(tq, 6)))),
               stiffness=sm.stiffness.tolist(), joint_s_m=sm.joint_s_m.tolist(),
               f_est=f_est, **curl_metrics(log, sm))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{design.name}__h{h}.npz", **log)
    return res


def law_metrics() -> list[dict]:
    """Same metrics for the stiffness-law runs of Experiment 1."""
    out = []
    for name in ("current_6p3", "phi10p0"):
        for law in ("cubic", "quadratic", "mjlab"):
            log = dict(np.load(STUDY / "results" / "curl" / f"{name}__{law}.npz"))
            d = next(x for x in [D.CURRENT, *D.SWEEP] if x.name == name)
            sm = M.make(d, law=law, tag=f"curl_{law}")
            out.append(dict(design=name, law=law, **curl_metrics(log, sm)))
    return out


def main() -> dict:
    from multiprocessing import Pool
    jobs = [(d, h) for d in (D.CURRENT, *D.SWEEP) for h in LAYERS]
    with Pool(2) as p:
        res = p.starmap(run_one, jobs)
    out = dict(print=res, laws=law_metrics())
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    o = main()
    for r in o["print"]:
        print(f"{r['design']:12s} h={r['layer_mm']}  distinct {r['distinct_thicknesses']:2d}  reversals {r['order_reversals']:2d} "
              f"distal@180 {r['distal_share_at_180']:.2f}  F_first {r['f_first']:.2f} F_last {r['f_last']:.2f}  "
              f"neck {r['neck_printed_mm'][0]:.2f}->{r['neck_printed_mm'][-1]:.2f}")
    for r in o["laws"]:
        print(f"law {r['design']:12s} {r['law']:9s} distal@180 {r['distal_share_at_180']:.2f} reversals {r['order_reversals']}")


# ---------------------------------------------------------------- friction
MUS = (0.0, 0.1, 0.2, 0.3)


def run_friction(design: D.Design, h: float | None, mu: float) -> dict:
    fac = lambda w: (quantised(CORE * w, h) / (CORE * w)) ** 3
    sm = M.make(design, f_max=200.0, tag=f"fric_{h}_{mu}", k_factor=fac)
    f_est = sim.f_pack_estimate(sm)
    log = sim.ramp_single_cable(sm, CABLE, 1.25 * f_est, t_ramp=6.0, mu=mu)
    np.savez_compressed(OUT / f"fric_{design.name}__h{h}__mu{mu}.npz", **log)
    return dict(design=design.name, phi_deg=design.phi_deg, layer_mm=h, mu=mu, f_est=f_est,
                **curl_metrics(log, sm))


def main_friction() -> list[dict]:
    from multiprocessing import Pool
    ds = [D.sweep_by_phi()[p] for p in (4.0, 5.0, 6.3, 8.0, 10.0, 12.0, 15.0)]
    jobs = [(d, h, mu) for d in ds for h in (None, 0.20) for mu in MUS]
    with Pool(2) as p:
        res = p.starmap(run_friction, jobs)
    (OUT / "friction.json").write_text(json.dumps(res, indent=1))
    return res
