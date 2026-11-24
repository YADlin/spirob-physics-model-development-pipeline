"""Part 3, experiment P2 - wrap and carry the 45 mm egg under each cable actuation.

Same grid, egg, release-then-move test and success measures as part 2
(exp_wrap / wrap.py), now with the four ways to drive the cables in
actuation.py. Every mode runs the same programme (the authors' reach-then-wrap,
written as cable tensions and translated to lengths through each body's
free-curl curve), so differences come from the actuator, not the programme.

Groups
  main     6 bodies x {force, position, capped, admittance} x egg in a holder
           (mu = 0.2); egg on its pedestal (mu = 0.2) for 3 of them
  nofric   3 bodies x {force, position, capped} x holder, mu = 0
  mjlab    current robot and 45 cm 6.3 deg, position servo with the
           spirob-mjlab gains (2.5 N/mm), holder and pedestal, mu = 0.2
  hybrid2  6 bodies x hybrids v2 (each cable commanded with its own length
           and force limit, so co-contraction is possible) x holder, mu = 0.2
  dtcheck  45 cm 6.3 deg, position and capped, holder mu = 0.2, timestep
           halved (1e-4 s), to check that the 2e-4 s results are not a
           time-step artefact
"""
from __future__ import annotations

import json
import os
from multiprocessing import Pool

import numpy as np

from . import actuation as A, designs as D, wrap as W
from .analytic import osr_full_design
from . import model as M
from .exp_wrap import EGG_D, TASK_EGG

OUT = A.OUT / "wrap"
CRUSH_N = 30.0            # N, hen egg quasi-static breaking force ~32 N (Koprivica et al. 2021)


def bodies() -> list[D.Design]:
    return [D.sweep_by_phi()[6.3], D.sweep_by_phi()[8.0], D.length_design(0.30, 6.3),
            D.length_design(0.375, 6.3), D.length_design(0.45, 6.3), osr_full_design()]


PEDESTAL_BODIES = ("phi6p3", "L450_phi6p3", "authors_45cm")


def grid(design, diam: float = EGG_D) -> list[tuple[float, float]]:
    """Part 2's grid narrowed to where any held-egg trial got the egg into the
    curl (x 0.08-0.30 L, y 0.37-0.73 L in part 2), plus a margin."""
    sm = M.make(design, tag=f"grid3_{os.getpid()}")
    L = sm.length_m
    pts = []
    for fx in (0.10, 0.13, 0.16, 0.20, 0.30):
        for fy in (0.40, 0.55, 0.70, 0.85):
            x, y = fx * L, fy * L
            w = float(np.interp(y, sm.joint_s_m, sm.joint_width_m)) if y <= L else 0.0
            if x - diam / 2 < w / 2 + 0.002:
                continue
            pts.append((round(x, 4), round(y, 4)))
    return pts


def _gains(mode: str) -> A.Gains:
    return A.Gains(mode)


def jobs(group: str, b: D.Design) -> list[W.Trial]:
    st = A.stroke(b)
    pts = grid(b) + [TASK_EGG]
    out = []
    if group == "main":
        for mode in A.MODES:
            for anc in ((True, False) if b.name in PEDESTAL_BODIES else (True,)):
                for p in pts:
                    out.append(W.Trial(b, p, EGG_D, "reach", 0.2, anc, st.f_pack, gains=_gains(mode),
                                       extra=dict(group=group)))
    elif group == "nofric":
        if b.name in ("phi6p3", "L450_phi6p3", "authors_45cm"):
            for mode in ("force", "position", "capped"):
                for p in pts:
                    out.append(W.Trial(b, p, EGG_D, "reach", 0.0, True, st.f_pack, gains=_gains(mode),
                                       extra=dict(group=group)))
    elif group == "mjlab":
        if b.name in ("phi6p3", "L450_phi6p3"):
            for anc in (True, False):
                for p in pts:
                    out.append(W.Trial(b, p, EGG_D, "reach", 0.2, anc, st.f_pack,
                                       gains=A.Gains("position", kp=A.KP_MJLAB), extra=dict(group=group)))
    elif group == "hybrid2":
        for g in (A.Gains("capped", per_cable=True), A.Gains("admittance", per_cable=True)):
            for p in pts:
                out.append(W.Trial(b, p, EGG_D, "reach", 0.2, True, st.f_pack, gains=g,
                                   extra=dict(group=group)))
    elif group == "dtcheck":
        if b.name == "L450_phi6p3":
            for mode in ("position", "capped"):
                for p in pts:
                    out.append(W.Trial(b, p, EGG_D, "reach", 0.2, True, st.f_pack, gains=_gains(mode),
                                       timestep=1e-4, extra=dict(group=group)))
    return out


def _run(tr: W.Trial) -> dict:
    try:
        r = W.run(tr)
    except Exception as e:
        r = dict(design=tr.design.name, x=tr.pos[0], y=tr.pos[1], mode=W._mode(tr), mu=tr.mu,
                 anchored=tr.anchored, error=repr(e), wrap=False, grasp=False, in_curl=False)
    r.pop("frames", None); r.pop("xml_text", None); r.pop("trace", None)
    r.update(tr.extra)
    r["crush_risk"] = bool(r.get("sustained_squeeze_N", 0.0) > CRUSH_N)
    return r


def main(group: str = "main") -> None:
    part = OUT / group
    part.mkdir(parents=True, exist_ok=True)
    for b in bodies():
        A.stroke(b)                      # cache strokes before forking
    with Pool(2) as pool:
        for b in bodies():
            f = part / f"{b.name}.json"
            if f.exists():
                continue
            js = jobs(group, b)
            if not js:
                continue
            res = pool.map(_run, js, chunksize=1)
            f.write_text(json.dumps(res, indent=1))
            print(group, "done", b.name, len(res), flush=True)
    res = []
    for b in bodies():
        f = part / f"{b.name}.json"
        if f.exists():
            res += json.loads(f.read_text())
    (OUT / f"{group}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    import sys
    for g in (sys.argv[1:] or ["main", "nofric", "mjlab", "dtcheck", "hybrid2"]):
        main(g)
