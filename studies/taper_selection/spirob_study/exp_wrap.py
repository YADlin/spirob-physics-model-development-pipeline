"""Experiment W - where each body can wrap and grasp a 45 mm egg.

Grid trials in the spirit of the paper's Fig. 3C: the object is placed on a
grid in the right half of the workspace (scaled by the body length), and the
body runs a fixed cable programme. Each trial reports the wrap angle (one
unbroken contact stretch, wrapping.py), whether the egg fell off its pedestal,
and, if it was wrapped, how far the cables could carry it while it stayed
wrapped and could carry its own weight.

Factors: body (length x taper, plus the authors' 45 cm robot), cable friction
(mu = 0 or 0.2), object free on a pedestal (the task) or fixed in place (what
the body itself can wrap), programme ("reach", the authors' strategy; "pull",
the naive one, on a subset).
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import designs as D, model as M, sim, wrap as W
from .analytic import osr_full_design
from .designs import STUDY

OUT = STUDY / "results" / "wrap"
EGG_D = 0.045
TASK_EGG = (0.05, 0.076)      # spirob-mjlab: egg relative to the robot base


def bodies() -> list[D.Design]:
    return [*D.length_grid(), osr_full_design()]


def f_pack(design: D.Design) -> float:
    return sim.f_pack_estimate(M.make(design, tag=f"fp_{os.getpid()}"))


def grid(design: D.Design, diam: float = EGG_D) -> list[tuple[float, float]]:
    sm = M.make(design, tag=f"grid_{os.getpid()}")
    L = sm.length_m
    pts = []
    for fx in (0.13, 0.20, 0.30, 0.40, 0.50):
        for fy in (0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00, 1.15):
            x, y = fx * L, fy * L
            w = float(np.interp(y, sm.joint_s_m, sm.joint_width_m)) if y <= L else 0.0
            if x - diam / 2 < w / 2 + 0.002:
                continue            # overlaps the straight body
            pts.append((round(x, 4), round(y, 4)))
    return pts


def _run(tr: W.Trial) -> dict:
    try:
        r = W.run(tr)
    except Exception as e:  # keep the sweep going, record the failure
        r = dict(design=tr.design.name, x=tr.pos[0], y=tr.pos[1], strategy=tr.strategy, mu=tr.mu,
                 anchored=tr.anchored, error=repr(e), wrap=False, grasp=False)
    r.pop("frames", None); r.pop("xml_text", None)
    r.update(tr.extra)
    return r


def jobs_main(only=None) -> list[W.Trial]:
    out = []
    for b in (only or bodies()):
        fp = f_pack(b)
        pts = grid(b) + [TASK_EGG]
        for mu in (0.0, 0.2):
            for anc in (False, True):
                for p in pts:
                    out.append(W.Trial(b, p, EGG_D, "reach", mu, anc, fp))
    return out


def jobs_pull() -> list[W.Trial]:
    out = []
    for b in (D.length_design(0.22628, 8.0), D.length_design(0.45, 8.0), osr_full_design()):
        fp = f_pack(b)
        for mu in (0.0, 0.2):
            for p in grid(b) + [TASK_EGG]:
                out.append(W.Trial(b, p, EGG_D, "pull", mu, False, fp))
    return out


SIZE_BODIES = ("phi8p0", "L450_phi8p0", "authors_45cm")
SIZES_MM = (15, 25, 35, 45, 60)
S_FRACS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def jobs_size(only=None) -> list[W.Trial]:
    """Object size x contact position (the paper's Eq. 15 question), with the
    object touching the straight body at s/L on the +x side."""
    out = []
    for b in (only or [x for x in bodies() if x.name in SIZE_BODIES]):
        sm = M.make(b, tag=f"size_{os.getpid()}")
        fp = sim.f_pack_estimate(sm)
        for dmm in SIZES_MM:
            R = dmm / 2000
            for fr in S_FRACS:
                s = fr * sm.length_m
                x = float(np.interp(s, sm.joint_s_m, sm.joint_width_m)) / 2 + R + 0.001
                for strat in ("pull", "reach"):
                    for mu in (0.0, 0.2):
                        out.append(W.Trial(b, (round(x, 4), round(s, 4)), dmm / 1000, strat, mu, True, fp,
                                           transport=False, extra=dict(s_frac=fr)))
                out.append(W.Trial(b, (round(x, 4), round(s, 4)), dmm / 1000, "reach", 0.2, False, fp,
                                   extra=dict(s_frac=fr)))
    return out


REFINE_BODIES = ("phi6p3", "phi8p0", "L300_phi6p3", "L375_phi6p3", "L450_phi6p3", "L450_phi8p0",
                 "L450_phi10p0", "authors_45cm")


def jobs_refine(only=None) -> list[W.Trial]:
    """3 x 3 positions (step 3 % of L) around each body's best fixed-egg
    position of the main grid (mu = 0.2), at mu = 0.2 and 0.3; free egg at 0.3."""
    main_res = json.loads((OUT / "main.json").read_text())
    out = []
    for b in (only or [x for x in bodies() if x.name in REFINE_BODIES]):
        s = [r for r in main_res if r["design"] == b.name and r["mu"] == 0.2 and r["anchored"]]
        best = max(s, key=lambda r: r["wrap_deg"])
        sm = M.make(b, tag=f"ref_{os.getpid()}")
        fp = sim.f_pack_estimate(sm)
        step = 0.03 * sm.length_m
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                p = (round(best["x"] + dx * step, 4), round(best["y"] + dy * step, 4))
                w = float(np.interp(p[1], sm.joint_s_m, sm.joint_width_m)) if p[1] <= sm.length_m else 0.0
                if p[0] - EGG_D / 2 < w / 2 + 0.002:
                    continue
                for mu, anc in ((0.2, True), (0.3, True), (0.3, False)):
                    out.append(W.Trial(b, p, EGG_D, "reach", mu, anc, fp))
    return out


RECHECK_FREE = ("phi6p3", "phi8p0", "L450_phi6p3", "authors_45cm")


def jobs_recheck(only=None) -> list[W.Trial]:
    """After two model fixes (24 Nov): (a) every fixed-egg case of the main
    grid and the size sweep that wrapped >= 150 deg, re-run with the egg in a
    welded holder that lets go for the carry test (the main grid used a static
    egg, which cannot be carried and which MuJoCo's parent filter hides from
    the first moving link); (b) the free egg on four bodies with the pedestal
    friction made non-creeping."""
    out = []
    main_res = json.loads((OUT / "main.json").read_text())
    size_res = json.loads((OUT / "size.json").read_text())
    for b in (only or bodies()):
        fp = None
        for r in main_res + size_res:
            if r["design"] == b.name and r["anchored"] and r["wrap_deg"] >= 150:
                fp = fp or f_pack(b)
                out.append(W.Trial(b, (r["x"], r["y"]), r["d_mm"] / 1000, r["strategy"], r["mu"], True, fp,
                                   extra=dict(source="size" if "s_frac" in r else "main",
                                              s_frac=r.get("s_frac"), old_wrap=r["wrap_deg"])))
        if b.name in RECHECK_FREE:
            fp = fp or f_pack(b)
            for p in grid(b) + [TASK_EGG]:
                out.append(W.Trial(b, p, EGG_D, "reach", 0.2, False, fp, extra=dict(source="free")))
        if b.name in SIZE_BODIES:      # the size sweep's free-object row
            out += [W.Trial(t.design, t.pos, t.diam, t.strategy, t.mu, False, t.f_pack,
                            extra=dict(source="size_free", s_frac=t.extra["s_frac"]))
                    for t in jobs_size([b]) if not t.anchored]
    return out


def jobs_carry(only=None) -> list[W.Trial]:
    """Final pass (24 Nov) with both success measures (one-stretch wrap and
    object inside the curl) and the release-then-move carry test, for every
    held-egg case that came within reach of success: main grid and size sweep
    with a contact sweep >= 90 deg (after recheck), and all refine cases."""
    main = json.loads((OUT / "main.json").read_text())
    size = json.loads((OUT / "size.json").read_text())
    rech = {(r["design"], r["x"], r["y"], r["d_mm"], r["strategy"], r["mu"]): r
            for r in json.loads((OUT / "recheck.json").read_text()) if r["anchored"]}
    refine = json.loads((OUT / "refine.json").read_text())
    src = []
    for r in main + size:
        if r["anchored"]:
            r = rech.get((r["design"], r["x"], r["y"], r["d_mm"], r["strategy"], r["mu"]), r)
            if r["wrap_deg"] >= 90:
                src.append(r)
    src += [r for r in refine if r["anchored"]]
    out, seen = [], set()
    for b in (only or bodies()):
        for r in src:
            if r["design"] != b.name:
                continue
            key = (r["x"], r["y"], r["d_mm"], r["strategy"], r["mu"])
            if key in seen:
                continue
            seen.add(key)
            out.append(W.Trial(b, (r["x"], r["y"]), r["d_mm"] / 1000, r["strategy"], r["mu"], True, r["f_pack"],
                               extra=dict(s_frac=r.get("s_frac"), first_wrap=r["wrap_deg"],
                                          source="size" if r.get("s_frac") is not None else
                                          ("refine" if r in refine else "main"))))
    return out


def main(which: str = "main") -> None:
    """Results are written per body (results/wrap/<which>/<body>.json) so a
    long sweep can be inspected and resumed; then merged into <which>.json."""
    from multiprocessing import Pool
    part = OUT / which
    part.mkdir(parents=True, exist_ok=True)
    blist = bodies() if which == "main" else (
        [b for b in bodies() if b.name in SIZE_BODIES] if which == "size" else
        [b for b in bodies() if b.name in REFINE_BODIES] if which == "refine" else
        bodies() if which in ("recheck", "carry") else
        [D.length_design(0.22628, 8.0), D.length_design(0.45, 8.0), osr_full_design()])
    with Pool(2) as p:
        for b in blist:
            f = part / f"{b.name}.json"
            if f.exists():
                continue
            jobs = (jobs_main([b]) if which == "main" else jobs_size([b]) if which == "size"
                    else jobs_refine([b]) if which == "refine" else jobs_recheck([b]) if which == "recheck"
                    else jobs_carry([b]) if which == "carry"
                    else [j for j in jobs_pull() if j.design.name == b.name])
            res = p.map(_run, jobs, chunksize=2)
            f.write_text(json.dumps(res, indent=1))
            print("done", b.name, sum(r.get("wrap", False) for r in res), "wraps of", len(res), flush=True)
    res = []
    for b in blist:
        res += json.loads((part / f"{b.name}.json").read_text())
    (OUT / f"{which}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
