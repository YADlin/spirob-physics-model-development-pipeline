"""Re-run selected trials with frames and render them (part 2 animations)."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from . import designs as D, model as M, sim, wrap as W
from .analytic import osr_full_design
from .media2 import render_trials


def _des(name):
    return osr_full_design() if name == "authors_45cm" else next(b for b in D.length_grid() if b.name == name)


def trial(name, pos, mu, anchored, strategy="reach", diam=0.045, **kw):
    des = _des(name)
    fp = sim.f_pack_estimate(M.make(des, tag="media_fp"))
    return W.run(W.Trial(des, pos, diam, strategy, mu, anchored, fp, keep_frames=True, **kw))


def anim(name, specs, labels):
    res = [trial(*s[:4], **(s[4] if len(s) > 4 else {})) for s in specs]
    for r, lab in zip(res, labels):
        print(name, lab, round(r["wrap_deg"]), r.get("carry_mm"), r.get("dropped"))
    render_trials(res, labels, name)
    return res


def main(which: str = "all"):
    if which in ("friction", "all"):
        anim("anim4_friction_wrap",
             [("L375_phi6p3", (0.0495, 0.2092), 0.0, True), ("L375_phi6p3", (0.0495, 0.2092), 0.2, True),
              ("L375_phi6p3", (0.0495, 0.2092), 0.2, False)],
             ["37.5 cm 6.3°, egg in holder, no friction", "same, cable friction 0.2", "same, egg free on pedestal"])
    if which in ("strategy", "all"):
        anim("anim5_pull_vs_reach",
             [("L375_phi6p3", (0.0495, 0.2092), 0.2, True, dict(strategy="pull")),
              ("L375_phi6p3", (0.0495, 0.2092), 0.2, True)],
             ["37.5 cm 6.3°, μ 0.2, holder: pull from straight", "same: curl away, then reach + wrap"])
    if which in ("task", "all"):
        anim("anim6_task_egg",
             [("phi6p3", (0.05, 0.076), 0.2, False), ("phi8p0", (0.05, 0.076), 0.2, False),
              ("phi6p3", (0.05, 0.076), 0.2, True)],
             ["22.6 cm 6.3° (current), task egg, μ 0.2", "22.6 cm 8°, task egg, μ 0.2",
              "22.6 cm 6.3°, same egg in a holder"])
    if which in ("carry", "all"):
        anim("anim7_wrap_and_carry",
             [("phi6p3", (0.0447, 0.0895), 0.2, True), ("L450_phi6p3", (0.0584, 0.2469), 0.2, True),
              ("authors_45cm", (0.0742, 0.311), 0.2, True)],
             ["22.6 cm 6.3° (current), μ 0.2", "45 cm 6.3°, μ 0.2", "authors' 45 cm 10°, μ 0.2"])

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
