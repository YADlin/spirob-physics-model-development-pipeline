"""Reproduce the whole taper-selection study.

    uv run python studies/taper_selection/run_all.py --osr ../Open-Spiral-Robots

Steps (approximate wall time on 2 CPU cores):
  build      12 models through build.py                       ~2 min
  curl       Experiment 1, single-cable ramps                  ~1 min
  workspace  Experiment 2, envelopes and random forcing        ~5 min
  grasp      Experiment 3, size-position map and egg map       ~45 min
  print      Experiment 5, layer quantisation and friction     ~2 min
  validate   Experiment 4, needs a clone of Open-Spiral-Robots <1 min
  figures    PNG figures, GIF animations and summary table     ~5 min

Part 2 (cable friction, length, wrap and grasp; 24 Nov 2026):
  build2     length grid (4 lengths x 3 tapers) + authors' 45 cm  ~7 min
  friction   load-unload and workspaces with cable friction        ~40 min
  wrap       egg grid, pull baseline, size x position              ~4 h
  figures2   figures 10-14 and animations 4-7                      ~10 min
The grasp step of part 1 used an enclosure metric that counted an object
pinched between two separate parts of the body as wrapped; its results are
superseded by the wrap step (see README).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

from spirob_study import (analytic, designs as D, exp_curl, exp_friction, exp_grasp,  # noqa: E402
                          exp_print, exp_validation, exp_workspace, exp_wrap, figures, figures2, media,
                          media_runs, summary, summary2)

STEPS = ("build", "curl", "workspace", "grasp", "print", "validate", "figures",
         "build2", "friction", "wrap", "figures2")
PART1 = STEPS[:7]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osr", default=str(HERE.parents[2] / "Open-Spiral-Robots"),
                    help="local clone of ZhanchiWang/Open-Spiral-Robots (validation only)")
    ap.add_argument("--only", nargs="*", choices=STEPS, help="run a subset of steps")
    a = ap.parse_args()
    steps = a.only or STEPS
    if "build" in steps:
        for d in (D.CURRENT, *D.SWEEP, *D.PAPER, analytic.osr_print_design()):
            D.build(d)
    if "curl" in steps:
        exp_curl.main()
    if "workspace" in steps:
        exp_workspace.main()
    if "grasp" in steps:
        exp_grasp.main("both")
    if "print" in steps:
        exp_print.main(); exp_print.main_friction()
    if "validate" in steps:
        exp_validation.main(Path(a.osr))
    if "figures" in steps:
        figures.main(); summary.main()
        media.stills(); media.curl_same_force(); media.limits_demo(); media.egg_grasp()
    if "build2" in steps:
        for d in (*D.length_grid(), analytic.osr_full_design()):
            D.build(d)
    if "friction" in steps:
        exp_friction.main(); exp_friction.main_ws(); exp_friction.main_wss()
    if "wrap" in steps:
        for w in ("main", "pull", "size", "refine", "recheck", "carry"):
            exp_wrap.main(w)
    if "figures2" in steps:
        figures2.main(); summary2.table()
        media_runs.main()


if __name__ == "__main__":
    main()
