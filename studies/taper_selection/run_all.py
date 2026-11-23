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
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

from spirob_study import (analytic, designs as D, exp_curl, exp_grasp, exp_print,  # noqa: E402
                          exp_validation, exp_workspace, figures, media, summary)

STEPS = ("build", "curl", "workspace", "grasp", "print", "validate", "figures")


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


if __name__ == "__main__":
    main()
