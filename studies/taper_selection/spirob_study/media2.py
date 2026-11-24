"""Animations for part 2: every reported wrap or grasp has a video.

Each tile is one trial rendered from the overhead camera, with a live caption:
the cable forces and the wrap angle measured on that very frame (one unbroken
contact stretch, wrap.wrap_angle). Tiles play in sync; a finished trial holds
its last frame.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

from . import sim, wrap as W
from .media import _caption, _save_gif


def _links(m):
    names = sorted(m.body(i).name for i in range(m.nbody) if m.body(i).name.startswith("link_"))
    out = []
    for n in names:
        bid = m.body(n).id
        out.append(np.array([g for g in range(m.ngeom)
                             if m.geom_bodyid[g] == bid and m.geom(g).name.startswith("collision_")]))
    return out


def flatten_object(m) -> None:
    """Draw the object as a thin disc at the body's height. The simulated
    cylinder is 120 mm tall; seen from the overhead camera its top face is
    magnified and shifted by perspective, which makes it look misplaced
    relative to the body. Rendering only."""
    g = m.geom("obj_geom").id
    m.geom_size[g, 1] = 0.004


def render_trials(results: list[dict], labels: list[str], name: str, fps: int = 15,
                  size: int = 300, scale: float = 0.85) -> None:
    sim.gl_env()
    tiles = []
    for r, lab in zip(results, labels):
        xp = os.path.join(r["build_dir"], f"render_{name}.xml")
        with open(xp, "w") as f:
            f.write(r["xml_text"])
        m = mujoco.MjModel.from_xml_path(xp)
        d = mujoco.MjData(m)
        flatten_object(m)
        L = r["L"]
        tv = sim.TopView(m, size, size, center=(0.2 * L, 0.45 * L), extent=1.35 * L)
        links = _links(m)
        bid = m.body("obj").id
        R = r["d_mm"] / 2000
        seq = []
        step = max(1, int(round(1.0 / (fps * 0.02))))
        for (t, q, ten, ph) in r["frames"][::step]:
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            w = W.wrap_angle(m, d, links, d.xpos[bid, :2], R)["wrap_deg"]
            cu = W.curl_angle(m, d, d.xpos[bid, :2])
            phase = {"pack": "curl away", "reach": "reach + wrap", "wrap": "pull",
                     "release": "holder lets go"}.get(ph, ph.replace("carry:", "move: "))
            seq.append(_caption(tv(d), [lab, f"t {t:4.1f} s   {phase}",
                                        f"cable +x {ten[1]:4.1f} N   -x {ten[0]:4.1f} N",
                                        f"wrap {w:4.0f}° (≥180)   curl {cu:4.0f}° (≥270)"]))
        tiles.append(seq)
    n = max(len(s) for s in tiles) + fps
    frames = [np.concatenate([s[min(i, len(s) - 1)] for s in tiles], axis=1) for i in range(n)]
    _save_gif(frames, name, fps=fps, scale=scale)
