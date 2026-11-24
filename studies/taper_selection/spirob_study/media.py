"""Animated GIFs rendered with MuJoCo (overhead camera, like the rig's)."""
from __future__ import annotations

import math

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import designs as D, model as M, sim
from .designs import STUDY

FIG = STUDY / "figures"


def _font(size=15):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _caption(img: np.ndarray, lines: list[str]) -> np.ndarray:
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    f = _font(15)
    y = 8
    for ln in lines:
        dr.text((10, y), ln, fill=(20, 20, 20), font=f)
        y += 19
    return np.asarray(im)


def _save_gif(frames: list[np.ndarray], name: str, fps: int = 20, scale: float = 1.0):
    ims = [Image.fromarray(f) for f in frames]
    if scale != 1.0:
        ims = [im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS) for im in ims]
    pal = [im.convert("P", palette=Image.ADAPTIVE, colors=96) for im in ims]
    FIG.mkdir(parents=True, exist_ok=True)
    pal[0].save(FIG / f"{name}.gif", save_all=True, append_images=pal[1:], duration=int(1000 / fps),
                loop=0, optimize=True)


def curl_same_force(f_end: float = 12.0, t: float = 5.0, fps: int = 15):
    """Four bodies, the same cable force ramp: who curls, where, how easily."""
    sim.gl_env()
    phis = (4.0, 6.3, 10.0, 15.0)
    runs = []
    for phi in phis:
        sm = M.make(D.sweep_by_phi()[phi], f_max=200.0, tag="media")
        m, d = sm.load()
        runs.append((phi, m, d, sim.TopView(m, 300, 340, center=(0.0, 0.10), extent=0.30)))
    dt = runs[0][1].opt.timestep
    n = int(t / dt); every = int(1.0 / (fps * dt))
    frames = []
    for k in range(n + int(1.0 / dt)):
        F = f_end * min(1.0, k / n)
        tiles = []
        for phi, m, d, tv in runs:
            d.ctrl[:] = 0.0; d.ctrl[1] = -F
            mujoco.mj_step(m, d)
            if k % every == 0:
                tiles.append(_caption(tv(d), [f"taper {phi:g}°" + ("  (current)" if phi == 6.3 else ""),
                                              f"cable force {F:4.1f} N"]))
        if tiles:
            frames.append(np.concatenate(tiles, axis=1))
    _save_gif(frames, "anim1_curl_same_force", fps=fps, scale=0.8)


def egg_grasp(phis=(6.3, 10.0, 15.0), fps: int = 15):
    """Single-cable pull with the 40 mm egg cap at the spirob-mjlab task position."""
    sim.gl_env()
    from .exp_grasp import PEDESTAL_FRICTION
    import json
    curl = {r["design"]: r for r in json.loads((STUDY / "results" / "curl" / "summary.json").read_text())
            if r["law"] == "cubic"}
    runs = []
    for phi in phis:
        d0 = D.sweep_by_phi()[phi]
        F = 1.25 * curl[d0.name]["f_est"]
        sm = M.make(d0, f_max=2 * F, tag="media_egg",
                    objects=[dict(name="obj", pos=(0.05, 0.076), radius=0.02, free=True)])
        m, d = sm.load()
        for j in ("obj_x", "obj_y"):
            m.dof_frictionloss[m.joint(j).dofadr[0]] = PEDESTAL_FRICTION
        runs.append((phi, F, m, d, sim.TopView(m, 300, 340, center=(0.02, 0.09), extent=0.28)))
    dt = runs[0][2].opt.timestep
    n = int(4.0 / dt); every = int(1.0 / (fps * dt))
    frames = []
    for k in range(n + int(1.0 / dt)):
        tiles = []
        for phi, F, m, d, tv in runs:
            f = F * min(1.0, k / n)
            d.ctrl[:] = 0.0; d.ctrl[1] = -f
            mujoco.mj_step(m, d)
            if k % every == 0:
                tiles.append(_caption(tv(d), [f"taper {phi:g}°", f"cable {f:4.1f} N of {F:.0f} N"]))
        if tiles:
            frames.append(np.concatenate(tiles, axis=1))
    _save_gif(frames, "anim2_egg_grasp", fps=fps, scale=0.8)


def limits_demo(fps: int = 15):
    """Generator default (+/- pi) vs paper limits (+/- Delta_theta)."""
    sim.gl_env()
    d0 = D.CURRENT
    runs = []
    for lim, lab in (("pipeline", "generator default: ±π"), ("dtheta", "paper: ±Δθ")):
        sm = M.make(d0, limits=lim, f_max=200.0, tag=f"media_{lim}")
        m, d = sm.load()
        runs.append((lab, sim.f_pack_estimate(sm) * 1.25, m, d, sim.TopView(m, 320, 340, center=(0.03, 0.06), extent=0.20)))
    dt = runs[0][2].opt.timestep
    n = int(4.0 / dt); every = int(1.0 / (fps * dt))
    frames = []
    for k in range(n + int(0.6 / dt)):
        tiles = []
        for lab, F, m, d, tv in runs:
            d.ctrl[:] = 0.0; d.ctrl[1] = -F * min(1.0, k / n)
            mujoco.mj_step(m, d)
            if k % every == 0:
                tiles.append(_caption(tv(d), [lab]))
        if tiles:
            frames.append(np.concatenate(tiles, axis=1))
    _save_gif(frames, "anim3_joint_limits", fps=fps, scale=0.8)


def stills():
    """Straight and packed stills of the four key bodies (Fig. 1)."""
    sim.gl_env()
    rows = []
    for packed in (False, True):
        tiles = []
        for phi in (4.0, 6.3, 10.0, 15.0):
            sm = M.make(D.sweep_by_phi()[phi], f_max=200.0, tag="media")
            m, d = sm.load()
            if packed:
                qi = sim.qpos_index(m, sm)
                d.qpos[qi] = math.radians(30.0)
            mujoco.mj_forward(m, d)
            if packed:
                p = sim.hull_points_world(m, d)[:, :2]
                ctr, ext = (p.max(0) + p.min(0)) / 2, 1.25 * (p.max(0) - p.min(0)).max()
            else:
                ctr, ext = (0.0, 0.11), 0.26
            tv = sim.TopView(m, 300, 300, center=tuple(ctr), extent=ext)
            tiles.append(_caption(tv(d), [f"taper {phi:g}°" + ("" if packed else f" | {sm.mass_kg*1e3:.0f} g")]))
        rows.append(np.concatenate(tiles, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(FIG / "fig1_bodies.png")


if __name__ == "__main__":
    stills(); curl_same_force(); limits_demo(); egg_grasp()
