"""Experiment 2 - where the tip can go.

Three workspaces are computed for every design:

1. Kinematic envelope (paper, Note S2 / Fig. S4A): joints are turned to their
   limit one at a time from the base while the rest stays straight; the tip
   draws a chain of arcs. This is the largest region the body geometry allows.
2. Quasi-static curl path: the tip path when one cable is pulled slowly
   (from Experiment 1). With frictionless cables the equilibrium shape depends
   almost only on the force difference F1 - F0, so this path is the whole
   quasi-static workspace of the model.
3. Dynamic exploration: random, band-limited force pairs (as in the paper's
   Fig. 2E) for 60 s; inertia lets the tip leave the quasi-static path.
"""
from __future__ import annotations

import json
import math

import mujoco
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from . import designs as D, model as M, sim
from .designs import STUDY

OUT = STUDY / "results" / "workspace"
EGG = (0.05, 0.076)       # egg position relative to the robot base, spirob-mjlab Stage 1
BUCKET = (-0.05, 0.076)   # bucket position, mirror side


def kinematic_envelope(sm: M.StudyModel, n_sub: int = 12) -> np.ndarray:
    m, d = sm.load()
    qi = sim.qpos_index(m, sm)
    lim = math.radians(sm.design.dtheta_deg)
    path = []
    for sign in (+1, -1):
        side = []
        q = np.zeros(len(qi))
        for k in range(len(qi)):
            for a in np.linspace(0, lim, n_sub, endpoint=(k == len(qi) - 1)):
                q[k] = sign * a
                d.qpos[qi] = q
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                side.append(d.site("tip").xpos[:2].copy())
            q[k] = sign * lim
        path.append(np.array(side))
    # closed outline: +side forward, -side backward
    return np.vstack([path[0], path[1][::-1]])


def envelope_polygon(outline: np.ndarray) -> Polygon:
    poly = Polygon(outline)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def random_forces(f_max: float, t_total: float, dt: float, hold: float, tau: float,
                  rng: np.random.Generator) -> np.ndarray:
    n = int(t_total / dt)
    n_hold = int(hold / dt)
    targets = rng.uniform(0.0, f_max, size=(n // n_hold + 2, 2))
    raw = np.repeat(targets, n_hold, axis=0)[:n]
    out = np.empty_like(raw)
    a = dt / tau
    y = np.zeros(2)
    for k in range(n):
        y += a * (raw[k] - y)
        out[k] = y
    return out


def dynamic_cloud(sm: M.StudyModel, f_max: float, t_total: float = 60.0, seed: int = 0) -> np.ndarray:
    m, d = sm.load()
    dt = m.opt.timestep
    forces = random_forces(f_max, t_total, dt, hold=0.25, tau=0.06, rng=np.random.default_rng(seed))
    every = int(0.01 / dt)
    pts = []
    for k in range(len(forces)):
        d.ctrl[:] = -forces[k]
        mujoco.mj_step(m, d)
        if k % every == 0:
            pts.append(d.site("tip").xpos[:2].copy())
    return np.array(pts)


def cell_area(pts: np.ndarray, cell: float = 0.004) -> float:
    cells = {tuple(c) for c in np.floor(pts / cell).astype(int)}
    return len(cells) * cell * cell


def run_one(design: D.Design) -> dict:
    sm = M.make(design, f_max=200.0, tag="workspace")
    f_est = sim.f_pack_estimate(sm)
    env = kinematic_envelope(sm)
    poly = envelope_polygon(env)
    cloud = dynamic_cloud(sm, 1.2 * f_est)
    curl = np.load(STUDY / "results" / "curl" / f"{design.name}__cubic.npz")["tip"]
    curl_both = np.vstack([curl, curl[::-1] * [-1, 1]])
    inside = np.mean([poly.buffer(0.002).contains(Point(p)) for p in cloud[::5]])
    res = dict(
        design=design.name, phi_deg=design.phi_deg, f_est=f_est,
        envelope_area_cm2=poly.area * 1e4,
        dynamic_area_cm2=cell_area(cloud) * 1e4,
        dynamic_fraction_inside_envelope=float(inside),
        egg_in_envelope=bool(poly.contains(Point(EGG))),
        egg_clearance_from_curl_path_mm=float(np.min(np.linalg.norm(curl_both - EGG, axis=1)) * 1e3),
        max_reach_m=float(np.max(np.linalg.norm(env, axis=1))),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{design.name}.npz", envelope=env, cloud=cloud, curl=curl_both)
    (OUT / f"{design.name}.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> list[dict]:
    from multiprocessing import Pool
    from .analytic import osr_print_design
    ds = [D.CURRENT, *D.SWEEP, *D.PAPER, osr_print_design()]
    with Pool(2) as p:
        res = p.map(run_one, ds)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    for r in main():
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
