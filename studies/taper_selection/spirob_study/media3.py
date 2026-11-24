"""Explainer animations for part 3 (cable length control and hybrids).

Each animation shows one idea: the robot on the left (or top) and the graph
that explains it beside it, drawn in step with the motion, so the reader sees
the cause and the measurement together.

anim8   reeling in cable curls the body joint by joint (length control)
anim9   the same held egg under force, length, and the hybrids
anim10  friction memory: force control vs length control
"""
from __future__ import annotations

import io
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image

from . import actuation as A, designs as D, model as M, sim
from .exp_position import force_cycle, ramp
from .figures import INK, INK2, MUTED, GRID
from .media import FIG

C_FORCE, C_POS, C_CAP, C_ADM = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_LOCK = "#eb6834"
MODE_COLOR = {"force": C_FORCE, "position-stiff": C_POS, "capped-stiff": C_CAP, "admittance-mjlab": C_ADM,
              "position-mjlab": C_POS, "capped2-stiff": C_CAP, "admittance2-mjlab": C_ADM}
MODE_NAME = {"force": "force control", "position-stiff": "length control",
             "capped-stiff": "hybrid A: length + force limit", "admittance-mjlab": "hybrid B: length + force loop",
             "position-mjlab": "length control, mjlab servo",
             "capped2-stiff": "hybrid A: length + force limit", "admittance2-mjlab": "hybrid B: length + force loop"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.7})


def _fig_to_array(fig) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi, facecolor="white")
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def _save(frames, name, fps, hold_last: int = 0):
    frames = frames + [frames[-1]] * hold_last
    ims = [Image.fromarray(f) for f in frames]
    pal = [im.convert("P", palette=Image.ADAPTIVE, colors=128) for im in ims]
    FIG.mkdir(parents=True, exist_ok=True)
    pal[0].save(FIG / f"{name}.gif", save_all=True, append_images=pal[1:], duration=int(1000 / fps),
                loop=0, optimize=True)
    print("wrote", FIG / f"{name}.gif", len(frames), "frames")


def _visual_geoms_by_link(m, sm):
    out = []
    for n in sm.link_names:
        bid = m.body(n).id
        out.append([g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_group[g] == 1])
    return out


# ------------------------------------------------------------------ anim 8
def anim8_reel_in(fps: int = 20):
    """Reeling in cable: the tip curls first, joints lock one by one, then the wall."""
    sim.gl_env()
    des = D.sweep_by_phi()[6.3]
    st = A.stroke(des)
    over = 0.006
    log = ramp(des, A.KP_STIFF, 0.0, speed=0.015, over=over, hold=0.6)
    sm = M.make(des, tag="anim8", actuation="position", kp=A.KP_STIFF, kv=A.kv_for(A.KP_STIFF),
                ctrlrange=(st.full - over * st.stroke, st.ctrlrange()[1]))
    m, d = sm.load()
    qi = sim.qpos_index(m, sm)
    vis = _visual_geoms_by_link(m, sm)
    base_rgba = m.geom_rgba[vis[1][0]].copy() if vis[1] else np.array([0.11, 0.36, 0.67, 1])
    lock_rgba = np.array([0.92, 0.41, 0.20, 1.0])
    tv = sim.TopView(m, 480, 480, center=(0.035, 0.105), extent=0.25)
    dth = math.radians(des.dtheta_deg)
    reel_cmd = (st.rest - log["cmd"]) * 1e3
    reel_act = (st.rest - log["len"]) * 1e3
    T = log["T"]
    q = np.abs(log["q"])
    first_lock = np.array([np.argmax(q[:, j] >= 0.95 * dth) if (q[:, j] >= 0.95 * dth).any() else -1
                           for j in range(q.shape[1])])
    S = st.stroke * 1e3
    frames = []
    step = max(1, int(round(1.0 / (fps * 0.01) / 1.0)))     # one frame per 50 ms of sim
    idx = list(range(0, len(log["t"]), 5))
    for i in idx:
        d.qpos[qi] = log["q"][i]
        mujoco.mj_forward(m, d)
        locked = q[i] >= 0.95 * dth
        for j, lk in enumerate(locked):        # joint j sits between link j and link j+1
            for g in vis[j + 1]:
                m.geom_rgba[g] = lock_rgba if lk else base_rgba
        img = tv(d)
        fig = plt.figure(figsize=(10.4, 5.6), dpi=100)
        axr = fig.add_axes([0.0, 0.0, 0.5, 0.88]); axr.imshow(img); axr.axis("off")
        n_l = int(locked.sum())
        frac = reel_cmd[i] / S
        if n_l == 0:
            msg = "1  The tip curls first. Almost no tension is needed."
        elif n_l < len(locked):
            msg = f"2  Joints reach their stops one by one, tip to base ({n_l}/{len(locked)} locked)."
        else:
            msg = "3  Fully curled. Reeling in more only raises the tension."
        fig.text(0.02, 0.95, "Length control: reel in cable, the body follows", fontsize=13, color=INK,
                 weight="bold")
        fig.text(0.02, 0.905, msg, fontsize=10.5, color=INK2)
        axr.text(0.03, 0.04, "orange = unit at its stop", transform=axr.transAxes, color=INK2, fontsize=9)
        # bar: commanded vs actual
        axb = fig.add_axes([0.58, 0.70, 0.38, 0.12])
        axb.barh([1], [reel_cmd[i]], color="white", edgecolor=C_POS, lw=1.8, height=0.6)
        axb.barh([0], [reel_act[i]], color=C_FORCE, height=0.6)
        axb.set_yticks([0, 1], ["actual (in the body)", "commanded (spool)"])
        axb.set_xlim(0, S * 1.1); axb.axvline(S, color=INK, lw=1, ls="--")
        axb.text(S, 1.55, "full curl", ha="center", fontsize=8.5, color=INK2)
        axb.set_xlabel("cable reeled in (mm)"); axb.grid(axis="y", visible=False)
        axb.set_title(f"stroke {S:.0f} mm  ·  difference {abs(reel_cmd[i]-reel_act[i]):.2f} mm (stiff spool)",
                      fontsize=9, color=INK2, loc="left")
        # curve: tension vs reel
        axc = fig.add_axes([0.58, 0.12, 0.38, 0.45])
        axc.plot(reel_act[:i + 1], T[:i + 1], color=C_FORCE, lw=2)
        lk = [k for k in first_lock if 0 <= k <= i]
        axc.plot(reel_act[lk], T[lk], "o", ms=5, mfc="white", mec=C_LOCK, mew=1.6)
        axc.plot([reel_act[i]], [T[i]], "o", color=C_FORCE, ms=7)
        axc.set_xlim(0, S * 1.1); axc.set_ylim(0, 25)
        axc.set_xlabel("cable reeled in (mm)"); axc.set_ylabel("tension at the base (N)")
        axc.axvline(S, color=INK, lw=1, ls="--")
        if len(lk):
            k0 = min(lk)
            axc.annotate("first stop: the tip", (reel_act[k0], T[k0]), xytext=(-40, 45),
                         textcoords="offset points", fontsize=8.5, color=INK2,
                         arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        if n_l == len(locked):
            k1 = max(lk)
            axc.annotate("last stop: the base", (reel_act[k1], T[k1]), xytext=(-120, 25),
                         textcoords="offset points", fontsize=8.5, color=INK2,
                         arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        if n_l == len(locked):
            axc.annotate("the wall: tension rises,\nthe body cannot curl further", (S, min(T[i], 24)),
                         xytext=(-170, 0), textcoords="offset points", fontsize=8.5, color=INK2,
                         arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
        axc.text(0.02, 0.92, "○ a joint reaches its stop", transform=axc.transAxes, fontsize=8.5, color=C_LOCK)
        frames.append(_fig_to_array(fig))
        plt.close(fig)
    _save(frames, "anim8_reel_in", fps, hold_last=fps * 2)


# ------------------------------------------------------------------ anim 10
def anim10_memory(fps: int = 20, mu: float = 0.2):
    """Same body, same friction: cable force up and down vs cable length in and out."""
    sim.gl_env()
    des = D.sweep_by_phi()[6.3]
    st = A.stroke(des)
    frc = force_cycle(des, mu, t_ramp=5.0, hold=0.4)
    pos = ramp(des, A.KP_STIFF, mu, speed=st.stroke / 5.0, back=True, hold=0.4)
    runs = []
    for name, log in (("force", frc), ("length", pos)):
        if name == "force":
            sm = M.make(des, tag=f"anim10_{name}")
        else:
            lo, hi = st.ctrlrange()
            sm = M.make(des, tag=f"anim10_{name}", actuation="position", kp=A.KP_STIFF,
                        kv=A.kv_for(A.KP_STIFF), ctrlrange=(lo, hi))
        m, d = sm.load()
        runs.append((name, log, m, d, sim.qpos_index(m, sm), sim.TopView(m, 400, 400, center=(0.03, 0.095), extent=0.25)))
    n = min(len(frc["t"]), len(pos["t"]))
    frames = []
    for i in range(0, n, 5):
        fig = plt.figure(figsize=(10.4, 6.6), dpi=100)
        fig.text(0.02, 0.955, "Cable friction (μ = 0.2): which command remembers its past?", fontsize=13,
                 color=INK, weight="bold")
        for col, (name, log, m, d, qi, tv) in enumerate(runs):
            d.qpos[qi] = log["q"][i]
            mujoco.mj_forward(m, d)
            img = tv(d)
            x0 = 0.02 + col * 0.5
            ax = fig.add_axes([x0, 0.36, 0.46, 0.54]); ax.imshow(img); ax.axis("off")
            tips = np.asarray(log["tip"])
            half = int(np.searchsorted(log["t"], (log["t_ramp"] if name == "force" else log["t_in"]) + log["hold"]))
            # tip trails, drawn in image coordinates of the top view
            def to_px(p):
                ext, c = 0.25, (0.03, 0.095)
                return ((p[:, 0] - c[0]) / ext + 0.5) * 400, (0.5 - (p[:, 1] - c[1]) / ext) * 400
            a, b = to_px(tips[:min(i, half) + 1])
            ax.plot(a, b, color=C_FORCE, lw=2)
            if i > half:
                a2, b2 = to_px(tips[half:i + 1]); ax.plot(a2, b2, color=C_POS, lw=2)
            ax.set_xlim(0, 400); ax.set_ylim(400, 0)
            title = "force control: tension up, then down" if name == "force" else "length control: reel in, then out"
            ax.set_title(title, fontsize=10.5, color=INK)
            # bottom plot: tip height vs command
            axp = fig.add_axes([x0 + 0.06, 0.08, 0.38, 0.22])
            cmd = log["F"] if name == "force" else (st.rest - log["cmd"]) * 1e3
            y = tips[:, 1] * 1e3
            axp.plot(cmd[:min(i, half) + 1], y[:min(i, half) + 1], color=C_FORCE, lw=2, label="on the way in")
            if i > half:
                axp.plot(cmd[half:i + 1], y[half:i + 1], color=C_POS, lw=2, label="on the way out")
            axp.plot([cmd[i]], [y[i]], "o", color=INK, ms=5)
            axp.set_xlim(0, cmd.max() * 1.03); axp.set_ylim(min(y.min(), 0) - 5, y.max() + 5)
            axp.set_xlabel("cable tension (N)" if name == "force" else "cable reeled in (mm)")
            axp.set_ylabel("tip height (mm)")
            if i > half:
                axp.legend(loc="upper right", fontsize=8, frameon=False)
        frames.append(_fig_to_array(fig))
        plt.close(fig)
    _save(frames, "anim10_memory", fps, hold_last=fps * 2)


if __name__ == "__main__":
    import sys
    for a in (sys.argv[1:] or ["anim8", "anim10"]):
        {"anim8": anim8_reel_in, "anim10": anim10_memory}[a]()


# ------------------------------------------------------------------ anim 9
def run_cases(design, pos, mu=0.2, anchored=True, modes=("force", "position", "capped", "admittance")):
    """Run one held-egg trial per mode (name or Gains) with frames and cable traces."""
    from . import wrap as W
    st = A.stroke(design)
    out = []
    for mode in modes:
        g = mode if isinstance(mode, A.Gains) else (None if mode == "force" else A.Gains(mode))
        r = W.run(W.Trial(design, pos, 0.045, "reach", mu, anchored, st.f_pack, gains=g, keep_frames=True,
                          trace=True))
        out.append(r)
    return out


def anim9_actuators(results, name="anim9_actuators", fps=12, dt_frame=0.1, title=None):
    """Columns: one actuator each. Render, base tension (log), cable reeled in."""
    from . import wrap as W
    from .media2 import flatten_object, _links
    sim.gl_env()
    cols = []
    for r in results:
        xp = os.path.join(r["build_dir"], f"render_{name}_{len(cols)}.xml")
        with open(xp, "w") as f:
            f.write(r["xml_text"])
        m = mujoco.MjModel.from_xml_path(xp)
        d = mujoco.MjData(m)
        flatten_object(m)
        L = r["L"]
        tv = sim.TopView(m, 300, 300, center=(0.22 * L, 0.45 * L), extent=1.25 * L)
        tr = r["trace"]
        tt = np.array([row[0] for row in tr])
        cols.append(dict(r=r, m=m, d=d, tv=tv, links=_links(m), bid=m.body("obj").id, tr=tr, tt=tt,
                         T=np.array([row[6:8] for row in tr]), cmd=np.array([row[2:4] for row in tr], float),
                         act=np.array([row[4:6] for row in tr]), N=np.array([row[8] for row in tr]),
                         ft=np.array([f[0] for f in r["frames"]])))
        Nn = cols[-1]["N"]
        over = np.convolve(Nn > 30.0, np.ones(5), "valid") >= 5          # > 30 N for 0.1 s
        cols[-1]["t_crack"] = float(cols[-1]["tt"][int(np.argmax(over))]) if over.any() else None
    t_end = max(c["tt"][-1] for c in cols)
    frames = []
    for t in np.arange(0.0, t_end + 1e-9, dt_frame):
        fig = plt.figure(figsize=(12.8, 8.2), dpi=100)
        fig.text(0.01, 0.965, title or "The same held egg, four ways to drive the cable", fontsize=13,
                 color=INK, weight="bold")
        n = len(cols)
        for k, c in enumerate(cols):
            r = c["r"]
            mode = r["mode"]
            col = MODE_COLOR.get(mode, INK)
            x0 = 0.04 + k * (0.95 / n)
            w = 0.95 / n - 0.03
            i = int(min(np.searchsorted(c["ft"], t), len(c["ft"]) - 1))
            fr = r["frames"][i]
            c["d"].qpos[:] = fr[1]
            mujoco.mj_forward(c["m"], c["d"])
            img = c["tv"](c["d"])
            ax = fig.add_axes([x0, 0.55, w, 0.36]); ax.imshow(img); ax.axis("off")
            ph = fr[3]
            phase = {"pack": "curl away", "reach": "reach + wrap", "release": "holder lets go"}.get(
                ph, ph.replace("carry:", "move: "))
            ctr = c["d"].xpos[c["bid"], :2]
            wd = W.wrap_angle(c["m"], c["d"], c["links"], ctr, 0.0225)["wrap_deg"]
            ax.set_title(MODE_NAME.get(mode, mode), fontsize=10, color=col, weight="bold")
            ax.text(0.02, 0.02, f"{fr[0]:4.1f} s  {phase}\nwrap {wd:3.0f}°", transform=ax.transAxes,
                    fontsize=8.5, color=INK, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
            if c["t_crack"] is not None and t >= c["t_crack"]:
                msg = "a real egg cracks" if c["N"].max() > 100 else "a real egg may crack"
                ax.text(0.5, 0.93, f"squeeze > 30 N for 0.1 s at {c['t_crack']:.1f} s:\n{msg}",
                        transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=INK, weight="bold",
                        bbox=dict(fc="#fde3d6", ec=C_POS, lw=1, pad=2))
            j = int(min(np.searchsorted(c["tt"], t), len(c["tt"]) - 1))
            # tension
            a1 = fig.add_axes([x0 + 0.02, 0.30, w - 0.02, 0.20])
            a1.semilogy(c["tt"][:j + 1], np.maximum(c["T"][:j + 1, 1], 0.1), color=col, lw=1.8, label="pulled cable")
            a1.semilogy(c["tt"][:j + 1], np.maximum(c["T"][:j + 1, 0], 0.1), color=col, lw=1.0, ls="--",
                        label="other cable")
            a1.semilogy(c["tt"][:j + 1], np.maximum(c["N"][:j + 1], 0.1), color=INK, lw=0.9, alpha=0.7,
                        label="squeeze on egg")
            a1.axhline(30, color=INK, lw=0.7, ls=":")
            a1.set_xlim(0, t_end); a1.set_ylim(0.3, 3000)
            if k == 0:
                a1.set_ylabel("force (N)")
                a1.text(0.3, 38, "egg cracks ≈ 30 N", fontsize=7, color=INK2)
                a1.legend(fontsize=6.5, loc="upper left", frameon=False)
            a1.tick_params(labelsize=7.5)
            # cable reeled in
            a2 = fig.add_axes([x0 + 0.02, 0.06, w - 0.02, 0.19])
            r0 = c["act"][0, 1]
            a2.plot(c["tt"][:j + 1], (r0 - c["act"][:j + 1, 1]) * 1e3, color=col, lw=1.8, label="actual")
            if mode != "force":
                a2.plot(c["tt"][:j + 1], (r0 - c["cmd"][:j + 1, 1]) * 1e3, color=INK, lw=1.0, ls="--",
                        label="commanded")
            else:
                a2.text(0.5, 0.85, "length is not commanded", transform=a2.transAxes, ha="center",
                        fontsize=7.5, color=INK2)
            a2.set_xlim(0, t_end)
            lo = min((r0 - c["act"][:, 1]).min(), np.nanmin(r0 - c["cmd"][:, 1]) if mode != "force" else 0) * 1e3
            hi = max((r0 - c["act"][:, 1]).max(), np.nanmax(r0 - c["cmd"][:, 1]) if mode != "force" else 0) * 1e3
            a2.set_ylim(lo - 5, hi + 5)
            a2.set_xlabel("time (s)", fontsize=8); a2.tick_params(labelsize=7.5)
            if k == 0:
                a2.set_ylabel("pulled cable reeled in\n(mm, below 0 = paid out)")
                a2.legend(fontsize=6.5, loc="lower right", frameon=False)
        frames.append(_fig_to_array(fig))
        plt.close(fig)
    _save(frames, name, fps, hold_last=fps * 2)


# ------------------------------------------------------------------ anim 11
def anim11_operating_point(results, name="anim11_operating_point", fps=12, dt_frame=0.1):
    """Where each actuator sits on its own law while the body closes on the egg.

    x: how far the body holds the pulled cable back from the command
    (actual - commanded length); y: tension at the base. Length control rides
    up a steep line to the 1000 N limit; the force cap slides sideways along a
    flat line instead: the extra command turns into slack, not squeeze."""
    from .media2 import flatten_object
    sim.gl_env()
    use = [r for r in results if r["mode"] in ("position-stiff", "capped-stiff")]
    cols = []
    for k, r in enumerate(use):
        xp = os.path.join(r["build_dir"], f"render_{name}_{k}.xml")
        with open(xp, "w") as f:
            f.write(r["xml_text"])
        m = mujoco.MjModel.from_xml_path(xp)
        d = mujoco.MjData(m)
        flatten_object(m)
        L = r["L"]
        tr = r["trace"]
        cols.append(dict(r=r, m=m, d=d, tv=sim.TopView(m, 360, 360, center=(0.22 * L, 0.45 * L), extent=1.2 * L),
                         tt=np.array([row[0] for row in tr]), ph=[row[1] for row in tr],
                         hold=np.array([row[4] - row[2] for row in tr]) * 1e3,     # cable_0? no: pulled = cable_1
                         hold1=np.array([row[5] - row[3] for row in tr]) * 1e3,
                         T1=np.array([row[7] for row in tr]), N=np.array([row[8] for row in tr]),
                         ft=np.array([f[0] for f in r["frames"]])))
    f_cap = A.stroke(D.length_design(0.45, 6.3)).f_grip if use and use[0]["design"] == "L450_phi6p3" else 22.4
    t_rel = [c["tt"][c["ph"].index("release")] for c in cols if "release" in c["ph"]]
    t_end = (min(t_rel) - 0.05) if t_rel else max(c["tt"][-1] for c in cols)   # up to the release
    x = np.linspace(0, 100, 400)
    laws = {"position-stiff": np.minimum(100 * x, 1000), "capped-stiff": np.minimum(100 * x, f_cap)}
    frames = []
    for t in np.arange(0.0, t_end + 1e-9, dt_frame):
        fig = plt.figure(figsize=(11.6, 7.4), dpi=100)
        fig.text(0.01, 0.965, "Why the force cap protects the egg: the operating point on each actuator's law",
                 fontsize=13, color=INK, weight="bold")
        for k, c in enumerate(cols):
            mode = c["r"]["mode"]
            col = MODE_COLOR[mode]
            x0 = 0.03 + k * 0.49
            i = int(min(np.searchsorted(c["ft"], t), len(c["ft"]) - 1))
            c["d"].qpos[:] = c["r"]["frames"][i][1]
            mujoco.mj_forward(c["m"], c["d"])
            ax = fig.add_axes([x0 + 0.08, 0.50, 0.30, 0.40]); ax.imshow(c["tv"](c["d"])); ax.axis("off")
            ax.set_title(MODE_NAME[mode], color=col, fontsize=11, weight="bold")
            j = int(min(np.searchsorted(c["tt"], t), len(c["tt"]) - 1))
            a = fig.add_axes([x0 + 0.04, 0.08, 0.40, 0.34])
            a.semilogy(x, np.maximum(laws[mode], 0.3), color=col, lw=1.2, alpha=0.5)
            lo = max(0, j - 15)
            a.semilogy(np.maximum(c["hold1"][lo:j + 1], 0), np.maximum(c["T1"][lo:j + 1], 0.3), color=col, lw=2)
            a.semilogy([max(c["hold1"][j], 0)], [max(c["T1"][j], 0.3)], "o", color=col, ms=9, mec=INK, mew=0.8)
            a.semilogy([max(c["hold1"][j], 0)], [max(c["N"][j], 0.3)], "s", color="white", ms=7, mec=INK, mew=1.2)
            a.axhline(30, color=INK, lw=0.7, ls=":")
            a.text(98, 36, "egg cracks ≈ 30 N", fontsize=7.5, color=INK2, ha="right")
            a.set_xlim(-2, 100); a.set_ylim(0.3, 2500)
            a.set_xlabel("body holds the pulled cable back from the command (mm)", fontsize=8.5)
            a.set_ylabel("force (N)")
            a.text(0.02, 0.92, f"● cable tension {c['T1'][j]:6.1f} N\n□ squeeze on egg {c['N'][j]:6.1f} N",
                   transform=a.transAxes, fontsize=8.5, color=INK, va="top",
                   bbox=dict(fc="white", ec="none", alpha=0.85))
            a.text(0.98, 0.04, f"t = {c['tt'][j]:.1f} s", transform=a.transAxes, ha="right", fontsize=8.5, color=INK2)
        frames.append(_fig_to_array(fig))
        plt.close(fig)
    _save(frames, name, fps, hold_last=fps * 2)
