"""Figures for part 3: cable length control and force/length hybrids."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import actuation as A, designs as D
from .figures import INK, INK2, MUTED, GRID, RAMP, _save  # noqa: F401  (shared style)

R = D.STUDY / "results" / "position"
# categorical, validated (dataviz validator, light): blue, orange, aqua, yellow
C_FORCE, C_POS, C_CAP, C_ADM = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MODE_C = {"force": C_FORCE, "position-stiff": C_POS, "capped-stiff": C_CAP, "admittance-mjlab": C_ADM,
          "position-mjlab": C_POS, "capped2-stiff": C_CAP, "admittance2-mjlab": C_ADM}
MODE_LABEL = {"force": "force", "position-stiff": "length (stiff spool)",
              "position-mjlab": "length (mjlab servo)",
              "capped-stiff": "hybrid A1: one fixed force limit", "capped2-stiff": "hybrid A2: force limit per cable",
              "admittance-mjlab": "hybrid B1: one fixed force loop", "admittance2-mjlab": "hybrid B2: force loop per cable"}
NAMES = {"phi4p0": "22.6 cm 4°", "phi5p0": "22.6 cm 5°", "phi6p3": "22.6 cm 6.3° (current)",
         "phi8p0": "22.6 cm 8°", "phi10p0": "22.6 cm 10°", "phi12p0": "22.6 cm 12°", "phi15p0": "22.6 cm 15°",
         "L300_phi6p3": "30 cm 6.3°", "L375_phi6p3": "37.5 cm 6.3°", "L450_phi6p3": "45 cm 6.3°",
         "authors_45cm": "authors' 45 cm"}
SHORT = {"phi6p3": "22.6 cm\n6.3° (current)", "phi8p0": "22.6 cm\n8°", "L300_phi6p3": "30 cm\n6.3°",
         "L375_phi6p3": "37.5 cm\n6.3°", "L450_phi6p3": "45 cm\n6.3°", "authors_45cm": "authors'\n45 cm"}


def _j(name):
    return json.loads((R / name).read_text())


# ------------------------------------------------------------------ fig 16
def fig_actuator_laws():
    """Tension against how far the body holds the cable back from the command."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    x = np.linspace(0, 6, 400)                  # mm the body is held back (actual - commanded length)
    F_cap = 7.6                                 # current robot's grip force
    lines = [
        ("length, mjlab servo (2.5 N/mm)", np.minimum(2.5 * x, 1000), C_POS, "--"),
        ("length, stiff spool (100 N/mm)", np.minimum(100 * x, 1000), C_POS, "-"),
        ("hybrid A: length + force limit", np.minimum(100 * x, F_cap), C_CAP, "-"),
    ]
    for a, ylim, title in ((ax[0], (0, 25), "(a) close up: the first 25 N"),
                           (ax[1], (0.5, 2000), "(b) the whole range (log scale)")):
        for lab, y, c, ls in lines:
            a.plot(x, y, color=c, ls=ls, lw=2.2 if "hybrid" not in lab else 4.0, label=lab,
                   zorder=3 if "hybrid" in lab else 2)
        a.plot(x, np.full_like(x, F_cap), color=C_FORCE, ls=(0, (2, 2)), lw=2.2, zorder=4,
               label="force control (hybrid A matches it once blocked)")
        a.set_xlabel("how far the body holds the cable back from the command (mm)")
        a.set_ylabel("tension at the base (N)")
        a.set_title(title, loc="left")
        a.set_ylim(*ylim)
    ax[1].set_yscale("log")
    ax[1].axhline(30, color=INK, lw=0.8, ls=":")
    ax[1].text(4.3, 36, "a hen's egg cracks ≈ 30 N", fontsize=8, color=INK2)
    ax[1].axhline(1000, color=MUTED, lw=0.8, ls=":")
    ax[1].text(3.3, 1150, "spirob-mjlab force limit 1000 N", fontsize=8, color=INK2)
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].annotate("in free space the body barely resists:\nthe length modes track the command", (0.05, 1),
                   xytext=(0.9, 13), fontsize=8, color=INK2, arrowprops=dict(arrowstyle="->", color=MUTED))
    fig.suptitle("What each actuator does when the body is blocked (example: current robot, grip force 7.6 N)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    _save(fig, "fig16_actuator_laws")


# ------------------------------------------------------------------ fig 17
def fig_stroke():
    a = {r["design"]: r for r in _j("p1a_strokes.json")}
    st = {n: A.Stroke(**json.loads((R / "strokes" / f"{n}.json").read_text())) for n in a}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.1), gridspec_kw=dict(width_ratios=[1.15, 1]))
    for n, c in (("phi6p3", RAMP[6.3]), ("L450_phi6p3", RAMP[10.0]), ("authors_45cm", RAMP[15.0])):
        s = st[n]
        reel = (s.rest - np.array(s.L_near)) * 1e3
        ax[0].plot(reel, s.F, color=c, lw=2.2, label=NAMES[n])
        ax[0].plot(reel[-1], s.F[-1], "o", color=c, ms=6)
        ax[0].annotate(f"full curl\n{reel[-1]:.0f} mm", (reel[-1], s.F[-1]), xytext=(6, -4),
                       textcoords="offset points", fontsize=8, color=INK2)
    ax[0].set(xlabel="cable reeled in (mm)", ylabel="tension at the base (N)")
    ax[0].set_title("(a) the free-curl curve: little tension for most of the stroke", loc="left")
    ax[0].set_xlim(0, 215)
    ax[0].legend(fontsize=8, loc="upper left")
    order = ["phi4p0", "phi5p0", "phi6p3", "phi8p0", "phi10p0", "phi12p0", "phi15p0",
             "L300_phi6p3", "L375_phi6p3", "L450_phi6p3", "authors_45cm"]
    y = np.arange(len(order))[::-1]
    pull = [a[n]["stroke_mm"] for n in order]
    pay = [a[n]["far_payout_mm"] for n in order]
    ax[1].barh(y + 0.18, pull, height=0.34, color=C_FORCE, label="pulled cable: reeled in")
    ax[1].barh(y - 0.18, pay, height=0.34, color="#86b6ef", label="other cable: paid out")
    for yy, v in zip(y, pull):
        ax[1].text(v - 2, yy + 0.18, f"{v:.0f}", va="center", ha="right", fontsize=7, color="white")
    ax[1].set_yticks(y, [NAMES[n] for n in order], fontsize=8)
    # spirob-mjlab range for the current robot (0.15-0.29 m about 0.22 m rest)
    k = order.index("phi6p3")
    ax[1].plot([70], [y[k] + 0.18], marker="|", ms=14, color=INK, mew=2)
    ax[1].text(73, y[k] + 0.45, "spirob-mjlab range:\n70 mm each way", fontsize=7.5, color=INK2, va="bottom")
    ax[1].set_xlabel("cable length change from straight to fully curled (mm)")
    ax[1].set_title("(b) the stroke each body needs", loc="left")
    ax[1].legend(fontsize=8, loc="upper right")
    ax[1].grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "fig17_stroke")


# ------------------------------------------------------------------ fig 18
def fig_lock_order():
    """Fig. 2b of part 1, re-read in cable length."""
    a = {r["design"]: r for r in _j("p1a_strokes.json")}
    b = {(r["design"], r["kp"], r["mu"]): r for r in _j("p1b_ramps.json")}
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.9), sharex=True)
    for phi, n in ((4.0, "phi4p0"), (6.3, "phi6p3"), (10.0, "phi10p0"), (15.0, "phi15p0")):
        r = b[(n, A.KP_STIFF, 0.0)]
        s = np.array(r["joint_s_mm"]) / 224.0
        ax[0].plot(s, a[n]["lock_F"], "-o", ms=3, color=RAMP[phi], label=f"{phi:g}°" + (" (current)" if phi == 6.3 else ""))
        ax[1].plot(s, np.array(r["reel_mm"]) / r["stroke_mm"] * 100, "-o", ms=3, color=RAMP[phi])
    ax[0].set(xlabel="joint position s/L (0 base, 1 tip)", ylabel="tension when the joint locks (N)")
    ax[1].set(xlabel="joint position s/L (0 base, 1 tip)", ylabel="cable reeled in when it locks (% of stroke)")
    ax[0].set_title("(a) force view (part 1, Fig. 2b): the tip locks at the lowest force", loc="left", fontsize=9)
    ax[1].set_title("(b) length view: the tip locks after 15–35 % of the stroke, the base at 100 %", loc="left", fontsize=9)
    ax[0].legend(fontsize=8, title="22.6 cm bodies", title_fontsize=8)
    ax[1].set_ylim(0, 105)
    fig.tight_layout()
    _save(fig, "fig18_lock_order")


# ------------------------------------------------------------------ fig 19
def fig_spool_stiffness():
    a = {r["design"]: r for r in _j("p1a_strokes.json")}
    b = {(r["design"], r["kp"], r["mu"]): r for r in _j("p1b_ramps.json")}
    order = ["phi6p3", "phi8p0", "phi10p0", "L300_phi6p3", "L375_phi6p3", "L450_phi6p3", "authors_45cm"]
    x = np.arange(len(order))
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.9))
    sm = [a[n]["stretch_mjlab_mm"] / a[n]["stroke_mm"] * 100 for n in order]
    ss = [a[n]["stretch_stiff_mm"] / a[n]["stroke_mm"] * 100 for n in order]
    ax[0].bar(x - 0.2, sm, 0.38, color=C_POS, alpha=0.45, label="mjlab servo, 2.5 N/mm")
    ax[0].bar(x + 0.2, ss, 0.38, color=C_POS, label="stiff spool, 100 N/mm")
    ax[0].axhline(1, color=INK, lw=0.8, ls="--")
    ax[0].text(len(order) - 0.6, 1.25, "1 % of the stroke", fontsize=8, color=INK2, ha="right")
    for xi, v, n in zip(x, sm, order):
        ax[0].text(xi - 0.2, v + 0.2, f"{a[n]['kp_1pct'] / 1000:.0f}", ha="center", fontsize=7, color=INK2)
    ax[0].set_xticks(x, [NAMES[n].replace(" (current)", "\n(current)") for n in order], fontsize=7.5)
    ax[0].set_ylabel("cable stretch at the grip force (% of stroke)")
    ax[0].set_title("(a) is the cable inextensible? numbers: N/mm needed for 1 %", loc="left", fontsize=9)
    ax[0].legend(fontsize=8)
    for k, (kp, lab, alpha) in enumerate(((A.KP_MJLAB, "mjlab servo", 0.45), (A.KP_STIFF, "stiff spool", 1.0))):
        v0 = [b[(n, kp, 0.0)]["end_locked"] * 100 for n in order]
        v2 = [b[(n, kp, 0.2)]["end_locked"] * 100 for n in order]
        off = -0.3 + 0.2 * k * 2
        ax[1].bar(x + off, v0, 0.18, color=C_FORCE, alpha=alpha, label=f"{lab}, no friction")
        ax[1].bar(x + off + 0.2, v2, 0.18, color=C_POS, alpha=alpha, label=f"{lab}, friction μ = 0.2")
    ax[1].set_xticks(x, [NAMES[n].replace(" (current)", "\n(current)") for n in order], fontsize=7.5)
    ax[1].set_ylabel("joints at their stop at full command (%)")
    ax[1].set_ylim(0, 115)
    ax[1].set_title("(b) commanding the full stroke: does the body actually curl fully?", loc="left", fontsize=9)
    ax[1].legend(fontsize=7, ncol=2, loc="upper center")
    fig.tight_layout()
    _save(fig, "fig19_spool_stiffness")


# ------------------------------------------------------------------ fig 20
def _mid_gap(cmd, tip, frac=(0.1, 0.9)):
    cmd, tip = np.asarray(cmd), np.asarray(tip)
    n = int(np.argmax(cmd))
    lo, hi = frac[0] * cmd.max(), frac[1] * cmd.max()
    c_in, t_in = cmd[:n + 1], tip[:n + 1]
    out = []
    for c, p in zip(cmd[n:], tip[n:]):
        if lo <= c <= hi:
            i = int(np.argmin(np.abs(c_in - c)))
            out.append(np.linalg.norm(p - t_in[i]) * 1e3)
    return float(np.median(out)), float(np.max(out))


def fig_memory():
    rs = _j("p1c_memory.json")
    order = ["phi6p3", "phi10p0", "L450_phi6p3", "authors_45cm"]
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 3.6))
    x = np.arange(len(order))
    for k, (mu, alpha) in enumerate(((0.0, 0.45), (0.2, 1.0))):
        fm = [_mid_gap(r["force"]["F"], r["force"]["tip"])[1] for n in order for r in rs if r["design"] == n and r["mu"] == mu]
        pm = [_mid_gap(r["pos"]["reel_mm"], r["pos"]["tip"])[1] for n in order for r in rs if r["design"] == n and r["mu"] == mu]
        ax.bar(x - 0.3 + 0.2 * k, fm, 0.18, color=C_FORCE, alpha=alpha, label=f"force command, μ = {mu:g}")
        ax.bar(x + 0.1 + 0.2 * k, pm, 0.18, color=C_POS, alpha=alpha, label=f"length command, μ = {mu:g}")
        for xi, v in zip(x, fm):
            ax.text(xi - 0.3 + 0.2 * k, v + 1, f"{v:.0f}", ha="center", fontsize=7, color=INK2)
        for xi, v in zip(x, pm):
            ax.text(xi + 0.1 + 0.2 * k, v + 1, f"{v:.0f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, [NAMES[n] for n in order], fontsize=8)
    ax.set_ylabel("tip position difference, in vs out (mm)")
    ax.set_title("Same command on the way in and on the way out: how far apart is the tip? "
                 "(largest gap, 10–90 % of the command range)", loc="left", fontsize=9)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    _save(fig, "fig20_memory")


def main_p1():
    fig_actuator_laws()
    fig_stroke()
    fig_lock_order()
    fig_spool_stiffness()
    fig_memory()


if __name__ == "__main__":
    main_p1()


# ------------------------------------------------------------------ fig 21-23 (sweep)
BODY_ORDER = ["phi6p3", "phi8p0", "L300_phi6p3", "L375_phi6p3", "L450_phi6p3", "authors_45cm"]
MODE_ORDER = ["force", "position-stiff", "position-mjlab", "capped-stiff", "capped2-stiff",
              "admittance-mjlab", "admittance2-mjlab"]


def fig_outcomes():
    from . import summary3 as S
    tab = S.table()
    t = {(r["design"], r["mode"], r["egg"]): r for g in ("main", "hybrid2", "mjlab") for r in tab[g] if r["mu"] == 0.2}
    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    ax.set_xlim(-0.5, len(BODY_ORDER) - 0.5); ax.set_ylim(len(MODE_ORDER) - 0.5, -0.5)
    cmap = plt.get_cmap("Blues")
    for i, mode in enumerate(MODE_ORDER):
        for j, b in enumerate(BODY_ORDER):
            r = t.get((b, mode, "holder"))
            if r is None:
                continue
            frac = r["in_curl"] / max(r["n"], 1)
            ax.add_patch(plt.Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, color=cmap(0.08 + 0.8 * frac), lw=0))
            txt = f"in the curl {r['in_curl']}/{r['n']}\nmoved ≥ 45 mm: {r['grasp']}"
            if r["crush_risk"]:
                txt += f"\n⚠ squeeze > 30 N: {r['crush_risk']}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if frac > 0.55 else INK)
    ax.set_xticks(range(len(BODY_ORDER)), [SHORT[b] for b in BODY_ORDER], fontsize=8.5)
    ax.set_yticks(range(len(MODE_ORDER)), [MODE_LABEL[m] for m in MODE_ORDER], fontsize=8.5)
    for i, m in enumerate(MODE_ORDER):
        ax.get_yticklabels()[i].set_color(MODE_C[m])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Held egg (holder lets go before the carry), cable friction μ = 0.2: positions where the egg "
                 "ends inside the curl. Shade = fraction.", loc="left", fontsize=9.5)
    fig.tight_layout()
    _save(fig, "fig21_outcomes")


TICK = {"force": "force", "position-stiff": "length\nstiff spool", "position-mjlab": "length\nmjlab servo",
        "capped-stiff": "hybrid A1\nfixed limit", "capped2-stiff": "hybrid A2\nlimit per cable",
        "admittance-mjlab": "hybrid B1\nfixed loop", "admittance2-mjlab": "hybrid B2\nloop per cable"}


def fig_forces():
    from . import summary3 as S
    rs = [r for g in ("main", "hybrid2", "mjlab") for r in S.load(g)
          if r.get("anchored") and r["mu"] == 0.2 and not r.get("infeasible") and "error" not in r]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.3))
    rng = np.random.default_rng(0)
    for k, m in enumerate(MODE_ORDER):
        sq = np.array([r["sustained_squeeze_N"] for r in rs if r["mode"] == m and r["sustained_squeeze_N"] > 0.5])
        T = np.array([max(r["peak_T_N"]) for r in rs if r["mode"] == m])
        for a, v in ((ax[0], sq), (ax[1], T)):
            if len(v):
                a.scatter(k + rng.uniform(-0.22, 0.22, len(v)), v, s=12, color=MODE_C[m], alpha=0.7, lw=0)
                a.plot([k - 0.3, k + 0.3], [np.median(v)] * 2, color=INK, lw=1.6)
    for a in ax:
        a.set_yscale("log"); a.set_xticks(range(len(MODE_ORDER)), [TICK[m] for m in MODE_ORDER], fontsize=8)
        a.grid(axis="x", visible=False)
    ax[0].axhline(30, color=INK, lw=0.8, ls=":"); ax[0].text(6.45, 34, "egg cracks ≈ 30 N", fontsize=7.5, color=INK2, ha="right")
    ax[0].set_ylabel("squeeze on the egg held ≥ 0.1 s (N)")
    ax[0].set_title("(a) how hard the body squeezes the egg (all bodies and positions; bar = median)", loc="left", fontsize=9)
    ax[1].axhline(1000, color=MUTED, lw=0.8, ls=":"); ax[1].text(6.45, 1150, "mjlab limit 1000 N", fontsize=7.5, color=INK2, ha="right")
    ax[1].set_ylabel("largest cable tension at the base (N)")
    ax[1].set_title("(b) the tension the motor must supply", loc="left", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig22_forces")


def fig_checks():
    """Does the answer change with friction off, the mjlab servo, or half the time step?"""
    from . import summary3 as S
    t = S.table()
    main = {(r["design"], r["mode"], r["egg"]): r for r in t["main"]}
    rows = []
    for r in t["nofric"]:
        m = main.get((r["design"], r["mode"], "holder"))
        if m:
            rows.append((f"{NAMES[r['design']]}\n{MODE_LABEL[r['mode']]}", "friction off (μ = 0)", r, m))
    for r in t["mjlab"]:
        if r["egg"] != "holder":
            continue
        m = main.get((r["design"], "position-stiff", "holder"))
        if m:
            rows.append((f"{NAMES[r['design']]}\nlength control", "mjlab servo 2.5 N/mm", r, m))
    for r in t["dtcheck"]:
        m = main.get((r["design"], r["mode"], "holder"))
        if m:
            rows.append((f"{NAMES[r['design']]}\n{MODE_LABEL[r['mode']]}", "time step 0.1 ms", r, m))
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    ax.barh(y + 0.18, [m["in_curl"] for _, _, _, m in rows], 0.34, color=C_FORCE, label="reference run (μ = 0.2, stiff spool, 0.2 ms)")
    ax.barh(y - 0.18, [r["in_curl"] for _, _, r, _ in rows], 0.34, color="#86b6ef", label="changed run")
    for yy, (lab, chg, r, m) in zip(y, rows):
        ax.text(max(r["in_curl"], m["in_curl"]) + 0.3, yy, f"{chg}:  {m['in_curl']} → {r['in_curl']} in curl,  "
                f"{m['grasp']} → {r['grasp']} moved,  crack risk {m['crush_risk']} → {r['crush_risk']}",
                va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [lab for lab, *_ in rows], fontsize=7.5)
    ax.set_xlabel("positions where the held egg ends in the curl")
    ax.set_xlim(0, max(max(r["in_curl"], m["in_curl"]) for _, _, r, m in rows) + 14)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_title("Checks: do the conclusions survive switching friction off, the mjlab servo, or a smaller time step?",
                 loc="left", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig23_checks")


def main_p2():
    fig_outcomes()
    fig_forces()
    fig_checks()
