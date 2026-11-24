"""Figures for the taper-selection study (static PNG, 200 dpi)."""
from __future__ import annotations

import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle

from . import analytic, designs as D
from .designs import STUDY

R = STUDY / "results"
FIG = STUDY / "figures"

# palette (validated with the dataviz validator): ordinal blue ramp for the
# four key tapers, categorical slots for unordered comparisons
KEY = (4.0, 6.3, 10.0, 15.0)
RAMP = {4.0: "#6da7ec", 6.3: "#2a78d6", 10.0: "#1c5cab", 15.0: "#0d366b"}
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
OK, PART = "#2a78d6", "#b7d3f6"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "legend.frameon": False, "font.family": "DejaVu Sans",
})


def _label(phi: float) -> str:
    return f"{phi:g}°" + (" (current)" if phi == 6.3 else "") + (" (authors' egg robot)" if phi == 10.0 else "")


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _curl(name, law="cubic"):
    return json.loads((R / "curl" / f"{name}__{law}.json").read_text()), np.load(R / "curl" / f"{name}__{law}.npz")


# ------------------------------------------------------------------ Fig. 2
def fig_curl():
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5), gridspec_kw=dict(width_ratios=[1.1, 1, 1.25]))
    for phi in KEY:
        r, z = _curl(D.sweep_by_phi()[phi].name)
        s = np.array(r["joint_s_m"]) / r["length_m"]
        a = np.array(r["lock_force_analytic"]); sm = np.array(r["lock_force_sim"])
        c = RAMP[phi]
        ax[0].plot(s, a, color=c, lw=1.6)
        ax[0].plot(s[1:], sm[1:], "o", ms=4.5, mfc="white", mec=c, mew=1.4)
        ax[0].annotate(f"{phi:g}°", (s[1], a[1]), xytext=(-4, 0), textcoords="offset points",
                       ha="right", va="center", color=INK2, fontsize=8)
        lim = math.radians(30.0)
        locked = (np.abs(z["q"]) >= 0.9 * lim).mean(axis=1)
        ax[1].plot(z["F"], locked, color=c)
        k = np.argmax(locked >= 0.999) if np.any(locked >= 0.999) else -1
        ax[1].annotate(f"{phi:g}°", (z["F"][k], locked[k]), xytext=(0, 4), textcoords="offset points",
                       ha="center", va="bottom", color=INK2, fontsize=8)
    ax[0].set_yscale("log"); ax[0].set_xlabel("joint position along the body, s / L  (0 = base, 1 = tip)")
    ax[0].set_ylabel("force to lock the joint (N)")
    ax[0].set_title("(a) Lock force of every joint", loc="left")
    ax[0].legend(handles=[Line2D([], [], color=MUTED, lw=1.6, label="closed form  k·Δθ / h"),
                          Line2D([], [], color=MUTED, ls="", marker="o", mfc="white", label="MuJoCo")],
                 loc="upper right", fontsize=8)
    ax[1].set_xscale("log"); ax[1].set_xlabel("cable force (N)"); ax[1].set_ylabel("fraction of joints locked")
    ax[1].set_title("(b) Curling progresses with force", loc="left")
    ax[1].set_xlim(0.3, 40); ax[1].set_ylim(-0.03, 1.12)
    # (c) shapes at equal absolute force
    forces = (1.0, 2.0, 4.0, 8.0)
    off = 0.0
    for phi in KEY:
        r, z = _curl(D.sweep_by_phi()[phi].name)
        for j, F in enumerate(forces):
            k = int(np.argmin(np.abs(z["F"] - F)))
            if z["F"].max() < F * 0.98:
                k = len(z["F"]) - 1
            line = z["line"][k]
            alpha = 0.35 + 0.65 * j / (len(forces) - 1)
            ax[2].plot(line[:, 0] + off, line[:, 1], color=RAMP[phi], alpha=alpha, lw=1.6)
        ax[2].text(off, -0.03, f"{phi:g}°", ha="center", color=INK2, fontsize=8.5)
        off += 0.14
    ax[2].set_aspect("equal"); ax[2].set_xticks([]); ax[2].set_ylabel("y (m)")
    ax[2].set_title("(c) Shape at 1, 2, 4, 8 N (light → dark)", loc="left")
    ax[2].grid(False); ax[2].spines["bottom"].set_visible(False)
    _save(fig, "fig2_curling")


# ------------------------------------------------------------------ Fig. 3
def fig_robustness():
    pr = json.loads((R / "print" / "summary.json").read_text())
    fr = json.loads((R / "print" / "friction.json").read_text())
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4))
    # (a) stiffness profiles of the current robot
    cur = [p for p in pr["print"] if p["design"] == "current_6p3"]
    for p, c, lab in zip(cur, CAT, ("ideal 5 % flexure (K ∝ W³)", "printed, 0.20 mm layers", "printed, 0.12 mm layers")):
        s = np.array(p["joint_s_m"]) / p["joint_s_m"][-1]
        ax[0].step(s, p["stiffness"], where="mid", color=c, lw=1.6, label=lab)
    rr, _ = _curl("current_6p3", "mjlab")
    s = np.array(rr["joint_s_m"]) / rr["joint_s_m"][-1]
    ax[0].plot(s, rr["stiffness"], color=MUTED, ls="--", lw=1.4, label="spirob-mjlab law (0.915ⁱ)")
    ax[0].set_yscale("log"); ax[0].set_xlabel("s / L"); ax[0].set_ylabel("joint stiffness (N·m/rad)")
    ax[0].set_title("(a) Current robot: stiffness profiles", loc="left"); ax[0].legend(fontsize=7.5, loc="lower left")
    # (b) distal share vs taper for printing
    for h, c, lab in zip((None, 0.20, 0.12), CAT, ("ideal", "0.20 mm layers", "0.12 mm layers")):
        rows = sorted([p for p in pr["print"] if p["layer_mm"] == h and p["design"] != "current_6p3"],
                      key=lambda p: p["phi_deg"])
        ax[1].plot([p["phi_deg"] for p in rows], [p["order_reversals"] for p in rows], "-o", ms=4.5,
                   color=c, label=lab)
    ax[1].set_xlabel("taper angle φ (deg)"); ax[1].set_ylabel("joints that lock out of order")
    ax[1].set_title("(b) Printing disorders the curl sequence", loc="left"); ax[1].legend(fontsize=8)
    # (c) friction
    for phi in KEY:
        rows = sorted([f for f in fr if f["phi_deg"] == phi and f["layer_mm"] is None], key=lambda f: f["mu"])
        ax[2].plot([f["mu"] for f in rows], [f["distal_share_at_180"] for f in rows], "-o", ms=4.5,
                   color=RAMP[phi])
        ax[2].annotate(f"{phi:g}°", (rows[-1]["mu"], rows[-1]["distal_share_at_180"]), xytext=(5, 0),
                       textcoords="offset points", va="center", color=INK2, fontsize=8)
    lm = [l for l in pr["laws"] if l["law"] == "mjlab"][0]["distal_share_at_180"]
    ax[2].axhline(lm, color=MUTED, ls="--", lw=1.2)
    ax[2].text(0.0, lm + 0.012, "spirob-mjlab law (any taper)", color=INK2, fontsize=7.5)
    ax[2].set_xlabel("cable friction coefficient μ"); ax[2].set_ylabel("share of bending in distal third")
    ax[2].set_title("(c) Tip-first curling vs cable friction", loc="left"); ax[2].set_xlim(-0.01, 0.36)
    _save(fig, "fig3_robustness")


# ------------------------------------------------------------------ Fig. 4
def fig_workspace():
    from .exp_workspace import BUCKET, EGG
    fig, ax = plt.subplots(1, 5, figsize=(13.5, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.0], wspace=0.35))
    for a, phi in zip(ax[:4], KEY):
        z = np.load(R / "workspace" / f"{D.sweep_by_phi()[phi].name}.npz")
        e = z["envelope"]
        a.fill(e[:, 0], e[:, 1], color="#f1f0ec", lw=0)
        a.plot(e[:, 0], e[:, 1], color=MUTED, lw=0.9)
        a.plot(z["cloud"][:, 0], z["cloud"][:, 1], ".", ms=0.8, color=RAMP[phi], alpha=0.35)
        a.plot(z["curl"][:, 0], z["curl"][:, 1], color=CAT[1], lw=1.4)
        a.plot([0, 0], [0, 0.226], color=INK, lw=2.5, solid_capstyle="butt")
        a.add_patch(Circle(EGG, 0.02, fc="none", ec=INK, lw=1.0))
        a.add_patch(Circle(BUCKET, 0.02, fc="none", ec=INK, lw=1.0, ls=":"))
        a.set_aspect("equal"); a.set_xlim(-0.22, 0.22); a.set_ylim(-0.13, 0.25)
        a.set_title(f"φ = {_label(phi)}", loc="left", fontsize=8.5)
        a.set_xlabel("x (m)")
    ax[0].set_ylabel("y (m)")
    fig.legend(handles=[Patch(fc="#f1f0ec", ec=MUTED, label="kinematic envelope (paper construction)"),
                        Line2D([], [], color=RAMP[6.3], ls="", marker=".", ms=8, label="tip, random cable forces (dynamic)"),
                        Line2D([], [], color=CAT[1], label="tip, one cable pulled slowly (quasi-static)"),
                        Line2D([], [], color=INK, ls="", marker="o", mfc="none", label="egg (solid) / bucket (dotted)")],
               loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.45, -0.08))
    ws = sorted(json.loads((R / "workspace" / "summary.json").read_text()), key=lambda r: r["phi_deg"])
    ws = [w for w in ws if w["design"].startswith("phi")]
    ax[4].plot([w["phi_deg"] for w in ws], [w["dynamic_area_cm2"] for w in ws], "-o", color=CAT[0], ms=4.5)
    ax[4].set_xlabel("taper angle φ (deg)"); ax[4].set_ylabel("cm²")
    ax[4].set_title("Area the tip visited\n(random forces, 60 s)", loc="left", fontsize=8.5)
    ax[4].set_ylim(0, None)
    _save(fig, "fig4_workspace")


# ------------------------------------------------------------------ Fig. 5
def fig_size_position():
    res = json.loads((R / "grasp" / "size_position.json").read_text())
    phis = sorted({r["phi_deg"] for r in res})
    fig, ax = plt.subplots(1, len(phis), figsize=(2.0 * len(phis), 3.3), sharey=True)
    for a, phi in zip(ax, phis):
        d = D.sweep_by_phi()[phi]
        rows = [r for r in res if r["phi_deg"] == phi]
        for r in rows:
            col = OK if r["success"] else (PART if r["enclosure_deg"] >= 180 else "white")
            a.add_patch(Rectangle((r["s_frac"] - 0.045, r["d_mm"] - 4.5), 0.09, 9, fc=col, ec=GRID, lw=0.6))
        s = np.linspace(0.25, 0.95, 50)
        dm = [analytic.d_max(d, d.L * (1 - x)) * 1e3 for x in s]
        a.plot(s, dm, color=CAT[1], lw=1.5)
        a.axhline(analytic.d_min(d) * 1e3, color=CAT[1], lw=1.0, ls=":")
        a.axhline(40, color=INK, lw=0.8, ls="--")
        a.set_xlim(0.25, 0.95); a.set_ylim(5, 85); a.set_title(f"φ = {phi:g}°", loc="left")
        a.set_xlabel("contact point s / L"); a.grid(False)
    ax[0].set_ylabel("cylinder diameter (mm)")
    ax[0].text(0.27, 41.5, "egg 40 mm", fontsize=7, color=INK2)
    fig.legend(handles=[Patch(fc=OK, label="wrapped ≥ 180° and held"),
                        Patch(fc=PART, ec=GRID, label="wrapped, not held"),
                        Patch(fc="white", ec=GRID, label="failed"),
                        Line2D([], [], color=CAT[1], label="paper bound d_max(L − s), Eq. 15"),
                        Line2D([], [], color=CAT[1], ls=":", label="paper d_min, Eq. 14")],
               loc="lower center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, -0.07))
    _save(fig, "fig5_size_position")


# ------------------------------------------------------------------ Fig. 6
def fig_egg_map():
    from .exp_workspace import EGG
    res = json.loads((R / "grasp" / "egg_map.json").read_text())
    phis = sorted({r["phi_deg"] for r in res})
    fig, ax = plt.subplots(1, len(phis), figsize=(2.0 * len(phis), 3.6), sharey=True)
    for a, phi in zip(ax, phis):
        rows = [r for r in res if r["phi_deg"] == phi and r["s_frac"] != "task"]
        for r in rows:
            col = OK if r["success"] else (PART if r["enclosure_deg"] >= 180 else "#f6f5f2")
            a.add_patch(Rectangle((r["x"] - 0.009, r["y"] - 0.009), 0.018, 0.018, fc=col, ec="white", lw=0.8))
        w = analytic.spiral(D.sweep_by_phi()[phi])
        L = D.sweep_by_phi()[phi].L
        a.fill([-w.base_width_mm / 2e3, w.base_width_mm / 2e3, w.tip_width_mm / 2e3, -w.tip_width_mm / 2e3],
               [0, 0, L, L], color=INK, alpha=0.85, lw=0)
        t = [r for r in res if r["phi_deg"] == phi and r["s_frac"] == "task"][0]
        a.plot(*EGG, marker="*", ms=11, color=CAT[1] if t["success"] else "white", mec=CAT[1], mew=1.2)
        a.set_aspect("equal"); a.set_xlim(-0.04, 0.155); a.set_ylim(-0.035, 0.255)
        n_ok = sum(r["success"] for r in rows)
        a.set_title(f"φ = {phi:g}°\n{n_ok} / {len(rows)} positions", loc="left", fontsize=8.5)
        a.set_xlabel("x (m)"); a.grid(False)
    ax[0].set_ylabel("y (m)")
    fig.legend(handles=[Patch(fc=OK, label="egg wrapped and held"), Patch(fc=PART, label="wrapped, not held"),
                        Patch(fc="#f6f5f2", label="failed"),
                        Line2D([], [], color=CAT[1], marker="*", ls="", ms=10, label="task egg position (filled = success)")],
               loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    _save(fig, "fig6_egg_map")


# ------------------------------------------------------------------ Fig. 7
def fig_validation():
    v = json.loads((R / "validation" / "validation.json").read_text())
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4))
    g = [x for x in v["geometry"] if x["design"] == "osr_print_phi10"][0]
    n = np.arange(1, len(g["pipeline_chords_mm"]) + 1)
    ax[0].plot(n, g["osr_chords_mm"], color=CAT[1], lw=3.5, alpha=0.5, label="Open-Spiral-Robots: chord")
    ax[0].plot(n, g["pipeline_chords_mm"], "o", ms=3.5, color=CAT[0], label="this repository: chord")
    ax[0].plot(n, g["osr_width_mm"], color=CAT[2], lw=3.5, alpha=0.5, label="Open-Spiral-Robots: width")
    ax[0].plot(n, g["pipeline_width_mm"], "s", ms=3.2, color=CAT[0], label="this repository: width")
    worst = max(max(x["max_rel_err_chord"], x["max_rel_err_width"]) for x in v["geometry"])
    ax[0].set_xlabel("unit number from the tip"); ax[0].set_ylabel("mm")
    ax[0].set_title(f"(a) Authors' print, unit by unit\n(largest relative difference {worst:.0e})", loc="left")
    ax[0].legend(fontsize=7)
    p = [x for x in v["packed"] if x["design"] == "phi10p0"][0]
    li, ls_ = np.array(p["line_ideal"]), np.array(p["line_sim"])
    ax[1].plot(li[:, 0], li[:, 1], color=CAT[1], lw=3.5, alpha=0.5, label="analytic spiral (all joints at Δθ)")
    ax[1].plot(ls_[:, 0], ls_[:, 1], "o-", ms=3, lw=1.0, color=CAT[0], label="MuJoCo, cable pulled")
    ax[1].set_aspect("equal"); ax[1].legend(fontsize=7, loc="lower left")
    ax[1].set_title(f"(b) Packed shape, φ = 10° (RMS {p['rms_mm']:.1f} mm)", loc="left")
    lim = v["limits"]
    for key, c, lab in (("pipeline", CAT[1], "generator default ±π"), ("dtheta", CAT[0], "±Δθ (paper)")):
        q = np.array(lim[key]["q_over_dtheta"])
        ax[2].plot(np.arange(2, len(q) + 2), q, "o-", ms=3.5, lw=1.2, color=c,
                   label=f"{lab}: max {q.max():.2f}")
    ax[2].axhline(1.0, color=INK, lw=0.8, ls="--")
    ax[2].text(2, 1.03, "units touch", fontsize=7.5, color=INK2)
    ax[2].set_xlabel("joint number (base → tip)"); ax[2].set_ylabel("|joint angle| / Δθ")
    ax[2].set_title("(c) Joint limits: current robot pulled to 1.25 F_pack", loc="left"); ax[2].legend(fontsize=7.5)
    _save(fig, "fig7_validation")


# ------------------------------------------------------------------ Fig. 8
def fig_summary(table: list[dict]):
    keys = [("f_pack_N", "Packing force (N)"), ("mass_g", "Body mass (g)"),
            ("dyn_area_cm2", "Dynamic reach area (cm²)"), ("egg_success", "Egg positions grasped"),
            ("dmax40_frac", "Contact span that holds a 40 mm object (s/L)"),
            ("distal_ideal", "Distal-third share of bending")]
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 5.6))
    for a, (k, t) in zip(ax.ravel(), keys):
        a.plot([r["phi_deg"] for r in table], [r[k] for r in table], "-o", ms=4.5, color=CAT[0])
        a.axvline(6.3, color=MUTED, lw=0.8, ls=":")
        a.set_title(t, loc="left"); a.set_xlabel("taper angle φ (deg)")
    ax[0, 0].text(6.5, ax[0, 0].get_ylim()[1] * 0.9, "current", fontsize=7, color=INK2)
    fig.tight_layout()
    _save(fig, "fig8_summary")


def main():
    fig_curl(); fig_robustness(); fig_workspace(); fig_validation()
    if (R / "grasp" / "size_position.json").exists():
        fig_size_position()
    if (R / "grasp" / "egg_map.json").exists():
        fig_egg_map()


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------ Fig. 9
def fig_extra():
    ex = json.loads((R / "grasp" / "extra.json").read_text())
    rf = json.loads((R / "grasp" / "refine_35_45.json").read_text())
    sp = json.loads((R / "grasp" / "size_position.json").read_text())
    panels = [("phi6p3", "cubic", "Current geometry (6.3°)\nphysical stiffness law K ∝ W³", sp),
              ("current_6p3", "mjlab", "Current robot as modelled\nin spirob-mjlab (0.915ⁱ law)", ex),
              ("phi8p0", "cubic", "Recommended 8°\n(K ∝ W³)", sp),
              ("osr_print_phi10", "cubic", "Authors' print you hold\n(10°, 24 cm, 5.2 mm tip)", ex)]
    fig, ax = plt.subplots(1, 4, figsize=(9.5, 3.4), sharey=True)
    for a, (name, law, title, src) in zip(ax, panels):
        rows = [r for r in src if r["design"] == name and r.get("law", "cubic") == law and isinstance(r.get("s_frac"), float)]
        rows += [r for r in rf if r["design"] == name and law == "cubic"]
        for r in rows:
            col = OK if r["success"] else (PART if r["enclosure_deg"] >= 180 else "white")
            h = 2.3
            a.add_patch(Rectangle((r["s_frac"] - 0.045, r["d_mm"] - h), 0.09, 2 * h, fc=col, ec=GRID, lw=0.6))
        a.axhspan(41.5, 45.5, color=CAT[1], alpha=0.10, lw=0)
        a.set_yticks([10, 20, 30, 35, 40, 45, 50, 60, 70, 80])
        a.set_xlim(0.25, 0.95); a.set_ylim(5, 85); a.grid(False)
        a.set_title(title, loc="left", fontsize=8.5); a.set_xlabel("contact point s / L")
    ax[0].set_ylabel("cylinder diameter (mm)")
    fig.legend(handles=[Patch(fc=OK, label="wrapped ≥ 180° and held"), Patch(fc=PART, ec=GRID, label="wrapped, not held"),
                        Patch(fc="white", ec=GRID, label="failed"),
                        Patch(fc=CAT[1], alpha=0.10, label="hen's egg, 42-45 mm")], loc="lower center", ncol=4, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.08))
    _save(fig, "fig9_grasp_extra")
