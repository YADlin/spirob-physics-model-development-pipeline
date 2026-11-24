"""Figures for part 2 of the study (cable friction, length, wrap and grasp)."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from . import designs as D
from .figures import CAT, INK, INK2, MUTED, GRID, _save  # noqa: F401  (shared style)

R = D.STUDY / "results"
MU_RAMP = {0.0: "#b7d3f6", 0.1: "#6da7ec", 0.2: "#2a78d6", 0.3: "#0d366b"}   # ordinal blue
PHI_COL = {6.3: CAT[0], 8.0: CAT[1], 10.0: CAT[2]}                         # categorical
L_LABEL = {0.22628: "22.6 cm (current)", 0.30: "30 cm", 0.375: "37.5 cm", 0.45: "45 cm"}


def _lu():
    return json.loads((R / "friction" / "load_unload.json").read_text())


def fig_friction_curl():
    lu = _lu()
    by = {(r["design"], r["mu"]): r for r in lu}
    fig, ax = plt.subplots(1, 4, figsize=(14.5, 3.6))
    # (a) lock force along the body, 45 cm / 8 deg
    name = "L450_phi8p0"
    for mu, c in MU_RAMP.items():
        r = by[(name, mu)]
        s, f = np.array(r["s_frac"]), np.array(r["lock_F"], float)
        ax[0].semilogy(s, f, "-o", ms=3, color=c, label=f"μ = {mu:g}")
    ax[0].set(xlabel="joint position s/L (0 base, 1 tip)", ylabel="cable force at which the joint locks (N)",
              title="(a) 45 cm, 8°: every joint still locks tip-first")
    ax[0].legend(fontsize=8)
    # (b) bend centroid when the body has turned 180 deg
    for phi, c in PHI_COL.items():
        for mu, ls in ((0.0, "-"), (0.3, "--")):
            xs, ys = [], []
            for L in D.LENGTHS:
                r = by[(D.length_design(L, phi).name, mu)]
                xs.append(r["length_m"] * 100); ys.append(r["centroid_180"])
            ax[1].plot(xs, ys, ls, marker="o", ms=3.5, color=c)
    ax[1].set(xlabel="body length (cm)", ylabel="where the bending is (0 base, 1 tip)", ylim=(0.6, 1.0),
              title="(b) bending centre at 180° of curl")
    ax[1].legend(handles=[Line2D([], [], color=PHI_COL[p], label=f"{p:g}°") for p in PHI_COL] +
                 [Line2D([], [], color=MUTED, ls="-", label="no friction"),
                  Line2D([], [], color=MUTED, ls="--", label="μ = 0.3")], fontsize=7.5, ncol=2)
    # (c) packed fraction at 1.25 x packing force
    for phi, c in PHI_COL.items():
        for L, mk in zip(D.LENGTHS, "os^D"):
            ys = [by[(D.length_design(L, phi).name, mu)]["locked_frac_peak"] for mu in MU_RAMP]
            ax[2].plot(list(MU_RAMP), ys, "-", marker=mk, ms=4, color=c, alpha=0.85)
    ax[2].set(xlabel="cable friction coefficient μ", ylabel="fraction of joints locked",
              title="(c) does it still pack? (1.25 × frictionless force)")
    ax[2].legend(handles=[Line2D([], [], color=PHI_COL[p], label=f"{p:g}°") for p in PHI_COL] +
                 [Line2D([], [], color=MUTED, marker=mk, ls="", label=L_LABEL[L]) for L, mk in zip(D.LENGTHS, "os^D")],
                 fontsize=7, ncol=1, loc="lower left")
    # (d) hysteresis: tip path out and back, 45 cm 8 deg
    for mu, c in ((0.0, MU_RAMP[0.1]), (0.2, MU_RAMP[0.3])):
        z = np.load(R / "friction" / f"{name}__mu{mu}.npz")
        tip, F = z["tip"] * 100, z["F"]
        k = int(np.argmax(F))
        ax[3].plot(tip[:k, 0], tip[:k, 1], "-", color=c, lw=1.8, label=f"μ = {mu:g}, pulling")
        ax[3].plot(tip[k:, 0], tip[k:, 1], ":", color=c, lw=1.8, label=f"μ = {mu:g}, releasing")
    ax[3].plot(0, 0, "s", color=INK2, ms=6)
    ax[3].set(xlabel="x (cm)", ylabel="y (cm)", title="(d) tip path, 45 cm 8°: friction = memory")
    ax[3].set_aspect("equal")
    ax[3].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig10_friction_curl")


def fig_friction_workspace():
    ws = json.loads((R / "friction" / "workspace.json").read_text())
    by = {(r["design"], r["mu"]): r for r in ws}
    names = ["phi6p3", "phi8p0", "phi10p0", "L450_phi6p3", "L450_phi8p0", "L450_phi10p0", "authors_45cm"]
    labels = ["22.6 cm 6.3°", "22.6 cm 8°", "22.6 cm 10°", "45 cm 6.3°", "45 cm 8°", "45 cm 10°", "authors 45 cm"]
    fig = plt.figure(figsize=(14, 3.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.5])
    for i, (nm, lab) in enumerate((("phi8p0", "22.6 cm, 8°"), ("L450_phi8p0", "45 cm, 8°"), ("authors_45cm", "authors' 45 cm, 10°"))):
        a = fig.add_subplot(gs[i])
        for mu, c, msz in ((0.2, MU_RAMP[0.2], 1.4), (0.0, CAT[1], 1.0)):
            z = np.load(R / "friction" / f"ws_{nm}__mu{mu}.npz")["tips"] * 100
            a.plot(z[:, 0], z[:, 1], ".", ms=msz, color=c, alpha=0.5)
        a.plot(0, 0, "s", color=INK2, ms=6)
        a.set_aspect("equal"); a.set(title=f"({'abc'[i]}) {lab}", xlabel="x (cm)", ylabel="y (cm)" if i == 0 else None)
    a.legend(handles=[Line2D([], [], color=CAT[1], marker=".", ls="", ms=8, label="no cable friction"),
                      Line2D([], [], color=MU_RAMP[0.2], marker=".", ls="", ms=8, label="μ = 0.2")], fontsize=7.5, loc="lower right")
    a = fig.add_subplot(gs[3])
    x = np.arange(len(names))
    a.bar(x - 0.2, [by[(n, 0.0)]["area_cm2"] for n in names], 0.38, color=CAT[1], label="no cable friction")
    a.bar(x + 0.2, [by[(n, 0.2)]["area_cm2"] for n in names], 0.38, color=MU_RAMP[0.2], label="μ = 0.2")
    a.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    a.set(ylabel="area the tip visited (cm²)", title="(d) same random cable programme, 40 s")
    a.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig11_friction_workspace")




# ------------------------------------------------------------------ wrapping
WRAP_BINS = [(0, 45, "#e6e5e1"), (45, 90, "#b7d3f6"), (90, 135, "#6da7ec"), (135, 180, "#2a78d6"),
             (180, 1e9, "#0d366b")]
CONDS = [(0.0, True, "egg in a holder, no friction"), (0.2, True, "egg in a holder, μ = 0.2"),
         (0.0, False, "egg on pedestal, no friction"), (0.2, False, "egg on pedestal, μ = 0.2")]


def _wrap_col(w):
    for lo, hi, c in WRAP_BINS:
        if lo <= w < hi:
            return c
    return WRAP_BINS[-1][2]


def _body_outline(ax, name):
    from . import model as M
    from .analytic import osr_full_design
    des = osr_full_design() if name == "authors_45cm" else next(b for b in D.length_grid() if b.name == name)
    sm = M.make(des, tag="fig_outline")
    s = np.r_[0, sm.joint_s_m, sm.length_m] * 100
    w = np.r_[sm.joint_width_m[0], sm.joint_width_m, sm.joint_width_m[-1] * 0.8] * 50
    ax.fill_betweenx(s, -w, w, color="#52514e", alpha=0.85, lw=0)
    return sm.length_m


def _key(r):
    return (r["design"], round(r["x"], 4), round(r["y"], 4), round(r["d_mm"]), r["strategy"], r["mu"], r["anchored"])


def _wrap_results(which="main"):
    """Results with the 24 Nov re-runs (holder weld, non-creeping pedestal)
    substituted wherever a case was re-run."""
    res = json.loads((R / "wrap" / f"{which}.json").read_text())
    f = R / "wrap" / "recheck.json"
    if which in ("main", "size") and f.exists():
        src = {"main": ("main", "free"), "size": ("size", "size_free")}[which]
        new = {_key(r): r for r in json.loads(f.read_text()) if r.get("source") in src}
        c = R / "wrap" / "carry.json"            # final release-then-move test of every wrap
        if c.exists():
            new.update({_key(r): r for r in json.loads(c.read_text())})
        res = [new.get(_key(r), r) for r in res]
    return res


def fig_wrap_maps(names=("phi6p3", "L300_phi6p3", "L375_phi6p3", "L450_phi6p3", "authors_45cm"),
                  titles=("22.6 cm, 6.3° (current)", "30 cm, 6.3°", "37.5 cm, 6.3°", "45 cm, 6.3°", "authors' 45 cm, 10°"),
                  fname="fig12_wrap_maps"):
    res = [r for r in _wrap_results() if not r.get("error")]
    fig, axes = plt.subplots(len(CONDS), len(names), figsize=(2.75 * len(names), 2.9 * len(CONDS)))
    for j, (nm, tt) in enumerate(zip(names, titles)):
        for i, (mu, anc, lab) in enumerate(CONDS):
            a = axes[i, j]
            L = _body_outline(a, nm)
            for r in res:
                if r["design"] != nm or r["mu"] != mu or r["anchored"] != anc:
                    continue
                x, y = r["x"] * 100, r["y"] * 100
                rr = r["d_mm"] / 20
                if r.get("dropped"):
                    a.plot(x, y, "x", color=CAT[1], ms=6, mew=1.6)
                    continue
                c = _wrap_col(r["wrap_deg"])
                if r.get("in_curl") and not r.get("wrap"):
                    a.plot(x, y, "s", ms=8, color=WRAP_BINS[-1][2], mec=INK, mew=1.0)
                else:
                    a.plot(x, y, "o", ms=8, color=c, mec=INK if r.get("wrap") else "none", mew=1.6)
                if r.get("grasp"):
                    a.plot(x, y, "*", color="white", ms=7)
            # egg to scale, bottom right
            a.add_patch(plt.Circle((0.5 * L * 100, 0.02 * L * 100 + 2.25), 2.25, fill=False, ec=INK2, lw=0.8, ls="--"))
            if i == 0:
                a.text(0.5 * L * 100, 0.02 * L * 100 + 5.2, "egg", ha="center", fontsize=7, color=INK2)
            a.set_xlim(-0.15 * L * 100, 0.62 * L * 100)
            a.set_ylim(-0.05 * L * 100, 1.25 * L * 100)
            a.set_aspect("equal")
            a.tick_params(labelsize=7)
            if i == 0:
                a.set_title(tt, fontsize=9.5)
            if j == 0:
                a.set_ylabel(lab + "\ny (cm)", fontsize=8.5)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=c, label=f"{lo}–{hi:g}°" if hi < 1e8 else "≥ 180° (wrap)")
               for lo, hi, c in WRAP_BINS]
    handles[-1] = plt.Line2D([], [], marker="o", ls="", ms=8, color=WRAP_BINS[-1][2], mec=INK, mew=1.4,
                             label="≥ 180°, one stretch (wrap)")
    handles += [plt.Line2D([], [], marker="s", ls="", ms=8, color=WRAP_BINS[-1][2], mec=INK, label="inside the curl (body winds ≥ 270°)"),
                plt.Line2D([], [], marker="x", ls="", ms=7, mew=1.6, color=CAT[1], label="egg fell off the pedestal"),
                plt.Line2D([], [], marker="*", ls="", ms=8, color=INK2, label="released, held and moved ≥ 1 diameter")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, title="largest contact sweep of one unbroken stretch of body",
               title_fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, fname)


def fig_wrap_summary():
    res = [r for r in _wrap_results() if not r.get("error")]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    for k, anc in enumerate((True, False)):
        a = ax[k]
        for phi, c in PHI_COL.items():
            for mu, ls in ((0.0, "--"), (0.2, "-")):
                xs, ys = [], []
                for L in D.LENGTHS:
                    nm = D.length_design(L, phi).name
                    s = [r for r in res if r["design"] == nm and r["mu"] == mu and r["anchored"] == anc]
                    if s:   # free egg: best wrap even after it left the pedestal (all such are failures)
                        xs.append(L * 100); ys.append(max(r["wrap_deg"] for r in s))
                a.plot(xs, ys, ls, marker="o", ms=4, color=c)
        for mu, mk in ((0.0, "D"), (0.2, "s")):
            s = [r for r in res if r["design"] == "authors_45cm" and r["mu"] == mu and r["anchored"] == anc]
            if s:
                a.plot(46.4, max(r["wrap_deg"] for r in s), mk, color=INK, ms=6, mfc="white" if mu == 0 else INK)
        a.axhline(180, color=INK2, lw=1, ls=":")
        a.text(31, 184, "half wrap (success)", fontsize=7.5, color=INK2)
        a.set(xlabel="body length (cm)", title="(a) egg in a holder" if anc else
              "(b) egg free on its pedestal (every case above 66° had\n already pushed the egg off the pedestal)")
    ax[0].set_ylabel("best wrap over the grid (deg)")
    ax[1].legend(handles=[Line2D([], [], color=PHI_COL[p], label=f"{p:g}°") for p in PHI_COL] +
                 [Line2D([], [], color=MUTED, ls="--", label="no friction"), Line2D([], [], color=MUTED, ls="-", label="μ = 0.2"),
                  Line2D([], [], color=INK, marker="s", ls="", label="authors' 45 cm (open: μ = 0)")],
                 fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    _save(fig, "fig13_wrap_summary")


def fig_size_position():
    res = [r for r in _wrap_results("size") if not r.get("error")]
    from .analytic import d_max, osr_full_design
    names = ("phi8p0", "L450_phi8p0", "authors_45cm")
    titles = ("22.6 cm, 8°", "45 cm, 8°", "authors' 45 cm, 10°")
    conds = (("pull", 0.2, True, "pull from straight, held object"), ("reach", 0.2, True, "reach then wrap, held object"),
             ("reach", 0.2, False, "reach then wrap, free object"))
    fig, axes = plt.subplots(len(conds), len(names), figsize=(11, 8.2), sharex=True)
    for j, nm in enumerate(names):
        des = osr_full_design() if nm == "authors_45cm" else next(b for b in D.length_grid() if b.name == nm)
        for i, (st, mu, anc, lab) in enumerate(conds):
            a = axes[i, j]
            for r in res:
                if r["design"] != nm or r["strategy"] != st or r["mu"] != mu or r["anchored"] != anc:
                    continue
                if r.get("dropped"):
                    a.plot(r["s_frac"], r["d_mm"], "x", color=CAT[1], ms=7, mew=1.6)
                elif r.get("in_curl") and not r.get("wrap"):
                    a.plot(r["s_frac"], r["d_mm"], "s", ms=10, color=WRAP_BINS[-1][2], mec=INK, mew=1.0)
                else:
                    a.plot(r["s_frac"], r["d_mm"], "o", ms=11, color=_wrap_col(r["wrap_deg"]),
                           mec=INK if r.get("wrap") else "none", mew=1.4)
                    if r.get("grasp"):
                        a.plot(r["s_frac"], r["d_mm"], "*", color="white", ms=7)
            s = np.linspace(0.3, 0.95, 50)
            a.plot(s, [d_max(des, des.L * (1 - x)) * 1e3 for x in s], color=CAT[1], lw=1.3)
            a.axhspan(42, 46, color=CAT[2], alpha=0.15, lw=0)
            a.set_ylim(5, 70)
            if i == 0:
                a.set_title(titles[j])
            if j == 0:
                a.set_ylabel(lab + "\nobject diameter (mm)", fontsize=8.5)
            if i == len(conds) - 1:
                a.set_xlabel("contact position s/L (0 base, 1 tip)")
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=8, color=c, label=f"{lo}–{hi:g}°") for lo, hi, c in WRAP_BINS[:-1]] +
               [plt.Line2D([], [], marker="o", ls="", ms=8, color=WRAP_BINS[-1][2], mec=INK, mew=1.4, label="≥ 180° wrap"),
                plt.Line2D([], [], marker="s", ls="", ms=8, color=WRAP_BINS[-1][2], mec=INK, label="inside the curl (≥ 270°)"),
                plt.Line2D([], [], marker="x", ls="", color=CAT[1], label="object pushed off"),
                plt.Line2D([], [], marker="*", ls="", color=INK2, label="released, held and moved ≥ 1 diameter"),
                plt.Line2D([], [], color=CAT[1], label="paper's bound, Eq. 15 with free length L − s"),
                plt.Rectangle((0, 0), 1, 1, color=CAT[2], alpha=0.3, label="hen's egg")],
               loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "fig14_size_position")


def _spread_mm(tips, ref):
    """Mean distance (mm) from each settled tip to the nearest frictionless one."""
    d = np.linalg.norm(tips[:, None, :] - ref[None, :, :], axis=-1)
    return float(d.min(1).mean() * 1e3)


def wss_table():
    names = ["phi6p3", "phi8p0", "phi10p0", "L450_phi6p3", "L450_phi8p0", "L450_phi10p0", "authors_45cm"]
    out = []
    for nm in names:
        ref = np.load(R / "friction" / f"wss_{nm}__mu0.0.npz")["tips"]
        for mu in (0.0, 0.1, 0.2, 0.3):
            t = np.load(R / "friction" / f"wss_{nm}__mu{mu}.npz")["tips"]
            cells = len({tuple(c) for c in np.floor(t / 0.005).astype(int)})
            out.append(dict(design=nm, mu=mu, spread_mm=_spread_mm(t, ref) if mu else 0.0, cells=cells))
    (R / "friction" / "workspace_settled_summary.json").write_text(json.dumps(out, indent=1))
    return out


def fig_friction_workspace_settled():
    tab = wss_table()
    fig = plt.figure(figsize=(14, 3.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.4])
    for i, (nm, lab) in enumerate((("phi8p0", "22.6 cm, 8°"), ("L450_phi8p0", "45 cm, 8°"), ("authors_45cm", "authors' 45 cm, 10°"))):
        a = fig.add_subplot(gs[i])
        for mu, c, ms in ((0.3, MU_RAMP[0.3], 3.0), (0.0, CAT[1], 3.5)):
            z = np.load(R / "friction" / f"wss_{nm}__mu{mu}.npz")["tips"] * 100
            a.plot(z[:, 0], z[:, 1], "o", ms=ms, color=c, alpha=0.75, mec="none")
        a.plot(0, 0, "s", color=INK2, ms=6)
        a.set_aspect("equal"); a.set(title=f"({'abc'[i]}) {lab}", xlabel="x (cm)", ylabel="y (cm)" if i == 0 else None)
    a.legend(handles=[Line2D([], [], color=CAT[1], marker="o", ls="", ms=6, label="no cable friction"),
                      Line2D([], [], color=MU_RAMP[0.3], marker="o", ls="", ms=6, label="μ = 0.3")], fontsize=7.5, loc="lower right")
    a = fig.add_subplot(gs[3])
    names = ["phi6p3", "phi8p0", "phi10p0", "L450_phi6p3", "L450_phi8p0", "L450_phi10p0", "authors_45cm"]
    labels = ["22.6 cm 6.3°", "22.6 cm 8°", "22.6 cm 10°", "45 cm 6.3°", "45 cm 8°", "45 cm 10°", "authors 45 cm"]
    x = np.arange(len(names))
    for k, mu in enumerate((0.1, 0.2, 0.3)):
        a.bar(x + (k - 1) * 0.27, [next(t["spread_mm"] for t in tab if t["design"] == n and t["mu"] == mu) for n in names],
              0.26, color=MU_RAMP[mu], label=f"μ = {mu:g}")
    a.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    a.set(ylabel="mean distance off the frictionless curve (mm)", title="(d) new tip positions friction makes reachable")
    a.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig11_friction_workspace")



def fig_grasp_summary():
    """Fig. 15: every held-egg (45 mm) case that came near success, by body."""
    from .summary2 import table
    rows = [r for r in table() if r["d_mm"] == 45]
    order = ["phi6p3", "phi8p0", "phi10p0", "L300_phi6p3", "L300_phi8p0", "L375_phi6p3", "L375_phi8p0",
             "L375_phi10p0", "L450_phi6p3", "L450_phi8p0", "L450_phi10p0", "authors_45cm"]
    lab = {"phi6p3": "22.6 cm 6.3°", "phi8p0": "22.6 cm 8°", "phi10p0": "22.6 cm 10°", "L300_phi6p3": "30 cm 6.3°",
           "L300_phi8p0": "30 cm 8°", "L375_phi6p3": "37.5 cm 6.3°", "L375_phi8p0": "37.5 cm 8°",
           "L375_phi10p0": "37.5 cm 10°", "L450_phi6p3": "45 cm 6.3°", "L450_phi8p0": "45 cm 8°",
           "L450_phi10p0": "45 cm 10°", "authors_45cm": "authors 45 cm"}
    by = {r["design"]: r for r in rows}
    names = [n for n in order if n in by]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    x = np.arange(len(names))
    for k, (key, c, lb) in enumerate((("in_curl", "#6da7ec", "egg inside the curl"),
                                       ("secured", "#2a78d6", "still held after release"),
                                       ("grasp", "#0d366b", "moved ≥ 45 mm while held"))):
        ax[0].bar(x + (k - 1) * 0.27, [by[n][key] for n in names], 0.26, color=c, label=lb)
    ax[0].plot(x, [by[n]["cases"] for n in names], "_", color=INK, ms=14, mew=1.5, label="cases tried")
    ax[0].set_xticks(x, [lab[n] for n in names], rotation=35, ha="right", fontsize=8)
    ax[0].set(ylabel="positions (egg in a holder)", title="(a) 45 mm egg: positions near success")
    ax[0].set_ylim(0, 36)
    ax[0].legend(fontsize=7.5, loc="upper left", ncol=2)
    ax[1].bar(x, [by[n]["max_carry_mm"] for n in names], color=CAT[0])
    ax[1].axhline(45, color=INK2, ls=":", lw=1)
    ax[1].set_xticks(x, [lab[n] for n in names], rotation=35, ha="right", fontsize=8)
    ax[1].set(ylabel="mm", title="(b) furthest the cables moved the held egg")
    lo = [by[n]["push_N"][0] if by[n]["push_N"] else np.nan for n in names]
    hi = [by[n]["push_N"][1] if by[n]["push_N"] else np.nan for n in names]
    ax[2].vlines(x, lo, hi, color=CAT[1], lw=5)
    ax[2].axhline(0.24, color=INK2, ls=":", lw=1)
    ax[2].text(0, 0.3, "pedestal friction 0.24 N", fontsize=7.5, color=INK2)
    ax[2].set_yscale("log")
    ax[2].set_xticks(x, [lab[n] for n in names], rotation=35, ha="right", fontsize=8)
    ax[2].set(ylabel="N", title="(c) push on the egg while it is wrapped")
    fig.tight_layout()
    _save(fig, "fig15_grasp_summary")


def main():
    fig_grasp_summary()
    fig_friction_curl()
    fig_friction_workspace_settled()
    fig_wrap_maps()
    fig_wrap_summary()
    fig_size_position()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            globals()[name]()
    else:
        main()
