"""
preview.py  —  Interactive 2-D SpiRob geometry preview.

STANDALONE (interactive, with hover + click measuring):
    python preview.py --params params.json
    python preview.py --params params.json --nlobe    # also show cross-section
    python preview.py --params params.json --proceed  # skip yes/no prompt

Exit codes (standalone):
    0 = user approved  ("y") or --proceed flag set
    1 = user rejected  ("n") or window closed without approval
"""

import math
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import CheckButtons


# ══════════════════════════════════════════════════════════════════════════════
#  Geometry helpers  (duplicated lightly so preview.py is self-contained)
# ══════════════════════════════════════════════════════════════════════════════

def _phi_from_b(b):
    e = math.exp(2 * math.pi * b)
    return 2.0 * math.atan2(b * (e - 1.0), math.sqrt(1.0 + b * b) * (e + 1.0))

def _solve_b(phi_target, lo=1e-6, hi=2.0):
    f = lambda b: _phi_from_b(b) - phi_target
    flo, fhi = f(lo), f(hi)
    for _ in range(50):
        if flo * fhi <= 0:
            break
        hi *= 1.5; fhi = f(hi)
    else:
        return max(1e-6, math.tan(phi_target / 2.0))
    for _ in range(120):
        mid = 0.5 * (lo + hi); fmid = f(mid)
        if abs(fmid) < 1e-14 or (hi - lo) < 1e-14:
            return mid
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)

def _normalize(v):
    n = np.linalg.norm(v); return v / n if n > 0 else v

def _angle_between(v1, v2):
    return np.arccos(np.clip(np.dot(_normalize(v1), _normalize(v2)), -1, 1))

def _rotate2d(pts, angle):
    R = np.array([[math.cos(angle), -math.sin(angle)],
                  [math.sin(angle),  math.cos(angle)]])
    return pts @ R.T

def _build_quads(params):
    """Replicate the full spiral → straighten → invert pipeline from params dict."""
    L               = params["L"]
    d_tip           = params["d_tip"]
    phi_deg         = params["phi_deg"]
    Delta_theta_deg = params["Delta_theta_deg"]

    phi         = math.radians(phi_deg)
    delta_theta = math.radians(Delta_theta_deg)
    b           = _solve_b(phi)
    e2pb        = math.exp(2 * math.pi * b)
    a           = d_tip / (e2pb - 1.0)
    A           = (a / b) * math.sqrt(b ** 2 + 1)
    Q0          = (1.0 / b) * math.log(1.0 + L / A)
    N           = int(math.ceil(Q0 / delta_theta))
    theta       = np.array([min(k * delta_theta, Q0) for k in range(N + 1)])

    r1 = a * np.exp(b * theta)
    side_A = np.column_stack((r1 * np.cos(theta), r1 * np.sin(theta)))

    theta_s = theta - 2 * math.pi
    r2 = a * np.exp(b * theta_s)
    side_B = np.column_stack((r2 * np.cos(theta), r2 * np.sin(theta)))

    # Form quads [A0, A1, B1, B0]
    raw = [np.array([side_A[i], side_A[i+1], side_B[i+1], side_B[i]])
           for i in range(len(theta) - 1)]

    # Straighten
    straight = []
    q0 = raw[0].copy()
    hv = q0[1] - q0[0]
    v3 = np.array([hv[0], hv[1], 0.0])
    ref = np.array([0.0, 1.0, 0.0])
    angle = _angle_between(v3, ref)
    if np.cross(v3, ref)[2] < 0: angle = -angle
    q0r = _rotate2d(q0 - q0[0], angle)
    straight.append(q0r)
    cur = q0r[1].copy()
    for q in raw[1:]:
        hv = q[1] - q[0]
        h3 = np.array([hv[0], hv[1], 0.0])
        ang = _angle_between(h3, ref)
        if np.cross(h3, ref)[2] < 0: ang = -ang
        qr = _rotate2d(q - q[0], ang) + cur
        straight.append(qr)
        cur = qr[1].copy()

    # Invert (biggest at bottom, index 0)
    inv = []
    for q in straight:
        qi = q.copy()
        for j in range(4):
            qi[j][1] = -qi[j][1] + L
        inv.append(qi)
    inv.reverse()
    return inv   # each quad: [A1, A0, B0, B1]  (post-invert naming)


def _tendon_path(quads, tendon_inward_shift, phi_deg):
    """
    Return (xs, ys) arrays for the cable-0 tendon path in 2D side view.

    B0x/B1x are negative (outer edge is on the left after straightening),
    so we preserve the sign and shift *inward* (toward zero / the axis).
    The inward shift and taper correction from csv2xml are replicated here
    so the preview matches what the XML will actually use.

    Path order: tip → base (s2 of tip, s1 of tip, s2 of next, s1 of next …)
    """
    half_phi = math.radians(phi_deg / 2)
    xs, ys = [], []

    for quad in reversed(quads):          # tip first (quads[0]=base, quads[-1]=tip)
        A1, A0, B0, B1 = quad
        dz   = float(B1[1] - B0[1])
        b0x  = float(B0[0])              # raw value — negative
        sign = -1.0 if b0x < 0 else 1.0  # direction toward axis

        r0_old = abs(b0x)
        r0_new = max(r0_old - tendon_inward_shift, 1e-6)
        r1_new = max(r0_old - tendon_inward_shift - dz * math.tan(half_phi), 1e-6)

        # restore sign so tendon sits on same side as element bodies
        xs += [sign * r1_new, sign * r0_new]
        ys += [float(B1[1]), float(B0[1])]

    return np.array(xs), np.array(ys)


# ══════════════════════════════════════════════════════════════════════════════
#  Per-element stats (for hover / click info)
# ══════════════════════════════════════════════════════════════════════════════

def _element_stats(quads):
    """Return a list of dicts, one per element (index 0 = base)."""
    stats = []
    for i, quad in enumerate(quads):
        A1, A0, B0, B1 = quad
        height   = float(abs(A1[1] - A0[1]))
        w_bottom = float(abs(B0[0])) * 2
        w_top    = float(abs(B1[0])) * 2
        cx       = 0.0
        cy       = float((A0[1] + A1[1]) / 2)
        stats.append(dict(
            idx      = i + 1,
            height   = height,
            w_bottom = w_bottom,
            w_top    = w_top,
            cx       = cx,
            cy       = cy,
            quad     = quad,
        ))
    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  Main drawing function
# ══════════════════════════════════════════════════════════════════════════════

def draw_preview(
    quads,
    params: dict,
    out_path: str = "spirob_preview.png",
    interactive: bool = False,
) -> str:
    """
    Draw the 2-D annotated SpiRob preview.

    Layout
    ------
    Left margin   : total-length double arrow + label
    Robot body    : element fills + outlines (negative-x side)
    Right of robot: tip/base width arrows, taper arc annotation
    Right panel   : summary params box + hover info box
    Bottom strip  : tendon toggle checkbox

    interactive=False  → save PNG, return path  (pipeline use)
    interactive=True   → show live window with hover + click, return path after close
    """
    tendon_shift = params.get("tendon_inward_shift", 0.0015)
    phi_deg      = params.get("phi_deg", 5.7)
    n_elems      = len(quads)

    # ── Geometry pre-compute ──────────────────────────────────────────────────
    tx, ty   = _tendon_path(quads, tendon_shift, phi_deg)
    stats    = _element_stats(quads)
    all_y    = np.concatenate([q[:, 1] for q in quads])
    y_min, y_max = float(all_y.min()), float(all_y.max())
    y_span   = y_max - y_min

    tip_quad  = quads[-1]   # smallest / top
    base_quad = quads[0]    # largest / bottom

    # B-points are on the negative-x side; use their actual signed values
    base_B0   = base_quad[2]   # [x, y]  outer-bottom of base
    base_B1   = base_quad[3]   # [x, y]  outer-top   of base
    tip_B1    = tip_quad[3]    # [x, y]  outer-top   of tip

    base_r    = float(abs(base_B0[0]))   # magnitude of outer radius at base
    # annotation x-positions (elements are on negative-x side)
    x_body    = float(base_B0[0])        # most negative x (outer edge of base)
    x_ann_r   = abs(x_body) * 0.08      # small positive offset right of axis for width arrows
    x_length  = x_body * 1.55           # left of body for length arrow

    # ── Figure + axes ────────────────────────────────────────────────────────
    if interactive:
        matplotlib.use("TkAgg")
    fig = plt.figure(figsize=(8, 11))
    fig.patch.set_facecolor("#f5f6fa")

    # Main drawing axes (left portion)
    ax = fig.add_axes([0.08, 0.08, 0.52, 0.86])
    ax.set_aspect("equal")
    ax.axis("off")

    # Right info panel axis (no frame, just for placing text)
    ax_info = fig.add_axes([0.62, 0.08, 0.36, 0.86])
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis("off")

    # ── Title (figure-level, above everything) ────────────────────────────────
    fig.text(0.38, 0.97, "SpiRob — 2D Side View Preview",
             ha="center", va="top", fontsize=12, fontweight="bold", color="#1a2f5e")

    # ── 1. Element bodies ────────────────────────────────────────────────────
    element_patches = []
    for i, quad in enumerate(quads):
        pts   = np.array(quad)
        alpha = 0.25 + 0.50 * (i / max(n_elems - 1, 1))
        fill  = ax.fill(pts[:, 0], pts[:, 1],
                        color="#3a5fa0", alpha=alpha, zorder=1)[0]
        outline, = ax.plot(
            np.append(pts[:, 0], pts[0, 0]),
            np.append(pts[:, 1], pts[0, 1]),
            color="#1a2f5e", lw=0.9, zorder=2)
        element_patches.append((fill, outline, stats[i]))

    # ── 2. Tendon path ───────────────────────────────────────────────────────
    tendon_line, = ax.plot(tx, ty, color="#e74c3c", lw=1.5,
                           linestyle="--", zorder=4, label="Tendon (cable 0)")
    tendon_dots, = ax.plot(tx, ty, "o", color="#e74c3c", ms=3.0, zorder=5)
    tendon_artists = [tendon_line, tendon_dots]

    # ── 3. Total length arrow (left of body) ─────────────────────────────────
    ax.annotate("", xy=(x_length, y_max), xytext=(x_length, y_min),
                arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.5))
    ax.text(x_length - base_r * 0.08, (y_min + y_max) / 2,
            f"L = {params['L']*1e3:.0f} mm",
            ha="right", va="center", fontsize=8, color="#c0392b", rotation=90)

    # ── 4. Tip / base width arrows (right of axis, pointing left into body) ──
    # Arrow from axis (x=0) to the outer edge (negative x), label on right
    for bx, z, label in [
        (float(tip_B1[0]),  float(tip_B1[1]),  f"tip r = {abs(tip_B1[0])*1e3:.1f} mm"),
        (float(base_B0[0]), float(base_B0[1]), f"base r = {base_r*1e3:.1f} mm"),
    ]:
        ax.annotate("", xy=(bx, z), xytext=(0.0, z),
                    arrowprops=dict(arrowstyle="<->", color="#8e44ad", lw=1.1))
        ax.text(x_ann_r, z, label,
                ha="left", va="center", fontsize=7.5, color="#8e44ad")

    # ── 5. Taper arc (at outer-bottom of base, on the correct negative side) ──
    r_arc = base_r * 0.28
    # Arc runs from vertical (90°) to (90°+phi) since body tapers leftward
    t1_arc = 90.0
    t2_arc = 90.0 + phi_deg
    ax.add_patch(mpatches.Arc(
        (float(base_B0[0]), float(base_B0[1])),
        2 * r_arc, 2 * r_arc,
        angle=0, theta1=t1_arc, theta2=t2_arc,
        color="#27ae60", lw=1.3, zorder=5))
    mid_a = math.radians((t1_arc + t2_arc) / 2)
    ax.text(float(base_B0[0]) - r_arc * 0.3,
            float(base_B0[1]) - r_arc * 1.6,
            f"φ={phi_deg}°", ha="center", va="top",
            fontsize=7.5, color="#27ae60")

    # ── 6. Element count (above robot, inside drawing axes) ──────────────────
    ax.text(x_body * 0.5, y_max + y_span * 0.025,
            f"{n_elems} elements",
            ha="center", va="bottom", fontsize=9,
            fontweight="bold", color="#1a2f5e")

    # ── 7. Summary box in right panel ────────────────────────────────────────
    summary_lines = [
        "── Parameters ──",
        f"L            = {params['L']*1e3:.0f} mm",
        f"d_tip        = {params['d_tip']*1e3:.1f} mm",
        f"phi          = {phi_deg} °",
        f"Delta_theta  = {params['Delta_theta_deg']} °",
        f"n_cables     = {params['n_cables']}",
        f"tendon_shift = {tendon_shift*1e3:.2f} mm",
        "",
        "── Result ──",
        f"elements     = {n_elems}",
    ]
    ax_info.text(0.05, 0.98, "\n".join(summary_lines),
                 va="top", ha="left", fontsize=8, family="monospace",
                 transform=ax_info.transAxes,
                 bbox=dict(boxstyle="round,pad=0.5", fc="white",
                           ec="#cccccc", alpha=0.95))

    # ── 8. Hover info box in right panel (below summary) ─────────────────────
    info_text = ax_info.text(
        0.05, 0.42, "Hover or click an\nelement to inspect",
        va="top", ha="left", fontsize=8, family="monospace",
        transform=ax_info.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", fc="#fffde7",
                  ec="#f0c040", alpha=0.95))

    # ── 9. Legend in right panel ──────────────────────────────────────────────
    ax_info.plot([0.05, 0.25], [0.28, 0.28], color="#e74c3c",
                 lw=1.5, linestyle="--", transform=ax_info.transAxes)
    ax_info.plot([0.15], [0.28], "o", color="#e74c3c", ms=4,
                 transform=ax_info.transAxes)
    ax_info.text(0.28, 0.28, "Tendon\n(cable 0)",
                 va="center", ha="left", fontsize=8, color="#e74c3c",
                 transform=ax_info.transAxes)

    # ── 10. Tendon toggle (bottom strip) ─────────────────────────────────────
    ax_check = fig.add_axes([0.28, 0.015, 0.44, 0.045])
    ax_check.set_facecolor("#eef2f7")
    check = CheckButtons(ax_check, ["Show tendon path"], [True])
    for txt in check.labels:
        txt.set_fontsize(9)
        txt.set_color("#c0392b")

    def _on_toggle(label):
        visible = check.get_status()[0]
        for artist in tendon_artists:
            artist.set_visible(visible)
        fig.canvas.draw_idle()

    check.on_clicked(_on_toggle)

    # ── Interactive: hover + click ────────────────────────────────────────────
    if interactive:
        highlight_fill    = [None]
        highlight_outline = [None]
        selected          = [None]

        def _hit(quad, x, y):
            pts = np.array(quad)
            return (pts[:, 0].min() <= x <= pts[:, 0].max() and
                    pts[:, 1].min() <= y <= pts[:, 1].max())

        def _show_info(s, click=False):
            verb = "Selected" if click else "Hovered"
            info_text.set_text(
                f"{verb}: Elem {s['idx']}/{n_elems}\n"
                f"─────────────────\n"
                f"Height   : {s['height']*1e3:.3f} mm\n"
                f"Width top: {s['w_top']*1e3:.3f} mm\n"
                f"Width bot: {s['w_bottom']*1e3:.3f} mm\n"
                f"Centre Z : {s['cy']*1e3:.2f} mm"
            )
            fig.canvas.draw_idle()

        def _clear_highlight():
            if highlight_fill[0] is not None:
                highlight_fill[0].remove(); highlight_fill[0] = None
            if highlight_outline[0] is not None:
                highlight_outline[0].remove(); highlight_outline[0] = None

        def on_motion(event):
            if event.inaxes != ax: return
            for _, _, s in element_patches:
                if _hit(s["quad"], event.xdata, event.ydata):
                    if selected[0] is None:
                        _show_info(s, click=False)
                    return
            if selected[0] is None:
                info_text.set_text("Hover or click an\nelement to inspect")
                fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes != ax or event.button != 1: return
            _clear_highlight()
            for _, _, s in element_patches:
                if _hit(s["quad"], event.xdata, event.ydata):
                    selected[0] = s
                    pts = np.array(s["quad"])
                    highlight_fill[0] = ax.fill(
                        pts[:, 0], pts[:, 1],
                        color="#f39c12", alpha=0.55, zorder=3)[0]
                    highlight_outline[0], = ax.plot(
                        np.append(pts[:, 0], pts[0, 0]),
                        np.append(pts[:, 1], pts[0, 1]),
                        color="#e67e22", lw=2.0, zorder=6)
                    _show_info(s, click=True)
                    return
            selected[0] = None
            info_text.set_text("Hover or click an\nelement to inspect")
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_motion)
        fig.canvas.mpl_connect("button_press_event",  on_click)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"📐 Preview saved → {out_path}")

    if interactive:
        plt.show()
        plt.close(fig)

    return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  N-lobe cross-section preview  (n_cables >= 3)
# ══════════════════════════════════════════════════════════════════════════════

def draw_nlobe_section(ax, outer_radius, n, params, title=""):
    """
    Draw the n-lobe cross-section: circle ∩ n-gon, with notch holes.

    nlobe_t = 0  →  circumscribed (vertices on circle, no arcs)
    nlobe_t = 1  →  inscribed (edges tangent, arcs fully visible)
    nlobe_t = 0.5 → midpoint default
    """
    nlobe_t      = float(params.get("nlobe_t", 0.5))
    notch_factor = float(params.get("notch_factor", 0.25))

    circ_R = outer_radius
    poly_R = circ_R * (1.0 + nlobe_t * (1.0 / math.cos(math.pi / n) - 1.0))
    side   = 2.0 * poly_R * math.sin(math.pi / n)
    notch_r = (side / 2.0) * notch_factor

    # One vertex always points left (angle π) — aligns with cable-0 plane
    angle_offset = math.pi

    verts = [(poly_R * math.cos(2*math.pi*k/n + angle_offset),
              poly_R * math.sin(2*math.pi*k/n + angle_offset))
             for k in range(n)]

    def _mid(a, b): return ((a[0]+b[0])/2, (a[1]+b[1])/2)
    midpoints = [_mid(verts[k], verts[(k+1) % n]) for k in range(n)]

    ax.set_aspect("equal")
    ax.axis("off")
    pad = circ_R * 0.55
    ax.set_xlim(-circ_R - pad, circ_R + pad)
    ax.set_ylim(-circ_R - pad, circ_R + pad)

    # ── Draw the true intersection: sample points inside BOTH shapes ──────────
    # Build a dense point grid, keep points inside circle AND inside polygon
    # This correctly shows the arc segments + flat facets
    N = 400
    xs_in, ys_in = [], []
    for i in range(N):
        for j in range(N):
            px = -circ_R + 2*circ_R * i / (N-1)
            py = -circ_R + 2*circ_R * j / (N-1)
            # Inside circle?
            if px*px + py*py > circ_R*circ_R:
                continue
            # Inside polygon? Check all half-planes (inward normal test)
            inside_poly = True
            for k in range(n):
                ax1, ay1 = verts[k]
                ax2, ay2 = verts[(k+1) % n]
                # Inward normal: rotate edge vector 90° CW
                ex, ey = ax2-ax1, ay2-ay1
                nx, ny = ey, -ex   # outward normal
                # Point must be on inward side: dot(p - v1, outward_normal) <= 0
                if (px-ax1)*nx + (py-ay1)*ny > 0:
                    inside_poly = False
                    break
            if not inside_poly:
                continue
            # Outside all notch circles?
            in_notch = any((px-cx)**2 + (py-cy)**2 < notch_r**2
                           for cx, cy in midpoints)
            if in_notch:
                continue
            xs_in.append(px)
            ys_in.append(py)

    if xs_in:
        ax.scatter(xs_in, ys_in, s=1.2, c="#3a5fa0",
                   linewidths=0, zorder=2)

    # ── Ghost circle outline ───────────────────────────────────────────────────
    ax.add_patch(mpatches.Circle((0, 0), circ_R,
                                 fc="none", ec="#8899bb",
                                 lw=1.2, linestyle="--", zorder=3))

    # ── Polygon dashed outline ─────────────────────────────────────────────────
    ax.add_patch(mpatches.Polygon(verts, closed=True,
                                  fc="none", ec="#1a2f5e",
                                  lw=1.0, linestyle="--", zorder=4))

    # ── Notch outlines ─────────────────────────────────────────────────────────
    for cx, cy in midpoints:
        ax.add_patch(mpatches.Circle((cx, cy), notch_r,
                                     fc="white", ec="#c0392b",
                                     lw=1.1, zorder=5))

    # ── Annotations ───────────────────────────────────────────────────────────
    ax.annotate("", xy=(circ_R, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.2))
    ax.text(circ_R / 2, circ_R * 0.08,
            f"r = {circ_R*1e3:.2f} mm",
            ha="center", va="bottom", fontsize=7.5, color="#c0392b")

    cx0, cy0 = midpoints[0]
    ax.annotate("", xy=(cx0 + notch_r, cy0), xytext=(cx0, cy0),
                arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.0))
    ax.text(cx0 + notch_r * 1.15, cy0,
            f"notch r\n= {notch_r*1e3:.2f} mm",
            ha="left", va="center", fontsize=6.5, color="#8e44ad")

    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=8, color="#1a2f5e")


def draw_nlobe_section_preview(quads, params,
                               out_path="spirob_nlobe_preview.png",
                               interactive=False):
    """Second figure for n-lobe cross-section (n_cables >= 3)."""
    if interactive:
        matplotlib.use("TkAgg")

    n_cables = int(params.get("n_cables", 3))
    base_r   = float(abs(quads[0][2][0]))
    tip_r    = float(abs(quads[-1][3][0]))

    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    fig.patch.set_facecolor("#f5f6fa")
    fig.suptitle(f"SpiRob — {n_cables}-Lobe Cross-Section Preview",
                 fontsize=11, fontweight="bold", color="#1a2f5e", y=0.97)

    draw_nlobe_section(axes[0], base_r, n_cables, params,
                       title=f"Base element  (r = {base_r*1e3:.2f} mm)")
    draw_nlobe_section(axes[1], tip_r,  n_cables, params,
                       title=f"Tip element   (r = {tip_r*1e3:.2f} mm)")

    nlobe_t      = params.get("nlobe_t", 0.5)
    notch_factor = params.get("notch_factor", 0.25)
    fig.text(0.5, 0.02,
             f"nlobe_t = {nlobe_t}  (0=circumscribed, 1=inscribed)    "
             f"notch_factor = {notch_factor}",
             ha="center", va="bottom", fontsize=8, color="#555555", style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"📐 N-lobe section preview saved → {out_path}")

    if interactive:
        plt.show()
        plt.close(fig)

    return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  Flat cross-section preview  (n_cables <= 2)
# ══════════════════════════════════════════════════════════════════════════════

def draw_flat_section(ax, outer_radius, params, title="", quad=None):
    """
    Draw the flat cross-section on *ax*.

    If *quad* is supplied (a 4-point array [A1, A0, B0, B1] from _build_quads),
    the actual trapezoidal profile is drawn by projecting the quad into the XZ
    plane: width = 2 * |B0x|, and the inner/outer widths differ because B0 and
    B1 have slightly different x-magnitudes due to taper.  The thickness is
    derived from flat_thickness_ratio applied to the outer half-width.

    Falls back to a pure rectangle if quad is None.
    """
    thickness_ratio = float(params.get("flat_thickness_ratio", 0.3))

    if quad is not None:
        # A1, A0, B0, B1 — use the actual outer radii from the quad
        _, _, B0, B1 = quad
        r_bot = float(abs(B0[0]))   # outer radius at bottom of element
        r_top = float(abs(B1[0]))   # outer radius at top of element
        ht    = r_bot * thickness_ratio   # half-thickness (same both ends)
        # Trapezoid corners: (±r_bot at bottom, ±r_top at top)
        trap_x = [ r_bot,  r_top, -r_top, -r_bot]
        trap_y = [-ht,     ht,     ht,    -ht   ]
        circ_R = r_bot
    else:
        r_bot  = outer_radius
        r_top  = outer_radius
        ht     = outer_radius * thickness_ratio
        trap_x = [ r_bot,  r_top, -r_top, -r_bot]
        trap_y = [-ht,     ht,     ht,    -ht   ]
        circ_R = outer_radius

    ax.set_aspect("equal")
    ax.axis("off")
    pad = circ_R * 0.55
    ax.set_xlim(-circ_R - pad, circ_R + pad)
    ax.set_ylim(-circ_R - pad, circ_R + pad)

    # Ghost circle (original revolved profile for scale reference)
    ax.add_patch(mpatches.Circle((0, 0), circ_R,
                                 fc="#e8ecf4", ec="#8899bb",
                                 lw=1.2, linestyle="--", zorder=1))

    # Actual cross-section (trapezoid or rectangle)
    ax.add_patch(mpatches.Polygon(
        list(zip(trap_x, trap_y)), closed=True,
        fc="#3a5fa0", ec="#1a2f5e", lw=1.2, alpha=0.75, zorder=2))

    # Width annotation (bottom edge = widest)
    ax.annotate("", xy=(r_bot, -ht * 1.8), xytext=(-r_bot, -ht * 1.8),
                arrowprops=dict(arrowstyle="<->", color="#8e44ad", lw=1.2))
    ax.text(0, -ht * 2.2, f"w = {r_bot*2*1e3:.2f} mm",
            ha="center", va="top", fontsize=7.5, color="#8e44ad")

    # Thickness annotation (right side)
    ax.annotate("", xy=(r_bot * 1.5, ht), xytext=(r_bot * 1.5, -ht),
                arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.2))
    ax.text(r_bot * 1.6, 0, f"t = {ht*2*1e3:.2f} mm",
            ha="left", va="center", fontsize=7.5, color="#c0392b")

    # Outer radius reference arrow
    ax.annotate("", xy=(circ_R, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<->", color="#27ae60",
                                lw=1.1, linestyle="dashed"))
    ax.text(circ_R / 2, circ_R * 0.08,
            f"r = {circ_R*1e3:.2f} mm",
            ha="center", va="bottom", fontsize=7, color="#27ae60")

    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=8, color="#1a2f5e")


def draw_flat_section_preview(quads, params,
                              out_path="spirob_flat_preview.png",
                              interactive=False):
    """Second figure for flat cross-section (n_cables <= 2)."""
    if interactive:
        matplotlib.use("TkAgg")

    base_r = float(abs(quads[0][2][0]))
    tip_r  = float(abs(quads[-1][3][0]))

    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    fig.patch.set_facecolor("#f5f6fa")
    fig.suptitle("SpiRob — Flat Cross-Section Preview",
                 fontsize=11, fontweight="bold", color="#1a2f5e", y=0.97)

    draw_flat_section(axes[0], base_r, params,
                      title=f"Base element  (r = {base_r*1e3:.2f} mm)",
                      quad=quads[0])
    draw_flat_section(axes[1], tip_r,  params,
                      title=f"Tip element   (r = {tip_r*1e3:.2f} mm)",
                      quad=quads[-1])

    t_ratio = params.get("flat_thickness_ratio", 0.3)
    fig.text(0.5, 0.02,
             f"flat_thickness_ratio = {t_ratio}    "
             f"(dashed circle = original revolved profile for scale reference)",
             ha="center", va="bottom", fontsize=8, color="#555555", style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"📐 Flat section preview saved → {out_path}")

    if interactive:
        plt.show()
        plt.close(fig)

    return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SpiRob 2-D interactive preview. Exits 0 if user approves, 1 otherwise."
    )
    parser.add_argument("--params", required=True, help="Path to params.json")
    parser.add_argument("--out", default=None,
                        help="Save preview PNG to this path (default: auto in Geom_Data_CSV/)")
    parser.add_argument("--nlobe", action="store_true",
                        help="Also show n-lobe or flat cross-section preview window")
    parser.add_argument("--proceed", action="store_true",
                        help="Skip the yes/no prompt and exit 0 automatically")
    args = parser.parse_args()

    with open(args.params) as f:
        params = json.load(f)

    print("\n🔍 Building geometry preview from params.json …")
    quads = _build_quads(params)
    print(f"   → {len(quads)} elements")

    out_path = args.out or os.path.join("Geom_Data_CSV", "spirob_preview.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Main side-view preview
    draw_preview(quads, params, out_path=out_path, interactive=True)

    # Cross-section preview
    n_cables = int(params.get("n_cables", 3))

    if args.nlobe:
        if n_cables <= 2:
            # Show flat cross-section
            section_path = out_path.replace(".png", "_section.png")
            draw_flat_section_preview(quads, params,
                                      out_path=section_path, interactive=True)
        else:
            # n-lobe cross-section (n >= 3)
            section_path = out_path.replace(".png", "_section.png")
            draw_nlobe_section_preview(quads, params,
                                       out_path=section_path, interactive=True)

    if args.proceed:
        print("\n✅ --proceed flag set. Continuing pipeline.")
        sys.exit(0)

    # Gate: ask user
    print("\n" + "─" * 50)
    print("Review the preview window / saved PNG.")
    choice = input("Proceed with STL + XML generation? (y / n): ").strip().lower()
    if choice == "y":
        print("✅ Approved. Run:  python build.py --params params.json --noclean")
        sys.exit(0)
    else:
        print("❌ Rejected. Adjust params.json and re-run preview.py.")
        sys.exit(1)
