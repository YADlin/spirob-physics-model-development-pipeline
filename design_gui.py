"""
design_gui.py  —  Lightweight desktop GUI for the SpiRob pipeline.

A params-driven front end that wraps the existing command-line pipeline: edit
the physical parameters with live 2-D preview (side profile + cross-section),
then build the MuJoCo model, export a printable STEP/STL, generate a robot
array, or open the model in the MuJoCo viewer — all without touching a terminal.

It reuses this repo's own geometry and preview code (``spirob/geometry.py`` and
``preview.py``) for the live drawing, and shells out to ``build.py`` /
``cad_export.py`` / ``tools/multi_array.py`` for the heavy steps, so there is a
single source of truth for the maths.

Built on Tkinter + Matplotlib (both already available — no new dependency).

Design note
-----------
Clean-room reimplementation for this pipeline. The idea of a GUI design tool for
spiral robots is inspired by the OpenSpiRobs design tool (Zhanchi Wang et al.;
https://github.com/ZhanchiWang/Open-Spiral-Robots, PolyForm-Noncommercial). No
code from that project is used here; this is a thin Tk wrapper over this repo's
MIT-licensed pipeline. That tool is PySide6 and 2/3-cable only; this one is
stdlib Tk and supports the full n-cable range this repo generates.

Run
---
  python design_gui.py                 # loads params.json
  python design_gui.py --params my.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading

# --- numeric params shown as editable fields -----------------------------------
_FIELDS = [
    ("L",                   "Total length L (m)"),
    ("d_tip",               "Tip diameter (m)"),
    ("phi_deg",             "Taper angle φ (deg)"),
    ("Delta_theta_deg",     "Δθ per element (deg)"),
    ("n_cables",            "Number of cables"),
    ("tendon_inward_shift", "Tendon inward shift (m)"),
    ("nlobe_t",             "n-lobe t (0..1)"),
    ("notch_factor",        "Notch factor (0..0.4)"),
    ("flat_thickness_ratio","Flat thickness ratio"),
]


def launch(params_path: str = "params.json") -> None:
    import tkinter as tk
    from tkinter import ttk, messagebox

    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # Repo helpers (imported lazily so the module imports without a display).
    from spirob_csv_generator import validate_params
    import preview as pv

    repo_dir = os.path.dirname(os.path.abspath(__file__))

    with open(params_path, encoding="utf-8") as f:
        params = json.load(f)

    root = tk.Tk()
    root.title("SpiRob Design Tool")
    root.geometry("1120x720")

    # ── layout: left = controls, right = preview + log ────────────────────────
    left = ttk.Frame(root, padding=10); left.pack(side="left", fill="y")
    right = ttk.Frame(root, padding=6); right.pack(side="right", fill="both", expand=True)

    ttk.Label(left, text="Parameters", font=("", 12, "bold")).pack(anchor="w")
    entries = {}
    form = ttk.Frame(left); form.pack(fill="x", pady=6)
    for i, (key, label) in enumerate(_FIELDS):
        ttk.Label(form, text=label).grid(row=i, column=0, sticky="w", pady=2)
        var = tk.StringVar(value=str(params.get(key, "")))
        ent = ttk.Entry(form, textvariable=var, width=12)
        ent.grid(row=i, column=1, sticky="e", padx=4)
        entries[key] = var

    fig = Figure(figsize=(6.4, 5.4), dpi=100)
    ax_side = fig.add_subplot(1, 2, 1)
    ax_sec = fig.add_subplot(1, 2, 2)
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    log = tk.Text(right, height=8, wrap="word")
    log.pack(fill="x", pady=(6, 0))

    def logln(msg: str) -> None:
        log.insert("end", msg + "\n"); log.see("end"); root.update_idletasks()

    def collect_params() -> dict:
        p = dict(params)  # keep post_gen and other keys
        for key, var in entries.items():
            raw = var.get().strip()
            if raw == "":
                continue
            p[key] = int(raw) if key == "n_cables" else float(raw)
        return p

    def refresh_preview() -> None:
        try:
            p = collect_params()
            validate_params(p)
            quads = pv._build_quads(p)
            stats = pv._element_stats(quads)
            outer_radius = stats[0]["w_bottom"] / 2.0
            n = int(p["n_cables"])

            ax_side.clear(); ax_sec.clear()
            # Side profile: draw each quad polygon. Quads are 2-D (x, z).
            for st in stats:
                q = st["quad"]
                xs = [pt[0] for pt in q] + [q[0][0]]
                zs = [pt[1] for pt in q] + [q[0][1]]
                ax_side.plot(xs, zs, "-", lw=0.7, color="#1f77b4")
            ax_side.set_aspect("equal"); ax_side.set_title("Side profile")
            ax_side.set_xlabel("x (m)"); ax_side.set_ylabel("z (m)")

            if n >= 3:
                pv.draw_nlobe_section(ax_sec, outer_radius, n, p,
                                      title=f"{n}-lobe section")
            else:
                pv.draw_flat_section(ax_sec, outer_radius, p,
                                     title=f"{n}-cable flat section",
                                     quad=stats[0]["quad"])
            canvas.draw()
            logln(f"Preview OK — {len(stats)} elements, outer Ø="
                  f"{outer_radius*2*1000:.2f} mm")
        except Exception as e:
            logln(f"⚠ preview error: {e}")

    def save_params() -> None:
        try:
            p = collect_params()
            validate_params(p)
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(p, f, indent=2)
            params.update(p)
            logln(f"Saved → {params_path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _run_async(cmd: list, desc: str) -> None:
        def worker():
            logln(f"$ {' '.join(cmd)}")
            try:
                proc = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
                tail = (proc.stdout or "").strip().splitlines()[-6:]
                for ln in tail:
                    logln("  " + ln)
                logln(f"{'✅' if proc.returncode == 0 else '❌'} {desc} "
                      f"(exit {proc.returncode})")
                if proc.returncode != 0 and proc.stderr:
                    for ln in proc.stderr.strip().splitlines()[-4:]:
                        logln("  ! " + ln)
            except Exception as e:
                logln(f"❌ {desc}: {e}")
        threading.Thread(target=worker, daemon=True).start()

    def do_build() -> None:
        save_params()
        _run_async([sys.executable, "build.py", "--no-preview", "--params",
                    params_path], "Build (CSV→STL→XML)")

    def do_cad() -> None:
        _run_async([sys.executable, "cad_export.py", "--params", params_path],
                   "Export STEP + solid STL")

    def do_array() -> None:
        _run_async([sys.executable, "tools/multi_array.py",
                    "--in", "spirob_physics_model.xml", "--count", "6"],
                   "Generate 6-robot array")

    def do_viewer() -> None:
        _run_async([sys.executable, "-m", "mujoco.viewer",
                    "--mjcf=spirob_physics_model.xml"], "Open MuJoCo viewer")

    # ── buttons ───────────────────────────────────────────────────────────────
    btns = ttk.Frame(left); btns.pack(fill="x", pady=10)
    ttk.Button(btns, text="Update preview", command=refresh_preview).pack(fill="x", pady=2)
    ttk.Button(btns, text="Save params.json", command=save_params).pack(fill="x", pady=2)
    ttk.Separator(btns).pack(fill="x", pady=6)
    ttk.Button(btns, text="Build model", command=do_build).pack(fill="x", pady=2)
    ttk.Button(btns, text="Export STEP / STL", command=do_cad).pack(fill="x", pady=2)
    ttk.Button(btns, text="Generate array", command=do_array).pack(fill="x", pady=2)
    ttk.Button(btns, text="Open MuJoCo viewer", command=do_viewer).pack(fill="x", pady=2)

    logln("Ready. Edit parameters, Update preview, then Build.")
    refresh_preview()
    root.mainloop()


def main():
    p = argparse.ArgumentParser(description="SpiRob design GUI")
    p.add_argument("--params", default="params.json", help="Path to params.json")
    args = p.parse_args()
    launch(args.params)


if __name__ == "__main__":
    main()
