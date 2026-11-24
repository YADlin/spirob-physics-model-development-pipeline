"""Collect the per-taper metrics of all experiments into one table."""
from __future__ import annotations

import json

import numpy as np

from . import analytic, designs as D
from .designs import STUDY

R = STUDY / "results"


def table() -> list[dict]:
    curl = {r["design"]: r for r in json.loads((R / "curl" / "summary.json").read_text()) if r["law"] == "cubic"}
    ws = {r["design"]: r for r in json.loads((R / "workspace" / "summary.json").read_text())}
    pr = json.loads((R / "print" / "summary.json").read_text())["print"]
    fr = json.loads((R / "print" / "friction.json").read_text())
    sp = json.loads((R / "grasp" / "size_position.json").read_text())
    egg = json.loads((R / "grasp" / "egg_map.json").read_text())
    rows = []
    for d in D.SWEEP:
        n = d.name
        s = analytic.spiral(d)
        g = [r for r in sp if r["design"] == n]
        ok = [r for r in g if r["success"]]
        ok40 = sorted(r["s_frac"] for r in ok if r["d_mm"] == 40)
        e = [r for r in egg if r["design"] == n and r["s_frac"] != "task"]
        task = [r for r in egg if r["design"] == n and r["s_frac"] == "task"][0]
        p20 = [p for p in pr if p["design"] == n and p["layer_mm"] == 0.2][0]
        p12 = [p for p in pr if p["design"] == n and p["layer_mm"] == 0.12][0]
        f3 = [f for f in fr if f["design"] == n and f["layer_mm"] is None and f["mu"] == 0.3]
        rows.append(dict(
            design=n, phi_deg=d.phi_deg, base_width_mm=s.base_width_mm, width_ratio=s.width_ratio,
            n_units=curl[n]["n_joints"] + 1, mass_g=curl[n]["mass_kg"] * 1e3,
            f_first_N=curl[n]["f_first"], f_pack_N=curl[n]["f_est"],
            force_range=curl[n]["f_est"] / curl[n]["f_first"], stroke_mm=curl[n]["stroke_m"] * 1e3,
            d_min_theory_mm=s.d_min_mm, d_max_theory_mm=s.d_max_full_mm,
            dyn_area_cm2=ws[n]["dynamic_area_cm2"], envelope_area_cm2=ws[n]["envelope_area_cm2"],
            grasp_cells=len(ok), grasp_cells_total=len(g),
            d_held_min_mm=min((r["d_mm"] for r in ok), default=np.nan),
            d_held_max_mm=max((r["d_mm"] for r in ok), default=np.nan),
            dmax40_frac=(ok40[-1] - ok40[0] + 0.1) if ok40 else 0.0,
            s40_range=(ok40[0], ok40[-1]) if ok40 else None,
            egg_success=sum(r["success"] for r in e), egg_total=len(e),
            egg_task_success=bool(task["success"]), egg_task_enclosure=task["enclosure_deg"],
            egg_task_grip_N=task["grip_N"],
            distal_ideal=curl_distal(n), distal_mu03=f3[0]["distal_share_at_180"] if f3 else np.nan,
            reversals_020=p20["order_reversals"], reversals_012=p12["order_reversals"],
            neck_tip_mm=p20["neck_ideal_mm"][-1], neck_base_mm=p20["neck_ideal_mm"][0],
            distinct_020=p20["distinct_thicknesses"],
        ))
    return rows


def curl_distal(name: str) -> float:
    pr = json.loads((R / "print" / "summary.json").read_text())["print"]
    return [p for p in pr if p["design"] == name and p["layer_mm"] is None][0]["distal_share_at_180"]


def main():
    rows = table()
    (R / "summary_table.json").write_text(json.dumps(rows, indent=1, default=float))
    from .figures import fig_summary
    fig_summary(rows)
    return rows


if __name__ == "__main__":
    for r in main():
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
