"""
tools/geometry_audit.py — read-only analytical + discretization audit.

Phase 1/2 instrumentation. This module imports nothing from the pipeline's
production code paths and writes nothing into them; it recomputes the
published SpiRob equations independently and compares them against what
``helper_functions.py`` actually produces.

It is deliberately side-effect free so it can be used both as a CLI

    python tools/geometry_audit.py --params params.json --out build_manifest.json

and as a library by the future canonical geometry model and the test suite.

Units are SI (metres, radians) throughout. Millimetres appear only in
human-readable report strings, never in returned data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict, field
from typing import List, Optional

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ══════════════════════════════════════════════════════════════════════════
#  Reference equations — transcribed from the specification, independent of
#  helper_functions.py so that agreement is a genuine cross-check.
# ══════════════════════════════════════════════════════════════════════════

def E_of(b: float) -> float:
    """E = exp(2*pi*b)."""
    return math.exp(2.0 * math.pi * b)


def phi_of_b(b: float) -> float:
    """Full *included* taper angle phi(b), radians. NOT a half angle."""
    E = E_of(b)
    return 2.0 * math.atan((b * (E - 1.0)) / (math.sqrt(1.0 + b * b) * (E + 1.0)))


def b_of_phi(phi: float, tol: float = 1e-15, max_iter: int = 200) -> float:
    """Invert phi(b) by bisection. Deterministic, bracketed, bounded."""
    if not (0.0 < phi < math.pi):
        raise ValueError(f"phi must be in (0, pi) rad; got {phi}")
    lo, hi = 1e-12, 1.0
    for _ in range(200):
        if phi_of_b(hi) >= phi:
            break
        hi *= 2.0
    else:
        raise RuntimeError(f"could not bracket b for phi={phi}")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if phi_of_b(mid) < phi:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < tol:
            break
    return 0.5 * (lo + hi)


def a_of(d_tip: float, b: float) -> float:
    """a = d_tip / (E - 1)."""
    return d_tip / (E_of(b) - 1.0)


def r_c(theta, a: float, b: float):
    """Central spiral radius r_c(theta) = 0.5*a*(E+1)*exp(b*theta)."""
    return 0.5 * a * (E_of(b) + 1.0) * np.exp(b * np.asarray(theta, dtype=float))


def local_width(theta, a: float, b: float):
    """Local width d(theta) = a*(E-1)*exp(b*theta)."""
    return a * (E_of(b) - 1.0) * np.exp(b * np.asarray(theta, dtype=float))


def arc_length(q: float, a: float, b: float) -> float:
    """L_arc(q) = sqrt(1+b^2)/b * 0.5*a*(E+1) * (exp(b*q) - 1)."""
    return (math.sqrt(1.0 + b * b) / b) * 0.5 * a * (E_of(b) + 1.0) * (math.exp(b * q) - 1.0)


def q0_from_arc_length(L: float, a: float, b: float) -> float:
    """Closed-form inverse of arc_length()."""
    A = (math.sqrt(1.0 + b * b) / b) * 0.5 * a * (E_of(b) + 1.0)
    return (1.0 / b) * math.log(1.0 + L / A)


def beta_of(b: float, delta_theta: float) -> float:
    """Adjacent-unit scale beta = exp(b*Delta_theta)."""
    return math.exp(b * delta_theta)


def d_root_of(d_tip: float, b: float, q0: float) -> float:
    """d_root = d_tip*exp(b*q0)."""
    return d_tip * math.exp(b * q0)


# ══════════════════════════════════════════════════════════════════════════
#  Discretization
# ══════════════════════════════════════════════════════════════════════════

def theta_samples(q0: float, delta_theta: float, policy: str = "truncate") -> np.ndarray:
    """Sample angles for the current ('truncate') and proposed policies.

    truncate       theta[i] = min(i*Delta_theta, q0)   [current behaviour]
    uniform_dtheta N = ceil(q0/Delta_theta), Delta_eff = q0/N
    """
    if delta_theta <= 0:
        raise ValueError("delta_theta must be > 0")
    N = int(math.ceil(q0 / delta_theta))
    if policy == "truncate":
        return np.array([min(k * delta_theta, q0) for k in range(N + 1)], dtype=float)
    if policy == "uniform_dtheta":
        return np.linspace(0.0, q0, N + 1)
    raise ValueError(f"unknown terminal_unit_policy {policy!r}")


def centerline_points(theta: np.ndarray, a: float, b: float) -> np.ndarray:
    """Central-spiral sample points, shape (n, 2)."""
    r = r_c(theta, a, b)
    return np.column_stack((r * np.cos(theta), r * np.sin(theta)))


def discrete_backbone_length(theta: np.ndarray, a: float, b: float) -> float:
    """sum(norm(center(theta[i+1]) - center(theta[i])))."""
    C = centerline_points(theta, a, b)
    return float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())


def solve_q0_for_discrete_length(
    L: float,
    a: float,
    b: float,
    delta_theta: float,
    policy: str = "truncate",
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """Numerically solve q0 so the *chord* sum equals L.

    Deterministic bisection with explicit bracketing, tolerance and iteration
    limit, per the Phase 2 requirement. Raises with an actionable message if
    the root cannot be bracketed or the iteration limit is exhausted.
    """
    def f(q: float) -> float:
        return discrete_backbone_length(theta_samples(q, delta_theta, policy), a, b) - L

    lo = q0_from_arc_length(L, a, b)          # chord sum < arc, so this is a lower bound
    if f(lo) > 0.0:
        raise RuntimeError(
            "discrete_backbone solver: chord sum already exceeds L at the "
            "continuous-arc solution; check a, b, Delta_theta."
        )
    hi = lo
    for _ in range(64):
        hi *= 1.05
        if f(hi) > 0.0:
            break
    else:
        raise RuntimeError(
            f"discrete_backbone solver: could not bracket q0 for L={L} "
            f"(searched up to q0={hi}). Reduce Delta_theta_deg or check L."
        )
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < tol:
            return 0.5 * (lo + hi)
    raise RuntimeError(
        f"discrete_backbone solver: {max_iter} iterations exhausted without "
        f"reaching tol={tol} (bracket width {hi - lo:.3e})."
    )


# ══════════════════════════════════════════════════════════════════════════
#  Report structures
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class UnitRecord:
    index: int
    theta_start: float
    theta_end: float
    angular_span: float
    arc_length: float
    chord_length: float
    local_width_start: float
    local_width_end: float
    realized_width: float          # width the straightening step actually produces
    scale_ratio: Optional[float]   # width_i / width_{i-1}


@dataclass
class BuildManifest:
    schema_version: str = "1.0"
    params: dict = field(default_factory=dict)
    analytical: dict = field(default_factory=dict)
    dimensions: dict = field(default_factory=dict)
    discretization: dict = field(default_factory=dict)
    units: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _chord_offset_factor(P0: np.ndarray, P1: np.ndarray) -> float:
    """Perpendicular distance from the origin to the chord line P0->P1.

    The straightening step maps the inner edge to a constant lateral offset
    of (1-k)*this distance, with k = 2/(E+1); see GEOMETRY_AUDIT.md F-06.
    """
    e = P1 - P0
    e = e / np.linalg.norm(e)
    # 2-D cross product magnitude
    return abs(e[0] * (-P0[1]) - e[1] * (-P0[0]))


def audit(params: dict, length_definition: str = "continuous_arc",
          terminal_unit_policy: str = "truncate") -> BuildManifest:
    L = float(params["L"])
    d_tip = float(params["d_tip"])
    phi = math.radians(float(params["phi_deg"]))
    dth = math.radians(float(params["Delta_theta_deg"]))

    b = b_of_phi(phi)
    E = E_of(b)
    a = a_of(d_tip, b)

    warnings: List[str] = []
    if length_definition == "continuous_arc":
        q0 = q0_from_arc_length(L, a, b)
    elif length_definition == "discrete_backbone":
        q0 = solve_q0_for_discrete_length(L, a, b, dth, terminal_unit_policy)
    else:
        raise ValueError(f"unknown length_definition {length_definition!r}")

    theta = theta_samples(q0, dth, terminal_unit_policy)
    C = centerline_points(theta, a, b)
    chords = np.linalg.norm(np.diff(C, axis=0), axis=1)
    L_cont = arc_length(q0, a, b)
    L_disc = float(chords.sum())

    k = 2.0 / (E + 1.0)
    widths = local_width(theta, a, b)
    rc_vals = r_c(theta, a, b)

    units: List[UnitRecord] = []
    prev_w = None
    for i in range(len(theta) - 1):
        realized = (1.0 - k) * _chord_offset_factor(C[i], C[i + 1]) * 2.0
        units.append(UnitRecord(
            index=i,
            theta_start=float(theta[i]),
            theta_end=float(theta[i + 1]),
            angular_span=float(theta[i + 1] - theta[i]),
            arc_length=float(arc_length(theta[i + 1], a, b) - arc_length(theta[i], a, b)),
            chord_length=float(chords[i]),
            local_width_start=float(widths[i]),
            local_width_end=float(widths[i + 1]),
            realized_width=float(realized),
            scale_ratio=None if prev_w is None else float(realized / prev_w),
        ))
        prev_w = realized

    spans = np.diff(theta)
    truncated = bool(abs(spans[-1] - dth) > 1e-12)
    if truncated:
        warnings.append(
            f"terminal unit is truncated: span {math.degrees(spans[-1]):.4f} deg "
            f"vs nominal {math.degrees(dth):.4f} deg. Under Invert_pose() this "
            f"partial unit becomes link_001 (the BASE link), not the tip."
        )
    rel_err = (L_disc - L) / L
    if abs(rel_err) > 1e-3:
        warnings.append(
            f"discrete backbone length differs from requested L by "
            f"{rel_err * 100:+.4f}% ({(L_disc - L) * 1e3:+.4f} mm)."
        )

    m = BuildManifest()
    m.params = {"L": L, "d_tip": d_tip, "phi_deg": params["phi_deg"],
                "Delta_theta_deg": params["Delta_theta_deg"],
                "n_cables": params.get("n_cables"),
                "length_definition": length_definition,
                "terminal_unit_policy": terminal_unit_policy}
    m.analytical = {"a": a, "b": b, "E": E, "q0": q0,
                    "phi_rad": phi, "phi_deg_full_included": math.degrees(phi),
                    "beta_nominal": beta_of(b, dth)}
    m.dimensions = {
        "requested_length": L,
        "continuous_arc_length": L_cont,
        "discrete_backbone_length": L_disc,
        "relative_length_error": rel_err,
        "tip_width_nominal": d_tip,
        "root_width_nominal": d_root_of(d_tip, b, q0),
        "tip_width_realized": units[0].realized_width,
        "root_width_realized": units[-1].realized_width,
    }
    m.discretization = {
        "requested_delta_theta": dth,
        "effective_delta_theta": float(np.mean(spans[:-1])) if len(spans) > 1 else float(spans[0]),
        "n_units": len(units),
        "terminal_unit_truncated": truncated,
        "terminal_unit_span": float(spans[-1]),
    }
    m.units = [asdict(u) for u in units]
    m.warnings = warnings
    return m


def _fmt(m: BuildManifest) -> str:
    d, an, dc = m.dimensions, m.analytical, m.discretization
    out = [
        "SpiRob geometry audit",
        "=" * 62,
        f"  b                        = {an['b']:.10f}",
        f"  a                        = {an['a']:.10f} m",
        f"  q0                       = {an['q0']:.10f} rad",
        f"  phi (FULL included)      = {an['phi_deg_full_included']:.6f} deg",
        f"  beta (nominal)           = {an['beta_nominal']:.9f}",
        "",
        f"  requested length         = {d['requested_length'] * 1e3:10.4f} mm",
        f"  continuous arc length    = {d['continuous_arc_length'] * 1e3:10.4f} mm",
        f"  discrete backbone length = {d['discrete_backbone_length'] * 1e3:10.4f} mm",
        f"  relative length error    = {d['relative_length_error'] * 100:+10.4f} %",
        "",
        f"  tip width  nominal/realized = {d['tip_width_nominal'] * 1e3:8.4f} / "
        f"{d['tip_width_realized'] * 1e3:8.4f} mm",
        f"  root width nominal/realized = {d['root_width_nominal'] * 1e3:8.4f} / "
        f"{d['root_width_realized'] * 1e3:8.4f} mm",
        "",
        f"  units                    = {dc['n_units']}",
        f"  requested Delta_theta    = {math.degrees(dc['requested_delta_theta']):.4f} deg",
        f"  effective Delta_theta    = {math.degrees(dc['effective_delta_theta']):.4f} deg",
        f"  terminal unit truncated  = {dc['terminal_unit_truncated']} "
        f"(span {math.degrees(dc['terminal_unit_span']):.4f} deg)",
    ]
    if m.warnings:
        out.append("")
        out.append("  WARNINGS")
        for w in m.warnings:
            out.append(f"    ! {w}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="SpiRob analytical/discretization audit")
    ap.add_argument("--params", default="params.json")
    ap.add_argument("--out", default=None, help="write the build manifest as JSON")
    ap.add_argument("--length-definition", default="continuous_arc",
                    choices=["continuous_arc", "discrete_backbone"])
    ap.add_argument("--terminal-unit-policy", default="truncate",
                    choices=["truncate", "uniform_dtheta"])
    args = ap.parse_args()

    with open(args.params) as f:
        params = json.load(f)

    params.setdefault("length_definition", args.length_definition)
    params.setdefault("terminal_unit_policy", args.terminal_unit_policy)

    m = audit(params,
              length_definition=params["length_definition"],
              terminal_unit_policy=params["terminal_unit_policy"])
    print(_fmt(m))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(asdict(m), f, indent=2)
        print(f"\nWrote manifest: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
