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
#  Reporting — delegates to the canonical geometry model
#
#  Everything above this line is an INDEPENDENT transcription of the published
#  equations, kept deliberately separate from spirob/geometry.py so that the
#  test suite can use it as an oracle. Everything below consumes the canonical
#  model and does no geometry of its own.
# ══════════════════════════════════════════════════════════════════════════

from spirob.geometry import (  # noqa: E402
    SpiRobGeometry, TerminalUnitPolicy, Tolerances, from_params,
)


def audit(params: dict,
          terminal_unit_policy: str = "exact_requested_length",
          tolerances: Optional[Tolerances] = None) -> SpiRobGeometry:
    """Build the canonical geometry for ``params`` under the given policy."""
    p = dict(params)
    p["terminal_unit_policy"] = terminal_unit_policy
    return from_params(p, tolerances=tolerances)


def format_report(geo: SpiRobGeometry) -> str:
    sp, lr, inp = geo.spiral, geo.lengths, geo.inputs
    L = [
        "SpiRob geometry audit",
        "=" * 72,
        f"  terminal_unit_policy     = {inp.terminal_unit_policy.value}",
        "",
        "  Spiral parameters",
        f"    b                      = {sp.b:.10f}",
        f"    a                      = {sp.a_m:.10f} m",
        f"    q0 (requested)         = {sp.q0_requested_rad:.10f} rad",
        f"    q0 (effective)         = {sp.q0_rad:.10f} rad",
        f"    phi (FULL included)    = {inp.phi_deg_full_included:.6f} deg",
        f"    beta (nominal)         = {sp.beta_nominal:.9f}",
        "",
        "  Lengths  (three distinct quantities)",
        f"    requested continuous   = {lr.requested_continuous_length_m * 1e3:10.4f} mm",
        f"    effective continuous   = {lr.effective_continuous_length_m * 1e3:10.4f} mm",
        f"    discrete chord         = {lr.discrete_chord_length_m * 1e3:10.4f} mm",
        "",
        "    unit-completion effect   (requested -> effective)",
        f"      absolute             = {lr.completion_delta_m * 1e3:+10.4f} mm",
        f"      relative             = {lr.completion_delta_rel * 100:+10.4f} %",
        "    arc-vs-chord effect      (effective -> discrete)",
        f"      absolute             = {lr.chord_deficit_m * 1e3:+10.4f} mm",
        f"      relative             = {lr.chord_deficit_rel * 100:+10.4f} %",
        "",
        "  Units",
        f"    total                  = {lr.n_units_total}",
        f"    complete               = {lr.n_complete_units}",
        f"    partial unit present   = {lr.has_partial_unit}",
    ]
    if lr.has_partial_unit:
        L += [
            f"    partial unit span      = {math.degrees(lr.partial_unit_span_rad):.4f} deg "
            f"(nominal {inp.delta_theta_deg:.4f})",
            f"    partial unit location  = {geo.units[0].link_name} (BASE, by design)",
        ]
    L += [
        "",
        "  Widths          nominal      realized",
        f"    tip          {lr.nominal_tip_width_m * 1e3:9.4f} mm  {lr.realized_tip_width_m * 1e3:9.4f} mm",
        f"    root         {lr.nominal_root_width_m * 1e3:9.4f} mm  {lr.realized_root_width_m * 1e3:9.4f} mm",
        "",
        "  Ordering (traced, not inferred)",
        f"    {geo.units[0].link_name}  index_tip_to_base={geo.units[0].index_tip_to_base:2d}  "
        f"realized width {geo.units[0].realized_width_m * 1e3:8.4f} mm  <- BASE, largest",
        f"    {geo.units[-1].link_name}  index_tip_to_base={geo.units[-1].index_tip_to_base:2d}  "
        f"realized width {geo.units[-1].realized_width_m * 1e3:8.4f} mm  <- TIP,  smallest",
        "",
        "  Base frame (fabrication anchor)",
        f"    origin offset          = {geo.base_frame.origin_offset_m * 1e3:.4f} mm",
        f"    mount plane normal     = {geo.base_frame.mount_plane_normal}",
    ]
    if geo.warnings:
        L += ["", "  NOTES"]
        for w in geo.warnings:
            L.append(f"    * {w}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="SpiRob canonical geometry audit")
    ap.add_argument("--params", default="params.json")
    ap.add_argument("--out", default=None, help="write the build manifest as JSON")
    ap.add_argument("--policy", default=None,
                    choices=[m.value for m in TerminalUnitPolicy],
                    help="override terminal_unit_policy from params.json")
    ap.add_argument("--compare-policies", action="store_true",
                    help="report both policies side by side")
    args = ap.parse_args()

    with open(args.params) as f:
        params = json.load(f)

    policy = args.policy or params.get(
        "terminal_unit_policy", TerminalUnitPolicy.EXACT_REQUESTED_LENGTH.value)

    if args.compare_policies:
        rows = [(m.value, audit(params, m.value)) for m in TerminalUnitPolicy]
        print("Policy comparison")
        print("=" * 72)
        hdr = f"{'quantity':<34}" + "".join(f"{n:>19}" for n, _ in rows)
        print(hdr)
        print("-" * len(hdr))

        def row(label, fn, fmt="{:>19.4f}"):
            print(f"{label:<34}" + "".join(fmt.format(fn(g)) for _, g in rows))

        row("requested continuous (mm)", lambda g: g.lengths.requested_continuous_length_m * 1e3)
        row("effective continuous (mm)", lambda g: g.lengths.effective_continuous_length_m * 1e3)
        row("discrete chord (mm)", lambda g: g.lengths.discrete_chord_length_m * 1e3)
        row("completion delta (mm)", lambda g: g.lengths.completion_delta_m * 1e3)
        row("completion delta (%)", lambda g: g.lengths.completion_delta_rel * 100)
        row("chord deficit (mm)", lambda g: g.lengths.chord_deficit_m * 1e3)
        row("chord deficit (%)", lambda g: g.lengths.chord_deficit_rel * 100)
        row("units total", lambda g: g.lengths.n_units_total, "{:>19d}")
        row("complete units", lambda g: g.lengths.n_complete_units, "{:>19d}")
        row("partial unit", lambda g: str(g.lengths.has_partial_unit), "{:>19}")
        row("partial span (deg)", lambda g: math.degrees(g.lengths.partial_unit_span_rad))
        row("effective q0 (rad)", lambda g: g.lengths.effective_q0_rad)
        row("nominal root width (mm)", lambda g: g.lengths.nominal_root_width_m * 1e3)
        row("realized root width (mm)", lambda g: g.lengths.realized_root_width_m * 1e3)
        row("realized tip width (mm)", lambda g: g.lengths.realized_tip_width_m * 1e3)
        return 0

    geo = audit(params, policy)
    print(format_report(geo))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(geo.to_manifest(), f, indent=2)
        print(f"\nWrote manifest: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
