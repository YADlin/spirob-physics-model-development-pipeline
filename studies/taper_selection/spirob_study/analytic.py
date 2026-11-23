"""Closed-form relations from Wang, Freris & Wei (Device 3, 100646, 2025).

Equation numbers refer to the main text and Supplemental Note S2 / Table S1.
The spiral constants (b, a, q0) are computed with this repository's own
``spirob.geometry`` functions, so this module is also a check that the
generator implements the paper's Eqs. 1-5.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spirob import geometry as g

from .designs import Design


@dataclass(frozen=True)
class SpiralSummary:
    name: str
    phi_deg: float
    b: float
    a_mm: float
    E: float
    q0_rad: float
    n_units_continuous: float     # q0 / Delta_theta
    beta: float                   # adjacent-unit size ratio, Eq. 9
    tip_width_mm: float
    base_width_mm: float
    width_ratio: float            # base / tip
    d_min_mm: float               # Eq. 14
    d_max_full_mm: float          # Eq. 15, whole length wraps half the object
    packed_turns: float           # q0 / 2pi


def spiral(design: Design) -> SpiralSummary:
    phi = math.radians(design.phi_deg)
    b = g.b_from_phi(phi=phi)
    a = g.a_from_tip_width(design.d_tip, b)
    E = g._E(b)
    q0 = g.q_from_arc_length(design.L, a, b)
    dth = math.radians(design.dtheta_deg)
    wb = g.root_width(design.d_tip, b, q0)
    return SpiralSummary(
        name=design.name, phi_deg=design.phi_deg, b=b, a_mm=a * 1e3, E=E,
        q0_rad=q0, n_units_continuous=q0 / dth, beta=math.exp(b * dth),
        tip_width_mm=design.d_tip * 1e3, base_width_mm=wb * 1e3,
        width_ratio=wb / design.d_tip,
        d_min_mm=d_min(design) * 1e3,
        d_max_full_mm=d_max(design, design.L) * 1e3,
        packed_turns=q0 / (2 * math.pi),
    )


def d_min(design: Design) -> float:
    """Eq. 14: inscribed diameter of the packed spiral (smallest object)."""
    b = g.b_from_phi(phi=math.radians(design.phi_deg))
    return design.d_tip * (math.exp(math.pi * b) + 1.0) / (math.exp(2 * math.pi * b) - 1.0)


def wrap_length(design: Design, d: float, wrap_angle: float = math.pi) -> float:
    """Length needed to wrap ``wrap_angle`` of a cylinder of diameter d.

    Eqs. 12-13: while wrapping, the body's central axis follows a spiral
    rho'_c = (d + delta0)/2 * exp(b' theta) with b' = cot((pi - phi)/2)
    = tan(phi/2). Its arc length from theta = 0 to the wrap angle is closed
    form for a logarithmic spiral.
    """
    bp = math.tan(math.radians(design.phi_deg) / 2.0)
    r0 = 0.5 * (d + design.d_tip)
    return r0 * math.sqrt(1.0 + bp * bp) / bp * (math.exp(bp * wrap_angle) - 1.0)


def d_max(design: Design, free_length: float, wrap_angle: float = math.pi) -> float:
    """Eq. 15 generalised: largest diameter that ``free_length`` can half-wrap.

    With free_length = L this is the paper's d_max. With
    free_length = L - s (object touching the body at arc length s from the
    base) it gives the position-dependent limit: the closer the object sits to
    the tip, the shorter the free length, and the smaller the largest object.
    """
    bp = math.tan(math.radians(design.phi_deg) / 2.0)
    k = math.sqrt(1.0 + bp * bp) / bp * (math.exp(bp * wrap_angle) - 1.0)
    return max(0.0, 2.0 * free_length / k - design.d_tip)


def max_load_kg(design: Design, d: float, f_max: float) -> float:
    """Table S1 load bound M(phi, d, F_max) (gravity torque balance, Eq. 16)."""
    c = 1.0 / math.tan((math.pi - math.radians(design.phi_deg)) / 2.0)
    return f_max / 9.81 * (1.0 - d / ((d + design.d_tip) * math.exp(c * math.pi)))


def involute_envelope(design: Design, n: int = 400) -> np.ndarray:
    """Supplemental Note S2 workspace envelope rho_w (polar, pole frame).

    Returned as (theta, rho_w) samples over the robot's angular extent.
    """
    s = spiral(design)
    b, a, E = s.b, s.a_mm * 1e-3, s.E
    th = np.linspace(0.0, s.q0_rad, n)
    rho = a / (2 * b) * (E + 1.0) * np.exp(b * (th - math.pi / 2))
    return np.stack([th, rho], axis=1)


def osr_print_design() -> Design:
    """The authors' example print, expressed in this repository's inputs.

    The 3MF holds 24 units generated with b = 0.1764, a = 4.95 mm,
    Delta_theta = 30 deg and the build-plate scale 0.5164. Units span
    theta in [0, 4 pi] (24 x 30 deg), so the continuous length is
    L(0, 4 pi) of Table S1 and the tip width is a (E - 1).
    """
    from .designs import OSR_PRINT_NAME
    b, a = 0.1764, 4.95e-3 * 0.5164
    E = math.exp(2 * math.pi * b)
    L = g.continuous_arc_length(4 * math.pi, a, b)
    d_tip = a * (E - 1.0)
    phi = math.degrees(g.phi_from_b(b))
    return Design(OSR_PRINT_NAME, L, d_tip, phi, 30.0,
                  policy="exact_requested_length", group="osr",
                  note="authors' example print file, 24 units")
