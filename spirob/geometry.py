"""
Canonical SpiRob geometry model.

=============================================================================
CONVENTIONS OF RECORD
=============================================================================
Every ambiguity that the Phase 1 audit found spread across modules is resolved
here, once. Downstream consumers must not re-derive any of this.

UNITS
    All lengths are metres, all angles are radians, unless a name ends in
    ``_deg`` or ``_mm``. User-facing ``params.json`` keeps its historical
    degree-valued keys (``phi_deg``, ``Delta_theta_deg``); they are converted
    at the boundary in :func:`build_geometry` and never stored in degrees.

PHI
    ``phi`` is the FULL INCLUDED taper angle between the two boundary edges of
    the spiral, not a half-angle. The identity that fixes this is

        d(half-width)/d(arc length) = tan(phi/2)

    so the half-width slope is ``tan(phi/2)``. The parameter is spelled
    ``phi_deg`` in ``params.json`` for backward compatibility; the alias
    ``taper_angle_deg`` is accepted and must agree if both are present.

ANGLE DIRECTION AND REFERENCE AXIS
    ``theta`` increases counter-clockwise from the +X axis. ``theta = 0`` is
    the TIP of the robot (smallest radius); ``theta = q0`` is the BASE
    (largest radius). This is the *curled* frame, centred on the spiral pole.

ORDERING
    Two orderings exist and both are exposed explicitly.

      spiral order   index 0 = theta 0        = smallest = TIP
      link order     index 0 = ``link_001``   = largest  = BASE

    They are exact reverses of one another:  ``link[k] = spiral[N-1-k]``.
    :class:`UnitRecord` carries both ``index_base_to_tip`` (link order, the
    one the CSV/XML use) and ``index_tip_to_base`` (spiral order). Never infer
    ordering from a variable name.

WHAT ``Invert_pose()`` DOES — AND WHY IT IS CORRECT
    ``helper_functions.Invert_pose(quads, Length)`` performs three operations:

      1. reflects y about the requested continuous length:  y -> -y + Length
      2. permutes the vertex slots  [0,1,2,3] -> [1,0,3,2]
      3. reverses the element list

    The result is the intended physical arrangement: LARGEST element at the
    base, SMALLEST at the tip. When the requested length does not contain an
    integer number of nominal units, the partial unit is the last spiral
    interval, and after inversion it becomes ``link_001``, the base link.
    *This is correct by design and is preserved.* It is not an inversion
    error. It does mean a fabricated robot has a partial unit at its base,
    which is a fabrication note, not a geometry defect.

    Step (1) anchors the TIP at exactly ``y = requested_length_m``. The base
    therefore sits at ``requested_length_m - discrete_chord_length_m`` in the
    inverted frame. This is a documented frame convention, exposed as
    :attr:`BaseFrame.origin_offset_m`. It does not reach the simulation,
    because ``csv2xml.py`` overrides the root body position from
    ``post_gen.robot_pos``.

LENGTHS — THREE DIFFERENT QUANTITIES
    Do not conflate these. Unit completion and arc-versus-chord shortening are
    separate effects and are reported separately.

      requested_continuous_length_m   what the user asked for
      effective_continuous_length_m   spiral arc actually generated, after the
                                      terminal-unit policy has been applied
      discrete_chord_length_m         sum of straightened chords, i.e. what the
                                      built link chain actually measures

    ``requested -> effective`` is the unit-completion effect (zero under
    ``exact_requested_length``). ``effective -> discrete`` is the arc-to-chord
    shortening effect, always negative, second order in ``Delta_theta``.

WIDTHS
    ``nominal`` widths come from ``d(theta) = a*(E-1)*exp(b*theta)``.
    ``realized`` widths are what straightening actually produces:

        realized_width = d(theta_start) * dist(O, chord) / r_c(theta_start)

    The sagitta factor is < 1, so realized < nominal. Both are reported.

TENDON POINTS
    Defined once, here, in the inverted (CSV) frame. Two stages:

      attachment points   the raw inner-edge vertices, radius = local surface
      routing points      after ``tendon_inward_shift`` and the per-link
                          ``-dz*tan(phi/2)`` correction

    The routing stage reproduces ``csv2xml.py`` exactly (verified to 1e-15 by
    ``tests/test_canonical_geometry.py``) so that adopting this model changes
    no simulation output. See ``docs/GEOMETRY_AUDIT.md`` F-08 for the open
    question about whether that correction is the right long-term rule.

FABRICATION ANCHORS (deferred to a later phase)
    :class:`BaseFrame` and :attr:`UnitRecord.slit_reference_m` expose stable
    named locations for a future CAD exporter (base dovetail, inter-unit
    slits). No solids are built here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "TerminalUnitPolicy", "UserInputs", "SpiralParameters", "LengthReport",
    "UnitRecord", "TendonPoint", "TendonPath", "BaseFrame", "Tolerances",
    "SpiRobGeometry", "build_geometry",
]


# ══════════════════════════════════════════════════════════════════════════
#  Tolerances
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Tolerances:
    """Numerical tolerances, all documented and all overridable.

    angle_rad
        Two angles closer than this are considered equal. Used to decide
        whether ``q0/Delta_theta`` is an exact integer, i.e. whether a partial
        unit exists at all. 1e-9 rad is ~2e-4 arc-seconds.
    length_m
        Two lengths closer than this are equal. 1e-12 m is well below any
        printable or simulable feature at this scale.
    solver_rel
        Relative convergence target for iterative solvers.
    solver_max_iter
        Hard iteration cap; exceeding it raises rather than returning a
        half-converged root.
    """
    angle_rad: float = 1e-9
    length_m: float = 1e-12
    solver_rel: float = 1e-12
    solver_max_iter: int = 200


# ══════════════════════════════════════════════════════════════════════════
#  Policy
# ══════════════════════════════════════════════════════════════════════════

class TerminalUnitPolicy(str, Enum):
    """How to handle a requested length that is not a whole number of units.

    EXACT_REQUESTED_LENGTH  (default, current behaviour)
        Preserve the requested continuous spiral length exactly. The final
        spiral interval is truncated, producing one partial unit which — after
        ``Invert_pose()`` — is the base link. Effective length == requested.

    WHOLE_UNITS
        Extend the spiral forward to the next complete nominal unit boundary.
        Never shortens. Produces no partial unit. The generated robot is
        LONGER than requested; the excess is reported.
    """
    EXACT_REQUESTED_LENGTH = "exact_requested_length"
    WHOLE_UNITS = "whole_units"

    @classmethod
    def coerce(cls, value) -> "TerminalUnitPolicy":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        allowed = ", ".join(repr(m.value) for m in cls)
        raise ValueError(
            f"terminal_unit_policy must be one of {allowed}; got {value!r}"
        )


# ══════════════════════════════════════════════════════════════════════════
#  Closed-form spiral relations
# ══════════════════════════════════════════════════════════════════════════

def _E(b: float) -> float:
    return math.exp(2.0 * math.pi * b)


def phi_from_b(b: float) -> float:
    """Full included taper angle, radians.

    NUMERICAL CONTRACT. This must agree **bit-for-bit** with
    ``helper_functions.phi_from_b``, which every artefact the project has ever
    generated was built from. The ``atan2(num, den)`` form is therefore used
    rather than the algebraically identical ``atan(num/den)``: for ``b > 0``
    the denominator is positive so the two are mathematically the same, but
    they round differently, and the difference propagates into every CSV, STL
    and MJCF value. See docs/CANONICAL_GEOMETRY.md, "Numerical contract".
    """
    e = math.exp(2 * math.pi * b)
    num = b * (e - 1.0)
    den = math.sqrt(1.0 + b * b) * (e + 1.0)
    return 2.0 * math.atan2(num, den)


def b_from_phi(b_tol: float = 1e-15, *, phi: float, max_iter: int = 200) -> float:
    """Invert ``phi(b)``. Deterministic bracketed bisection.

    NUMERICAL CONTRACT. Reproduces ``helper_functions.solve_b_for_phi``
    bit-for-bit, including its bracket (``lo=1e-6``, ``hi=2.0``), its 1.5x
    expansion, and — critically — its early exit at ``abs(f(mid)) < 1e-14``.
    That early exit stops roughly 217 ULP from the value a pure interval
    bisection would reach. Bisecting further is *more* accurate in residual
    terms, but it changes ``b`` by 4.8e-14 relative, which moves every digit of
    the generated CSV. Backward compatibility wins: the canonical layer's job
    is to be the one true source of the numbers the pipeline actually uses.

    The two arguments below are retained for API compatibility and are
    deliberately not used to loosen the contract:

    ``b_tol``    kept in the signature; the legacy interval floor of 1e-14 is
                 what actually terminates the loop.
    ``max_iter`` capped at the legacy 120 iterations.

    Unlike the legacy routine, an unbracketable ``phi`` raises instead of
    silently returning ``max(1e-6, tan(phi/2))`` — a wrong ``b`` presented as a
    right one. That path is unreachable for the validated range ``phi`` in
    (0, 45) deg.
    """
    if not (0.0 < phi < math.pi):
        raise ValueError(f"phi must lie in (0, pi) rad; got {phi!r}")

    def f(b: float) -> float:
        return phi_from_b(b) - phi

    lo, hi = 1e-6, 2.0
    flo, fhi = f(lo), f(hi)
    tries = 0
    while flo * fhi > 0 and tries < 50:
        hi *= 1.5
        fhi = f(hi)
        tries += 1
    if flo * fhi > 0:
        raise RuntimeError(
            f"could not bracket b for phi={phi} rad "
            f"({math.degrees(phi)} deg) within lo=1e-6, hi={hi}")

    for _ in range(min(max_iter, 120)):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < 1e-14 or (hi - lo) < 1e-14:
            return mid
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def a_from_tip_width(tip_width_m: float, b: float) -> float:
    """``a = d_tip / (E - 1)``."""
    return tip_width_m / (_E(b) - 1.0)


def central_radius(theta, a: float, b: float):
    """``r_c(theta) = 0.5*a*(E+1)*exp(b*theta)``."""
    return 0.5 * a * (_E(b) + 1.0) * np.exp(b * np.asarray(theta, dtype=float))


def nominal_width(theta, a: float, b: float):
    """``d(theta) = a*(E-1)*exp(b*theta)``."""
    return a * (_E(b) - 1.0) * np.exp(b * np.asarray(theta, dtype=float))


def continuous_arc_length(q: float, a: float, b: float) -> float:
    """``L_arc(q) = sqrt(1+b^2)/b * 0.5*a*(E+1) * (exp(b*q)-1)``."""
    return ((math.sqrt(1.0 + b * b) / b) * 0.5 * a * (_E(b) + 1.0)
            * (math.exp(b * q) - 1.0))


def q_from_arc_length(length_m: float, a: float, b: float) -> float:
    """Exact closed-form inverse of :func:`continuous_arc_length`."""
    A = (math.sqrt(1.0 + b * b) / b) * 0.5 * a * (_E(b) + 1.0)
    return (1.0 / b) * math.log(1.0 + length_m / A)


def adjacent_unit_scale(b: float, delta_theta: float) -> float:
    """``beta = exp(b*Delta_theta)``."""
    return math.exp(b * delta_theta)


def root_width(tip_width_m: float, b: float, q0: float) -> float:
    """``d_root = d_tip*exp(b*q0)``."""
    return tip_width_m * math.exp(b * q0)


# ══════════════════════════════════════════════════════════════════════════
#  Records
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class UserInputs:
    """Exactly what the user asked for, in SI, before anything is derived."""
    requested_length_m: float
    tip_width_m: float
    phi_rad_full_included: float
    delta_theta_rad: float
    n_cables: int
    tendon_inward_shift_m: float
    terminal_unit_policy: TerminalUnitPolicy

    @property
    def phi_deg_full_included(self) -> float:
        return math.degrees(self.phi_rad_full_included)

    @property
    def delta_theta_deg(self) -> float:
        return math.degrees(self.delta_theta_rad)


@dataclass(frozen=True)
class SpiralParameters:
    """Derived global parameters of the logarithmic spiral."""
    b: float
    a_m: float
    E: float
    q0_rad: float                     # effective angular extent, after policy
    q0_requested_rad: float           # extent that matches the requested length
    beta_nominal: float


@dataclass(frozen=True)
class LengthReport:
    """The three lengths and both difference effects, kept strictly apart."""
    requested_continuous_length_m: float
    effective_continuous_length_m: float
    discrete_chord_length_m: float

    # unit-completion effect: requested -> effective
    completion_delta_m: float
    completion_delta_rel: float

    # arc-versus-chord effect: effective -> discrete
    chord_deficit_m: float
    chord_deficit_rel: float

    n_units_total: int
    n_complete_units: int
    has_partial_unit: bool
    partial_unit_span_rad: float
    effective_q0_rad: float

    nominal_tip_width_m: float
    nominal_root_width_m: float
    realized_tip_width_m: float
    realized_root_width_m: float

    @property
    def requested_vs_discrete_m(self) -> float:
        """Total shortfall the user actually experiences, both effects summed."""
        return self.discrete_chord_length_m - self.requested_continuous_length_m


@dataclass(frozen=True)
class UnitRecord:
    """One rigid unit. Immutable; arrays are stored as tuples of tuples."""
    index_base_to_tip: int          # 0 == link_001 == base   (CSV / XML order)
    index_tip_to_base: int          # 0 == theta 0  == tip    (spiral order)
    is_partial: bool

    theta_start_rad: float          # spiral order: start is nearer the tip
    theta_end_rad: float
    angular_span_rad: float

    arc_length_m: float
    chord_length_m: float

    curled_centerline_start_m: Tuple[float, float]
    curled_centerline_end_m: Tuple[float, float]
    straight_centerline_start_m: Tuple[float, float]
    straight_centerline_end_m: Tuple[float, float]
    inverted_centerline_start_m: Tuple[float, float]   # base end of the link
    inverted_centerline_end_m: Tuple[float, float]     # tip end of the link

    local_frame_origin_m: Tuple[float, float, float]
    local_frame_axis: Tuple[float, float, float]       # unit, base -> tip

    nominal_width_start_m: float
    nominal_width_end_m: float
    realized_width_m: float

    # Fabrication anchors (deferred phase). The slit that forms the compliant
    # hinge between this unit and the next one toward the tip.
    slit_reference_m: Tuple[float, float, float]
    slit_normal: Tuple[float, float, float]

    @property
    def link_name(self) -> str:
        return f"link_{self.index_base_to_tip + 1:03d}"


@dataclass(frozen=True)
class TendonPoint:
    """A single tendon site in the inverted (CSV) frame."""
    cable_index: int
    unit_index_base_to_tip: int
    slot: str                                   # "s1" (base end) | "s2" (tip end)
    attachment_m: Tuple[float, float, float]    # on the link surface
    routed_m: Tuple[float, float, float]        # after inward shift + correction
    surface_radius_m: float
    routed_radius_m: float


@dataclass(frozen=True)
class TendonPath:
    cable_index: int
    points: Tuple[TendonPoint, ...]             # base -> tip

    def routed_polyline(self) -> np.ndarray:
        return np.array([p.routed_m for p in self.points], dtype=float)

    def attachment_polyline(self) -> np.ndarray:
        return np.array([p.attachment_m for p in self.points], dtype=float)


@dataclass(frozen=True)
class BaseFrame:
    """Stable named anchor at the robot base, for assembly and fabrication.

    origin_offset_m
        Where the base sits along the inversion axis. Non-zero because
        ``Invert_pose()`` anchors the TIP at the requested length; see the
        module docstring. Equals ``requested - discrete`` under the exact
        policy.
    mount_plane_point_m / mount_plane_normal
        The flat annulus a dovetail or mount plate would attach to.
    root_width_m
        Realized outer width at the mounting face.
    """
    origin_offset_m: float
    mount_plane_point_m: Tuple[float, float, float]
    mount_plane_normal: Tuple[float, float, float]
    root_width_m: float


# ══════════════════════════════════════════════════════════════════════════
#  Internal helpers — pose construction
# ══════════════════════════════════════════════════════════════════════════

def _rotate2d(points: np.ndarray, angle: float) -> np.ndarray:
    """NUMERICAL CONTRACT: mirrors ``helper_functions.rotate``.

    ``np.cos``/``np.sin`` are used rather than ``math.cos``/``math.sin``. The
    two libraries do not always round identically, and the difference reaches
    the generated CSV.
    """
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return points @ R.T


def _signed_angle_to_up(v: np.ndarray) -> float:
    """Rotation that carries the 2-D vector ``v`` onto +Y.

    NUMERICAL CONTRACT: mirrors ``helper_functions.angle_between`` against the
    +Y reference, including its use of ``np.arccos`` rather than ``math.acos``.
    Those two differ by one ULP on some inputs — with the committed defaults,
    on exactly one element (the partial base unit) — and that single ULP moves
    11 fields of the generated CSV.
    """
    ref = np.array([0.0, 1.0, 0.0])
    v3 = np.array([v[0], v[1], 0.0], dtype=float)
    if np.linalg.norm(v3) == 0.0:
        raise ValueError("degenerate zero-length unit; check Delta_theta_deg")
    v1_u = v3 / np.linalg.norm(v3)
    v2_u = ref / np.linalg.norm(ref)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    angle = float(np.arccos(dot))
    if np.cross(v3, ref)[2] < 0:
        angle = -angle
    return angle


def _theta_samples(q0: float, delta_theta: float, tol: Tolerances) -> np.ndarray:
    """Spiral sample angles, tip (0) to base (q0), truncating the last span."""
    n_float = q0 / delta_theta
    n_round = round(n_float)
    if abs(n_float - n_round) * delta_theta <= tol.angle_rad and n_round >= 1:
        return np.linspace(0.0, q0, int(n_round) + 1)
    N = int(math.ceil(n_float))
    return np.array([min(k * delta_theta, q0) for k in range(N + 1)], dtype=float)


def _curled_quads(theta: np.ndarray, a: float, b: float) -> List[np.ndarray]:
    """Reproduces ``helper_functions.generate_spiral_pose`` exactly.

    Quad slots, curled frame: ``[A0, A1, B1, B0]`` where A is the central
    spiral and B the inner edge; subscript 0 is the lower theta.
    """
    rc = central_radius(theta, a, b)
    side_A = np.column_stack((rc * np.cos(theta), rc * np.sin(theta)))
    r_in = a * np.exp(b * theta)
    side_B = np.column_stack((r_in * np.cos(theta), r_in * np.sin(theta)))
    return [np.array([side_A[i], side_A[i + 1], side_B[i + 1], side_B[i]])
            for i in range(len(theta) - 1)]


def _straighten(quads: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Reproduces ``helper_functions.straighten_pose`` exactly."""
    out: List[np.ndarray] = []
    q0 = np.asarray(quads[0], dtype=float).copy()
    angle = _signed_angle_to_up(q0[1] - q0[0])
    placed = _rotate2d(q0 - q0[0], angle)
    out.append(placed)
    cursor = placed[1].copy()
    for q in quads[1:]:
        q = np.asarray(q, dtype=float).copy()
        angle = _signed_angle_to_up(q[1] - q[0])
        out.append(_rotate2d(q - q[0], angle) + cursor)
        cursor = out[-1][1].copy()
    return out


def _invert(straight: Sequence[np.ndarray], flip_length_m: float) -> List[np.ndarray]:
    """Reproduces ``helper_functions.Invert_pose`` exactly.

    Reflects about ``flip_length_m``, permutes slots ``[1,0,3,2]``, reverses
    the list. Result is base-first. See the module docstring for why the
    partial unit legitimately lands at the base.
    """
    out: List[np.ndarray] = []
    for quad in straight:
        q = np.asarray(quad, dtype=float).copy()
        q[:, 1] = -q[:, 1] + flip_length_m
        out.append(np.array([q[1], q[0], q[3], q[2]]))
    out.reverse()
    return out


def _chord_offset_factor(P0: np.ndarray, P1: np.ndarray) -> float:
    """Perpendicular distance from the spiral pole to the chord ``P0->P1``."""
    e = P1 - P0
    e = e / np.linalg.norm(e)
    return abs(float(e[0] * (-P0[1]) - e[1] * (-P0[0])))


def _frame_from_segment(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Right-handed frame with +Z along ``p0 -> p1``.

    Byte-identical to ``csv2xml._frame_from_segment``; duplicated here only so
    that the canonical layer keeps zero dependencies on the MJCF writer. A
    test asserts the two agree.
    """
    z = p1 - p0
    z = z / np.linalg.norm(z)
    fallback = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(fallback, z)
    nx = np.linalg.norm(x)
    if nx < 1e-12:
        fallback = np.array([0.0, 1.0, 0.0])
        x = np.cross(fallback, z)
        nx = np.linalg.norm(x)
    x = x / nx
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


# ══════════════════════════════════════════════════════════════════════════
#  The canonical model
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SpiRobGeometry:
    """One authoritative description of a SpiRob. Effectively immutable.

    Consumers read from this and never recompute spiral maths. Ordering is
    always explicit: :attr:`units` is base-to-tip (link order, matching the
    CSV and the MJCF), and :attr:`units_tip_to_base` is the spiral order.
    """
    inputs: UserInputs
    spiral: SpiralParameters
    lengths: LengthReport
    units: Tuple[UnitRecord, ...]                 # base -> tip (link order)
    tendon_paths: Tuple[TendonPath, ...]
    base_frame: BaseFrame
    tolerances: Tolerances
    warnings: Tuple[str, ...] = field(default_factory=tuple)

    # ── ordering ────────────────────────────────────────────────────────
    @property
    def units_base_to_tip(self) -> Tuple[UnitRecord, ...]:
        return self.units

    @property
    def units_tip_to_base(self) -> Tuple[UnitRecord, ...]:
        return tuple(reversed(self.units))

    @property
    def n_units(self) -> int:
        return len(self.units)

    def unit_by_link_name(self, name: str) -> UnitRecord:
        for u in self.units:
            if u.link_name == name:
                return u
        raise KeyError(f"no unit named {name!r}")

    # ── pose access (raw arrays, for legacy consumers) ───────────────────
    def curled_quads(self) -> List[np.ndarray]:
        """Spiral-order quads ``[A0, A1, B1, B0]``, tip first."""
        return [q.copy() for q in self._curled]

    def straight_quads(self) -> List[np.ndarray]:
        """Spiral-order straightened quads, tip first."""
        return [q.copy() for q in self._straight]

    def inverted_quads(self) -> List[np.ndarray]:
        """Link-order inverted quads, base first. Matches ``Invert_pose``."""
        return [q.copy() for q in self._inverted]

    # backing arrays, set by build_geometry via object.__setattr__
    _curled: Tuple[np.ndarray, ...] = field(default=(), repr=False, compare=False)
    _straight: Tuple[np.ndarray, ...] = field(default=(), repr=False, compare=False)
    _inverted: Tuple[np.ndarray, ...] = field(default=(), repr=False, compare=False)

    # ── tendons ─────────────────────────────────────────────────────────
    def tendon_path(self, cable_index: int) -> TendonPath:
        for p in self.tendon_paths:
            if p.cable_index == cable_index:
                return p
        raise KeyError(f"no tendon path for cable {cable_index}")

    def tendon_sites_for_unit(self, index_base_to_tip: int) -> Dict[int, Dict[str, np.ndarray]]:
        """``{cable: {"s1": xyz, "s2": xyz}}`` attachment points, CSV frame."""
        out: Dict[int, Dict[str, np.ndarray]] = {}
        for path in self.tendon_paths:
            for pt in path.points:
                if pt.unit_index_base_to_tip == index_base_to_tip:
                    out.setdefault(pt.cable_index, {})[pt.slot] = np.array(pt.attachment_m)
        return out

    # ── reporting ───────────────────────────────────────────────────────
    def to_manifest(self) -> dict:
        """Machine-readable build manifest. JSON-serialisable."""
        inp, sp, lr = self.inputs, self.spiral, self.lengths
        return {
            "schema_version": "2.0",
            "inputs": {
                "requested_length_m": inp.requested_length_m,
                "tip_width_m": inp.tip_width_m,
                "phi_deg_full_included": inp.phi_deg_full_included,
                "delta_theta_deg": inp.delta_theta_deg,
                "n_cables": inp.n_cables,
                "tendon_inward_shift_m": inp.tendon_inward_shift_m,
                "terminal_unit_policy": inp.terminal_unit_policy.value,
            },
            "spiral": {
                "b": sp.b, "a_m": sp.a_m, "E": sp.E,
                "q0_rad": sp.q0_rad, "q0_requested_rad": sp.q0_requested_rad,
                "beta_nominal": sp.beta_nominal,
            },
            "lengths": {
                "requested_continuous_length_m": lr.requested_continuous_length_m,
                "effective_continuous_length_m": lr.effective_continuous_length_m,
                "discrete_chord_length_m": lr.discrete_chord_length_m,
                "completion_delta_m": lr.completion_delta_m,
                "completion_delta_rel": lr.completion_delta_rel,
                "chord_deficit_m": lr.chord_deficit_m,
                "chord_deficit_rel": lr.chord_deficit_rel,
                "n_units_total": lr.n_units_total,
                "n_complete_units": lr.n_complete_units,
                "has_partial_unit": lr.has_partial_unit,
                "partial_unit_span_rad": lr.partial_unit_span_rad,
                "effective_q0_rad": lr.effective_q0_rad,
                "nominal_tip_width_m": lr.nominal_tip_width_m,
                "nominal_root_width_m": lr.nominal_root_width_m,
                "realized_tip_width_m": lr.realized_tip_width_m,
                "realized_root_width_m": lr.realized_root_width_m,
            },
            "base_frame": {
                "origin_offset_m": self.base_frame.origin_offset_m,
                "mount_plane_point_m": list(self.base_frame.mount_plane_point_m),
                "mount_plane_normal": list(self.base_frame.mount_plane_normal),
                "root_width_m": self.base_frame.root_width_m,
            },
            "units": [
                {
                    "link_name": u.link_name,
                    "index_base_to_tip": u.index_base_to_tip,
                    "index_tip_to_base": u.index_tip_to_base,
                    "is_partial": u.is_partial,
                    "theta_start_rad": u.theta_start_rad,
                    "theta_end_rad": u.theta_end_rad,
                    "angular_span_rad": u.angular_span_rad,
                    "arc_length_m": u.arc_length_m,
                    "chord_length_m": u.chord_length_m,
                    "nominal_width_start_m": u.nominal_width_start_m,
                    "nominal_width_end_m": u.nominal_width_end_m,
                    "realized_width_m": u.realized_width_m,
                    "slit_reference_m": list(u.slit_reference_m),
                }
                for u in self.units
            ],
            "tolerances": {
                "angle_rad": self.tolerances.angle_rad,
                "length_m": self.tolerances.length_m,
                "solver_rel": self.tolerances.solver_rel,
                "solver_max_iter": self.tolerances.solver_max_iter,
            },
            "warnings": list(self.warnings),
        }

    def with_policy(self, policy) -> "SpiRobGeometry":
        """Rebuild the same inputs under a different terminal-unit policy."""
        return build_geometry(replace(
            self.inputs, terminal_unit_policy=TerminalUnitPolicy.coerce(policy)),
            tolerances=self.tolerances)


# ══════════════════════════════════════════════════════════════════════════
#  Validation
# ══════════════════════════════════════════════════════════════════════════

def _validate(inputs: UserInputs) -> None:
    errs: List[str] = []
    if not math.isfinite(inputs.requested_length_m) or inputs.requested_length_m <= 0:
        errs.append(f"requested_length_m must be finite and > 0 (got {inputs.requested_length_m!r})")
    if not math.isfinite(inputs.tip_width_m) or inputs.tip_width_m <= 0:
        errs.append(f"tip_width_m must be finite and > 0 (got {inputs.tip_width_m!r})")
    if not (0.0 < inputs.phi_rad_full_included < math.radians(45.0)):
        errs.append(
            f"phi must lie in (0, 45) deg as a FULL included angle "
            f"(got {math.degrees(inputs.phi_rad_full_included)!r} deg)")
    if not (0.0 < inputs.delta_theta_rad < math.pi):
        errs.append(
            f"Delta_theta must lie in (0, 180) deg "
            f"(got {math.degrees(inputs.delta_theta_rad)!r} deg)")
    if inputs.n_cables < 2:
        errs.append(f"n_cables must be >= 2 (got {inputs.n_cables!r})")
    if inputs.tendon_inward_shift_m < 0:
        errs.append(f"tendon_inward_shift_m must be >= 0 (got {inputs.tendon_inward_shift_m!r})")
    if inputs.tendon_inward_shift_m >= inputs.tip_width_m / 2.0:
        errs.append(
            f"tendon_inward_shift_m ({inputs.tendon_inward_shift_m}) must be "
            f"< tip_width_m/2 ({inputs.tip_width_m / 2.0}) or the tendon exits the tip")
    if not isinstance(inputs.terminal_unit_policy, TerminalUnitPolicy):
        errs.append(f"terminal_unit_policy must be a TerminalUnitPolicy")
    if errs:
        raise ValueError("Invalid SpiRob inputs:\n" + "\n".join(f"  - {e}" for e in errs))


# ══════════════════════════════════════════════════════════════════════════
#  Builder
# ══════════════════════════════════════════════════════════════════════════

def build_geometry(inputs: UserInputs,
                   tolerances: Optional[Tolerances] = None) -> SpiRobGeometry:
    """Compute the canonical geometry once. Pure; no I/O, no globals."""
    tol = tolerances or Tolerances()
    _validate(inputs)

    b = b_from_phi(phi=inputs.phi_rad_full_included)
    a = a_from_tip_width(inputs.tip_width_m, b)
    E = _E(b)
    dth = inputs.delta_theta_rad

    q0_req = q_from_arc_length(inputs.requested_length_m, a, b)

    n_float = q0_req / dth
    n_round = round(n_float)
    lands_on_boundary = (abs(n_float - n_round) * dth <= tol.angle_rad and n_round >= 1)

    warnings: List[str] = []
    if inputs.terminal_unit_policy is TerminalUnitPolicy.WHOLE_UNITS and not lands_on_boundary:
        n_units = int(math.ceil(n_float))
        q0_eff = n_units * dth                      # extend forward, never back
    else:
        q0_eff = q0_req

    theta = _theta_samples(q0_eff, dth, tol)
    spans = np.diff(theta)
    has_partial = bool(spans[-1] < dth - tol.angle_rad)

    curled = _curled_quads(theta, a, b)
    straight = _straighten(curled)

    C = np.array([q[0] for q in curled] + [curled[-1][1]], dtype=float)
    chords = np.linalg.norm(np.diff(C, axis=0), axis=1)
    discrete_len = float(chords.sum())
    effective_len = continuous_arc_length(q0_eff, a, b)

    # Invert about the requested length; see module docstring. Under
    # whole_units the model is longer than requested, so the flip reference
    # becomes the effective length to keep the tip anchored consistently.
    flip_ref = (inputs.requested_length_m
                if inputs.terminal_unit_policy is TerminalUnitPolicy.EXACT_REQUESTED_LENGTH
                else effective_len)
    inverted = _invert(straight, flip_ref)

    n_total = len(curled)
    rc_vals = central_radius(theta, a, b)
    widths_nom = nominal_width(theta, a, b)
    k_scale = 2.0 / (E + 1.0)

    units: List[UnitRecord] = []
    for link_i in range(n_total):
        spiral_i = n_total - 1 - link_i             # traced, not assumed
        realized = (1.0 - k_scale) * _chord_offset_factor(C[spiral_i], C[spiral_i + 1]) * 2.0
        inv_q = inverted[link_i]
        base_pt = np.array([inv_q[0][0], 0.0, inv_q[0][1]])
        tip_pt = np.array([inv_q[1][0], 0.0, inv_q[1][1]])
        axis = tip_pt - base_pt
        axis = axis / np.linalg.norm(axis)
        units.append(UnitRecord(
            index_base_to_tip=link_i,
            index_tip_to_base=spiral_i,
            is_partial=bool(spiral_i == n_total - 1 and has_partial),
            theta_start_rad=float(theta[spiral_i]),
            theta_end_rad=float(theta[spiral_i + 1]),
            angular_span_rad=float(spans[spiral_i]),
            arc_length_m=float(continuous_arc_length(theta[spiral_i + 1], a, b)
                               - continuous_arc_length(theta[spiral_i], a, b)),
            chord_length_m=float(chords[spiral_i]),
            curled_centerline_start_m=tuple(map(float, C[spiral_i])),
            curled_centerline_end_m=tuple(map(float, C[spiral_i + 1])),
            straight_centerline_start_m=tuple(map(float, straight[spiral_i][0])),
            straight_centerline_end_m=tuple(map(float, straight[spiral_i][1])),
            inverted_centerline_start_m=tuple(map(float, inv_q[0])),
            inverted_centerline_end_m=tuple(map(float, inv_q[1])),
            local_frame_origin_m=tuple(map(float, base_pt)),
            local_frame_axis=tuple(map(float, axis)),
            nominal_width_start_m=float(widths_nom[spiral_i]),
            nominal_width_end_m=float(widths_nom[spiral_i + 1]),
            realized_width_m=float(realized),
            slit_reference_m=tuple(map(float, tip_pt)),
            slit_normal=tuple(map(float, axis)),
        ))

    realized_root = units[0].realized_width_m
    realized_tip = units[-1].realized_width_m

    completion_delta = effective_len - inputs.requested_length_m
    lengths = LengthReport(
        requested_continuous_length_m=inputs.requested_length_m,
        effective_continuous_length_m=effective_len,
        discrete_chord_length_m=discrete_len,
        completion_delta_m=completion_delta,
        completion_delta_rel=completion_delta / inputs.requested_length_m,
        chord_deficit_m=discrete_len - effective_len,
        chord_deficit_rel=(discrete_len - effective_len) / effective_len,
        n_units_total=n_total,
        n_complete_units=n_total - (1 if has_partial else 0),
        has_partial_unit=has_partial,
        partial_unit_span_rad=float(spans[-1]) if has_partial else 0.0,
        effective_q0_rad=q0_eff,
        nominal_tip_width_m=inputs.tip_width_m,
        nominal_root_width_m=root_width(inputs.tip_width_m, b, q0_eff),
        realized_tip_width_m=realized_tip,
        realized_root_width_m=realized_root,
    )

    if has_partial:
        warnings.append(
            "unit_completion: the requested length does not contain a whole "
            f"number of nominal units, so unit {n_total} spans "
            f"{math.degrees(spans[-1]):.4f} deg instead of "
            f"{math.degrees(dth):.4f} deg. After Invert_pose() this becomes "
            "link_001, the base link, by design — the intended large-at-base, "
            "small-at-tip ordering is preserved. Fabrication note: the printed "
            "part will have a partial unit at its base."
        )
    if inputs.terminal_unit_policy is TerminalUnitPolicy.WHOLE_UNITS and completion_delta > tol.length_m:
        warnings.append(
            f"whole_units: spiral extended to the next complete unit boundary. "
            f"The generated robot is {completion_delta * 1e3:.4f} mm "
            f"({completion_delta / inputs.requested_length_m * 100:+.4f} %) LONGER "
            f"than the {inputs.requested_length_m * 1e3:.4f} mm requested."
        )
    if abs(lengths.chord_deficit_rel) > 1e-3:
        warnings.append(
            f"arc_vs_chord: the straightened backbone measures "
            f"{discrete_len * 1e3:.4f} mm against an effective spiral arc of "
            f"{effective_len * 1e3:.4f} mm "
            f"({lengths.chord_deficit_rel * 100:+.4f} %). This is the "
            f"arc-to-chord shortening effect and is independent of unit completion."
        )

    tendons = _build_tendon_paths(inverted, inputs, tol)

    base_origin = min(float(np.asarray(q)[:, 1].min()) for q in inverted)
    base_frame = BaseFrame(
        origin_offset_m=base_origin,
        mount_plane_point_m=(0.0, 0.0, base_origin),
        mount_plane_normal=(0.0, 0.0, -1.0),
        root_width_m=realized_root,
    )

    geo = SpiRobGeometry(
        inputs=inputs,
        spiral=SpiralParameters(b=b, a_m=a, E=E, q0_rad=q0_eff,
                                q0_requested_rad=q0_req,
                                beta_nominal=adjacent_unit_scale(b, dth)),
        lengths=lengths,
        units=tuple(units),
        tendon_paths=tuple(tendons),
        base_frame=base_frame,
        tolerances=tol,
        warnings=tuple(warnings),
    )
    object.__setattr__(geo, "_curled", tuple(curled))
    object.__setattr__(geo, "_straight", tuple(straight))
    object.__setattr__(geo, "_inverted", tuple(inverted))
    return geo


def _build_tendon_paths(inverted: Sequence[np.ndarray],
                        inputs: UserInputs,
                        tol: Tolerances) -> List[TendonPath]:
    """Canonical tendon definition. One implementation, shared by all consumers.

    Stage 1 (attachment) reproduces
    ``helper_functions.generate_cable_sites_csv_zrot_from_P``.
    Stage 2 (routing) reproduces ``csv2xml.write_mjcf_from_sites_csv``'s
    inward-shift block. Both are verified to 1e-15 by the test suite, so
    adopting this model changes no existing output.
    """
    shift = inputs.tendon_inward_shift_m
    half_phi = inputs.phi_rad_full_included / 2.0
    paths: List[TendonPath] = []

    for c in range(inputs.n_cables):
        psi = 2.0 * math.pi * c / inputs.n_cables
        cos_p, sin_p = math.cos(psi), math.sin(psi)
        pts: List[TendonPoint] = []

        for link_i, quad in enumerate(inverted):
            # Slot meanings established by numeric trace, not by name:
            #   quad[0] centreline, base end      quad[1] centreline, tip end
            #   quad[2] inner edge, tip end       quad[3] inner edge, base end
            centre_base, centre_tip = quad[0], quad[1]
            inner_tip, inner_base = quad[2], quad[3]

            r_base, r_tip = float(inner_base[0]), float(inner_tip[0])
            s1_w = np.array([r_base * cos_p, r_base * sin_p, float(inner_base[1])])
            s2_w = np.array([r_tip * cos_p, r_tip * sin_p, float(inner_tip[1])])

            p0 = np.array([float(centre_base[0]), 0.0, float(centre_base[1])])
            p1 = np.array([float(centre_tip[0]), 0.0, float(centre_tip[1])])
            R = _frame_from_segment(p0, p1)

            s1_l = R.T @ (s1_w - p0)
            s2_l = R.T @ (s2_w - p0)
            r_old = float(np.linalg.norm(s1_l[:2]))
            if r_old < 1e-12:
                unit_r = np.array([1.0, 0.0])
                r_old = 1e-3
            else:
                unit_r = s1_l[:2] / r_old
            dz = float(s2_l[2] - s1_l[2])
            r1_new = max(r_old - shift, 1e-6)
            r2_new = max(r_old - shift - dz * math.tan(half_phi), 1e-6)
            s1_r = s1_l.copy()
            s2_r = s2_l.copy()
            s1_r[:2] = unit_r * r1_new
            s2_r[:2] = unit_r * r2_new

            pts.append(TendonPoint(
                cable_index=c, unit_index_base_to_tip=link_i, slot="s1",
                attachment_m=tuple(map(float, s1_w)),
                routed_m=tuple(map(float, R @ s1_r + p0)),
                surface_radius_m=abs(r_base), routed_radius_m=r1_new))
            pts.append(TendonPoint(
                cable_index=c, unit_index_base_to_tip=link_i, slot="s2",
                attachment_m=tuple(map(float, s2_w)),
                routed_m=tuple(map(float, R @ s2_r + p0)),
                surface_radius_m=abs(r_tip), routed_radius_m=r2_new))

        paths.append(TendonPath(cable_index=c, points=tuple(pts)))
    return paths


# ══════════════════════════════════════════════════════════════════════════
#  params.json boundary
# ══════════════════════════════════════════════════════════════════════════

def inputs_from_params(params: dict) -> UserInputs:
    """Convert a ``params.json`` dict into SI :class:`UserInputs`.

    Accepts ``taper_angle_deg`` as an alias for ``phi_deg``; if both are
    present they must agree, otherwise this raises rather than guessing.
    """
    missing = [k for k in ("L", "d_tip", "Delta_theta_deg", "n_cables",
                           "tendon_inward_shift") if k not in params]
    if "phi_deg" not in params and "taper_angle_deg" not in params:
        missing.append("phi_deg")
    if missing:
        raise ValueError("params is missing required keys: " + ", ".join(missing))

    phi_deg = params.get("phi_deg")
    alias = params.get("taper_angle_deg")
    if phi_deg is not None and alias is not None:
        if abs(float(phi_deg) - float(alias)) > 1e-12:
            raise ValueError(
                f"phi_deg ({phi_deg}) and taper_angle_deg ({alias}) disagree. "
                f"They are the same quantity — the FULL included taper angle. "
                f"Supply only one.")
    if phi_deg is None:
        phi_deg = alias

    policy = TerminalUnitPolicy.coerce(
        params.get("terminal_unit_policy", TerminalUnitPolicy.EXACT_REQUESTED_LENGTH))

    return UserInputs(
        requested_length_m=float(params["L"]),
        tip_width_m=float(params["d_tip"]),
        phi_rad_full_included=math.radians(float(phi_deg)),
        delta_theta_rad=math.radians(float(params["Delta_theta_deg"])),
        n_cables=int(params["n_cables"]),
        tendon_inward_shift_m=float(params["tendon_inward_shift"]),
        terminal_unit_policy=policy,
    )


def from_params(params: dict, tolerances: Optional[Tolerances] = None) -> SpiRobGeometry:
    """Convenience: ``params.json`` dict -> :class:`SpiRobGeometry`."""
    return build_geometry(inputs_from_params(params), tolerances=tolerances)
