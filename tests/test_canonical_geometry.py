"""
Phase 2 tests for the canonical geometry layer.

Covers the twelve required areas:

  1  default exact-length behaviour is unchanged
  2  the partial unit sits at the base after Invert_pose()
  3  the smallest unit is at the tip
  4  the largest unit is at the base
  5  whole_units extends and never shortens
  6  whole_units yields only complete units
  7  an exact-boundary length yields no partial unit under either policy
  8  arc length and chord length are reported separately
  9  both orderings are deterministic and reversible
 10  tendon points are shared, not independently recalculated
 11  input validation
 12  tolerances and dimensional units are documented

TOLERANCES USED HERE
    EXACT     0.0     bit-identity against the legacy implementation
    ABS_M     1e-12   metre-scale geometric identity
    ABS_TEND  1e-9    tendon agreement across a CSV text round trip
    REL       1e-12   relative agreement of closed-form quantities

UNITS
    Every numeric value in this module is SI (metres, radians) unless the
    name ends in ``_deg`` or ``_mm``.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import helper_functions as hf  # noqa: E402
from spirob.geometry import (  # noqa: E402
    BaseFrame, LengthReport, SpiRobGeometry, SpiralParameters, TendonPoint,
    TerminalUnitPolicy, Tolerances, UnitRecord, UserInputs,
    build_geometry, continuous_arc_length, from_params, inputs_from_params,
)

EXACT = 0.0
ABS_M = 1e-12
ABS_TEND = 1e-9
REL = 1e-12


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def params():
    with open(os.path.join(_ROOT, "params.json")) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def geo(params):
    return from_params(params)


@pytest.fixture(scope="module")
def geo_whole(params):
    p = dict(params)
    p["terminal_unit_policy"] = "whole_units"
    return from_params(p)


# ── 1. Default behaviour is unchanged ───────────────────────────────────────

def test_canonical_reproduces_legacy_curled_pose_exactly(geo, params):
    legacy = hf.generate_spiral_pose(geo.spiral.a_m, geo.spiral.b,
                                     Length=params["L"],
                                     delta_theta=geo.inputs.delta_theta_rad)
    got = geo.curled_quads()
    assert len(got) == len(legacy)
    for i, (g, l) in enumerate(zip(got, legacy)):
        assert np.array_equal(g, l), f"curled quad {i} differs from Invert-era output"


def test_canonical_reproduces_legacy_straight_and_inverted_pose_exactly(geo, params):
    legacy_raw = hf.generate_spiral_pose(geo.spiral.a_m, geo.spiral.b,
                                         Length=params["L"],
                                         delta_theta=geo.inputs.delta_theta_rad)
    legacy_straight = hf.straighten_pose(legacy_raw)
    legacy_inverted = hf.Invert_pose(legacy_straight, params["L"])
    for i, (g, l) in enumerate(zip(geo.straight_quads(), legacy_straight)):
        assert np.array_equal(g, l), f"straight quad {i} differs"
    for i, (g, l) in enumerate(zip(geo.inverted_quads(), legacy_inverted)):
        assert np.array_equal(g, l), f"inverted quad {i} differs"


def test_default_policy_is_exact_requested_length(params):
    p = dict(params)
    p.pop("terminal_unit_policy", None)
    g = from_params(p)
    assert g.inputs.terminal_unit_policy is TerminalUnitPolicy.EXACT_REQUESTED_LENGTH
    assert g.lengths.effective_continuous_length_m == pytest.approx(
        p["L"], rel=REL), "the default must preserve the requested length exactly"
    assert g.lengths.completion_delta_m == pytest.approx(0.0, abs=ABS_M)


def test_default_unit_count_unchanged(geo):
    assert geo.n_units == 21


# ── 2/3/4. Ordering after inversion ─────────────────────────────────────────

def test_partial_unit_is_at_the_base(geo):
    """Correct by design: Invert_pose() reverses, so the partial terminal
    spiral interval becomes link_001. Do not move it to the tip."""
    assert geo.lengths.has_partial_unit
    assert geo.units[0].is_partial, "the partial unit must be the base link"
    assert geo.units[0].link_name == "link_001"
    assert not any(u.is_partial for u in geo.units[1:])
    assert geo.units[0].index_tip_to_base == geo.n_units - 1


def test_smallest_unit_is_at_the_tip(geo):
    widths = [u.realized_width_m for u in geo.units_base_to_tip]
    assert widths[-1] == min(widths)
    assert geo.units[-1].theta_start_rad == pytest.approx(0.0, abs=ABS_M)


def test_largest_unit_is_at_the_base(geo):
    widths = [u.realized_width_m for u in geo.units_base_to_tip]
    assert widths[0] == max(widths)
    assert all(widths[i] > widths[i + 1] for i in range(len(widths) - 1)), \
        "width must decrease monotonically from base to tip"


def test_partial_base_unit_is_still_the_largest(geo):
    """The partial unit is short in ANGLE, not small in radius."""
    assert geo.units[0].angular_span_rad < geo.inputs.delta_theta_rad
    assert geo.units[0].realized_width_m == max(u.realized_width_m for u in geo.units)


# ── 5/6. whole_units policy ─────────────────────────────────────────────────

def test_whole_units_extends_never_shortens(geo, geo_whole):
    assert geo_whole.lengths.effective_continuous_length_m > \
        geo_whole.lengths.requested_continuous_length_m
    assert geo_whole.lengths.completion_delta_m > 0
    assert geo_whole.spiral.q0_rad > geo_whole.spiral.q0_requested_rad


def test_whole_units_has_no_partial_unit(geo_whole):
    assert not geo_whole.lengths.has_partial_unit
    assert not any(u.is_partial for u in geo_whole.units)
    assert geo_whole.lengths.n_complete_units == geo_whole.lengths.n_units_total


def test_whole_units_spans_are_all_nominal(geo_whole):
    dth = geo_whole.inputs.delta_theta_rad
    for u in geo_whole.units:
        assert u.angular_span_rad == pytest.approx(dth, rel=REL)


def test_whole_units_q0_is_an_integer_multiple(geo_whole):
    n = geo_whole.n_units
    assert geo_whole.spiral.q0_rad == pytest.approx(
        n * geo_whole.inputs.delta_theta_rad, rel=REL)


def test_whole_units_reports_the_excess(geo_whole):
    assert any("LONGER" in w for w in geo_whole.warnings)


@pytest.mark.parametrize("L", [0.05, 0.1, 0.22628, 0.4])
def test_whole_units_never_shortens_across_lengths(params, L):
    p = dict(params, L=L)
    exact = from_params(dict(p, terminal_unit_policy="exact_requested_length"))
    whole = from_params(dict(p, terminal_unit_policy="whole_units"))
    assert whole.lengths.effective_continuous_length_m >= \
        exact.lengths.effective_continuous_length_m - ABS_M
    assert whole.lengths.effective_continuous_length_m >= L - ABS_M


# ── 7. Exact-boundary length ────────────────────────────────────────────────

@pytest.fixture(scope="module")
def boundary_params(params):
    """A requested length that lands exactly on a unit boundary."""
    g = from_params(params)
    n = 12
    L_exact = continuous_arc_length(n * g.inputs.delta_theta_rad,
                                    g.spiral.a_m, g.spiral.b)
    return dict(params, L=L_exact), n


def test_exact_boundary_has_no_partial_unit_under_either_policy(boundary_params):
    p, n = boundary_params
    for policy in ("exact_requested_length", "whole_units"):
        g = from_params(dict(p, terminal_unit_policy=policy))
        assert not g.lengths.has_partial_unit, f"{policy} produced a partial unit"
        assert g.n_units == n, f"{policy} produced {g.n_units} units, expected {n}"
        assert g.lengths.n_complete_units == n


def test_exact_boundary_policies_are_identical(boundary_params):
    p, _ = boundary_params
    a = from_params(dict(p, terminal_unit_policy="exact_requested_length"))
    b = from_params(dict(p, terminal_unit_policy="whole_units"))
    assert a.spiral.q0_rad == pytest.approx(b.spiral.q0_rad, rel=REL)
    assert a.lengths.effective_continuous_length_m == pytest.approx(
        b.lengths.effective_continuous_length_m, rel=REL)
    assert b.lengths.completion_delta_m == pytest.approx(0.0, abs=ABS_M)


# ── 8. Arc versus chord, reported separately ────────────────────────────────

def test_arc_and_chord_lengths_are_distinct_fields(geo):
    lr = geo.lengths
    assert lr.effective_continuous_length_m != lr.discrete_chord_length_m
    assert lr.discrete_chord_length_m < lr.effective_continuous_length_m
    assert lr.chord_deficit_m < 0


def test_unit_completion_and_chord_deficit_are_independent(geo, geo_whole):
    """Under the default policy the completion effect is zero while the chord
    deficit is not. The two must never be conflated."""
    assert geo.lengths.completion_delta_m == pytest.approx(0.0, abs=ABS_M)
    assert abs(geo.lengths.chord_deficit_rel) > 1e-3
    # under whole_units BOTH are non-zero and have opposite signs
    assert geo_whole.lengths.completion_delta_m > 0
    assert geo_whole.lengths.chord_deficit_m < 0


def test_per_unit_arc_exceeds_per_unit_chord(geo):
    for u in geo.units:
        assert u.arc_length_m > u.chord_length_m > 0


def test_unit_chords_sum_to_the_discrete_length(geo):
    total = sum(u.chord_length_m for u in geo.units)
    assert total == pytest.approx(geo.lengths.discrete_chord_length_m, abs=ABS_M)


def test_unit_arcs_sum_to_the_effective_length(geo):
    total = sum(u.arc_length_m for u in geo.units)
    assert total == pytest.approx(geo.lengths.effective_continuous_length_m, rel=1e-10)


# ── 9. Ordering conventions ─────────────────────────────────────────────────

def test_both_orderings_are_exact_reverses(geo):
    b2t = geo.units_base_to_tip
    t2b = geo.units_tip_to_base
    assert len(b2t) == len(t2b)
    assert [u.index_base_to_tip for u in t2b] == list(range(geo.n_units - 1, -1, -1))
    for u in b2t:
        assert u.index_base_to_tip + u.index_tip_to_base == geo.n_units - 1


def test_ordering_is_reversible(geo):
    assert tuple(reversed(geo.units_tip_to_base)) == geo.units_base_to_tip


def test_ordering_is_deterministic(params):
    a = from_params(params)
    b = from_params(params)
    assert [u.index_tip_to_base for u in a.units] == [u.index_tip_to_base for u in b.units]
    assert [u.link_name for u in a.units] == [u.link_name for u in b.units]


def test_link_names_match_the_csv_and_xml_convention(geo):
    assert [u.link_name for u in geo.units][:3] == ["link_001", "link_002", "link_003"]
    assert geo.units[-1].link_name == f"link_{geo.n_units:03d}"


def test_theta_increases_toward_the_base(geo):
    t2b = geo.units_tip_to_base
    thetas = [u.theta_start_rad for u in t2b]
    assert thetas == sorted(thetas)
    assert thetas[0] == pytest.approx(0.0, abs=ABS_M)


# ── 10. Shared tendon points ────────────────────────────────────────────────

def test_canonical_tendon_points_match_csv2xml(geo, params, tmp_path):
    """The MJCF writer's independent computation must agree with the canonical
    definition. This is what makes 'shared' real rather than aspirational."""
    c2x = _load("c2x_canon", "csv2xml.py")
    csv_path = tmp_path / "sites.csv"
    hf.generate_cable_sites_csv_zrot_from_P(
        geo.inverted_quads(), n_cables=params["n_cables"], csv_path=str(csv_path))

    shift = params["tendon_inward_shift"]
    half = math.radians(params["phi_deg"] / 2.0)
    elements, _ = c2x._parse_sites_csv(str(csv_path))

    worst_att = worst_routed = 0.0
    for e in elements:
        link_i = e["idx"] - 1
        p0, p1 = e["p0"], e["p1"]
        R = c2x._frame_from_segment(p0, p1)
        for c, ss in e["sites"].items():
            s1 = c2x._world_to_local(R, p0, ss["s1"]).astype(float)
            s2 = c2x._world_to_local(R, p0, ss["s2"]).astype(float)
            r = float(np.linalg.norm(s1[:2]))
            u = s1[:2] / r
            dz = float(s2[2] - s1[2])
            s1[:2] = u * max(r - shift, 1e-6)
            s2[:2] = u * max(r - shift - dz * math.tan(half), 1e-6)
            expected = {"s1": (ss["s1"], R @ s1 + p0), "s2": (ss["s2"], R @ s2 + p0)}
            for pt in geo.tendon_path(c).points:
                if pt.unit_index_base_to_tip != link_i:
                    continue
                att, routed = expected[pt.slot]
                worst_att = max(worst_att, float(np.max(np.abs(np.array(pt.attachment_m) - att))))
                worst_routed = max(worst_routed, float(np.max(np.abs(np.array(pt.routed_m) - routed))))

    assert worst_att < ABS_TEND, f"attachment points differ by {worst_att:.3e} m"
    assert worst_routed < ABS_TEND, f"routed points differ by {worst_routed:.3e} m"


def test_preview_consumes_canonical_tendon_points(geo, params):
    """preview.py must not recompute; it must read the canonical points."""
    pv = _load("pv_canon", "preview.py")
    xs, ys = pv._tendon_path(pv._build_quads(params), params["tendon_inward_shift"],
                             params["phi_deg"], params)
    cable0 = geo.tendon_path(0)
    expected = []
    for unit in reversed(geo.units_base_to_tip):
        by_slot = {p.slot: p for p in cable0.points
                   if p.unit_index_base_to_tip == unit.index_base_to_tip}
        for slot in ("s2", "s1"):
            expected.append(by_slot[slot].routed_m)
    assert len(xs) == len(expected)
    for (x, y), e in zip(zip(xs, ys), expected):
        assert x == pytest.approx(e[0], abs=ABS_M)
        assert y == pytest.approx(e[2], abs=ABS_M)


def test_preview_build_quads_is_the_canonical_inverted_pose(geo, params):
    pv = _load("pv_canon2", "preview.py")
    for a, b in zip(pv._build_quads(params), geo.inverted_quads()):
        assert np.array_equal(a, b)


def test_preview_no_longer_forks_the_spiral_maths():
    """F-10 regression guard."""
    src = open(os.path.join(_ROOT, "preview.py"), encoding="utf-8").read()
    for symbol in ("def _phi_from_b", "def _solve_b", "def _rotate2d",
                   "def _angle_between", "def _normalize"):
        assert symbol not in src, f"{symbol} has reappeared in preview.py"


def test_tendon_points_cover_every_unit_and_cable(geo, params):
    for c in range(params["n_cables"]):
        path = geo.tendon_path(c)
        assert len(path.points) == 2 * geo.n_units
        slots = {(p.unit_index_base_to_tip, p.slot) for p in path.points}
        assert len(slots) == 2 * geo.n_units


def test_tendon_routed_radius_is_inside_the_surface(geo):
    for path in geo.tendon_paths:
        for p in path.points:
            assert 0.0 < p.routed_radius_m <= p.surface_radius_m + ABS_M


# ── 11. Input validation ────────────────────────────────────────────────────

def _base_inputs(**over):
    d = dict(requested_length_m=0.22628, tip_width_m=0.007139,
             phi_rad_full_included=math.radians(6.3),
             delta_theta_rad=math.radians(30.78), n_cables=3,
             tendon_inward_shift_m=0.0015,
             terminal_unit_policy=TerminalUnitPolicy.EXACT_REQUESTED_LENGTH)
    d.update(over)
    return UserInputs(**d)


@pytest.mark.parametrize("over,needle", [
    (dict(requested_length_m=0.0), "requested_length_m"),
    (dict(requested_length_m=-1.0), "requested_length_m"),
    (dict(requested_length_m=float("nan")), "requested_length_m"),
    (dict(requested_length_m=float("inf")), "requested_length_m"),
    (dict(tip_width_m=0.0), "tip_width_m"),
    (dict(tip_width_m=-0.001), "tip_width_m"),
    (dict(phi_rad_full_included=0.0), "phi"),
    (dict(phi_rad_full_included=math.radians(45.0)), "phi"),
    (dict(phi_rad_full_included=math.radians(90.0)), "phi"),
    (dict(delta_theta_rad=0.0), "Delta_theta"),
    (dict(delta_theta_rad=math.pi), "Delta_theta"),
    (dict(n_cables=1), "n_cables"),
    (dict(tendon_inward_shift_m=-0.001), "tendon_inward_shift_m"),
    (dict(tendon_inward_shift_m=0.05), "tendon_inward_shift_m"),
])
def test_invalid_inputs_are_rejected(over, needle):
    with pytest.raises(ValueError, match=needle):
        build_geometry(_base_inputs(**over))


@pytest.mark.parametrize("bad", ["", "whole", "WholeUnits ", "truncate", None, 3, True])
def test_invalid_policy_is_rejected(bad):
    with pytest.raises(ValueError, match="terminal_unit_policy"):
        TerminalUnitPolicy.coerce(bad)


@pytest.mark.parametrize("good,expected", [
    ("whole_units", TerminalUnitPolicy.WHOLE_UNITS),
    ("WHOLE_UNITS", TerminalUnitPolicy.WHOLE_UNITS),
    ("  exact_requested_length  ", TerminalUnitPolicy.EXACT_REQUESTED_LENGTH),
    (TerminalUnitPolicy.WHOLE_UNITS, TerminalUnitPolicy.WHOLE_UNITS),
])
def test_policy_coercion_accepts_reasonable_spellings(good, expected):
    assert TerminalUnitPolicy.coerce(good) is expected


def test_missing_params_keys_are_reported(params):
    p = dict(params)
    del p["d_tip"]
    with pytest.raises(ValueError, match="d_tip"):
        inputs_from_params(p)


def test_taper_angle_alias_is_accepted(params):
    p = dict(params)
    p["taper_angle_deg"] = p.pop("phi_deg")
    g = from_params(p)
    assert g.inputs.phi_deg_full_included == pytest.approx(6.3, rel=REL)


def test_conflicting_phi_aliases_raise(params):
    p = dict(params, taper_angle_deg=7.0)
    with pytest.raises(ValueError, match="disagree"):
        inputs_from_params(p)


def test_agreeing_phi_aliases_are_accepted(params):
    p = dict(params, taper_angle_deg=params["phi_deg"])
    assert inputs_from_params(p).phi_deg_full_included == pytest.approx(
        params["phi_deg"], rel=REL)


# ── 12. Tolerances and units are documented ─────────────────────────────────

def test_tolerances_are_explicit_and_documented():
    tol = Tolerances()
    assert tol.angle_rad > 0 and tol.length_m > 0
    assert tol.solver_max_iter >= 1
    doc = Tolerances.__doc__ or ""
    for field in dataclasses.fields(Tolerances):
        assert field.name in doc, f"Tolerances.{field.name} is undocumented"


def test_tolerances_are_overridable(params):
    g = from_params(params, tolerances=Tolerances(angle_rad=1e-6))
    assert g.tolerances.angle_rad == 1e-6


_DIMENSIONLESS = {
    "b", "E", "beta_nominal", "n_cables", "n_units_total", "n_complete_units",
    "has_partial_unit", "index_base_to_tip", "index_tip_to_base", "is_partial",
    "cable_index", "unit_index_base_to_tip", "slot", "solver_max_iter",
    "terminal_unit_policy", "local_frame_axis", "slit_normal",
    "mount_plane_normal", "points", "attachment_m", "routed_m",
}
_UNIT_TOKENS = ("m", "rad", "deg", "rel", "iter")


def _declares_units(name: str) -> bool:
    """True if any name segment is a unit token, e.g. phi_rad_full_included."""
    return any(part in _UNIT_TOKENS for part in name.split("_"))


@pytest.mark.parametrize("cls", [UserInputs, SpiralParameters, LengthReport,
                                 UnitRecord, TendonPoint, BaseFrame, Tolerances])
def test_every_dimensional_field_carries_its_unit_in_its_name(cls):
    for f in dataclasses.fields(cls):
        if f.name in _DIMENSIONLESS or f.name.startswith("_"):
            continue
        assert _declares_units(f.name), (
            f"{cls.__name__}.{f.name} does not declare its units; "
            f"the name must contain one of {_UNIT_TOKENS}")


def test_module_documents_the_conventions():
    import spirob.geometry as g
    doc = g.__doc__ or ""
    for heading in ("UNITS", "PHI", "ANGLE DIRECTION AND REFERENCE AXIS",
                    "ORDERING", "Invert_pose", "LENGTHS", "TENDON POINTS"):
        assert heading in doc, f"convention '{heading}' is not documented"


# ── Immutability and manifest ───────────────────────────────────────────────

def test_geometry_is_effectively_immutable(geo):
    with pytest.raises(dataclasses.FrozenInstanceError):
        geo.units = ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        geo.units[0].realized_width_m = 1.0


def test_mutating_returned_quads_does_not_affect_the_model(geo):
    q = geo.inverted_quads()
    before = geo.inverted_quads()[0].copy()
    q[0][0, 0] = 999.0
    assert np.array_equal(geo.inverted_quads()[0], before)


def test_manifest_is_json_serialisable(geo):
    blob = json.dumps(geo.to_manifest())
    round_trip = json.loads(blob)
    assert round_trip["lengths"]["n_units_total"] == geo.n_units


def test_with_policy_rebuilds_without_mutating(geo):
    other = geo.with_policy("whole_units")
    assert other.inputs.terminal_unit_policy is TerminalUnitPolicy.WHOLE_UNITS
    assert geo.inputs.terminal_unit_policy is TerminalUnitPolicy.EXACT_REQUESTED_LENGTH


# ── Fabrication anchors (deferred phase) ────────────────────────────────────

def test_base_frame_anchor_is_exposed(geo, params):
    bf = geo.base_frame
    assert bf.origin_offset_m == pytest.approx(
        params["L"] - geo.lengths.discrete_chord_length_m, abs=ABS_M)
    assert bf.root_width_m == pytest.approx(geo.units[0].realized_width_m, abs=ABS_M)
    assert len(bf.mount_plane_point_m) == 3 and len(bf.mount_plane_normal) == 3


def test_every_unit_exposes_a_slit_reference(geo):
    for u in geo.units:
        assert len(u.slit_reference_m) == 3
        assert len(u.slit_normal) == 3
        assert u.slit_reference_m == u.inverted_centerline_end_m[:1] + (0.0,) + \
            u.inverted_centerline_end_m[1:]


def test_slit_references_are_ordered_base_to_tip(geo):
    z = [u.slit_reference_m[2] for u in geo.units_base_to_tip]
    assert all(z[i] < z[i + 1] for i in range(len(z) - 1))


def test_canonical_layer_has_no_heavy_dependencies():
    """The canonical layer must stay importable in a headless, CAD-free env."""
    src = open(os.path.join(_ROOT, "spirob", "geometry.py"), encoding="utf-8",).read()
    for forbidden in ("matplotlib", "mujoco", "cadquery", "PySide6", "vtk"):
        assert forbidden not in src, f"spirob/geometry.py imports {forbidden}"
