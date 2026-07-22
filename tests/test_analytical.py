"""
Phase 1/2 analytical + discretization tests.

Reference values live *here*, in the test fixtures, and are deliberately not
importable by production code (Phase 1 requirement: "do not hard-code these
values into production logic").

Tolerances
----------
REL_TIGHT  1e-9   closed-form identities that must hold to solver precision
REL_REF    5e-9   agreement with the committed reference constants, which are
                  quoted to 10 significant figures
ABS_GEOM   1e-12  metre-scale geometric identities (machine precision at the
                  0.2 m scale of this model)
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from tools.geometry_audit import (  # noqa: E402
    E_of, phi_of_b, b_of_phi, a_of, r_c, local_width, arc_length,
    q0_from_arc_length, beta_of, d_root_of, theta_samples,
    centerline_points, discrete_backbone_length, solve_q0_for_discrete_length,
    audit,
)
import helper_functions as hf  # noqa: E402

REL_TIGHT = 1e-9
REL_REF = 5e-9
ABS_GEOM = 1e-12

# Committed default parameters (params.json as of the audit commit).
DEFAULTS = dict(L=0.22628, d_tip=0.007139, phi_deg=6.3, Delta_theta_deg=30.78,
                n_cables=3, tendon_inward_shift=0.0015)

# Reference constants for DEFAULTS, from the Phase 1 specification.
REF = dict(
    b=0.1369640839,
    a=0.0052319156,
    q0=10.9632013369,
    d_root=0.0320448838,
    beta=1.076353346,
    discrete_backbone_length=0.2236554,
    continuous_arc_length=0.22628,
)


@pytest.fixture(scope="module")
def derived():
    phi = math.radians(DEFAULTS["phi_deg"])
    b = b_of_phi(phi)
    a = a_of(DEFAULTS["d_tip"], b)
    q0 = q0_from_arc_length(DEFAULTS["L"], a, b)
    dth = math.radians(DEFAULTS["Delta_theta_deg"])
    return dict(phi=phi, b=b, a=a, q0=q0, dth=dth)


# ── Reference-value reproduction ────────────────────────────────────────────

@pytest.mark.parametrize("key", ["b", "a", "q0"])
def test_reference_constants(derived, key):
    assert derived[key] == pytest.approx(REF[key], rel=REL_REF)


def test_reference_d_root_and_beta(derived):
    d_root = d_root_of(DEFAULTS["d_tip"], derived["b"], derived["q0"])
    assert d_root == pytest.approx(REF["d_root"], rel=REL_REF)
    assert beta_of(derived["b"], derived["dth"]) == pytest.approx(REF["beta"], rel=REL_REF)


# ── phi(b) inversion ────────────────────────────────────────────────────────

@pytest.mark.parametrize("phi_deg", [0.5, 2.0, 6.3, 12.0, 30.0, 44.9])
def test_phi_b_roundtrip(phi_deg):
    phi = math.radians(phi_deg)
    assert phi_of_b(b_of_phi(phi)) == pytest.approx(phi, rel=REL_TIGHT)


def test_phi_is_monotonic_in_b():
    bs = np.linspace(1e-6, 2.0, 400)
    phis = np.array([phi_of_b(b) for b in bs])
    assert np.all(np.diff(phis) > 0)


def test_phi_is_the_full_included_angle(derived):
    """tan(phi/2) must equal d(half-width)/d(arc length) along the spiral.

    This is the identity that makes ``phi_deg`` the FULL included taper
    angle. If phi_deg were a half-angle, this test would fail by 2x.
    """
    a, b = derived["a"], derived["b"]
    for theta in (0.0, 1.0, 5.0, derived["q0"]):
        h = 0.5 * float(local_width(theta, a, b))          # half width
        dh_dtheta = b * h
        ds_dtheta = math.sqrt(1.0 + b * b) * float(r_c(theta, a, b))
        assert dh_dtheta / ds_dtheta == pytest.approx(
            math.tan(derived["phi"] / 2.0), rel=REL_TIGHT)


# ── Width / length identities ───────────────────────────────────────────────

def test_tip_width(derived):
    assert float(local_width(0.0, derived["a"], derived["b"])) == pytest.approx(
        DEFAULTS["d_tip"], rel=REL_TIGHT)


def test_outer_edge_is_inner_edge_one_turn_later(derived):
    """r_c + d/2 at theta equals r_c - d/2 at theta + 2*pi. Self-similarity."""
    a, b = derived["a"], derived["b"]
    for theta in (0.0, 2.0, 7.5):
        outer = float(r_c(theta, a, b)) + 0.5 * float(local_width(theta, a, b))
        inner_next = (float(r_c(theta + 2 * math.pi, a, b))
                      - 0.5 * float(local_width(theta + 2 * math.pi, a, b)))
        assert outer == pytest.approx(inner_next, rel=REL_TIGHT)


def test_arc_length_matches_numerical_quadrature(derived):
    a, b, q0 = derived["a"], derived["b"], derived["q0"]
    th = np.linspace(0.0, q0, 2_000_001)
    integrand = math.sqrt(1.0 + b * b) * r_c(th, a, b)
    numeric = float(np.trapezoid(integrand, th))
    assert arc_length(q0, a, b) == pytest.approx(numeric, rel=1e-9)


def test_q0_roundtrip(derived):
    a, b = derived["a"], derived["b"]
    for L in (0.05, 0.22628, 0.5):
        assert arc_length(q0_from_arc_length(L, a, b), a, b) == pytest.approx(L, rel=REL_TIGHT)


# ── Parameter round trips (Phase 10) ────────────────────────────────────────

def test_roundtrip_taper_angle_vs_root_width(derived):
    """(L, d_tip, phi) -> (L, d_tip, d_root) -> (L, d_tip, phi) is lossless."""
    a, b, q0 = derived["a"], derived["b"], derived["q0"]
    d_root = d_root_of(DEFAULTS["d_tip"], b, q0)

    def b_from_root_width(L, d_tip, d_root_target, tol=1e-15, max_iter=300):
        lo, hi = 1e-9, 3.0
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            a_m = a_of(d_tip, mid)
            q_m = q0_from_arc_length(L, a_m, mid)
            if d_root_of(d_tip, mid, q_m) < d_root_target:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return 0.5 * (lo + hi)

    b_back = b_from_root_width(DEFAULTS["L"], DEFAULTS["d_tip"], d_root)
    assert b_back == pytest.approx(b, rel=1e-8)
    assert math.degrees(phi_of_b(b_back)) == pytest.approx(DEFAULTS["phi_deg"], rel=1e-7)


# ── Discretization ──────────────────────────────────────────────────────────

def test_discrete_length_matches_reference(derived):
    th = theta_samples(derived["q0"], derived["dth"], "truncate")
    L_disc = discrete_backbone_length(th, derived["a"], derived["b"])
    assert L_disc == pytest.approx(REF["discrete_backbone_length"], abs=5e-7)


def test_discrete_length_is_below_continuous(derived):
    th = theta_samples(derived["q0"], derived["dth"], "truncate")
    L_disc = discrete_backbone_length(th, derived["a"], derived["b"])
    L_cont = arc_length(derived["q0"], derived["a"], derived["b"])
    assert L_disc < L_cont
    assert (L_cont - L_disc) / L_cont == pytest.approx(0.0115988, abs=1e-6)


@pytest.mark.parametrize("dth_deg", [60.0, 30.78, 10.0, 2.0, 0.5])
def test_discrete_converges_to_continuous(derived, dth_deg):
    """Chord sum -> arc length as Delta_theta -> 0, at second order."""
    a, b, q0 = derived["a"], derived["b"], derived["q0"]
    dth = math.radians(dth_deg)
    L_disc = discrete_backbone_length(theta_samples(q0, dth, "uniform_dtheta"), a, b)
    L_cont = arc_length(q0, a, b)
    rel = (L_cont - L_disc) / L_cont
    assert 0.0 <= rel < 0.2
    assert rel < 0.02 * (dth_deg / 30.78) ** 2 + 1e-6


def test_discrete_backbone_solver_converges(derived):
    a, b, dth = derived["a"], derived["b"], derived["dth"]
    q0d = solve_q0_for_discrete_length(DEFAULTS["L"], a, b, dth, "truncate")
    got = discrete_backbone_length(theta_samples(q0d, dth, "truncate"), a, b)
    assert got == pytest.approx(DEFAULTS["L"], rel=1e-10)
    assert q0d > derived["q0"]          # must curl further to make up the deficit


def test_discrete_backbone_solver_is_deterministic(derived):
    a, b, dth = derived["a"], derived["b"], derived["dth"]
    vals = {solve_q0_for_discrete_length(DEFAULTS["L"], a, b, dth) for _ in range(5)}
    assert len(vals) == 1


def test_solver_rejects_invalid_delta_theta(derived):
    with pytest.raises(ValueError, match="delta_theta"):
        solve_q0_for_discrete_length(DEFAULTS["L"], derived["a"], derived["b"], 0.0)


def test_solver_reports_iteration_exhaustion(derived):
    """The iteration limit must surface as an actionable error, not a silent
    return of a half-converged root."""
    with pytest.raises(RuntimeError, match="iterations exhausted"):
        solve_q0_for_discrete_length(
            DEFAULTS["L"], derived["a"], derived["b"], derived["dth"],
            tol=1e-18, max_iter=3)


# ── Terminal-unit policies ──────────────────────────────────────────────────

def test_truncate_policy_leaves_a_short_last_unit(derived):
    th = theta_samples(derived["q0"], derived["dth"], "truncate")
    spans = np.diff(th)
    assert np.allclose(spans[:-1], derived["dth"])
    assert spans[-1] < derived["dth"]
    assert math.degrees(spans[-1]) == pytest.approx(12.5452, abs=1e-3)


def test_uniform_dtheta_policy_gives_equal_spans(derived):
    th = theta_samples(derived["q0"], derived["dth"], "uniform_dtheta")
    spans = np.diff(th)
    assert np.allclose(spans, spans[0], atol=1e-14)
    assert len(spans) == len(np.diff(theta_samples(derived["q0"], derived["dth"], "truncate")))


def test_uniform_policy_gives_equal_scale_ratios(derived):
    """Under uniform_dtheta every adjacent-unit width ratio equals beta_eff."""
    a, b, q0, dth = derived["a"], derived["b"], derived["q0"], derived["dth"]
    th = theta_samples(q0, dth, "uniform_dtheta")
    w = local_width(th, a, b)
    ratios = w[1:] / w[:-1]
    assert np.allclose(ratios, ratios[0], rtol=1e-12)
    assert ratios[0] == pytest.approx(beta_of(b, th[1] - th[0]), rel=REL_TIGHT)


def test_unit_count_is_deterministic(derived):
    counts = {len(theta_samples(derived["q0"], derived["dth"], "truncate")) for _ in range(10)}
    assert counts == {22}          # 21 units + 1


# ── Agreement with the shipped implementation ───────────────────────────────

def test_helper_functions_agree_with_reference_equations(derived):
    """helper_functions.generate_spiral_pose must reproduce the spec equations."""
    a, b, q0, dth = derived["a"], derived["b"], derived["q0"], derived["dth"]
    quads = hf.generate_spiral_pose(a, b, Length=DEFAULTS["L"], delta_theta=dth)
    th = theta_samples(q0, dth, "truncate")
    C = centerline_points(th, a, b)
    assert len(quads) == len(th) - 1
    for i, q in enumerate(quads):
        assert np.allclose(q[0], C[i], atol=ABS_GEOM)        # A0 = centre at theta_i
        assert np.allclose(q[1], C[i + 1], atol=ABS_GEOM)    # A1 = centre at theta_i+1
        r_inner = a * math.exp(b * th[i])
        assert np.linalg.norm(q[3]) == pytest.approx(r_inner, rel=REL_TIGHT)


def test_helper_solve_b_matches_reference_inverter():
    for phi_deg in (1.0, 6.3, 20.0, 40.0):
        phi = math.radians(phi_deg)
        assert hf.solve_b_for_phi(phi) == pytest.approx(b_of_phi(phi), rel=1e-8)


def test_audit_manifest_shape():
    """The audit tool now emits the canonical model's manifest."""
    geo = audit(DEFAULTS)
    m = geo.to_manifest()
    assert m["schema_version"] == "2.0"
    assert m["lengths"]["n_units_total"] == 21
    assert m["lengths"]["n_complete_units"] == 20
    assert m["lengths"]["has_partial_unit"] is True
    assert m["lengths"]["effective_continuous_length_m"] == pytest.approx(
        DEFAULTS["L"], rel=1e-12)
    assert m["lengths"]["discrete_chord_length_m"] == pytest.approx(
        REF["discrete_backbone_length"], abs=5e-7)
    assert len(m["units"]) == 21
    assert m["units"][0]["link_name"] == "link_001"
    assert m["units"][0]["is_partial"] is True          # base, by design
    assert m["units"][-1]["is_partial"] is False        # tip
    assert m["inputs"]["terminal_unit_policy"] == "exact_requested_length"


def test_committed_params_json_still_matches_fixture():
    """Guards the fixture against silent drift of the committed defaults."""
    with open(os.path.join(_ROOT, "params.json")) as f:
        p = json.load(f)
    for k, v in DEFAULTS.items():
        assert p[k] == v, f"params.json[{k}] changed; update tests/REF too"
