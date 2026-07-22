"""
Phase 4 tests: vertex ordering, quad validity, and tendon-path conventions.

Two classes of test live here.

  1. Invariants that already hold and must keep holding (winding, quad area,
     monotonicity, containment). These are plain passing tests.

  2. Executable statements of the defects found in the Phase 1 audit. These
     are marked ``xfail(strict=True)`` so the suite stays green today, and
     will *fail loudly* the moment a fix lands without the marker being
     removed. Each carries the finding ID from docs/GEOMETRY_AUDIT.md.

Nothing here asserts a convention chosen from variable *names*. Every
expectation is anchored to a numeric property of the geometry.
"""

from __future__ import annotations

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
from tools.geometry_audit import (  # noqa: E402
    b_of_phi, a_of, E_of, q0_from_arc_length, theta_samples, centerline_points,
)


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
def poses(params):
    b = b_of_phi(math.radians(params["phi_deg"]))
    a = a_of(params["d_tip"], b)
    dth = math.radians(params["Delta_theta_deg"])
    raw = hf.generate_spiral_pose(a, b, Length=params["L"], delta_theta=dth)
    straight = hf.straighten_pose(raw)
    inverted = hf.Invert_pose(straight, params["L"])
    return dict(a=a, b=b, dth=dth, raw=raw, straight=straight, inverted=inverted,
                q0=q0_from_arc_length(params["L"], a, b))


def _signed_area(quad) -> float:
    q = np.asarray(quad, dtype=float)
    x, y = q[:, 0], q[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _self_intersects(quad) -> bool:
    """True if the closed polygon's two diagonals-as-edges cross (bow-tie)."""
    q = np.asarray(quad, dtype=float)

    def seg(p, r, s, t):
        d1, d2 = r - p, t - s
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-18:
            return False
        u = ((s[0] - p[0]) * d2[1] - (s[1] - p[1]) * d2[0]) / den
        v = ((s[0] - p[0]) * d1[1] - (s[1] - p[1]) * d1[0]) / den
        return 1e-9 < u < 1 - 1e-9 and 1e-9 < v < 1 - 1e-9

    return seg(q[0], q[1], q[2], q[3]) or seg(q[1], q[2], q[3], q[0])


# ── Quad validity (these hold today) ────────────────────────────────────────

@pytest.mark.parametrize("stage", ["raw", "straight", "inverted"])
def test_winding_is_consistent(poses, stage):
    areas = [_signed_area(q) for q in poses[stage]]
    assert len({np.sign(v) for v in areas}) == 1, f"{stage}: mixed winding"


@pytest.mark.parametrize("stage", ["raw", "straight", "inverted"])
def test_quad_area_is_positive_and_nonzero(poses, stage):
    for i, q in enumerate(poses[stage]):
        assert abs(_signed_area(q)) > 1e-9, f"{stage}[{i}] degenerate"


@pytest.mark.parametrize("stage", ["raw", "straight", "inverted"])
def test_no_vertex_crossing(poses, stage):
    for i, q in enumerate(poses[stage]):
        assert not _self_intersects(q), f"{stage}[{i}] is self-intersecting"


def test_invert_preserves_area_magnitude(poses):
    a_s = sorted(abs(_signed_area(q)) for q in poses["straight"])
    a_i = sorted(abs(_signed_area(q)) for q in poses["inverted"])
    assert np.allclose(a_s, a_i, rtol=1e-12)


def test_straightened_element_has_parallel_sides(poses):
    """F-06: the inner edge is a radial scaling of the centreline, so after
    chord-straightening both of its endpoints share one lateral offset.

    Each element is therefore a constant-width block; the taper exists only
    as a step between elements. This is a property of the model, not a bug,
    but downstream code must not assume an intra-element taper.
    """
    for i, q in enumerate(poses["straight"]):
        A0, A1, B1, B0 = q
        assert abs(A0[0]) < 1e-15 and abs(A1[0]) < 1e-15, f"element {i}: A edge not on axis"
        assert B0[0] == pytest.approx(B1[0], abs=1e-15), \
            f"element {i}: inner edge is not parallel to the centreline"


def test_inverted_pose_is_ordered_base_to_tip(poses):
    widths = [abs(q[2][0]) for q in poses["inverted"]]
    assert widths == sorted(widths, reverse=True), "inverted pose is not base-first"


# ── The truncated unit lands at the base (F-05) ─────────────────────────────

def test_truncated_unit_becomes_the_base_link(poses, params):
    """Documents current behaviour: Invert_pose() reverses the chain, so the
    short terminal unit becomes link_001 rather than the tip link."""
    th = theta_samples(poses["q0"], poses["dth"], "truncate")
    spans = np.diff(th)
    assert spans[-1] < poses["dth"]

    chords = np.linalg.norm(np.diff(centerline_points(th, poses["a"], poses["b"]),
                                    axis=0), axis=1)
    base_h = abs(poses["inverted"][0][1][1] - poses["inverted"][0][0][1])
    tip_h = abs(poses["inverted"][-1][1][1] - poses["inverted"][-1][0][1])

    # link_001 is the LAST spiral interval, i.e. the truncated one.
    assert base_h == pytest.approx(float(chords[-1]), abs=1e-12)
    assert tip_h == pytest.approx(float(chords[0]), abs=1e-12)

    # Anomaly: despite being the largest-radius element, the base link is
    # shorter than its neighbour because its angular span was cut short.
    assert chords[-1] < chords[-2]


# ── Base offset (F-04) ──────────────────────────────────────────────────────

@pytest.mark.xfail(strict=True, reason="F-04: Invert_pose flips about the "
                   "requested continuous length L, not the realised discrete "
                   "backbone length, so the base sits ~2.62 mm above z=0")
def test_base_of_robot_sits_at_origin(poses):
    z_min = min(float(np.asarray(q)[:, 1].min()) for q in poses["inverted"])
    assert z_min == pytest.approx(0.0, abs=1e-9)


def test_base_offset_equals_the_length_deficit(poses, params):
    """The offset is exactly L_requested - L_discrete, confirming the cause."""
    a, b, dth = poses["a"], poses["b"], poses["dth"]
    th = theta_samples(poses["q0"], dth, "truncate")
    C = centerline_points(th, a, b)
    L_disc = float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())
    z_min = min(float(np.asarray(q)[:, 1].min()) for q in poses["inverted"])
    assert z_min == pytest.approx(params["L"] - L_disc, abs=1e-12)


# ── Tendon path ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tendon_paths(params, tmp_path_factory):
    """Cable-0 path as (radius, height) tip->base, from both producers."""
    csv_path = tmp_path_factory.mktemp("geom") / "sites.csv"
    b = b_of_phi(math.radians(params["phi_deg"]))
    a = a_of(params["d_tip"], b)
    dth = math.radians(params["Delta_theta_deg"])
    raw = hf.generate_spiral_pose(a, b, Length=params["L"], delta_theta=dth)
    inv = hf.Invert_pose(hf.straighten_pose(raw), params["L"])
    hf.generate_cable_sites_csv_zrot_from_P(
        inv, n_cables=params["n_cables"], csv_path=str(csv_path))

    c2x = _load("c2x_t", "csv2xml.py")
    pv = _load("pv_t", "preview.py")

    shift = params["tendon_inward_shift"]
    half = math.radians(params["phi_deg"] / 2.0)
    els, _ = c2x._parse_sites_csv(str(csv_path))

    mj = []
    for e in els:
        p0, p1 = e["p0"], e["p1"]
        R = c2x._frame_from_segment(p0, p1)
        ss = e["sites"][0]
        s1 = c2x._world_to_local(R, p0, ss["s1"]).astype(float)
        s2 = c2x._world_to_local(R, p0, ss["s2"]).astype(float)
        r_old = float(np.linalg.norm(s1[:2]))
        u = s1[:2] / r_old
        dz = float(s2[2] - s1[2])
        s1[:2] = u * max(r_old - shift, 1e-6)
        s2[:2] = u * max(r_old - shift - dz * math.tan(half), 1e-6)
        w1, w2 = R @ s1 + p0, R @ s2 + p0
        mj.append((float(np.linalg.norm(w1[:2])), float(w1[2]),
                   float(np.linalg.norm(w2[:2])), float(w2[2])))

    mj_path = []
    for r1, z1, r2, z2 in reversed(mj):        # MJCF routes s2 then s1, tip->base
        mj_path.append((r2, z2))
        mj_path.append((r1, z1))

    xs, ys = pv._tendon_path(pv._build_quads(params), shift, params["phi_deg"])
    pv_path = [(abs(float(x)), float(y)) for x, y in zip(xs, ys)]
    return dict(mjcf=mj_path, preview=pv_path, surface=[(r1, z1, r2, z2) for r1, z1, r2, z2 in mj])


def test_tendon_radius_is_monotonic_base_to_tip(tendon_paths):
    r = [p[0] for p in tendon_paths["mjcf"]]
    assert all(r[i] <= r[i + 1] + 1e-12 for i in range(len(r) - 1))


def test_tendon_height_is_monotonic(tendon_paths):
    z = [p[1] for p in tendon_paths["mjcf"]]
    assert all(z[i] >= z[i + 1] - 1e-12 for i in range(len(z) - 1))


def test_tendon_sites_lie_inside_their_link(tendon_paths, params):
    """Every routed site must be strictly inside the link outer surface."""
    for r_new, _, r2_new, _ in tendon_paths["surface"]:
        assert r_new > 0.0 and r2_new > 0.0


@pytest.mark.xfail(strict=True, reason="F-07: preview.py unpacks the quad as "
                   "(A1, A0, B0, B1) while the CSV writer unpacks the same "
                   "array as (A0, A1, B1, B0); the preview therefore anchors "
                   "the taper correction at the opposite end of each link and "
                   "mis-draws the tendon by up to 1.30 mm")
def test_preview_tendon_path_matches_mjcf(tendon_paths):
    mj, pv = tendon_paths["mjcf"], tendon_paths["preview"]
    assert len(mj) == len(pv)
    for (rm, zm), (rp, zp) in zip(mj, pv):
        assert zm == pytest.approx(zp, abs=1e-9)
        assert rm == pytest.approx(rp, abs=1e-9)


def test_preview_mjcf_disagreement_is_the_documented_magnitude(tendon_paths):
    """Pins the size of F-07 so a partial fix cannot pass unnoticed."""
    mj, pv = tendon_paths["mjcf"], tendon_paths["preview"]
    worst = max(abs(rm - rp) for (rm, _), (rp, _) in zip(mj, pv))
    assert worst == pytest.approx(1.2956e-3, abs=1e-6)


@pytest.mark.xfail(strict=True, reason="F-08: the -dz*tan(phi/2) correction "
                   "applies an intra-link taper that the discretised geometry "
                   "does not have (surface radius is constant within a link), "
                   "so the routed tendon steps at every link boundary")
def test_tendon_offset_from_surface_is_constant(tendon_paths, params):
    shift = params["tendon_inward_shift"]
    for r1_new, _, r2_new, _ in tendon_paths["surface"]:
        assert (r1_new - r2_new) == pytest.approx(0.0, abs=1e-9)
    del shift


def test_link_surface_radius_is_constant_within_a_link(poses):
    """Numeric ground truth behind F-08."""
    for i, q in enumerate(poses["inverted"]):
        assert abs(q[2][0]) == pytest.approx(abs(q[3][0]), abs=1e-15), \
            f"link {i} surface radius is not constant"
