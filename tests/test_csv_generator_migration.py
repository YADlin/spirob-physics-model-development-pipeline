"""
Regression tests for the spirob_csv_generator migration onto the canonical
geometry model.

Scope is deliberately narrow: this module tests that the CSV generator consumes
``spirob/geometry.py`` rather than re-deriving the spiral, that the default
policy still produces the established bytes, and that ``whole_units`` genuinely
reaches the generated CSV.

All source-file reads use ``encoding="utf-8"`` explicitly so the suite behaves
identically on Windows, where the default encoding is cp1252 and the em-dashes
and bullets in these files would otherwise raise ``UnicodeDecodeError``.

TOLERANCES
    EXACT   bit-identity, via ``==`` on the rendered CSV text
    ABS_M   1e-12 m, for geometric identities at the 0.2 m scale of this model
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import helper_functions as hf  # noqa: E402
import spirob_csv_generator as gen  # noqa: E402
from spirob.geometry import TerminalUnitPolicy, from_params  # noqa: E402

ABS_M = 1e-12


def _read_text(path):
    """Windows-safe source read. See the module docstring."""
    with io.open(path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def params():
    return json.loads(_read_text(os.path.join(_ROOT, "params.json")))


def _write_csv(params_dict, path):
    """Drive the generator's serialisation path exactly as __main__ does."""
    geo = from_params(params_dict)
    hf.generate_cable_sites_csv_zrot_from_P(
        geo.inverted_quads(),
        n_cables=params_dict["n_cables"],
        csv_path=str(path),
        radial_scale=1.0,
    )
    return geo


def _rows(path):
    with io.open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ── The migration itself ────────────────────────────────────────────────────

def test_generator_no_longer_derives_the_spiral():
    """The duplicated b / a / q0 / pose calculations must be gone."""
    src = _read_text(os.path.join(_ROOT, "spirob_csv_generator.py"))
    for symbol in ("solve_b_for_phi", "generate_spiral_pose",
                   "straighten_pose", "Invert_pose"):
        assert symbol not in src, (
            f"{symbol} is still called in spirob_csv_generator.py; geometry "
            f"must come from spirob/geometry.py")
    assert "from helper_functions import *" not in src, \
        "the star import re-exposes the legacy geometry helpers"


def test_generator_imports_the_canonical_model():
    src = _read_text(os.path.join(_ROOT, "spirob_csv_generator.py"))
    assert "from spirob.geometry import" in src
    assert "from_params" in src


def test_generator_still_exposes_validate_params():
    """build.py imports this symbol; it is public API."""
    assert callable(gen.validate_params)
    assert callable(gen.load_params)


def test_csv_writer_shim_is_still_the_serialiser():
    """helper_functions remains necessary, but only for CSV serialisation."""
    src = _read_text(os.path.join(_ROOT, "spirob_csv_generator.py"))
    assert "generate_cable_sites_csv_zrot_from_P" in src


# ── Default policy: byte-level compatibility ────────────────────────────────

def test_default_csv_matches_legacy_pipeline_byte_for_byte(params, tmp_path):
    """Canonical geometry must reproduce the pre-migration CSV exactly.

    The legacy chain is reconstructed here from helper_functions, using the
    legacy b/a derivation, and compared as raw text.
    """
    legacy_csv = tmp_path / "legacy.csv"
    canon_csv = tmp_path / "canon.csv"

    b = hf.solve_b_for_phi(math.radians(params["phi_deg"]))
    a = params["d_tip"] / (math.exp(2 * math.pi * b) - 1.0)
    raw = hf.generate_spiral_pose(
        a, b, Length=params["L"],
        delta_theta=math.radians(params["Delta_theta_deg"]))
    legacy_pose = hf.Invert_pose(hf.straighten_pose(raw), params["L"])
    hf.generate_cable_sites_csv_zrot_from_P(
        legacy_pose, n_cables=params["n_cables"],
        csv_path=str(legacy_csv), radial_scale=1.0)

    _write_csv(params, canon_csv)

    assert _read_text(str(canon_csv)) == _read_text(str(legacy_csv)), \
        "migrated CSV differs from the legacy pipeline output"


def test_canonical_pose_is_bitwise_identical_to_legacy(params):
    """Guards the numerical contract in spirob/geometry.py.

    phi_from_b must use atan2, b_from_phi must keep the legacy early exit,
    _rotate2d must use np.cos/np.sin and _signed_angle_to_up must use
    np.arccos. Any of those drifting changes the generated CSV.
    """
    import numpy as np
    geo = from_params(params)
    b = hf.solve_b_for_phi(math.radians(params["phi_deg"]))
    a = params["d_tip"] / (math.exp(2 * math.pi * b) - 1.0)
    assert geo.spiral.b == b, "b is no longer bitwise equal to the legacy solver"
    assert geo.spiral.a_m == a, "a is no longer bitwise equal"

    raw = hf.generate_spiral_pose(
        a, b, Length=params["L"],
        delta_theta=math.radians(params["Delta_theta_deg"]))
    legacy = hf.Invert_pose(hf.straighten_pose(raw), params["L"])
    for i, (got, want) in enumerate(zip(geo.inverted_quads(), legacy)):
        assert np.array_equal(got, want), f"inverted quad {i} is not bitwise equal"


def test_default_policy_when_key_absent(params, tmp_path):
    """Omitting terminal_unit_policy must behave as exact_requested_length."""
    without = {k: v for k, v in params.items() if k != "terminal_unit_policy"}
    a = tmp_path / "absent.csv"
    b = tmp_path / "explicit.csv"
    _write_csv(without, a)
    _write_csv(dict(params, terminal_unit_policy="exact_requested_length"), b)
    assert _read_text(str(a)) == _read_text(str(b))


# ── Unit count and ordering ─────────────────────────────────────────────────

def test_default_unit_count_is_21(params, tmp_path):
    out = tmp_path / "c.csv"
    _write_csv(params, out)
    assert len(_rows(str(out))) == 21


def test_whole_units_does_not_add_a_twenty_second_unit(params, tmp_path):
    """whole_units COMPLETES the partial unit; it must not append one."""
    out = tmp_path / "wu.csv"
    geo = _write_csv(dict(params, terminal_unit_policy="whole_units"), out)
    assert geo.n_units == 21
    assert len(_rows(str(out))) == 21


def test_csv_is_ordered_base_to_tip(params, tmp_path):
    """elem 1 must be the widest; radius must decrease monotonically."""
    out = tmp_path / "c.csv"
    _write_csv(params, out)
    rows = _rows(str(out))
    radii = [abs(float(r["c0_s1_x"])) for r in rows]
    assert radii[0] == max(radii), "elem 1 is not the large-radius base"
    assert radii[-1] == min(radii), "the last elem is not the small-radius tip"
    assert all(radii[i] > radii[i + 1] for i in range(len(radii) - 1))


def test_elem_numbering_starts_at_one_and_is_contiguous(params, tmp_path):
    out = tmp_path / "c.csv"
    _write_csv(params, out)
    assert [int(r["elem"]) for r in _rows(str(out))] == list(range(1, 22))


@pytest.mark.parametrize("policy", ["exact_requested_length", "whole_units"])
def test_ordering_holds_under_both_policies(params, tmp_path, policy):
    out = tmp_path / f"{policy}.csv"
    _write_csv(dict(params, terminal_unit_policy=policy), out)
    radii = [abs(float(r["c0_s1_x"])) for r in _rows(str(out))]
    assert radii[0] == max(radii) and radii[-1] == min(radii)


# ── Partial versus complete base unit ───────────────────────────────────────

def test_default_base_unit_is_partial_and_shorter_than_its_neighbour(params, tmp_path):
    out = tmp_path / "c.csv"
    geo = _write_csv(params, out)
    assert geo.lengths.has_partial_unit
    assert geo.units[0].is_partial

    rows = _rows(str(out))

    def height(r):
        return abs(float(r["joint_s2_z"]) - float(r["joint_s1_z"]))

    assert height(rows[0]) < height(rows[1]), \
        "the partial base unit should be shorter than elem 2"
    assert height(rows[0]) == pytest.approx(
        geo.units[0].chord_length_m, abs=ABS_M)


def test_whole_units_base_unit_is_complete_and_longest(params, tmp_path):
    out = tmp_path / "wu.csv"
    geo = _write_csv(dict(params, terminal_unit_policy="whole_units"), out)
    assert not geo.lengths.has_partial_unit
    assert not any(u.is_partial for u in geo.units)
    assert geo.lengths.n_complete_units == geo.lengths.n_units_total == 21

    rows = _rows(str(out))

    def height(r):
        return abs(float(r["joint_s2_z"]) - float(r["joint_s1_z"]))

    assert height(rows[0]) > height(rows[1]), \
        "under whole_units the base unit is complete and therefore the longest"


def test_whole_units_actually_changes_the_csv(params, tmp_path):
    """Proof the policy reaches the generated artefact, not just the report."""
    a = tmp_path / "exact.csv"
    b = tmp_path / "whole.csv"
    _write_csv(dict(params, terminal_unit_policy="exact_requested_length"), a)
    _write_csv(dict(params, terminal_unit_policy="whole_units"), b)
    assert _read_text(str(a)) != _read_text(str(b))


def test_whole_units_extends_never_shortens(params, tmp_path):
    exact = _write_csv(dict(params, terminal_unit_policy="exact_requested_length"),
                       tmp_path / "e.csv")
    whole = _write_csv(dict(params, terminal_unit_policy="whole_units"),
                       tmp_path / "w.csv")
    assert (whole.lengths.effective_continuous_length_m
            > exact.lengths.effective_continuous_length_m)
    assert whole.lengths.completion_delta_m > 0


# ── Validation ──────────────────────────────────────────────────────────────

def test_validate_params_accepts_both_policies(params):
    for policy in ("exact_requested_length", "whole_units"):
        gen.validate_params(dict(params, terminal_unit_policy=policy))


def test_validate_params_rejects_an_unknown_policy(params):
    with pytest.raises(ValueError, match="terminal_unit_policy"):
        gen.validate_params(dict(params, terminal_unit_policy="truncate"))


def test_validate_params_accepts_params_without_the_policy_key(params):
    gen.validate_params({k: v for k, v in params.items()
                         if k != "terminal_unit_policy"})


# ── End-to-end through the real CLI ─────────────────────────────────────────

def test_generator_cli_runs_and_reports_both_policies(params, tmp_path):
    """Exercises __main__, including the reporting block."""
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"

    for policy, expect_partial in (("exact_requested_length", True),
                                   ("whole_units", False)):
        pfile = tmp_path / f"params_{policy}.json"
        with io.open(str(pfile), "w", encoding="utf-8") as f:
            json.dump(dict(params, terminal_unit_policy=policy), f)
        out = tmp_path / f"{policy}.csv"
        proc = subprocess.run(
            [sys.executable, "spirob_csv_generator.py",
             "--params", str(pfile), "--out", str(out), "--yes"],
            cwd=_ROOT, capture_output=True, text=True, encoding="utf-8", env=child_env,)
        assert proc.returncode == 0, proc.stderr
        assert "Number of elements generated: 21" in proc.stdout
        assert f"terminal_unit_policy= {policy}" in proc.stdout
        assert ("partial               = 1" if expect_partial
                else "partial               = 0") in proc.stdout
        assert len(_rows(str(out))) == 21
