"""
Regression tests for the csv2geom_nlobe migration onto the canonical geometry
model.

Scope: that the STL generator takes its unit structure, ordering, naming,
draft-angle convention and validation from ``spirob/geometry.py``, that the
default output is unchanged, and that both terminal-unit policies work.

CAD DEPENDENCY
    Tests that invoke CadQuery carry the ``@requires_cad`` skipif decorator and
    are skipped automatically when CadQuery is unavailable, so the analytical
    suite still runs in a CAD-free CI. No custom pytest marker is registered,
    deliberately: registering one would need a pytest.ini this project does not
    currently have.

TOLERANCES
    ABS_M   1e-12 m   geometric identity at the 0.2 m scale of this model
    CSV_TOL 1e-9  m   the generator's own CSV-vs-canonical guard band

WINDOWS / PYTHON 3.13
    Every source read pins ``encoding="utf-8"``; without it Windows decodes
    with cp1252 and the box-drawing characters and em-dashes in these modules
    raise UnicodeDecodeError.

    Subprocess tests additionally set ``PYTHONIOENCODING=utf-8`` in the child
    environment. Passing ``encoding="utf-8"`` to ``subprocess.run`` only tells
    the parent how to DECODE the pipe; it does not tell the child how to ENCODE
    it. On a cp1252 console the child would fail writing the ✓ and → characters
    that csv2geom_nlobe prints, so the environment variable is the part that
    actually makes this reliable.
"""

from __future__ import annotations

import importlib.util
import io
import json
import math
import os
import struct
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from spirob.geometry import from_params  # noqa: E402

ABS_M = 1e-12

_HAS_CAD = importlib.util.find_spec("cadquery") is not None
requires_cad = pytest.mark.skipif(not _HAS_CAD, reason="CadQuery not installed")


def _read_text(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def params():
    return json.loads(_read_text(os.path.join(_ROOT, "params.json")))


@pytest.fixture(scope="module")
def stl_mod():
    """Import csv2geom_nlobe, or skip if CadQuery is absent."""
    if not _HAS_CAD:
        pytest.skip("CadQuery not installed")
    spec = importlib.util.spec_from_file_location(
        "csv2geom_nlobe_under_test", os.path.join(_ROOT, "csv2geom_nlobe.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory, params):
    """A CSV generated the way the production pipeline generates it."""
    import helper_functions as hf
    out = tmp_path_factory.mktemp("geom") / "sites.csv"
    geo = from_params(params)
    hf.generate_cable_sites_csv_zrot_from_P(
        geo.inverted_quads(), n_cables=params["n_cables"],
        csv_path=str(out), radial_scale=1.0)
    return str(out)


def _stl_triangle_count(path):
    with open(path, "rb") as f:
        data = f.read()
    return struct.unpack("<I", data[80:84])[0]


def _stl_vertices(path):
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack("<I", data[80:84])[0]
    verts = []
    for i in range(n):
        off = 84 + 50 * i
        verts.extend(struct.unpack("<9f", data[off + 12:off + 48]))
    return verts


# ── The migration is real ───────────────────────────────────────────────────

def test_stl_generator_imports_the_canonical_model():
    src = _read_text(os.path.join(_ROOT, "csv2geom_nlobe.py"))
    assert "from spirob.geometry import" in src
    assert "from_params" in src
    assert "SpiRobGeometry" in src


def test_stale_phi_default_is_gone():
    """Audit F-11: phi_deg=5.7 silently disagreed with params.json's 6.3."""
    src = _read_text(os.path.join(_ROOT, "csv2geom_nlobe.py"))
    assert "phi_deg=5.7" not in src
    assert "phi_deg = 5.7" not in src


def test_column_mapping_is_declared_once():
    """The column->quad-slot mapping must not be re-spelled per call site."""
    src = _read_text(os.path.join(_ROOT, "csv2geom_nlobe.py"))
    assert "_SLOT_COLUMNS" in src
    # the old inline literal list must not survive
    assert 'point_sets = ["joint_s1", "joint_s2", "c0_s2", "c0_s1"]' not in src


@requires_cad
def test_slot_columns_match_canonical_quad_order(stl_mod, params, csv_path):
    """_SLOT_COLUMNS must be the canonical inverted-quad slot order."""
    import pandas as pd
    geo = from_params(params)
    df = pd.read_csv(csv_path)
    units = stl_mod.build_unit_inputs(df, geo)
    for i, unit in enumerate(units):
        quad = geo.inverted_quads()[i]
        for slot, (cx, _cy, cz) in enumerate(unit.profile_xyz):
            assert cx == pytest.approx(float(quad[slot][0]), abs=1e-9)
            assert cz == pytest.approx(float(quad[slot][1]), abs=1e-9)


@requires_cad
def test_process_csv_passes_the_canonical_draft_angle_to_cad(
        stl_mod, params, csv_path, tmp_path, monkeypatch):
    """Behavioural: capture the draft angle actually handed to the CAD stage.

    φ is the FULL included taper angle, so the CAD draft must be φ/2 taken from
    the canonical model. Asserting the arithmetic alone would not prove
    process_csv uses it, so the real call is intercepted.
    """
    seen = []
    real_cut = stl_mod.add_nlobe_cut

    def spy(solid, n, outer_radius, height_z, draft_angle_deg, **kw):
        seen.append((n, draft_angle_deg))
        return real_cut(solid, n, outer_radius, height_z, draft_angle_deg, **kw)

    monkeypatch.setattr(stl_mod, "add_nlobe_cut", spy)

    geo = from_params(params)
    stl_mod.process_csv(csv_file=csv_path, outdir=str(tmp_path / "spy"),
                        geometry=geo)

    assert len(seen) == geo.n_units
    expected_draft = geo.inputs.phi_deg_full_included / 2.0
    for n_cables_seen, draft in seen:
        assert draft == expected_draft
        assert n_cables_seen == geo.inputs.n_cables


@requires_cad
def test_conflicting_legacy_phi_raises_instead_of_being_used_silently(
        stl_mod, params, csv_path, tmp_path, monkeypatch):
    """The historical 5.7 default must not be able to override canonical 6.3."""
    seen = []
    monkeypatch.setattr(stl_mod, "add_nlobe_cut",
                        lambda *a, **k: seen.append(a) or a[0])

    with pytest.raises(ValueError, match="canonical geometry says"):
        stl_mod.process_csv(csv_file=csv_path, outdir=str(tmp_path / "c1"),
                            geometry=from_params(params), phi_deg=5.7)
    assert seen == [], "CAD ran despite a conflicting phi_deg"


@requires_cad
def test_conflicting_legacy_n_cables_raises(stl_mod, params, csv_path, tmp_path):
    with pytest.raises(ValueError, match="canonical geometry says"):
        stl_mod.process_csv(csv_file=csv_path, outdir=str(tmp_path / "c2"),
                            geometry=from_params(params), n_cables=5)


@requires_cad
def test_agreeing_legacy_values_are_accepted(stl_mod, params, csv_path, tmp_path):
    """Passing the same values the canonical model holds must not be an error."""
    stl_mod.process_csv(csv_file=csv_path, outdir=str(tmp_path / "ok"),
                        geometry=from_params(params),
                        phi_deg=float(params["phi_deg"]),
                        n_cables=int(params["n_cables"]))
    assert len(os.listdir(str(tmp_path / "ok"))) == 21


def _extract_call_args(source, func_name):
    """Return the argument text of the first `func_name(...)` call.

    Naive splitting on ')' breaks on nested calls such as
    `params.get("nlobe_t", 0.5)`, so the parentheses are balanced properly.
    """
    start = source.index(func_name + "(") + len(func_name) + 1
    depth = 1
    for i in range(start, len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                return source[start:i]
    raise AssertionError(f"unbalanced parentheses after {func_name}(")


def test_cli_does_not_pass_legacy_n_cables_or_phi_deg():
    """The __main__ call must rely on the canonical model, not params scalars."""
    src = _read_text(os.path.join(_ROOT, "csv2geom_nlobe.py"))
    main_block = src.split('if __name__ == "__main__":', 1)[1]
    call = _extract_call_args(main_block, "process_csv")
    assert "n_cables" not in call, "CLI still passes a legacy n_cables"
    assert "phi_deg" not in call, "CLI still passes a legacy phi_deg"
    assert "geometry" in call, "CLI does not pass the canonical geometry"


@requires_cad
def test_process_csv_refuses_to_guess_phi(stl_mod, csv_path, tmp_path):
    with pytest.raises(ValueError, match="phi_deg"):
        stl_mod.process_csv(csv_file=csv_path, outdir=str(tmp_path))


# ── Canonical drives structure ──────────────────────────────────────────────

@requires_cad
def test_unit_inputs_are_named_and_ordered_base_to_tip(stl_mod, params, csv_path):
    import pandas as pd
    geo = from_params(params)
    units = stl_mod.build_unit_inputs(pd.read_csv(csv_path), geo)

    assert [u.link_name for u in units][:2] == ["link_001", "link_002"]
    assert units[-1].link_name == f"link_{geo.n_units:03d}"
    assert [u.element_id for u in units] == list(range(1, geo.n_units + 1))

    radii = [u.outer_radius_m for u in units]
    assert radii[0] == max(radii), "link_001 is not the wide base"
    assert radii[-1] == min(radii), "the last link is not the narrow tip"
    assert all(radii[i] > radii[i + 1] for i in range(len(radii) - 1))


@requires_cad
def test_partial_unit_is_flagged_at_the_base(stl_mod, params, csv_path):
    import pandas as pd
    geo = from_params(params)
    units = stl_mod.build_unit_inputs(pd.read_csv(csv_path), geo)
    assert units[0].is_partial is True
    assert not any(u.is_partial for u in units[1:])


@requires_cad
def test_mismatched_csv_is_rejected(stl_mod, params, csv_path):
    """A stale CSV against whole_units params used to pass silently."""
    import pandas as pd
    geo_wu = from_params(dict(params, terminal_unit_policy="whole_units"))
    with pytest.raises(ValueError, match="disagrees with the canonical model"):
        stl_mod.build_unit_inputs(pd.read_csv(csv_path), geo_wu)


@requires_cad
def test_wrong_unit_count_is_rejected(stl_mod, params, csv_path):
    import pandas as pd
    geo = from_params(params)
    df = pd.read_csv(csv_path).iloc[:-1]
    with pytest.raises(ValueError, match="Regenerate the CSV"):
        stl_mod.build_unit_inputs(df, geo)


# ── Both policies ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("policy,expect_partial", [
    ("exact_requested_length", 1),
    ("whole_units", 0),
])
@requires_cad
def test_both_policies_yield_21_units(stl_mod, params, tmp_path,
                                      policy, expect_partial):
    import helper_functions as hf
    import pandas as pd
    geo = from_params(dict(params, terminal_unit_policy=policy))
    csv_file = tmp_path / f"{policy}.csv"
    hf.generate_cable_sites_csv_zrot_from_P(
        geo.inverted_quads(), n_cables=params["n_cables"],
        csv_path=str(csv_file), radial_scale=1.0)

    units = stl_mod.build_unit_inputs(pd.read_csv(str(csv_file)), geo)
    assert len(units) == 21
    assert sum(u.is_partial for u in units) == expect_partial


@requires_cad
def test_whole_units_never_creates_a_22nd_unit(stl_mod, params, tmp_path):
    import helper_functions as hf
    import pandas as pd
    geo = from_params(dict(params, terminal_unit_policy="whole_units"))
    assert geo.n_units == 21

    csv_file = tmp_path / "wu.csv"
    hf.generate_cable_sites_csv_zrot_from_P(
        geo.inverted_quads(), n_cables=params["n_cables"],
        csv_path=str(csv_file), radial_scale=1.0)
    units = stl_mod.build_unit_inputs(pd.read_csv(str(csv_file)), geo)

    assert len(units) == 21
    assert all(u.link_name != "link_022" for u in units)
    assert max(u.element_id for u in units) == 21


@requires_cad
def test_whole_units_base_unit_is_complete_and_tallest(stl_mod, params, tmp_path):
    import helper_functions as hf
    import pandas as pd
    heights = {}
    for policy in ("exact_requested_length", "whole_units"):
        geo = from_params(dict(params, terminal_unit_policy=policy))
        f = tmp_path / f"{policy}_h.csv"
        hf.generate_cable_sites_csv_zrot_from_P(
            geo.inverted_quads(), n_cables=params["n_cables"],
            csv_path=str(f), radial_scale=1.0)
        u = stl_mod.build_unit_inputs(pd.read_csv(str(f)), geo)
        heights[policy] = [x.height_z_m for x in u]

    assert heights["exact_requested_length"][0] < heights["exact_requested_length"][1]
    assert heights["whole_units"][0] > heights["whole_units"][1]


# ── Output equivalence ──────────────────────────────────────────────────────

@pytest.mark.parametrize("policy", ["exact_requested_length", "whole_units"])
@requires_cad
def test_stl_output_matches_the_legacy_path(stl_mod, params, tmp_path, policy):
    """Meshes built through the canonical path must match the legacy path.

    The legacy path is reconstructed verbatim from the same CSV using the
    pre-migration argument derivation, and compared byte-for-byte. Run under
    both terminal-unit policies, because whole_units shifts every absolute
    coordinate and so exercises a different set of inputs entirely.
    """
    import cadquery as cq
    import helper_functions as hf
    import pandas as pd

    geo = from_params(dict(params, terminal_unit_policy=policy))
    csv_file = tmp_path / f"{policy}_eq.csv"
    hf.generate_cable_sites_csv_zrot_from_P(
        geo.inverted_quads(), n_cables=params["n_cables"],
        csv_path=str(csv_file), radial_scale=1.0)

    new_dir = tmp_path / f"{policy}_new"
    old_dir = tmp_path / f"{policy}_old"
    nlobe_t = float(params.get("nlobe_t", 0.5))
    notch = float(params.get("notch_factor", 0.25))

    stl_mod.process_csv(csv_file=str(csv_file), outdir=str(new_dir),
                        geometry=geo, nlobe_t=nlobe_t, notch_factor=notch)

    # Legacy derivation, verbatim.
    os.makedirs(str(old_dir), exist_ok=True)
    df = pd.read_csv(str(csv_file))
    draft = float(params["phi_deg"]) / 2.0
    for _, row in df.iterrows():
        pts = stl_mod.extract_points(row)
        solid = stl_mod.revolve_profile(
            stl_mod.make_profile_from_points(pts), 360, "y")
        solid = solid.translate((-float(row["joint_s1_x"]),
                                 -float(row["joint_s1_y"]),
                                 -float(row["joint_s1_z"])))
        solid = stl_mod.add_nlobe_cut(
            solid, int(params["n_cables"]), abs(float(row["c0_s1_x"])),
            abs(float(row["joint_s2_z"]) - float(row["joint_s1_z"])), draft,
            nlobe_t=nlobe_t, notch_factor=notch)
        cq.exporters.export(
            solid, os.path.join(str(old_dir), f"link_{int(row['elem']):03d}.stl"),
            tolerance=1e-4)

    produced = sorted(os.listdir(str(new_dir)))
    assert produced == sorted(os.listdir(str(old_dir)))
    assert len(produced) == 21
    for name in produced:
        with open(new_dir / name, "rb") as f:
            got = f.read()
        with open(old_dir / name, "rb") as f:
            want = f.read()
        assert got == want, f"{policy}/{name} differs from the pre-migration output"


@requires_cad
def test_filenames_follow_the_link_naming_convention(stl_mod, params,
                                                     csv_path, tmp_path):
    out = tmp_path / "m"
    stl_mod.process_csv(csv_file=csv_path, outdir=str(out),
                        geometry=from_params(params))
    names = sorted(os.listdir(str(out)))
    assert names == [f"link_{i:03d}.stl" for i in range(1, 22)]


@requires_cad
def test_meshes_are_manifold_sized_and_ordered(stl_mod, params,
                                               csv_path, tmp_path):
    """Topology sanity plus base-to-tip size ordering of the actual solids."""
    out = tmp_path / "m2"
    stl_mod.process_csv(csv_file=csv_path, outdir=str(out),
                        geometry=from_params(params),
                        nlobe_t=float(params.get("nlobe_t", 0.5)),
                        notch_factor=float(params.get("notch_factor", 0.25)))
    extents = []
    for i in range(1, 22):
        path = str(out / f"link_{i:03d}.stl")
        assert _stl_triangle_count(path) > 0
        xs = _stl_vertices(path)[0::3]
        extents.append(max(xs) - min(xs))
    assert extents[0] == max(extents), "link_001 is not the widest solid"
    assert extents[-1] == min(extents), "link_021 is not the narrowest solid"


# ── CLI / build.py compatibility ────────────────────────────────────────────

@requires_cad
def test_cli_still_accepts_the_documented_flags(params, csv_path, tmp_path):
    """build.py invokes: --in <csv> --params <json> [--plain]."""
    pfile = tmp_path / "p.json"
    with io.open(str(pfile), "w", encoding="utf-8") as f:
        json.dump(params, f)
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "csv2geom_nlobe.py",
         "--in", csv_path, "--params", str(pfile),
         "--outdir", str(tmp_path / "cli")],
        cwd=_ROOT, capture_output=True, text=True, encoding="utf-8",
        env=child_env)
    assert proc.returncode == 0, proc.stderr
    assert "terminal_unit_policy = exact_requested_length" in proc.stdout
    assert len(os.listdir(str(tmp_path / "cli"))) == 21


def test_build_py_invocation_is_unchanged():
    """The STL step's command line must not have shifted."""
    src = _read_text(os.path.join(_ROOT, "build.py"))
    assert "python csv2geom_nlobe.py" in src
    assert "--in Geom_Data_CSV/Spirob_geom_data.csv" in src
    assert "--params {args.params}" in src


def test_f08_tendon_rule_untouched():
    """This phase must not alter tendon routing."""
    src = _read_text(os.path.join(_ROOT, "csv2xml.py"))
    assert "math.tan(HALF_PHI)" in src
    stl_src = _read_text(os.path.join(_ROOT, "csv2geom_nlobe.py"))
    assert "tendon" not in stl_src.lower().replace("tendon_inward_shift", "")
