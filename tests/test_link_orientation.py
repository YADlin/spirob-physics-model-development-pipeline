"""The partial cut belongs at the mount; mesh preprocessing must preserve pose."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def params():
    return json.loads((ROOT / 'params.json').read_text())


def test_partial_base_has_flat_mount_and_nominal_joint_face(params):
    from spirob.geometry import from_params
    g = from_params(params)
    base, neighbour = g.inverted_quads()[:2]
    assert base[3, 1] == base[0, 1]
    assert base[0, 1] < base[2, 1] < base[1, 1]
    def slope(q):
        return (q[1, 1] - q[2, 1]) / abs(q[2, 0] - q[1, 0])
    assert slope(base) == pytest.approx(slope(neighbour), abs=1e-12)
    assert base[1] == pytest.approx(neighbour[0], abs=1e-12)
    assert g.lengths.discrete_chord_length_m == pytest.approx(.2236554047704, abs=1e-12)
    assert g.units[0].realized_width_m == pytest.approx(.031087574, abs=1e-9)


def test_corrected_surface_is_consistent_across_all_poses(params):
    import helper_functions as hf
    from spirob.geometry import from_params
    g = from_params(params)
    straight = hf.straighten_pose(g.curled_quads())
    inverted = hf.Invert_pose(straight, params['L'])
    np.testing.assert_allclose(straight, g.straight_quads(), atol=1e-12, rtol=0)
    np.testing.assert_allclose(inverted, g.inverted_quads(), atol=1e-12, rtol=0)


@pytest.mark.parametrize('fraction', [.3, .75, .98])
def test_partial_base_correction_across_lengths(params, fraction):
    from spirob.geometry import from_params, continuous_arc_length
    g = from_params(params)
    params['L'] = continuous_arc_length((20 + fraction)*g.inputs.delta_theta_rad,
                                       g.spiral.a_m, g.spiral.b)
    test = from_params(params)
    base, neighbour = test.inverted_quads()[:2]
    assert base[3, 1] == base[0, 1]
    assert base[0, 1] < base[2, 1] < base[1, 1]
    assert test.n_units == 21


def test_too_short_partial_base_is_rejected_without_inverted_surface(params):
    from spirob.geometry import from_params, continuous_arc_length
    g = from_params(params)
    params['L'] = continuous_arc_length(20.01*g.inputs.delta_theta_rad,
                                       g.spiral.a_m, g.spiral.b)
    with pytest.raises(ValueError, match='Partial base is too short'):
        from_params(params)
    params['terminal_unit_policy'] = 'whole_units'
    assert not from_params(params).lengths.has_partial_unit


@pytest.fixture(scope='module', params=[2, 3, 4])
def built_model(request, tmp_path_factory):
    pytest.importorskip('cadquery')
    mj = pytest.importorskip('mujoco')
    folder = tmp_path_factory.mktemp(f'orientation-{request.param}')
    p = json.loads((ROOT/'params.json').read_text())
    p['n_cables'] = request.param
    params_path = folder/'params.json'
    params_path.write_text(json.dumps(p))
    result = subprocess.run([sys.executable, str(ROOT/'build.py'), '--params', str(params_path),
                             '--no-preview', '--output-dir', str(folder/'out')],
                            capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stdout + result.stderr
    model = mj.MjModel.from_xml_path(str(folder/'out'/'spirob_physics_model.xml'))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    return folder/'out', model, data


def test_compiled_meshes_preserve_authored_vertices_and_link_axes(built_model):
    mj = pytest.importorskip('mujoco')
    tm = pytest.importorskip('trimesh')
    from scipy.spatial import cKDTree
    from tools.inspect_model import frame_report
    output, model, data = built_model
    report = frame_report(model, data)
    assert report['links_checked'] == 21 and report['rest_frames_aligned']
    for body_id in range(1, model.nbody):
        gid = model.body_geomadr[body_id]
        mid = model.geom_dataid[gid]
        start = model.mesh_vertadr[mid]
        vertices = model.mesh_vert[start:start+model.mesh_vertnum[mid]].astype(float)
        world = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        local = (world-data.xpos[body_id]) @ data.xmat[body_id].reshape(3, 3)
        source = tm.load(output/'meshes'/f'{model.body(body_id).name}.stl', force='mesh')
        assert cKDTree(source.vertices).query(local)[0].max() < 5e-9
        assert cKDTree(local).query(source.vertices)[0].max() < 5e-9
    # Check the audit detects a real authoring error instead of blessing any frame.
    gid = model.body_geomadr[2]
    previous = model.geom_quat[gid].copy()
    model.geom_quat[gid] = [1, 0, 0, 0]
    try:
        assert not frame_report(model, data)['rest_frames_aligned']
    finally:
        model.geom_quat[gid] = previous


def test_generated_base_mesh_cut_is_on_mount_side(built_model):
    tm = pytest.importorskip('trimesh')
    output, model, data = built_model
    mesh = tm.load(output/'meshes'/'link_001.stl', force='mesh')
    height = model.body('link_002').pos[2]
    assert abs(mesh.bounds[0, 2]) < 1e-9
    assert mesh.bounds[1, 2] <= height + 1e-9
    mount = mesh.triangles[np.all(np.abs(mesh.triangles[:, :, 2]) < 1e-9, axis=1)]
    assert len(mount) >= 2
    assert np.ptp(mount[:, :, 0]) > .02
