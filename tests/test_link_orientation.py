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
                             '--no-preview', '--mesh-layout', 'individual', '--output-dir', str(folder/'out')],
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
    assert report['links_checked'] == 21 and report['authored_rest_frames_aligned']
    assert not report['geom_axes_aligned_with_bodies']
    assert not report['rest_frames_aligned']
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
        assert not frame_report(model, data)['authored_rest_frames_aligned']
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


def _world_mesh(model, data, gid, normals=False):
    mid = model.geom_dataid[gid]
    start = model.mesh_normaladr[mid] if normals else model.mesh_vertadr[mid]
    count = model.mesh_normalnum[mid] if normals else model.mesh_vertnum[mid]
    values = model.mesh_normal if normals else model.mesh_vert
    world = values[start:start + count].astype(float) @ data.geom_xmat[gid].reshape(3, 3).T
    return world if normals else world + data.geom_xpos[gid]


def test_actual_geom_alignment_preserves_shape_physics_and_binary_reload(built_model, tmp_path):
    import mujoco as mj
    from spirob.mujoco_frames import aligned_geom_model
    from tools.inspect_model import frame_report
    _, original, _ = built_model
    original_vertices = original.mesh_vert.copy()
    aligned = aligned_geom_model(original)
    np.testing.assert_array_equal(original.mesh_vert, original_vertices)
    # Check all compiled body/joint/site/cable fields, not only total mass.
    for name in dir(original):
        if name.startswith(('body_', 'jnt_', 'dof_', 'site_', 'tendon_', 'wrap_', 'actuator_')):
            value = getattr(original, name)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, getattr(aligned, name), err_msg=name)
    np.testing.assert_array_equal(original.geom_rbound, aligned.geom_rbound)
    binary = tmp_path/'aligned.mjb'
    mj.mj_saveModel(aligned, str(binary), None)
    aligned = mj.MjModel.from_binary_path(str(binary))
    da, db = mj.MjData(original), mj.MjData(aligned)
    mj.mj_forward(original, da); mj.mj_forward(aligned, db)
    assert frame_report(aligned, db)['rest_frames_aligned']
    twice = aligned_geom_model(aligned)
    np.testing.assert_array_equal(twice.mesh_vert, aligned.mesh_vert)
    np.testing.assert_array_equal(twice.bvh_aabb, aligned.bvh_aabb)
    for bent in (False, True):
        if bent:
            dq = np.random.default_rng(71).normal(0, .12, original.nv)
            mj.mj_integratePos(original, da.qpos, dq, 1)
            db.qpos[:] = da.qpos
        mj.mj_forward(original, da); mj.mj_forward(aligned, db)
        for gid in np.flatnonzero(original.geom_type == mj.mjtGeom.mjGEOM_MESH):
            bid = original.geom_bodyid[gid]
            np.testing.assert_allclose(db.geom_xmat[gid], db.xmat[bid], atol=1e-12, rtol=0)
            np.testing.assert_allclose(_world_mesh(original, da, gid),
                                       _world_mesh(aligned, db, gid), atol=5e-9, rtol=0)
            np.testing.assert_allclose(_world_mesh(original, da, gid, True),
                                       _world_mesh(aligned, db, gid, True), atol=2e-7, rtol=0)
            # A ray aimed through each geom exercises the mesh BVH and geom_size.
            center = da.geom_xpos[gid]
            # Oblique rays avoid the 3.3.5 slab test's zero-direction/zero-width
            # boundary degeneracy on exactly axis-parallel flat mesh faces.
            directions = np.array([[1., .23, .31], [.17, 1., .29], [.19, .37, 1.]])
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            for axis in directions:
                start, direction = center + .1*axis, -axis
                ra = mj.mj_rayMesh(original, da, gid, start, direction)
                rb = mj.mj_rayMesh(aligned, db, gid, start, direction)
                assert ra > 0 and rb > 0
                assert rb == pytest.approx(ra, abs=5e-9)
                # The scene ray also traverses body BVHs.
                ids = [np.empty(1, dtype=np.int32), np.empty(1, dtype=np.int32)]
                ra = mj.mj_ray(original, da, start, direction, None, 1, -1, ids[0])
                rb = mj.mj_ray(aligned, db, start, direction, None, 1, -1, ids[1])
                assert rb == pytest.approx(ra, abs=5e-9)
                np.testing.assert_array_equal(ids[0], ids[1])
        np.testing.assert_allclose(da.ten_length, db.ten_length, atol=1e-14, rtol=0)
    # Actual nonzero tendon actuation, using the generator's negative control range.
    mj.mj_resetData(original, da); mj.mj_resetData(aligned, db)
    da.ctrl[:] = db.ctrl[:] = np.linspace(-.05, -.2, original.nu)
    for _ in range(200):
        mj.mj_step(original, da); mj.mj_step(aligned, db)
    assert np.max(np.abs(da.qpos - original.qpos0)) > 1e-6
    np.testing.assert_allclose(da.qpos, db.qpos, atol=1e-8, rtol=0)
    np.testing.assert_allclose(da.qvel, db.qvel, atol=1e-7, rtol=0)


def test_alignment_preserves_contacts_with_external_objects(built_model):
    import mujoco as mj
    import xml.etree.ElementTree as ET
    from spirob.mujoco_frames import aligned_geom_model
    output, base, data = built_model
    tree = ET.parse(output/'spirob_physics_model.xml')
    for mesh in tree.findall('./asset/mesh'):
        mesh.set('file', str(output/mesh.get('file')))
    world = tree.find('./worldbody')
    # Touch each link at its extreme +X vertex; include both curved and box
    # obstacles so contact tests cannot pass simply because ncon == 0.
    for gid in np.flatnonzero(base.geom_type == mj.mjtGeom.mjGEOM_MESH):
        vertices = _world_mesh(base, data, gid)
        pos = vertices[np.argmax(vertices[:, 0])] + [.0005, 0, 0]
        kind = 'sphere' if gid % 2 else 'box'
        size = '.001' if kind == 'sphere' else '.001 .001 .001'
        ET.SubElement(world, 'geom', type=kind, size=size, pos=' '.join(map(str, pos)),
                      name=f'obstacle_{gid}')
    original = mj.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding='unicode'))
    aligned = aligned_geom_model(original)
    da, db = mj.MjData(original), mj.MjData(aligned)
    mj.mj_forward(original, da); mj.mj_forward(aligned, db)
    assert da.ncon >= base.nmesh
    assert db.ncon == da.ncon
    for a, b in zip(da.contact, db.contact):
        np.testing.assert_array_equal(a.geom, b.geom)
        assert a.dist == pytest.approx(b.dist, abs=5e-9)
        np.testing.assert_allclose(a.pos, b.pos, atol=5e-9, rtol=0)
        np.testing.assert_allclose(a.frame, b.frame, atol=1e-6, rtol=0)
    # Check a short trajectory with contacts, as well as static contact geometry.
    for _ in range(20):
        mj.mj_step(original, da); mj.mj_step(aligned, db)
    np.testing.assert_allclose(da.qpos, db.qpos, atol=1e-7, rtol=0)
    np.testing.assert_allclose(da.qvel, db.qvel, atol=1e-6, rtol=0)


def test_alignment_rejects_noncanonical_and_fluid_geoms_without_mutation(built_model):
    import copy
    from spirob.mujoco_frames import aligned_geom_model
    _, original, _ = built_model
    model = copy.copy(original)
    gid = model.body_geomadr[2]
    model.geom_pos[gid, 0] += .01
    vertices = model.mesh_vert.copy()
    with pytest.raises(ValueError, match='not canonical'):
        aligned_geom_model(model)
    np.testing.assert_array_equal(model.mesh_vert, vertices)
    model.geom_pos[:] = original.geom_pos
    model.geom_fluid[gid, 0] = 1
    with pytest.raises(ValueError, match='fluid'):
        aligned_geom_model(model)
    model.geom_fluid[:] = original.geom_fluid
    model.geom_dataid[model.body_geomadr[3]] = model.geom_dataid[gid]
    with pytest.raises(ValueError, match='shared meshes'):
        aligned_geom_model(model)
