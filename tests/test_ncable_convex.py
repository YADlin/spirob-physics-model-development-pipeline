"""N-cable hull fitting, dynamics/interface compatibility and real contacts."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import trimesh

from csv2xml import MJCFConfig, write_mjcf_from_sites_csv
from spirob.geometry import from_params
from tools.inspect_collision import collision_report
from tools.inspect_collision_surface import collider_surface, inspection_mesh
from tools.inspect_section import compiled_surface
from tools.multi_array import build_array

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {3: 'three', 4: 'four', 6: 'six'}


@pytest.fixture(scope='module', params=[3, 4, 6])
def ncable(request, tmp_path_factory):
    n = request.param
    folder = tmp_path_factory.mktemp(f'convex-{n}-cable')
    result = subprocess.run([sys.executable, str(ROOT/'build.py'), '--params',
        str(ROOT/f'examples/params-{EXAMPLES[n]}-cable.json'), '--collision-mode', 'convex',
        '--timestep', '.0001', '--no-preview', '--output-dir', str(folder)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    params = json.loads((folder/'build_params.json').read_text())
    cfg = MJCFConfig(physics_mode='mesh', mesh_layout='shared', joint_type='ball',
        timestep=.0001, phi_deg=params['phi_deg'], post_gen=params['post_gen'],
        tendon_inward_shift=params['tendon_inward_shift'])
    write_mjcf_from_sites_csv(str(folder/'Geom_Data_CSV/Spirob_geom_data.csv'),
        str(folder/'mesh-reference.xml'), str(folder/'meshes'), config=cfg, geometry=from_params(params))
    return folder, n


def test_ncable_inertia_joints_cables_names_and_assets(ncable):
    folder, n = ncable
    path = folder/'spirob_physics_model.xml'
    model = mujoco.MjModel.from_xml_path(str(path))
    reference = mujoco.MjModel.from_xml_path(str(folder/'mesh-reference.xml'))
    report = collision_report(path, model)
    assert report['links'] == report['total_link_contact_geoms'] == 21
    assert set(report['contact_geoms_per_link'].values()) == {1}
    assert report['all_proxies_zero_mass'] and report['explicit_inertias'] == 21
    assert report['nativeccd'] and report['multiccd']
    assert report['target_site_retained'] and not report['target_marker_visible']
    assert model.nu == model.ntendon == n
    assert np.all(model.jnt_type == mujoco.mjtJoint.mjJNT_BALL)
    assert model.nv == 3*21 and model.nq == 4*21
    assert model.opt.timestep == .0001 and np.all(model.geom_margin == 0)
    assert len(report['source_stl_files']) == len(list((folder/'meshes').glob('*.stl'))) == 2
    for attr in ('body_mass', 'body_ipos', 'body_inertia', 'body_iquat', 'body_pos', 'body_quat',
                 'jnt_stiffness', 'dof_damping', 'site_pos', 'actuator_ctrlrange', 'actuator_gear',
                 'tendon_adr', 'tendon_num', 'wrap_objid'):
        np.testing.assert_allclose(getattr(model, attr), getattr(reference, attr), rtol=1e-13, atol=1e-16)
    for obj, count in [('body', model.nbody), ('joint', model.njnt), ('site', model.nsite),
                       ('tendon', model.ntendon), ('actuator', model.nu)]:
        assert [getattr(model, obj)(i).name for i in range(count)] == [getattr(reference, obj)(i).name for i in range(count)]
    data, refdata = mujoco.MjData(model), mujoco.MjData(reference)
    mujoco.mj_forward(model, data); mujoco.mj_forward(reference, refdata)
    np.testing.assert_allclose(data.ten_length, refdata.ten_length, rtol=1e-13)
    # Disturb every rotational DOF, rather than comparing paths only at rest.
    mujoco.mj_integratePos(model, data.qpos, np.sin(np.arange(model.nv))*.04, 1.)
    refdata.qpos[:] = data.qpos
    mujoco.mj_forward(model, data); mujoco.mj_forward(reference, refdata)
    np.testing.assert_allclose(data.ten_length, refdata.ten_length, rtol=1e-13)


def test_ncable_hull_is_cad_envelope_in_body_axes(ncable):
    from scipy.spatial import ConvexHull
    folder, _ = ncable
    model = mujoco.MjModel.from_xml_path(str(folder/'spirob_physics_model.xml'))
    directions = np.random.default_rng(123).normal(size=(256, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    for i in range(1, model.nbody):
        name = model.body(i).name
        v, f = compiled_surface(model, name); cv, _ = collider_surface(model, name)
        surface, removed = inspection_mesh(v, f)
        assert surface.is_volume
        raw = trimesh.Trimesh(vertices=v, faces=f, process=True)
        assert removed == np.count_nonzero(raw.area_faces == 0)
        assert surface.volume == pytest.approx(raw.volume, rel=1e-13)
        eq = ConvexHull(cv).equations
        assert np.max(v @ eq[:, :3].T+eq[:, 3]) < 1e-5  # mm, compiled float vertices
        # A hull must share the CAD's support function in every direction.
        np.testing.assert_allclose(np.max(v @ directions.T, axis=0),
                                   np.max(cv @ directions.T, axis=0), rtol=0, atol=1e-5)


def isolated_sphere_probe(path, name):
    root = ET.Element('mujoco'); source = ET.parse(path).getroot()
    for tag in ('compiler', 'default', 'option', 'asset'):
        root.append(copy.deepcopy(source.find(tag)))
    for asset in root.findall('./asset/mesh'):
        if asset.get('file'):
            asset.set('file', str((path.parent/asset.get('file')).resolve()))
    world = ET.SubElement(root, 'worldbody')
    body = ET.SubElement(world, 'body', name='probe_link')
    ET.SubElement(body, 'freejoint')
    for element in source.find(f'.//body[@name="{name}"]'):
        if element.tag in ('geom', 'inertial'):
            body.append(copy.deepcopy(element))
    probe = ET.SubElement(world, 'body', name='probe', mocap='true')
    ET.SubElement(probe, 'geom', name='probe_sphere', type='sphere', size='.0002', margin='0')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def test_ncable_contacts_around_lobes_caps_and_separates(ncable):
    folder, n = ncable; path = folder/'spirob_physics_model.xml'
    original = mujoco.MjModel.from_xml_path(str(path))
    for name in ('link_001', 'link_002', 'link_021'):
        v, f = collider_surface(original, name)
        surface = trimesh.Trimesh(vertices=v/1000, faces=f, process=True)
        # Probe supports across all azimuth sectors, plus both end caps.
        angles = np.arange(2*n)*np.pi/n
        directions = np.vstack((np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles))),
                                [[0, 0, 1], [0, 0, -1]]))
        faces = np.argmax(surface.face_normals @ directions.T, axis=0)
        model = isolated_sphere_probe(path, name)
        for face in np.unique(faces):
            normal = surface.face_normals[face]; point = surface.triangles_center[face]
            for distance, touching in ((.00018, True), (.0005, False)):
                data = mujoco.MjData(model); data.mocap_pos[0] = point+normal*distance
                mujoco.mj_forward(model, data)
                assert bool(data.ncon) == touching, (n, name, face, distance)
                assert not np.any(data.warning.number)


def test_ncable_arrays_keep_shared_hulls_and_inertias(ncable):
    folder, n = ncable; source = folder/'spirob_physics_model.xml'
    output = folder/'array.xml'
    build_array(str(source), 2, radius_m=.12, out_path=str(output))
    single = mujoco.MjModel.from_xml_path(str(source)); array = mujoco.MjModel.from_xml_path(str(output))
    assert array.nmesh == single.nmesh and array.nu == 2*n
    assert sum(array.body_mass) == pytest.approx(2*sum(single.body_mass), rel=1e-13)


def test_target_marker_opt_in_preserves_physics_and_site(ncable):
    folder, _ = ncable
    result = subprocess.run([sys.executable, str(ROOT/'csv2xml.py'), '--in',
        str(folder/'Geom_Data_CSV/Spirob_geom_data.csv'), '--meshdir', str(folder/'meshes'),
        '--params', str(folder/'build_params.json'), '--collision-mode', 'convex',
        '--phi-deg', '6.3', '--show-target-marker', '--out', str(folder/'visible.xml')],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    hidden = mujoco.MjModel.from_xml_path(str(folder/'spirob_physics_model.xml'))
    shown = mujoco.MjModel.from_xml_path(str(folder/'visible.xml'))
    target = hidden.site('target').id
    assert target == shown.site('target').id and hidden.site_rgba[target, 3] == 0
    assert shown.site_rgba[target, 3] == 1
    for attr in ('body_mass', 'body_ipos', 'body_inertia', 'jnt_stiffness', 'dof_damping', 'site_pos'):
        np.testing.assert_array_equal(getattr(hidden, attr), getattr(shown, attr))


def test_build_forwards_target_marker_switch(tmp_path):
    params = json.loads((ROOT/'examples/params-three-cable.json').read_text())
    params['L'] = .012
    pfile = tmp_path/'short.json'; pfile.write_text(json.dumps(params))
    result = subprocess.run([sys.executable, str(ROOT/'build.py'), '--params', str(pfile),
        '--collision-mode', 'convex', '--no-preview', '--show-target-marker',
        '--output-dir', str(tmp_path/'build')], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    model = mujoco.MjModel.from_xml_path(str(tmp_path/'build/spirob_physics_model.xml'))
    assert model.site_rgba[model.site('target').id, 3] == 1
