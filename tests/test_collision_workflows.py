"""Check collision fit, physical contacts, invariant inertia, and shared assets."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from csv2xml import MJCFConfig, write_mjcf_from_sites_csv
from spirob.collision import flat_outline, rounded_flat_primitives
from spirob.geometry import from_params
from spirob.mesh_assets import mesh_assets
from spirob.sections import resolve_flat_thickness_ratio

ROOT = Path(__file__).resolve().parents[1]


def body_tensor(body):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, body.iquat)
    rotation = rotation.reshape(3, 3)
    return rotation @ np.diag(body.inertia) @ rotation.T


def primitive_outline(shapes):
    parts = []
    for shape in shapes:
        if shape['type'] == 'cylinder':
            p = np.fromstring(shape['fromto'], sep=' ')
            parts.append(Point(p[[0, 2]]).buffer(float(shape['size']), quad_segs=128))
        else:
            p = np.fromstring(shape['pos'], sep=' ')
            s = np.fromstring(shape['size'], sep=' ')
            q = np.fromstring(shape.get('quat', '1 0 0 0'), sep=' ')
            angle = -2*np.arctan2(q[2], q[0])
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            vertices = np.array([[-s[0], -s[2]], [s[0], -s[2]], [s[0], s[2]], [-s[0], s[2]]])
            parts.append(Polygon(vertices @ rot.T + p[[0, 2]]))
    return unary_union(parts)


@pytest.fixture(scope='module', params=[2, 3])
def models(request, tmp_path_factory):
    folder = tmp_path_factory.mktemp(f'contact-{request.param}')
    params = json.loads((ROOT/f'examples/params-{["", "", "two", "three"][request.param]}-cable.json').read_text())
    pfile = folder/'params.json'
    pfile.write_text(json.dumps(params))
    result = subprocess.run([sys.executable, str(ROOT/'build.py'), '--params', str(pfile),
                             '--no-preview', '--output-dir', str(folder)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    geo = from_params(params)
    paths = {'mesh': folder/'spirob_physics_model.xml'}
    for mode in ['capsule'] + (['compound'] if request.param == 2 else []):
        cfg = MJCFConfig(physics_mode=mode, joint_type='hinge' if request.param == 2 else 'ball',
                         mesh_layout='shared', phi_deg=params['phi_deg'], post_gen=params['post_gen'],
                         tendon_inward_shift=params['tendon_inward_shift'],
                         flat_thickness_ratio=resolve_flat_thickness_ratio(params) if params['n_cables']==2 else .3)
        paths[mode] = folder/f'{mode}.xml'
        write_mjcf_from_sites_csv(str(folder/'Geom_Data_CSV/Spirob_geom_data.csv'), str(paths[mode]),
                                 str(folder/'meshes'), config=cfg, geometry=geo)
    return params, geo, paths


def test_shared_sources_and_geometric_scaling(models):
    _, geo, paths = models
    assert {p.name for p in (paths['mesh'].parent/'meshes').glob('*.stl')} == {'link_001.stl', 'link_template.stl'}
    root = ET.parse(paths['mesh']).getroot()
    meshes = root.findall('./asset/mesh')
    assert len(meshes) == geo.n_units
    assert len({m.get('file') for m in meshes}) == 2
    model = mujoco.MjModel.from_xml_path(str(paths['mesh']))
    ref = model.body('link_002')
    for asset in mesh_assets(geo):
        if asset.filename != 'link_template.stl':
            continue
        body = model.body(asset.link_name)
        np.testing.assert_allclose(body.mass, ref.mass*asset.scale**3, rtol=2e-6)
        np.testing.assert_allclose(body.ipos, ref.ipos*asset.scale, rtol=2e-6, atol=1e-10)
        # Compare the tensor in body axes; principal eigenvalue order may swap.
        np.testing.assert_allclose(body_tensor(body), body_tensor(ref)*asset.scale**5, rtol=2e-6, atol=1e-14)


def test_contact_shape_changes_preserve_inertia_and_interfaces(models):
    _, _, paths = models
    reference = mujoco.MjModel.from_xml_path(str(paths['mesh']))
    refdata = mujoco.MjData(reference)
    mujoco.mj_forward(reference, refdata)
    for path in paths.values():
        root = ET.parse(path).getroot()
        assert root.find('compiler').get('inertiafromgeom') == 'auto'
        assert len(root.findall('.//inertial')) == reference.nbody-1
        model = mujoco.MjModel.from_xml_path(str(path))
        for name in ['body_mass', 'body_ipos', 'body_iquat', 'body_inertia', 'body_pos', 'body_quat',
                     'jnt_pos', 'jnt_axis', 'jnt_stiffness', 'dof_damping', 'site_pos',
                     'actuator_ctrlrange', 'actuator_gear', 'tendon_adr', 'wrap_objid']:
            np.testing.assert_allclose(getattr(model, name), getattr(reference, name), rtol=1e-13, atol=1e-16)
        for obj in ['body', 'joint', 'site', 'tendon', 'actuator']:
            n = getattr(reference, {'body':'nbody', 'joint':'njnt', 'site':'nsite', 'tendon':'ntendon', 'actuator':'nu'}[obj])
            assert [getattr(model, obj)(i).name for i in range(n)] == [getattr(reference, obj)(i).name for i in range(n)]
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.ten_length, refdata.ten_length, rtol=1e-13)
        for geom in root.findall('.//geom'):
            if geom.get('name', '').startswith('collision_'):
                assert geom.get('mass') == '0' and geom.get('density') == '0'
                # Even an accidental later mass edit must not affect the body.
                geom.set('mass', '123')
                geom.set('density', '456')
        mutated = path.with_name(path.stem+'-edited.xml')
        ET.ElementTree(root).write(mutated)
        edited = mujoco.MjModel.from_xml_path(str(mutated))
        np.testing.assert_array_equal(edited.body_mass, model.body_mass)
        np.testing.assert_array_equal(edited.body_inertia, model.body_inertia)
        # Users must still be able to add a normal, geom-inferred task object.
        task = ET.SubElement(root.find('worldbody'), 'body', name='task_object', pos='1 0 1')
        ET.SubElement(task, 'freejoint')
        ET.SubElement(task, 'geom', type='box', size='.01 .01 .01', mass='.05')
        ET.ElementTree(root).write(mutated)
        with_object = mujoco.MjModel.from_xml_path(str(mutated))
        np.testing.assert_array_equal(with_object.body_mass[:model.nbody], model.body_mass)
        assert with_object.body('task_object').mass[0] == pytest.approx(.05)


@pytest.mark.parametrize('policy', ['whole_units', 'exact_requested_length'])
@pytest.mark.parametrize('radius_ratio', [0.02, 0.04, 0.08])
def test_compound_shape_fits_every_flat_link(policy, radius_ratio):
    params = json.loads((ROOT/'examples/params-two-cable.json').read_text())
    params['terminal_unit_policy'] = policy
    geo = from_params(params)
    for i in range(geo.n_units):
        cad = Polygon(flat_outline(geo, i))
        shapes = rounded_flat_primitives(geo, i, resolve_flat_thickness_ratio(params), radius_ratio)
        union = primitive_outline(shapes)
        assert union.geom_type == 'Polygon' and len(union.interiors) == 0
        assert union.difference(cad).area < cad.area*1e-12
        assert union.area/cad.area > 0.99
        radius = geo.units[i].realized_width_m/2 * radius_ratio
        assert cad.hausdorff_distance(union) < 0.5*radius
        expected_half_y = geo.units[i].realized_width_m/2 * resolve_flat_thickness_ratio(params)
        for shape in shapes:
            if shape['type'] == 'cylinder':
                p = np.fromstring(shape['fromto'], sep=' ')
                np.testing.assert_allclose(p[[1, 4]], [-expected_half_y, expected_half_y])
            else:
                assert np.fromstring(shape['size'], sep=' ')[1] == pytest.approx(expected_half_y)


@pytest.mark.parametrize('index', [0, 1, 10, 20])
@pytest.mark.parametrize('surface', ['face', 'corner'])
@pytest.mark.parametrize('probe_type', ['sphere', 'box'])
def test_native_contacts_on_flat_faces_and_rounded_corners(models, index, surface, probe_type):
    params, geo, paths = models
    if params['n_cables'] != 2:
        pytest.skip('Flat compound collider is explicitly limited to two cables')
    source = ET.parse(paths['compound']).getroot()
    # Isolate one link to exercise external contact without adjacent links/floor.
    root = ET.Element('mujoco')
    for tag in ['compiler', 'default', 'asset']:
        root.append(copy.deepcopy(source.find(tag)))
    world = ET.SubElement(root, 'worldbody')
    body = ET.SubElement(world, 'body', name='test_link')
    ET.SubElement(body, 'freejoint')
    original = source.find(f'.//body[@name="link_{index+1:03d}"]')
    for element in original:
        if element.tag in ['inertial', 'geom']:
            body.append(copy.deepcopy(element))
    p = flat_outline(geo, index)
    shapes = rounded_flat_primitives(geo, index, resolve_flat_thickness_ratio(params))
    radius = float(shapes[1]['size'])
    center = np.fromstring(shapes[1]['fromto'], sep=' ')[[0, 2]]
    if surface == 'corner':
        normal = p[0]-center
        normal /= np.linalg.norm(normal)
        point = center+radius*normal
        expected = mujoco.mjtGeom.mjGEOM_CYLINDER
    else:
        delta = p[1]-p[0]
        normal = np.array([delta[1], -delta[0]])/np.linalg.norm(delta)
        point = np.fromstring(shapes[0]['pos'], sep=' ')[[0, 2]] + radius*normal
        expected = mujoco.mjtGeom.mjGEOM_BOX
    probe_radius = radius/2
    probe = ET.SubElement(world, 'geom', name='probe', type=probe_type,
                          size=' '.join([str(probe_radius)]*(3 if probe_type == 'box' else 1)), margin='0')
    testpath = paths['compound'].with_name('contact-probe.xml')
    for distance, touching in [(probe_radius-0.05*radius, True), (3*probe_radius, False)]:
        c = point+distance*normal
        probe.set('pos', f'{c[0]} 0 {c[1]}')
        ET.ElementTree(root).write(testpath)
        model = mujoco.MjModel.from_xml_path(str(testpath))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        probe_id = model.geom('probe').id
        contacts = [c for c in data.contact if probe_id in c.geom]
        assert bool(contacts) == touching
        if touching:
            assert any(model.geom_type[c.geom[0] if c.geom[1] == probe_id else c.geom[1]] == expected for c in contacts)
            assert all(model.geom_type[c.geom[0] if c.geom[1] == probe_id else c.geom[1]] != mujoco.mjtGeom.mjGEOM_MESH for c in contacts)


def test_array_preserves_shared_meshes_and_body_inertias(models):
    from tools.multi_array import build_array
    _, _, paths = models
    source = paths.get('compound', paths['capsule'])
    output = source.with_name('array.xml')
    build_array(str(source), count=2, radius_m=0.12, out_path=str(output))
    model = mujoco.MjModel.from_xml_path(str(source))
    array = mujoco.MjModel.from_xml_path(str(output))
    assert array.nmesh == model.nmesh
    assert array.nu == 2*model.nu
    assert sum(array.body_mass) == pytest.approx(2*sum(model.body_mass), rel=1e-13)


def test_reject_unsupported_compound_shapes_and_radius():
    p = json.loads((ROOT/'examples/params-three-cable.json').read_text())
    with pytest.raises(ValueError, match='two-cable'):
        rounded_flat_primitives(from_params(p), 0, 0.3)
    p['n_cables'] = 2
    with pytest.raises(ValueError, match='radius ratio'):
        rounded_flat_primitives(from_params(p), 0, 0.3, float('nan'))


def test_shared_whole_units_need_only_one_stl():
    p = json.loads((ROOT/'examples/params-two-cable.json').read_text())
    p['terminal_unit_policy'] = 'whole_units'
    assets = mesh_assets(from_params(p))
    assert {a.filename for a in assets} == {'link_template.stl'}
