"""External contacts, few-geom counts, fitted envelope and inertia regressions."""
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
from tools.benchmark_collision import benchmark, isolated_patch
from tools.inspect_collision import collision_report
from tools.inspect_collision_surface import collider_surface, max_surface_distance, surface_samples
from tools.inspect_section import compiled_surface
from tools.multi_array import build_array

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module', params=[('rectangular','stepped'), ('rectangular','linear'),
                                       ('hex','stepped'), ('hex','linear')])
def convex(request, tmp_path_factory):
    section, profile = request.param
    folder = tmp_path_factory.mktemp(f'convex-{section}-{profile}')
    args = [sys.executable, str(ROOT/'build.py'), '--params', str(ROOT/'examples/params-two-cable.json'),
            '--thickness-profile', profile, '--base-thickness-mm', 'auto', '--timestep', '.0001',
            '--collision-mode', 'convex', '--no-preview', '--output-dir', str(folder)]
    if section == 'hex': args.append('--hex-section')
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    params = json.loads((folder/'build_params.json').read_text())
    config = MJCFConfig(physics_mode='mesh', joint_type='hinge', mesh_layout='shared',
        flat_section=section, thickness_profile=profile, timestep=.0001,
        tendon_inward_shift=params['tendon_inward_shift'], phi_deg=params['phi_deg'], post_gen=params['post_gen'])
    write_mjcf_from_sites_csv(str(folder/'Geom_Data_CSV/Spirob_geom_data.csv'), str(folder/'reference.xml'),
        str(folder/'meshes'), config=config, geometry=from_params(params))
    return folder, section, profile


def test_one_collider_shared_files_and_unchanged_dynamics(convex):
    folder, _, _ = convex
    model = mujoco.MjModel.from_xml_path(str(folder/'spirob_physics_model.xml'))
    reference = mujoco.MjModel.from_xml_path(str(folder/'reference.xml'))
    report = collision_report(folder/'spirob_physics_model.xml', model)
    assert report['total_link_contact_geoms'] == report['links'] == 21
    assert set(report['contact_geoms_per_link'].values()) == {1}
    assert report['all_proxies_zero_mass'] and report['nativeccd'] and report['multiccd']
    assert report['max_contact_margin_m'] == 0
    assert len(report['source_stl_files']) == len(list((folder/'meshes').glob('*.stl'))) == 2
    for attr in ('body_mass','body_ipos','body_iquat','body_inertia','jnt_stiffness','dof_damping',
                 'site_pos','jnt_axis','body_pos','body_quat','actuator_gear','actuator_ctrlrange'):
        np.testing.assert_allclose(getattr(model, attr), getattr(reference, attr), rtol=1e-13, atol=1e-16)
    for obj, count in [('body',model.nbody),('joint',model.njnt),('site',model.nsite),
                       ('actuator',model.nu),('tendon',model.ntendon)]:
        assert [getattr(model,obj)(i).name for i in range(count)] == [getattr(reference,obj)(i).name for i in range(count)]
    root = ET.parse(folder/'spirob_physics_model.xml').getroot()
    for geom in root.findall('.//geom'):
        if geom.get('name','').startswith('collision_'): geom.set('mass','5')
    changed = folder/'changed.xml'; ET.ElementTree(root).write(changed)
    np.testing.assert_array_equal(mujoco.MjModel.from_xml_path(str(changed)).body_mass, model.body_mass)


def test_envelope_covers_cad_with_small_documented_hex_error(convex):
    from scipy.spatial import ConvexHull
    folder, section, profile = convex
    model = mujoco.MjModel.from_xml_path(str(folder/'spirob_physics_model.xml'))
    for name in ('link_001','link_002','link_021'):
        v,f = compiled_surface(model,name); cv,cf = collider_surface(model,name)
        cad = trimesh.Trimesh(vertices=v,faces=f,process=True)
        collider = trimesh.Trimesh(vertices=cv,faces=cf,process=True)
        eq = ConvexHull(cv).equations
        assert np.max(v @ eq[:,:3].T+eq[:,3]) < 2e-6  # mm; float mesh compilation
        excess = collider.volume/cad.volume-1
        if section == 'hex' and profile == 'linear':
            assert 0 <= excess < .002
            assert max_surface_distance(cad,surface_samples(collider)) < .060  # supplied geometry, mm
        else:
            assert abs(excess) < 1e-6
            assert max_surface_distance(cad,surface_samples(collider)) < 2e-5


def sphere_probe(xml_path, body_name):
    path = Path(xml_path); source = ET.parse(path).getroot(); root = ET.Element('mujoco')
    for tag in ('compiler','default','option','asset'): root.append(copy.deepcopy(source.find(tag)))
    for asset in root.findall('./asset/mesh'):
        if asset.get('file'): asset.set('file',str((path.parent/asset.get('file')).resolve()))
    world = ET.SubElement(root,'worldbody')
    body = ET.SubElement(world,'body',name='probe_link'); ET.SubElement(body,'freejoint')
    for element in source.find(f'.//body[@name="{body_name}"]'):
        if element.tag in ('geom','inertial'): body.append(copy.deepcopy(element))
    probe = ET.SubElement(world,'body',name='probe_body',mocap='true')
    ET.SubElement(probe,'geom',name='probe',type='sphere',size='.0002',margin='0')
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))


def test_sphere_contacts_each_external_face_and_separates(convex):
    folder,_,_ = convex; xml = folder/'spirob_physics_model.xml'
    original = mujoco.MjModel.from_xml_path(str(xml))
    for name in ('link_001','link_002','link_021'):
        v,f = compiled_surface(original,name)
        cad = trimesh.Trimesh(vertices=v/1000,faces=f,process=True)
        model = sphere_probe(xml,name); gid = model.geom('probe').id
        _,ids = np.unique(np.round(cad.face_normals,1),axis=0,return_index=True)
        for face in ids:
            point = cad.triangles_center[face]; normal = cad.face_normals[face]
            if np.linalg.norm(normal) < .9: continue
            for distance,touching in ((.00018,True),(.0005,False)):
                data = mujoco.MjData(model)
                data.mocap_pos[0] = point+normal*distance
                mujoco.mj_forward(model,data)
                assert bool(data.ncon) == touching, (name,face,point,normal,distance)


def test_flat_patch_counts_and_bounded_support(convex):
    folder,section,profile = convex
    if (section,profile) != ('rectangular','stepped'):
        pytest.skip('Flat-face benchmark uses legacy constant-thickness section')
    params = json.loads((folder/'build_params.json').read_text())
    from spirob.sections import resolve_flat_thickness_ratio
    cfg = MJCFConfig(physics_mode='compound',joint_type='hinge',mesh_layout='shared',
        timestep=.0001,flat_thickness_ratio=resolve_flat_thickness_ratio(params),
        post_gen=params['post_gen'],phi_deg=params['phi_deg'],tendon_inward_shift=params['tendon_inward_shift'])
    write_mjcf_from_sites_csv(str(folder/'Geom_Data_CSV/Spirob_geom_data.csv'),str(folder/'compound.xml'),
        str(folder/'meshes'),config=cfg,geometry=from_params(params))
    report = benchmark(folder/'compound.xml',folder/'spirob_physics_model.xml')
    for kind in ('plane','box','pair'):
        a = report['cases']['compound']['probes'][kind]; b = report['cases']['candidate']['probes'][kind]
        assert 2 <= b['contacts'] <= 5
        assert b['scalar_constraints'] < a['scalar_constraints']/10
        assert not any(b['warnings'])
    model = isolated_patch(folder/'spirob_physics_model.xml','link_002','plane')
    model.opt.gravity[:] = [0,9.81,0]  # Press the +Y face against its support.
    data = mujoco.MjData(model)
    displacement = 0.
    for _ in range(5000):
        mujoco.mj_step(model,data)
        displacement = max(displacement, np.linalg.norm(data.qpos[:3]))
    mujoco.mj_forward(model,data)
    assert not np.any(data.warning.number)
    # Mesh-plane manifold changes can rock slightly. Check bounded support,
    # not exact zero velocity (which this contact representation does not give).
    assert data.ncon > 0 and displacement < .001
    assert abs(data.qpos[3]) > np.cos(.05/2)


def test_array_shares_convex_assets(convex):
    folder,_,_ = convex; xml = folder/'spirob_physics_model.xml'; array = folder/'array.xml'
    build_array(str(xml),2,radius_m=.12,out_path=str(array))
    single = mujoco.MjModel.from_xml_path(str(xml)); multiple = mujoco.MjModel.from_xml_path(str(array))
    assert multiple.nmesh == single.nmesh
    assert sum(multiple.body_mass) == pytest.approx(2*sum(single.body_mass),rel=1e-13)


def test_convex_rejects_n_cable_and_plain(tmp_path):
    for options in (['--params',str(ROOT/'examples/params-three-cable.json')],
                    ['--params',str(ROOT/'examples/params-two-cable.json'),'--plain']):
        result = subprocess.run([sys.executable,str(ROOT/'build.py'),*options,'--collision-mode','convex',
            '--no-preview','--output-dir',str(tmp_path)],capture_output=True,text=True)
        assert result.returncode != 0
