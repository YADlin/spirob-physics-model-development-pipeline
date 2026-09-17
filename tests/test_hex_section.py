"""Six-sided XY section, shared scaling, inertias, and unchanged interfaces."""
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
import pytest
import trimesh

from spirob.sections import resolve_section_params
from tools.inspect_section import compiled_surface, projection

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('edge', [0, 1, -1, float('nan'), float('inf'), True])
def test_invalid_edge_ratio_rejected(edge):
    with pytest.raises(ValueError, match='hex_edge_ratio'):
        resolve_section_params({'n_cables': 2}, hex_section=True, hex_edge_ratio=edge)


def test_hex_is_explicit_two_cable_only():
    p = {'n_cables': 2}
    assert resolve_section_params(p) == p
    assert resolve_section_params(p, hex_section=True)['hex_edge_ratio'] == .75
    assert p == {'n_cables': 2}
    for args in [dict(hex_edge_ratio=.5), dict(hex_section=True, plain=True)]:
        with pytest.raises(ValueError): resolve_section_params(p, **args)
    with pytest.raises(ValueError): resolve_section_params({'n_cables': 3}, hex_section=True)


@pytest.fixture(scope='module')
def hex_build(tmp_path_factory):
    folder = tmp_path_factory.mktemp('hex')
    result = subprocess.run([sys.executable, str(ROOT/'build.py'), '--params', str(ROOT/'examples/params-two-cable.json'),
                             '--hex-section', '--timestep', '.0001', '--collision-mode', 'mesh',
                             '--collision-margin-m', '0', '--no-preview', '--cad', '--cad-profile', 'simulation',
                             '--output-dir', str(folder)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    return folder, mujoco.MjModel.from_xml_path(str(folder/'spirob_physics_model.xml'))


def test_every_compiled_link_has_six_transverse_sides(hex_build):
    folder, model = hex_build
    assert model.opt.timestep == .0001
    assert {p.name for p in (folder/'meshes').glob('*.stl')} == {'link_001.stl', 'link_template.stl'}
    params = json.loads((folder/'build_params.json').read_text())
    assert params['flat_section'] == 'hex' and params['hex_edge_ratio'] == .75
    for bid in range(1, model.nbody):
        vertices, faces = compiled_surface(model, model.body(bid).name)
        poly = projection(vertices, faces, (0, 1))
        outline = np.asarray(poly.exterior.coords)[:-1]
        assert len(outline) == 6
        radius, height = np.max(np.abs(outline), axis=0)
        assert height/radius == pytest.approx(params['flat_thickness_ratio'], rel=1e-6)
        outer = outline[np.abs(outline[:, 0]) > radius*.9]
        assert len(outer) == 4
        np.testing.assert_allclose(np.abs(outer[:, 1])/height, .75, rtol=1e-6)
        surface = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        assert surface.is_volume
    base, faces = compiled_surface(model, 'link_001')
    assert abs(base[:, 2].min()) < 1e-5
    mount = base[faces][np.all(np.abs(base[faces, 2]) < 1e-5, axis=1)]
    assert len(mount) >= 4 and np.ptp(mount[:, :, 0]) > 30


def test_hex_inertias_match_exact_cad_and_scale(hex_build):
    from tools.audit_inertia import audit, body_inertia
    from spirob.geometry import from_params
    from spirob.mesh_assets import mesh_assets
    folder, model = hex_build
    params = json.loads((folder/'build_params.json').read_text())
    report = audit(folder/'spirob_physics_model.xml', params=params)
    for link in report['links']:
        assert 'mesh_reference_error' not in link
        for ref in link['references'].values():
            assert abs(ref['mass_relative_error']) < 2e-6
            assert ref['inertia_relative_frobenius_error'] < 2e-6
    for asset in mesh_assets(from_params(params)):
        if asset.filename != 'link_template.stl': continue
        bid = model.body(asset.link_name).id
        np.testing.assert_allclose(model.body_mass[bid], model.body('link_002').mass*asset.scale**3, rtol=2e-6)
        np.testing.assert_allclose(body_inertia(model, bid), body_inertia(model, model.body('link_002').id)*asset.scale**5,
                                   rtol=2e-6, atol=1e-14)


def test_legacy_shape_and_interfaces_unchanged(hex_build, tmp_path):
    folder, new = hex_build
    result = subprocess.run([sys.executable, str(ROOT/'build.py'), '--params', str(ROOT/'examples/params-two-cable.json'),
                             '--no-preview', '--output-dir', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    old = mujoco.MjModel.from_xml_path(str(tmp_path/'spirob_physics_model.xml'))
    for bid in range(1, old.nbody):
        v, f = compiled_surface(old, old.body(bid).name)
        assert len(projection(v, f, (0, 1)).exterior.coords)-1 == 4
        vnew, fnew = compiled_surface(new, new.body(bid).name)
        assert projection(v, f, (0, 2)).symmetric_difference(projection(vnew, fnew, (0, 2))).area < 1e-4
    for field in ['body_pos', 'body_quat', 'jnt_pos', 'jnt_axis', 'jnt_stiffness', 'dof_damping',
                  'site_pos', 'site_quat', 'actuator_ctrlrange', 'actuator_gear', 'tendon_adr', 'wrap_objid']:
        np.testing.assert_array_equal(getattr(new, field), getattr(old, field), err_msg=field)
    for obj, count in [('body', 'nbody'), ('joint', 'njnt'), ('site', 'nsite'), ('tendon', 'ntendon'), ('actuator', 'nu')]:
        assert [getattr(new, obj)(i).name for i in range(getattr(new, count))] == [getattr(old, obj)(i).name for i in range(getattr(old, count))]
    assert np.all(new.body_mass[1:] < old.body_mass[1:])  # Material was actually removed.


def test_compound_hex_rejected_without_replacing_previous_output(tmp_path):
    dest = tmp_path/'spirob_physics_model.xml'; dest.write_text('previous model')
    result = subprocess.run([sys.executable, str(ROOT/'build.py'), '--params', str(ROOT/'examples/params-two-cable.json'),
                             '--hex-section', '--collision-mode', 'compound', '--no-preview', '--output-dir', str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode != 0 and 'not implemented' in result.stderr
    assert dest.read_text() == 'previous model'


def test_section_inspection_uses_body_axes(hex_build, tmp_path):
    from tools.inspect_section import inspect
    folder, _ = hex_build
    report = inspect(folder/'spirob_physics_model.xml', tmp_path/'sections.png')
    assert (tmp_path/'sections.png').stat().st_size > 1000
    assert all(link['end_on_sides'] == 6 for link in report['links'])


@pytest.mark.parametrize('edge', [.25, .9])
def test_custom_edge_ratio_and_whole_units(edge):
    from spirob.geometry import from_params
    from csv2geom_nlobe import build_flat_element
    params = json.loads((ROOT/'examples/params-two-cable.json').read_text())
    params.update(flat_section='hex', hex_edge_ratio=edge, terminal_unit_policy='whole_units')
    geo = from_params(params)
    quad = geo.inverted_quads()[0]
    row = {prefix+'_'+axis: value for prefix, (x,z) in zip(('joint_s1','joint_s2','c0_s2','c0_s1'), quad)
           for axis,value in [('x',x),('y',0.),('z',z)]}
    solid = build_flat_element(row, .3, hex_edge_ratio=edge).val()
    assert solid.isValid() and solid.Volume() > 0
    v = np.array([vertex.toTuple() for vertex in solid.Vertices()])*1000
    r = np.max(np.abs(v[:,0])); h = np.max(np.abs(v[:,1]))
    np.testing.assert_allclose(np.abs(v[np.abs(v[:,0]) > r*.9,1])/h, edge, rtol=1e-6)


@pytest.mark.parametrize('hole', [0, 1])
def test_fabrication_hex_is_connected_and_closed(hex_build, tmp_path, hole):
    import cadquery as cq
    from cad_export import process_cad
    folder, _ = hex_build
    params = json.loads((folder/'build_params.json').read_text())
    result = process_cad(folder/'Geom_Data_CSV/Spirob_geom_data.csv', params,
                         outdir=tmp_path, profile='fabrication', cable_hole_diameter_mm=hole)
    restored = cq.importers.importStep(result.step_path).val()
    assert restored.isValid() and len(restored.Solids()) == 1
    assert trimesh.load(result.stl_path, force='mesh').is_volume
    # Full-length rectangular cores used to flatten the taper in Y. Check a
    # centreline point in the last unit: it must lie outside this tapered solid.
    bb = restored.BoundingBox()
    assert not restored.isInside(cq.Vector(0, bb.ymax*.8, bb.zmax-2), 1e-6)
    report = json.loads(Path(result.report_path).read_text())
    assert report['flat_section'] == 'hex' and report['thickness_scales_with_link']


def test_timestep_override_beats_preset(hex_build, tmp_path):
    folder, _ = hex_build
    result = subprocess.run([sys.executable, str(ROOT/'csv2xml.py'), '--in', str(folder/'Geom_Data_CSV/Spirob_geom_data.csv'),
                             '--meshdir', str(folder/'meshes'), '--params', str(folder/'build_params.json'), '--hinge',
                             '--high', '--timestep', '.00005', '--out', str(tmp_path/'test.xml')], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    assert mujoco.MjModel.from_xml_path(str(tmp_path/'test.xml')).opt.timestep == .00005
