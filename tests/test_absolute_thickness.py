"""Absolute base dimensions propagate to meshes, proxies, CAD and GUI."""
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from spirob.geometry import from_params
from spirob.sections import resolve_section_params, resolve_flat_thickness_ratio, section_dimensions
from tools.inspect_section import compiled_surface

ROOT = Path(__file__).resolve().parents[1]


def two_params():
    return dict(json.loads((ROOT/'examples/params-two-cable.json').read_text()), thickness_profile='stepped')


def test_default_width_equals_thickness_and_explicit_base_stays_fixed():
    params = two_params()
    report = section_dimensions(params)
    for link in report['links']:
        assert link['centre_thickness_m'] == link['width_m']
    for policy in ['exact_requested_length', 'whole_units']:
        params['terminal_unit_policy'] = policy
        p = resolve_section_params(params, base_thickness_mm='20')
        dimensions = section_dimensions(p)
        assert dimensions['base']['centre_thickness_m'] == pytest.approx(.020)
        assert dimensions['tip']['centre_thickness_m'] == pytest.approx(.020*dimensions['tip']['width_m']/dimensions['base']['width_m'])
        p['L'] *= 1.1
        assert section_dimensions(p)['base']['centre_thickness_m'] == pytest.approx(.020)
        assert section_dimensions(p)['base']['width_m'] != dimensions['base']['width_m']


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True, '20'])
def test_invalid_json_base_thickness(value):
    p = two_params(); p['base_thickness_m'] = value
    with pytest.raises(ValueError, match='base_thickness_m'): resolve_section_params(p)


def test_legacy_ratio_and_cli_override_are_unambiguous():
    p = two_params(); p.pop('base_thickness_m'); p['flat_thickness_ratio'] = .3
    assert resolve_flat_thickness_ratio(p) == .3
    q = resolve_section_params(p, base_thickness_mm='20')
    assert 'flat_thickness_ratio' not in q and q['base_thickness_m'] == .020
    assert resolve_flat_thickness_ratio(resolve_section_params(p, base_thickness_mm='auto')) == 1.
    with pytest.raises(ValueError, match='not both'):
        resolve_section_params(dict(p, base_thickness_m=.020))
    with pytest.raises(ValueError, match='requires n_cables=2'):
        resolve_section_params(dict(p,n_cables=3), base_thickness_mm='20')
    with pytest.raises(ValueError, match='requires n_cables=2'):
        resolve_section_params(p, base_thickness_mm='20', plain=True)


def test_gui_roundtrip_and_switch_to_n_cables():
    from design_gui import collect_params, thickness_field_value
    p = two_params()
    assert thickness_field_value(p) == 'auto'
    q = collect_params(p, dict(base_thickness_mm='20', flat_section='hex', hex_edge_ratio='.6'))
    assert q['base_thickness_m'] == .020 and q['hex_edge_ratio'] == .6
    assert thickness_field_value(q) == '20'
    q = collect_params(q, dict(n_cables='3', base_thickness_mm='20', flat_section='hex', hex_edge_ratio='.6'))
    assert 'base_thickness_m' not in q and 'hex_edge_ratio' not in q and 'flat_section' not in q
    p.pop('base_thickness_m');p['flat_thickness_ratio'] = .3
    actual = float(thickness_field_value(p))
    assert actual/1000 == pytest.approx(from_params(p).units[0].realized_width_m*.3)


@pytest.fixture(scope='module', params=['hex', 'rectangular'])
def absolute_build(request, tmp_path_factory):
    folder=tmp_path_factory.mktemp('absolute-'+request.param)
    args=[sys.executable,str(ROOT/'build.py'),'--params',str(ROOT/'examples/params-two-cable.json'),
          '--thickness-profile','stepped',
          '--base-thickness-mm','20','--timestep','.0001','--no-preview','--output-dir',str(folder)]
    args += ['--hex-section','--collision-mode','mesh'] if request.param == 'hex' else ['--collision-mode','compound']
    run=subprocess.run(args,capture_output=True,text=True)
    assert run.returncode == 0, run.stdout+run.stderr
    return folder, request.param, mujoco.MjModel.from_xml_path(str(folder/'spirob_physics_model.xml'))


def test_every_mesh_and_compound_proxy_follows_base_dimension(absolute_build):
    folder, kind, model = absolute_build
    params=json.loads((folder/'build_params.json').read_text())
    assert params['base_thickness_m'] == .020 and 'flat_thickness_ratio' not in params
    report=json.loads((folder/'section_dimensions.json').read_text())
    assert report['reference'].startswith('largest base link')
    for link in report['links']:
        verts,_=compiled_surface(model,link['link'])
        assert np.ptp(verts[:,1])/1000 == pytest.approx(link['centre_thickness_m'],rel=2e-6)
        if kind == 'rectangular':
            bid=model.body(link['link']).id
            for gid in range(model.body_geomadr[bid],model.body_geomadr[bid]+model.body_geomnum[bid]):
                if model.geom_group[gid] != 3: continue
                if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
                    assert model.geom_size[gid,1]*2 == pytest.approx(link['centre_thickness_m'])
                elif model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    assert model.geom_size[gid,1]*2 == pytest.approx(link['centre_thickness_m'])
    assert report['base']['centre_thickness_m'] == pytest.approx(.020)
    assert len(list((folder/'meshes').glob('*.stl'))) == 2


def test_absolute_cad_audit_and_fabrication_match(absolute_build,tmp_path):
    import cadquery as cq
    import trimesh
    from tools.audit_inertia import audit
    from cad_export import process_cad
    folder,kind,model=absolute_build
    params=json.loads((folder/'build_params.json').read_text())
    result=audit(folder/'spirob_physics_model.xml',params=params,links=['link_001','link_002','link_021'])
    for link in result['links']:
        assert link['references']['cad']['inertia_relative_frobenius_error'] < 2e-6
    cad=process_cad(folder/'Geom_Data_CSV/Spirob_geom_data.csv',params,outdir=tmp_path,cable_hole_diameter_mm=1.)
    shape=cq.importers.importStep(cad.step_path).val()
    assert shape.isValid() and len(shape.Solids()) == 1
    assert shape.BoundingBox().ylen == pytest.approx(20.,abs=1e-5)
    assert trimesh.load(cad.stl_path,force='mesh').is_volume


def test_preview_displays_the_requested_section():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from preview import draw_flat_section
    p=resolve_section_params(two_params(),base_thickness_mm='20',hex_section=True)
    g=from_params(p); fig,ax=plt.subplots()
    draw_flat_section(ax,g.units[0].realized_width_m/2,p)
    polygon=ax.patches[-1]
    assert len(polygon.get_xy()) == 7
    assert np.ptp(polygon.get_xy()[:,1]) == pytest.approx(20.)
    plt.close(fig)
