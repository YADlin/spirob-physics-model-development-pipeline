"""Continuous thickness measured on exported, compiled solids, not metadata."""
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
import pytest
import trimesh

from spirob.geometry import from_params
from spirob.sections import linear_thickness_law, thickness_at_z, section_dimensions, resolve_section_params
from tools.inspect_section import compiled_surface
from tools.inspect_taper import inspect

ROOT=Path(__file__).resolve().parents[1]


def params(**overrides):
    return dict(json.loads((ROOT/'examples/params-two-cable.json').read_text()),**overrides)


@pytest.mark.parametrize('policy,length', [('exact_requested_length',.22628),('whole_units',.22628),
                                         ('exact_requested_length',.002),('exact_requested_length',.006)])
def test_taper_law_continuity_similarity_and_absolute_base(policy,length):
    p=params(terminal_unit_policy=policy,L=length,base_thickness_m=.020)
    g=from_params(p);law=linear_thickness_law(p,g);report=section_dimensions(p,g)
    assert report['base']['centre_thickness_m'] == pytest.approx(.020)
    assert 0 < report['tip']['centre_thickness_m'] < .020
    for a,b in zip(report['links'],report['links'][1:]):
        assert a['distal_centre_thickness_m'] == pytest.approx(b['proximal_centre_thickness_m'],abs=1e-14)
    complete=[u for u in g.units if not u.is_partial]
    for u in complete:
        ref=complete[0];scale=u.realized_width_m/ref.realized_width_m
        for attr in ['local_frame_origin_m','slit_reference_m']:
            assert thickness_at_z(law,getattr(u,attr)[2]) == pytest.approx(thickness_at_z(law,getattr(ref,attr)[2])*scale,rel=1e-12)


def test_new_default_and_profile_validation():
    p=params();p.pop('thickness_profile')
    report=section_dimensions(p)
    assert report['thickness_profile'] == 'linear'
    assert report['base']['centre_thickness_m'] == report['base']['width_m']
    assert report['tip']['centre_thickness_m'] == pytest.approx(.00692366277630555)
    for p in [dict(p,thickness_profile='bad'),dict(p,n_cables=3,thickness_profile='linear')]:
        with pytest.raises(ValueError):resolve_section_params(p)
    with pytest.raises(ValueError):resolve_section_params(params(),plain=True)


@pytest.fixture(scope='module',params=['hex','rectangular'])
def tapered(request,tmp_path_factory):
    folder=tmp_path_factory.mktemp('linear-'+request.param)
    args=[sys.executable,str(ROOT/'build.py'),'--params',str(ROOT/'examples/params-two-cable.json'),
          '--base-thickness-mm','20','--timestep','.0001','--collision-mode','mesh','--collision-margin-m','0',
          '--no-preview','--output-dir',str(folder)]
    if request.param == 'hex':args+=['--hex-section']
    result=subprocess.run(args,capture_output=True,text=True)
    assert result.returncode == 0,result.stdout+result.stderr
    return folder,request.param


def test_all_compiled_links_taper_and_join_continuously(tapered):
    folder,kind=tapered;p=json.loads((folder/'build_params.json').read_text())
    report=inspect(folder/'spirob_physics_model.xml',p,folder/'taper.png')
    assert report['passed']
    assert report['max_expected_error_mm'] < 1e-4
    assert report['max_joint_thickness_jump_mm'] < 1e-4
    assert all(row['slope_mm_per_mm'] < 0 for row in report['links'])
    assert all(row['max_line_fit_residual_mm'] < 1e-4 for row in report['links'])
    assert len(list((folder/'meshes').glob('*.stl'))) == 2
    model=mujoco.MjModel.from_xml_path(str(folder/'spirob_physics_model.xml'))
    for u in from_params(p).units:
        v,f=compiled_surface(model,u.link_name)
        mesh=trimesh.Trimesh(vertices=v,faces=f,process=True)
        assert mesh.is_volume
        # Cut between the two outer-corner Z coordinates: full-width XY section.
        q=from_params(p).inverted_quads()[u.index_base_to_tip]
        z=(q[2,1]+q[3,1])/2-q[0,1]
        cut=trimesh.intersections.mesh_plane(mesh,[0,0,1],[0,0,z*1000])
        points=cut.reshape(-1,3)
        h=np.max(np.abs(points[:,1]));r=u.realized_width_m*500
        outer=points[np.abs(np.abs(points[:,0])-r)<1e-4]
        assert np.max(np.abs(outer[:,1])) == pytest.approx(h*(.75 if kind=='hex' else 1),abs=1e-4)
        # Triangle cuts add collinear points on the short vertical outer sides.
        # Every boundary point must lie on either those sides or a sloping face.
        face_y=h*(1-(1-(.75 if kind=='hex' else 1))*np.abs(points[:,0])/r)
        boundary_error=np.minimum(np.abs(np.abs(points[:,0])-r),np.abs(np.abs(points[:,1])-face_y))
        # STL triangles approximate the ruled side faces; allow one micrometre
        # here. The centre-ridge continuity checks above retain 0.1 micrometre.
        assert np.max(boundary_error)<1e-3
    base,faces=compiled_surface(model,'link_001')
    assert abs(base[:,2].min()) < 1e-4  # mounting plane still flat at proximal end
    mount=base[faces][np.all(np.abs(base[faces,2])<1e-4,axis=1)]
    assert len(mount)>0 and np.ptp(mount[:,:,0])>30


def test_cad_inertia_and_connected_fabrication(tapered,tmp_path):
    import cadquery as cq
    from cad_export import process_cad
    from tools.audit_inertia import audit
    folder,kind=tapered;p=json.loads((folder/'build_params.json').read_text())
    result=audit(folder/'spirob_physics_model.xml',params=p,links=['link_001','link_002','link_021'])
    for link in result['links']:
        for reference in link['references'].values():
            assert abs(reference['mass_relative_error']) < 5e-6
            assert reference['inertia_relative_frobenius_error'] < 5e-6
    output=process_cad(folder/'Geom_Data_CSV/Spirob_geom_data.csv',p,outdir=tmp_path,cable_hole_diameter_mm=1.)
    shape=cq.importers.importStep(output.step_path).val()
    assert shape.isValid() and len(shape.Solids())==1
    assert shape.BoundingBox().ylen == pytest.approx(20,abs=1e-5)
    assert trimesh.load(output.stl_path,force='mesh').is_volume


def test_collision_refit_is_not_silently_claimed(tmp_path):
    xml=tmp_path/'spirob_physics_model.xml';xml.write_text('previous model')
    result=subprocess.run([sys.executable,str(ROOT/'build.py'),'--params',str(ROOT/'examples/params-two-cable.json'),
                           '--collision-mode','compound','--no-preview','--output-dir',str(tmp_path)],capture_output=True,text=True)
    assert result.returncode != 0 and 'not implemented' in result.stderr
    assert xml.read_text() == 'previous model'


def test_taper_changes_only_solid_and_inertia(tapered,tmp_path):
    folder,kind=tapered
    args=[sys.executable,str(ROOT/'build.py'),'--params',str(folder/'build_params.json'),
          '--thickness-profile','stepped','--no-preview','--output-dir',str(tmp_path)]
    result=subprocess.run(args,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    new=mujoco.MjModel.from_xml_path(str(folder/'spirob_physics_model.xml'))
    old=mujoco.MjModel.from_xml_path(str(tmp_path/'spirob_physics_model.xml'))
    for field in ['body_pos','body_quat','jnt_pos','jnt_axis','jnt_stiffness','dof_damping','site_pos',
                  'actuator_ctrlrange','actuator_gear','tendon_adr','wrap_objid']:
        np.testing.assert_array_equal(getattr(new,field),getattr(old,field))
    for obj,count in [('body','nbody'),('joint','njnt'),('site','nsite'),('tendon','ntendon'),('actuator','nu')]:
        assert [getattr(new,obj)(i).name for i in range(getattr(new,count))] == [getattr(old,obj)(i).name for i in range(getattr(old,count))]
    assert not np.array_equal(new.body_mass,old.body_mass)
