"""Measure core taper in exported solids, not just its parameter report."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import cadquery as cq
import mujoco
import numpy as np
import pytest
import trimesh

from spirob.core import core_dimensions, reference_width_at, reference_width_law
from spirob.geometry import from_params
from spirob.parameters import normalize_params

ROOT = Path(__file__).resolve().parents[1]


def parameters(n=3):
    name={2:'two-cable-hex',3:'three-cable',4:'four-cable',6:'six-cable'}[n]
    return json.loads((ROOT/f'examples/params-{name}.json').read_text())


@pytest.mark.parametrize('n',[2,3,4,6])
@pytest.mark.parametrize('length,policy',[(.22628,'exact_requested_length'),(.08,'whole_units'),(.002,'exact_requested_length')])
def test_width_percent_linear_distance_geometric_link_spacing(n,length,policy):
    p=parameters(n);p.update(L=length,terminal_unit_policy=policy)
    g=from_params(p);r=core_dimensions(g,7.5);law=r['width_law']
    z=np.linspace(law['z_base_m'],law['z_tip_m'],31)
    w=np.array([reference_width_at(law,x) for x in z])
    assert np.all(np.diff(w)<0)
    np.testing.assert_allclose(np.diff(w,2),0,atol=1e-16)
    for s in r['stations']:
        assert s['core_width_mm']==pytest.approx(.075*s['reference_width_mm'])
    complete=[u for u in g.units if not u.is_partial]
    widths=[reference_width_at(law,u.local_frame_origin_m[2]) for u in complete]
    for a,b in zip(widths,widths[1:]):assert a/b==pytest.approx(g.spiral.beta_nominal,rel=1e-12)


@pytest.mark.parametrize('bad',[0,-1,100,float('nan'),True])
def test_bad_core_percent_rejected(bad):
    with pytest.raises(ValueError):core_dimensions(from_params(parameters()),bad)


def test_old_core_migrates_to_taper_and_conflicting_definitions_fail():
    p=parameters();p['build'].pop('elastic_core_percent');p['build']['neck_width_mm']=1
    original=copy.deepcopy(p)
    with pytest.warns(UserWarning,match='tapered core'):new=normalize_params(p)
    assert p==original and 'neck_width_mm' not in new['build']
    r=core_dimensions(from_params(new),new['build']['elastic_core_percent'])
    assert r['stations'][0]['core_width_mm']==pytest.approx(1)
    assert r['stations'][-1]['core_width_mm']<1
    p['build']['elastic_core_percent']=5
    with pytest.raises(ValueError,match='Choose'):normalize_params(p)


@pytest.fixture(scope='module',params=[2,3,4])
def exported(request,tmp_path_factory):
    p=parameters(request.param);p.update(L=.04,terminal_unit_policy='whole_units')
    p['build'].update(cad=True,iges=True,elastic_core_percent=7,cable_hole_diameter_mm=.5)
    folder=tmp_path_factory.mktemp(f'core-{request.param}');source=folder/'params.json'
    source.write_text(json.dumps(p));output=folder/'model'
    result=subprocess.run([sys.executable,str(ROOT/'build.py'),'--params',str(source),'--no-preview',
                           '--output-dir',str(output)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    return p,output


def test_all_three_exports_have_the_requested_taper_at_joint_planes(exported):
    p,folder=exported
    report=json.loads((folder/'elastic_core_dimensions.json').read_text())
    cad=json.loads((folder/'cad/spirob_cad_report.json').read_text())
    assert cad['solid_count']==1 and cad['stl_oriented_volume']
    assert cad['elastic_core']['elastic_core_percent']==7
    shape=cq.importers.importStep(str(folder/'cad/spirob.step')).val()
    mesh=trimesh.load(folder/'cad/spirob.stl',force='mesh')
    from OCP.IGESControl import IGESControl_Reader
    reader=IGESControl_Reader();reader.ReadFile(str(folder/'cad/spirob.iges'));reader.TransferRoots()
    iges=cq.Shape.cast(reader.OneShape())
    # At a centreline joint, the link slit faces meet only at the axis;
    # the finite transverse section therefore measures the fabricated core.
    for s in report['stations'][1:-1]:
        z,w=s['z_from_base_mm'],s['core_width_mm']
        for solid in (shape,iges):
            from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
            plane=cq.Face.makePlane(100,100,cq.Vector(0,0,z))
            operation=BRepAlgoAPI_Section(solid.wrapped,plane.wrapped,False)
            operation.Build()
            assert operation.IsDone()
            cut=cq.Shape.cast(operation.Shape())
            assert cut.BoundingBox().xlen==pytest.approx(w,abs=3e-4)
        segments=trimesh.intersections.mesh_plane(mesh,[0,0,1],[0,0,z])
        assert np.ptp(segments[:,:,0])==pytest.approx(w,abs=.004)
    model=mujoco.MjModel.from_xml_path(str(folder/'spirob_physics_model.xml'))
    post=p['post_gen']
    for i in range(1,model.njnt):
        assert model.jnt_stiffness[i]==pytest.approx(post['joint_stiffness_base']/post['joint_beta']**(3*i))
        assert model.dof_damping[model.jnt_dofadr[i]]==pytest.approx(post['joint_damping_base']/post['joint_beta']**(3*i))
    assert model.jnt_stiffness[0]==post['first_joint_stiffness']


def test_browser_and_python_core_dimensions_agree():
    cases=[parameters(n) for n in (2,3,4,6)]
    script="""import fs from 'node:fs';import {derive,coreWidthAt} from './site/geometry.mjs';
console.log(JSON.stringify(JSON.parse(fs.readFileSync(0,'utf8')).map(p=>{const g=derive(p);return g.units.map(u=>[0,.25,.5,.75,1].map(f=>coreWidthAt(p,g,u.z0+(u.z1-u.z0)*f)));})));"""
    result=subprocess.run(['node','--input-type=module','-e',script],cwd=ROOT,input=json.dumps(cases),capture_output=True,text=True,check=True)
    for p,values in zip(cases,json.loads(result.stdout)):
        g=from_params(p);law=reference_width_law(g)
        expect=[[.05*reference_width_at(law,u.local_frame_origin_m[2]+f*(u.slit_reference_m[2]-u.local_frame_origin_m[2]))
                 for f in (0,.25,.5,.75,1)] for u in g.units]
        np.testing.assert_allclose(values,expect,atol=2e-14,rtol=0)
