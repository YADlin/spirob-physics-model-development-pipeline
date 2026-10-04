"""Constant Y must survive shared mesh scaling and all fabrication formats."""
import json
from pathlib import Path
import subprocess
import sys

import cadquery as cq
import mujoco
import numpy as np
import pytest
import trimesh
from spirob.geometry import from_params
from spirob.sections import section_dimensions
from tools.inspect_section import compiled_surface

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('section,layout',[('rectangular','shared'),('rectangular','individual'),('hex','shared')])
def test_constant_exports_and_browser_match(tmp_path,section,layout):
    p=json.loads((ROOT/'examples/params-two-cable.json').read_text())
    p.update(L=.04,thickness_profile='constant',base_thickness_m=.003,flat_section=section)
    if section=='hex':p['hex_edge_ratio']=.75
    p['build'].update(cad=True,iges=True,mesh_layout=layout,collision_mode='convex')
    source=tmp_path/'params.json';source.write_text(json.dumps(p));folder=tmp_path/'model'
    result=subprocess.run([sys.executable,str(ROOT/'build.py'),'--params',str(source),'--no-preview','--output-dir',str(folder)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    g=from_params(p);dims=section_dimensions(p,g)
    assert all(r['proximal_centre_thickness_m']==pytest.approx(.003) and r['distal_centre_thickness_m']==pytest.approx(.003) for r in dims['links'])
    model=mujoco.MjModel.from_xml_path(str(folder/'spirob_physics_model.xml'))
    for unit in g.units:
        v,_=compiled_surface(model,unit.link_name)
        assert np.ptp(v[:,1])==pytest.approx(3,abs=1e-5)
    shape=cq.importers.importStep(str(folder/'cad/spirob.step')).val()
    from OCP.IGESControl import IGESControl_Reader
    reader=IGESControl_Reader();reader.ReadFile(str(folder/'cad/spirob.iges'));reader.TransferRoots()
    iges=cq.Shape.cast(reader.OneShape())
    mesh=trimesh.load(folder/'cad/spirob.stl',force='mesh')
    length=g.lengths.discrete_chord_length_m*1000
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
    for z in [length*.1,length*.5,length*.9]:
        for solid in [shape,iges]:
            plane=cq.Face.makePlane(100,100,cq.Vector(0,0,z))
            op=BRepAlgoAPI_Section(solid.wrapped,plane.wrapped,False);op.Build()
            assert cq.Shape.cast(op.Shape()).BoundingBox().ylen==pytest.approx(3,abs=3e-4)
        cut=trimesh.intersections.mesh_plane(mesh,[0,0,1],[0,0,z])
        assert np.ptp(cut[:,:,1])==pytest.approx(3,abs=.01)
    core=json.loads((folder/'elastic_core_dimensions.json').read_text())
    assert core['stations'][0]['core_width_mm']>core['stations'][-1]['core_width_mm']
    report=json.loads((folder/'design_report/params.json').read_text())
    assert report['thickness_profile']=='constant'
    js="import fs from 'node:fs';import {derive} from './site/geometry.mjs';console.log(JSON.stringify(derive(JSON.parse(fs.readFileSync(0,'utf8'))).units.map(u=>[u.t0,u.t1])));"
    run=subprocess.run(['node','--input-type=module','-e',js],cwd=ROOT,input=json.dumps(p),text=True,capture_output=True,check=True)
    np.testing.assert_allclose(json.loads(run.stdout),.003,atol=1e-15)
