"""Verify the manufacturing route independently of the MJCF polyline."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import cadquery as cq
import trimesh
from spirob.geometry import from_params
from spirob.fabrication_routes import cable_hole_axes

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('n',[2,3,4])
@pytest.mark.parametrize('policy',['exact_requested_length','whole_units'])
def test_straight_endpoints_match_browser(n,policy):
    p=json.loads((ROOT/'params.json').read_text());p.update(n_cables=n,terminal_unit_policy=policy)
    g=from_params(p);axes=cable_hole_axes(g)
    script="import fs from 'node:fs';import {derive} from './site/geometry.mjs';import {boreAxes} from './site/holes.mjs';console.log(JSON.stringify(boreAxes(derive(JSON.parse(fs.readFileSync(0,'utf8'))))));"
    r=subprocess.run(['node','--input-type=module','-e',script],cwd=ROOT,input=json.dumps(p),text=True,capture_output=True,check=True)
    for a,b,path in zip(axes,json.loads(r.stdout),g.tendon_paths):
        np.testing.assert_allclose([a['base_m'],a['tip_m']],[b['base'],b['tip']],atol=1e-13)
        direction=np.array(a['tip_m'])-a['base_m']
        for pt in [path.points[0],path.points[-1]]:
            assert np.linalg.norm(np.cross(np.array(pt.routed_m)-a['base_m'],direction))<1e-15

@pytest.mark.parametrize('profile,policy',[('constant','exact_requested_length'),('linear','exact_requested_length'),('constant','whole_units')])
def test_exports_have_open_bores_without_round_transition_faces(tmp_path,profile,policy):
    p=json.loads((ROOT/'examples/params-two-cable.json').read_text())
    p.update(L=.06,thickness_profile=profile,base_thickness_m=.015,terminal_unit_policy=policy)
    p['build'].update(cad=True,iges=True,cable_hole_diameter_mm=1.6)
    src=tmp_path/'params.json';src.write_text(json.dumps(p));out=tmp_path/'model'
    r=subprocess.run([sys.executable,str(ROOT/'build.py'),'--params',str(src),'--no-preview','--output-dir',str(out)],capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    report=json.loads((out/'cad/spirob_cad_report.json').read_text())
    assert max(report['bore_check']['residual_volumes_mm3'])<1e-6
    assert report['stl_bore_check']['passed']
    step=cq.importers.importStep(str(out/'cad/spirob.step')).val()
    assert not any(f.geomType() in ('SPHERE','TORUS','REVOLUTION') for f in step.Faces())
    from OCP.IGESControl import IGESControl_Reader
    reader=IGESControl_Reader();reader.ReadFile(str(out/'cad/spirob.iges'));reader.TransferRoots()
    iges=cq.Shape.cast(reader.OneShape())
    assert len(iges.Solids())==len(step.Solids())==1
    assert len(iges.Faces())==len(step.Faces())
    for shape in [step,iges]:
        b=shape.BoundingBox()
        assert b.zmin>=-1e-5
        assert b.zmax==pytest.approx(from_params(p).lengths.discrete_chord_length_m*1000,abs=1e-5)
    mesh=trimesh.load(out/'cad/spirob.stl',force='mesh')
    assert mesh.is_volume and len(mesh.split())==1
    # Reimported IGES must also contain a full-length unobstructed bore.
    from spirob.fabrication_routes import cylinder_for_axis
    g=from_params(p)
    for axis in cable_hole_axes(g):
        pin=cylinder_for_axis(axis,1.599,g.units[0].local_frame_origin_m[2])
        assert iges.intersect(pin).Volume()<1e-6
