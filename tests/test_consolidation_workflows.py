"""Behavioral regression tests for the consolidation's broken user workflows."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def test_gui_commands_keep_interpreter_and_paths_with_spaces(tmp_path):
    from design_gui import build_command,collect_params
    cmd=build_command(tmp_path/'my params.json',tmp_path/'my outputs',cad=True)
    assert cmd[0]==sys.executable
    assert cmd[cmd.index('--params')+1]==str(tmp_path/'my params.json')
    assert '--cad' in cmd
    p=json.loads((ROOT/'params.json').read_text())
    with pytest.raises(ValueError): collect_params(p,{'L':'nan'})
    with pytest.raises(ValueError): collect_params(p,{'n_cables':'2.5'})


def test_missing_or_truncated_meshes_fail(tmp_path):
    from build import verify_meshes
    from spirob.geometry import from_params
    g=from_params(json.loads((ROOT/'params.json').read_text()))
    with pytest.raises(ValueError,match='Incomplete'):verify_meshes(tmp_path,g)
    for u in g.units:(tmp_path/(u.link_name+'.stl')).write_bytes(b'broken')
    with pytest.raises(ValueError,match='incomplete binary'):verify_meshes(tmp_path,g)


def test_failed_build_preserves_previous_outputs(tmp_path):
    p=json.loads((ROOT/'params.json').read_text());p['L']=-1
    params=tmp_path/'bad params.json';params.write_text(json.dumps(p))
    output=tmp_path/'outputs';output.mkdir()
    old=output/'spirob_physics_model.xml';old.write_text('previous model')
    r=subprocess.run([sys.executable,str(ROOT/'build.py'),'--params',str(params),
                      '--no-preview','--output-dir',str(output)],capture_output=True)
    assert r.returncode!=0
    assert old.read_text()=='previous model'


@pytest.mark.parametrize('cuts',[[-1],[0],[10],[5,5],[float('nan')]])
def test_invalid_cuts_rejected(cuts):
    from fabrication.part_splitter import plan_cuts
    with pytest.raises(ValueError):plan_cuts(0,10,cut_positions=cuts)


def test_translated_step_split_conserves_volume(tmp_path):
    cq=pytest.importorskip('cadquery')
    from fabrication.part_splitter import split_file
    source=tmp_path/'translated.step'
    original=cq.Workplane().box(20,30,210).translate((10000,-5000,300))
    cq.exporters.export(original,str(source))
    r=split_file(str(source),max_span_mm=80,build_volume_mm=(80,40,40))
    assert r['n_parts']==3 and r['all_fit']
    assert sum(p['volume'] for p in r['parts'])==pytest.approx(original.val().Volume(),rel=1e-7)
    assert all(p['valid'] and p['span'][2]<=80.000001 for p in r['parts'])


def test_stl_split_closed_and_volume_conserved(tmp_path):
    tm=pytest.importorskip('trimesh')
    from fabrication.part_splitter import split_file
    source=tmp_path/'box.stl';tm.creation.box(extents=(20,30,210)).export(source)
    r=split_file(str(source),max_span_mm=80,build_volume_mm=(80,40,40))
    assert r['n_parts']==3 and r['all_fit']
    assert all(p['watertight'] for p in r['parts'])
    assert sum(p['volume'] for p in r['parts'])==pytest.approx(20*30*210,rel=1e-6)


def test_array_rewrites_sensors_contacts_and_relocated_mesh(tmp_path):
    cq=pytest.importorskip('cadquery'); mj=pytest.importorskip('mujoco')
    from tools.multi_array import build_array
    assets=tmp_path/'source'/'meshes';assets.mkdir(parents=True)
    cq.exporters.export(cq.Workplane().box(.02,.02,.02),str(assets/'box.stl'))
    source=assets.parent/'robot.xml'
    source.write_text('''<mujoco><compiler meshdir="meshes"/>
    <asset><mesh name="shared_mesh" file="box.stl"/></asset>
    <worldbody><geom name="ground" type="plane" size="1 1 .1"/>
    <site name="target" pos="0 0 .3"/>
    <body name="link_001" pos=".2 .3 .1"><joint name="hinge"/>
    <geom name="visual" type="mesh" mesh="shared_mesh"/><site name="tip_site" pos="0 0 .02"/>
    <body name="link_002" pos="0 0 .02"><joint name="hinge2"/><geom name="ball" size=".01"/></body>
    </body></worldbody><tendon><fixed name="cable_0"><joint joint="hinge" coef="1"/></fixed></tendon>
    <actuator><motor name="motor_c0" tendon="cable_0"/></actuator>
    <sensor><jointpos name="angle" joint="hinge"/><framepos name="tip" objtype="site" objname="tip_site" reftype="site" refname="target"/></sensor>
    <contact><exclude body1="link_001" body2="link_002"/></contact>
    <equality><joint joint1="hinge" joint2="hinge2"/></equality></mujoco>''')
    dest=tmp_path/'elsewhere'/'array.xml'
    build_array(str(source),3,radius_m=.1,out_path=str(dest))
    model=mj.MjModel.from_xml_path(str(dest))
    assert (model.nbody,model.njnt,model.ntendon,model.nu,model.nsensor,model.nexclude,model.neq)==(7,6,3,3,6,3,3)
    assert model.nmesh==1
    tree=ET.parse(dest)
    for k in range(3):
        assert tree.find(f'.//framepos[@name="tip_r{k}"]').get('refname')=='target'
        assert tree.find(f'.//joint[@joint="hinge_r{k}"]') is not None
    before=source.read_bytes()
    with pytest.raises(ValueError):build_array(str(source),0,radius_m=.1)
    with pytest.raises(ValueError):build_array(str(source),2,radius_m=.1,out_path=str(source))
    assert source.read_bytes()==before


@pytest.fixture(scope='module')
def cad_two(tmp_path_factory):
    pytest.importorskip('cadquery')
    from spirob.geometry import from_params
    from helper_functions import generate_cable_sites_csv_zrot_from_P
    from cad_export import process_cad
    d=tmp_path_factory.mktemp('cad2');p=json.loads((ROOT/'params.json').read_text());p['n_cables']=2
    g=from_params(p);csv=d/'geometry.csv'
    generate_cable_sites_csv_zrot_from_P(g.inverted_quads(),n_cables=2,csv_path=str(csv),radial_scale=1.0)
    result=process_cad(str(csv),p,outdir=str(d/'cad'))
    return result,g,p


def test_fabrication_scale_connectivity_and_slits(cad_two):
    import cadquery as cq
    result,g,p=cad_two
    shape=cq.importers.importStep(result.step_path).val()
    assert shape.isValid() and len(shape.Solids())==1
    assert shape.BoundingBox().zlen==pytest.approx(g.lengths.discrete_chord_length_m*1000,abs=1e-5)
    # Probe between the two segment edge stations: old endpoint loft filled this gap.
    q0,q1=g.inverted_quads()[1:3]
    x=8.0
    z_joint=q0[1][1]*1000
    z_left=q0[2][1]*1000; z_right=q1[3][1]*1000
    z=(z_left+z_right)/2-g.units[0].local_frame_origin_m[2]*1000
    assert not shape.isInside(cq.Vector(x,0,z),1e-6)
    # The new finite ligament connects the otherwise zero-width hinge.
    assert shape.isInside(cq.Vector(.1,0,z_joint-g.units[0].local_frame_origin_m[2]*1000),1e-6)
    report=json.loads(Path(result.report_path).read_text())
    assert report['units']=='mm' and report['solid_count']==1
    tm=pytest.importorskip('trimesh');mesh=tm.load(result.stl_path,force='mesh')
    assert mesh.is_volume
    assert mesh.extents[2]==pytest.approx(shape.BoundingBox().zlen,abs=.01)


def test_cable_holes_preserve_closed_fabrication_mesh(cad_two,tmp_path):
    import cadquery as cq
    import trimesh
    from cad_export import process_cad
    result,g,p=cad_two
    csv=Path(result.step_path).parent.parent/'geometry.csv'
    drilled=process_cad(str(csv),p,outdir=str(tmp_path),cable_hole_diameter_mm=1)
    shape=cq.importers.importStep(drilled.step_path).val()
    assert shape.isValid() and len(shape.Solids())==1
    assert trimesh.load(drilled.stl_path,force='mesh').is_volume
    assert shape.Volume()<cq.importers.importStep(result.step_path).val().Volume()
    # A midpoint on each per-element cable path must now be empty.
    offset=g.units[0].local_frame_origin_m[2]*1000
    for path in g.tendon_paths:
        a,b=path.points[8:10]
        point=[(x+y)*500 for x,y in zip(a.routed_m,b.routed_m)];point[2]-=offset
        assert not shape.isInside(cq.Vector(*point),1e-6)
