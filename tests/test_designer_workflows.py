"""User workflow regressions for the consolidated CLI and local designer."""
import copy
import http.client
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from spirob.parameters import normalize_params
from tools.add_touch_sensors import add_touch_sensors
from tools.designer import BuilderServer

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('change',[
    {'L':True}, {'notch_factor':float('nan')}, {'n_cables':True},
    {'post_gen':{'robot_quat':[0,0,0,0]}},
    {'post_gen':{'robot_pos':[0,float('inf'),0]}},
    {'post_gen':{'joint_damping_base':True}},
    {'build':{'collision_mod':'convex'}}, {'build':{'cad':'yes'}},
])
def test_bad_parameters_are_rejected_before_geometry(change):
    p=json.loads((ROOT/'params.json').read_text());p.update(change)
    with pytest.raises(ValueError):normalize_params(p)


def test_retired_target_is_removed_without_mutating_user_input():
    p=json.loads((ROOT/'params.json').read_text());p['post_gen']['target_site_pos']=[1,2,3]
    old=copy.deepcopy(p)
    with pytest.warns(UserWarning,match='retired'):
        clean=normalize_params(p)
    assert 'target_site_pos' not in clean['post_gen'] and p==old


@pytest.fixture(scope='module')
def model(tmp_path_factory):
    directory=tmp_path_factory.mktemp('model with spaces')
    p=json.loads((ROOT/'examples/params-three-cable.json').read_text())
    p.update(L=.025,terminal_unit_policy='whole_units',notch_factor=0)
    p['build'].update(cad=True,iges=True,neck_width_mm=.6)
    params=directory/'my params.json';params.write_text(json.dumps(p))
    output=directory/'my model'
    result=subprocess.run([sys.executable,str(ROOT/'build.py'),'--params',str(params),
                           '--output-dir',str(output),'--no-preview'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    return output,params


def test_build_json_generates_zero_notch_cad_and_iges(model):
    folder,_=model
    xml=folder/'spirob_physics_model.xml'
    model=mujoco.MjModel.from_xml_path(str(xml))
    assert model.opt.timestep==.0001
    assert mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'target')==-1
    assert model.nu==3
    report=json.loads((folder/'cad/spirob_cad_report.json').read_text())
    assert report['stl_oriented_volume'] and report['solid_count']==1
    assert report['iges']['units']=='mm'
    assert (folder/'cad/spirob.iges').stat().st_size>1000
    saved=json.loads((folder/'build_params.json').read_text())
    assert saved['build']['collision_mode']=='convex' and saved['build']['iges']


def test_touch_regions_preserve_existing_sensors_and_physics(model,tmp_path):
    folder,_=model;source=folder/'spirob_physics_model.xml'
    # Use the existing safe path rebasing helper to keep real assets resolvable.
    from spirob.xml_tools import read_standalone,write_validated
    tree=read_standalone(source);sensor=tree.getroot().find('sensor')
    if sensor is None:sensor=ET.SubElement(tree.getroot(),'sensor')
    ET.SubElement(sensor,'framepos',name='existing_position',objtype='body',objname='link_001')
    original=tmp_path/'existing.xml';write_validated(tree,source,original)
    output=tmp_path/'sensors.xml'
    count=add_touch_sensors(original,output,links=[1,2])
    result=mujoco.MjModel.from_xml_path(str(output));baseline=mujoco.MjModel.from_xml_path(str(original))
    assert count==6 and result.nsensor==baseline.nsensor+6
    assert result.sensor('existing_position').dim==3
    assert result.ngeom==baseline.ngeom
    np.testing.assert_array_equal(result.body_mass,baseline.body_mass)
    with pytest.raises(ValueError,match='already exists'):
        add_touch_sensors(output,tmp_path/'duplicate.xml',links=[1])
    assert not (tmp_path/'duplicate.xml').exists()


def test_rebuild_retires_stale_cad_and_cli_overrides_json(model,tmp_path):
    source,params=model
    folder=tmp_path/'rebuild'
    shutil.copytree(source,folder)
    result=subprocess.run([sys.executable,str(ROOT/'build.py'),'--params',str(params),
                           '--output-dir',str(folder),'--no-preview','--no-cad','--no-iges',
                           '--collision-mode','capsule'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert not (folder/'cad').exists()
    saved=json.loads((folder/'build_params.json').read_text())
    assert not saved['build']['cad'] and saved['build']['collision_mode']=='capsule'
    mujoco.MjModel.from_xml_path(str(folder/'spirob_physics_model.xml'))


def test_builder_rejects_other_origins_and_bad_host(tmp_path):
    with BuilderServer(('127.0.0.1',0),tmp_path) as server:
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            port=server.server_port;connection=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
            connection.request('GET','/api/status');response=connection.getresponse()
            assert response.status==200 and json.loads(response.read())['builder']
            connection.request('POST','/api/build',body='{}',headers={'Origin':'https://other.example','Content-Type':'application/json'})
            response=connection.getresponse();assert response.status==403;response.read()
            connection.request('GET','/api/status',headers={'Host':'other.example'})
            response=connection.getresponse();assert response.status==403;response.read()
            connection.request('GET','/../pyproject.toml');response=connection.getresponse()
            assert response.status==404;response.read();connection.close()
        finally:
            server.shutdown();thread.join()


def test_site_presets_are_the_repository_examples():
    for p in (ROOT/'examples').glob('params*.json'):
        assert json.loads(p.read_text())==json.loads((ROOT/'site/presets'/p.name).read_text())
