"""Experimental j_002 convention, cross-language parity and compiled preservation."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from spirob.joint_gains import resolve_gains
from spirob.parameters import normalize_params
from tools.set_joint_gains import update_gains

ROOT = Path(__file__).resolve().parents[1]
RHO = 1.03 ** -3


def browser(p, n):
    result = subprocess.run(['node', '--input-type=module', '-e',
        "import {resolveGains} from './site/gains.mjs'; console.log(JSON.stringify(resolveGains(JSON.parse(process.argv[1]),Number(process.argv[2]))));",
        json.dumps(p), str(n)], cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize('post', [dict(), dict(k2=.2*RHO, rho_k=RHO),
    dict(k2=.34,rho_k=.8),dict(k2=.34,rho_k=.8,d2=.032,rho_d=.97),dict(rho_k=.7)])
@pytest.mark.parametrize('preset', ['safe','fast','high','default'])
def test_python_website_values(post, preset):
    p = {'post_gen':post, 'build':{'physics_preset':preset}}
    defaults={'safe':(.2,.2),'fast':(.01,.001),'high':(.1,.02),'default':(.2,.01)}[preset]
    py=resolve_gains(post, 21, *defaults); js=browser(p,21)
    for key in ('k2','rho_k','beta_j','d2','rho_d'):
        assert js[key] == pytest.approx(py[key], rel=1e-14)
    np.testing.assert_allclose([v['k'] for v in js['gains']],py['stiffness'],rtol=1e-14)
    np.testing.assert_allclose([v['d'] for v in js['gains']],py['damping'],rtol=1e-14)


def test_legacy_unchanged_and_conversion():
    post={'joint_stiffness_base':.2,'joint_damping_base':.013,'joint_beta':1.03,
          'first_joint_stiffness':100,'first_joint_damping':50}
    legacy=resolve_gains(post,21)
    assert legacy['stiffness'][1:] == [.2/((1.03**3)**i) for i in range(1,21)]
    new=resolve_gains(dict(post,k2=.2*RHO,rho_k=RHO,d2=.013*RHO,rho_d=RHO),21)
    np.testing.assert_allclose(new['stiffness'],legacy['stiffness'],rtol=1e-14)
    np.testing.assert_allclose(new['damping'],legacy['damping'],rtol=1e-14)
    fitted=resolve_gains(dict(post,k2=.7,rho_k=.75),21)
    assert fitted['damping']==legacy['damping']
    assert fitted['stiffness'][0]==100
    assert .2*RHO == pytest.approx(.1830283319,abs=5e-11)
    assert RHO == pytest.approx(.9151416594,abs=5e-11)


@pytest.mark.parametrize('joint_type',['hinge','ball'])
@pytest.mark.parametrize('automatic',[False,True])
def test_stiffness_only_preserves_compiled_model(tmp_path,joint_type,automatic):
    source=tmp_path/'source.xml'
    default='springdamper=".3 .7"' if automatic else 'damping=".04"'
    source.write_text(f'''<mujoco><default><joint {default}/></default><worldbody>
    <body><joint name="j_001" type="{joint_type}" stiffness="100"/><geom size=".01"/>
      <body pos="0 0 .1"><joint name="j_002" type="{joint_type}"/><geom size=".01"/>
        <body pos="0 0 .1"><joint name="j_003" type="{joint_type}"/><geom size=".01"/>
          <body pos="0 0 .1"><joint name="j_004" type="{joint_type}"/><geom size=".01"/></body>
        </body></body></body></worldbody><actuator><motor joint="j_003" gear="2"/></actuator></mujoco>''')
    before=source.read_bytes(); a=mujoco.MjModel.from_xml_path(str(source))
    report=update_gains(source,tmp_path/'new.xml',k2=.3,rho_k=.8,exclude=['j_002'])
    b=mujoco.MjModel.from_xml_path(report['output'])
    assert source.read_bytes()==before
    assert report['reference_joint']=='j_002' and report['damping_preserved']
    assert b.joint('j_003').stiffness[0]==pytest.approx(.3*.8)
    assert b.joint('j_004').stiffness[0]==pytest.approx(.3*.8**2)
    for j in ('j_001','j_002'):
        np.testing.assert_array_equal(a.joint(j).stiffness,b.joint(j).stiffness)
    # All compiled arrays except the intended stiffness change (and solver bookkeeping)
    for field in ('dof_damping','body_mass','body_inertia','body_pos','body_quat','body_ipos','body_iquat',
                  'geom_size','geom_pos','geom_quat','jnt_pos','jnt_axis','actuator_gainprm',
                  'actuator_biasprm','actuator_dynprm','actuator_gear','actuator_trnid','actuator_ctrlrange'):
        np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
    alias=update_gains(source,tmp_path/'alias.xml',.3,None,1.03,anchor='first-flexible',exclude=['j_002'])
    assert alias['reference_joint']=='j_002'
    assert alias['joints'][2]['new_stiffness']==pytest.approx(.3*RHO)


def test_independent_damping_and_missing_reference(tmp_path):
    source=tmp_path/'a.xml'
    source.write_text('<mujoco><worldbody>'+''.join(f'<body pos="{i} 0 0"><joint name="j_{i:03d}"/><geom size=".1"/></body>' for i in (1,2,3))+'</worldbody></mujoco>')
    legacy=update_gains(source,tmp_path/'legacy.xml',.2,.01,1.03)
    converted=update_gains(source,tmp_path/'converted.xml',k2=.2*RHO,rho_k=RHO,d2=.01*RHO,rho_d=RHO)
    a=mujoco.MjModel.from_xml_path(legacy['output'])
    b=mujoco.MjModel.from_xml_path(converted['output'])
    np.testing.assert_allclose(a.jnt_stiffness,b.jnt_stiffness,rtol=1e-14)
    np.testing.assert_allclose(a.dof_damping,b.dof_damping,rtol=1e-14)
    report=update_gains(source,tmp_path/'b.xml',k2=.2,rho_k=.8,d2=.05,rho_d=.99)
    assert report['joints'][2]['new_damping']==pytest.approx([.0495])
    with pytest.raises(ValueError,match='Damping updates require'):
        update_gains(source,tmp_path/'bad.xml',k2=.2,rho_k=.8,d2=.05)
    with pytest.raises(ValueError,match='Reference joint not found'):
        update_gains(source,tmp_path/'bad.xml',.2,None,1.03,reference_joint='j_005')


def test_generated_xml_report_and_website_match(tmp_path):
    p=json.loads((ROOT/'examples/params-two-cable.json').read_text())
    p['L']=.035
    p['post_gen'].update(joint_stiffness_base=.2,joint_damping_base=.01,joint_beta=1.03)
    models=[]
    for label in ('legacy','canonical'):
        if label=='canonical':p['post_gen'].update(k2=.2*RHO,rho_k=RHO,d2=.01*RHO,rho_d=RHO)
        normalize_params(p)
        path=tmp_path/f'{label}.json';path.write_text(json.dumps(p))
        out=tmp_path/label
        proc=subprocess.run([sys.executable,'build.py','--params',str(path),'--output-dir',str(out),'--no-preview'],cwd=ROOT,capture_output=True,text=True)
        assert proc.returncode==0,proc.stdout+proc.stderr
        m=mujoco.MjModel.from_xml_path(str(out/'spirob_physics_model.xml'));models.append(m)
        js=browser(p,m.njnt)
        np.testing.assert_allclose(m.jnt_stiffness,[v['k'] for v in js['gains']],rtol=1e-14)
        for j,row in enumerate(js['gains']):assert m.dof_damping[m.jnt_dofadr[j]]==pytest.approx(row['d'])
        md=next((out/'design_report').glob('*.md')).read_text()
        assert 'Fixed reference j_002' in md and 'Joint law rho_k' in md
    for field in ('jnt_stiffness','dof_damping','body_mass','body_inertia','geom_size','actuator_gear'):
        np.testing.assert_allclose(getattr(models[0],field),getattr(models[1],field),rtol=1e-14)


@pytest.mark.parametrize('key,value',[('k2',-1),('rho_k',0),('rho_d',True),('d2',float('nan'))])
def test_invalid_canonical_values(key,value):
    p=json.loads((ROOT/'params.json').read_text());p['post_gen'][key]=value
    with pytest.raises(ValueError):normalize_params(p)
    with pytest.raises(ValueError):resolve_gains(p['post_gen'],3)
