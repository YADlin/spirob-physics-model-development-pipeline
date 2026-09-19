"""Validate contact-force signs, frames, moments and spatial filtering."""
import mujoco
import numpy as np
import pytest
from tools.inspect_contact_forces import contact_report


@pytest.fixture
def supported_sphere():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep=".0001" integrator="implicit"/>
      <worldbody><geom name="floor" type="plane" size="1 1 .1" condim="6"/>
      <body name="link_001" pos="0 0 .0099" quat=".7071067811865476 0 0 .7071067811865476">
      <freejoint/><geom name="ball" type="sphere" pos=".003 0 0" size=".01" mass=".1" condim="6"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    for _ in range(6000): mujoco.mj_step(model,data)
    mujoco.mj_forward(model,data)
    assert not np.any(data.warning.number)
    return model,data


def test_weight_balance_local_frame_and_moment(supported_sphere):
    model,data = supported_sphere
    body = contact_report(model,data)['bodies'][0]
    assert body['count'] == 1
    np.testing.assert_allclose(body['net_force_world_N'],[0,0,.981],atol=1e-5)
    contact = body['contacts'][0]
    bid = model.body('link_001').id; rotation = data.xmat[bid].reshape(3,3)
    local = np.asarray(contact['position_body_m']); world = np.asarray(contact['position_world_m'])
    np.testing.assert_allclose(rotation @ local+data.xpos[bid],world,atol=1e-14)
    force = np.asarray(body['net_force_world_N'])
    np.testing.assert_allclose(rotation @ body['net_force_body_N'],force,atol=1e-14)
    expected = np.cross(world-data.xpos[bid],force)+contact['torque_at_contact_world_Nm']
    np.testing.assert_allclose(body['net_moment_world_Nm'],expected,atol=1e-14)
    assert np.linalg.norm(expected) > .002
    near = contact_report(model,data,'link_001',local,.001)['bodies'][0]
    far = contact_report(model,data,'link_001',[1,1,1],.001)['bodies'][0]
    assert near['count'] == 1 and far['count'] == 0
    np.testing.assert_allclose(near['net_moment_world_Nm'],contact['torque_at_contact_world_Nm'],atol=1e-14)


def test_equal_opposite_forces_on_contact_pair():
    model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
      <worldbody><body name="link_001"><freejoint/><geom type="sphere" size=".01" mass=".1"/></body>
      <body name="link_002" pos=".0199 0 0"><freejoint/><geom type="sphere" size=".01" mass=".1"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model); mujoco.mj_forward(model,data)
    a,b = contact_report(model,data)['bodies']
    assert a['net_force_world_N'][0] < 0 < b['net_force_world_N'][0]
    np.testing.assert_allclose(np.array(a['net_force_world_N'])+b['net_force_world_N'],0,atol=1e-14)


@pytest.mark.parametrize('body,point,radius',[(None,[0,0,0],.01),('link_001',[0,0,0],None),
    ('link_001',[0,0,float('nan')],.01),('link_001',[0,0,0],-1),('missing',None,None)])
def test_invalid_selection_rejected(body,point,radius):
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="link_001"/></worldbody></mujoco>')
    with pytest.raises(ValueError): contact_report(model,mujoco.MjData(model),body,point,radius)
