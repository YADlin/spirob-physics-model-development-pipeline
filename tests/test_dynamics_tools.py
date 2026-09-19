"""Scientific references and preservation checks for the diagnostic/edit tools."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import cadquery as cq
import mujoco
import numpy as np
import pytest
import trimesh

from tools.audit_inertia import audit, body_inertia, shift_inertia
from tools.inspect_collision import stress_report
from tools.set_joint_gains import update_gains

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def compound(tmp_path_factory):
    folder = tmp_path_factory.mktemp('dynamics-tools')
    p = subprocess.run([sys.executable, str(ROOT/'build.py'), '--params', str(ROOT/'examples/params-two-cable.json'),
                        '--no-preview', '--thickness-profile', 'stepped', '--collision-mode', 'compound', '--output-dir', str(folder)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout+p.stderr
    return folder/'spirob_physics_model.xml'


def test_rotated_offset_box_against_analytic_tensor(tmp_path):
    size = np.array([.02, .03, .04])
    trimesh.creation.box(extents=size).export(tmp_path/'box.stl')
    density = 1200.
    mass = density*np.prod(size)
    # Rotate the geom 30 degrees about Z and offset its COM in the body frame.
    a = np.pi/6
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    exact = R @ np.diag(mass/12*(sum(size**2)-size**2)) @ R.T
    path = tmp_path/'box.xml'
    path.write_text(f'''<mujoco><asset><mesh name="box" file="box.stl" inertia="exact"/></asset>
      <worldbody><body name="link_001"><freejoint/>
      <geom name="gmesh_001" type="mesh" mesh="box" group="1" density="{density}"
        pos=".01 -.02 .03" quat="{np.cos(a/2)} 0 0 {np.sin(a/2)}"/>
      </body></worldbody></mujoco>''')
    report = audit(path, density=density)['links'][0]
    np.testing.assert_allclose(report['mujoco']['inertia_about_com_body_axes_kg_m2'], exact, rtol=2e-6, atol=1e-13)
    np.testing.assert_allclose(report['mujoco']['com_in_body_m'], [.01, -.02, .03], atol=1e-10)
    assert report['references']['mesh']['inertia_relative_frobenius_error'] < 1e-6
    assert abs(exact[0, 1]) > 1e-8  # Off-diagonal sign/orientation matters here.
    # Deliberately corrupt the explicit mass/tensor; the audit must detect it.
    root = ET.parse(path).getroot()
    principal = mujoco.MjModel.from_xml_path(str(path)).body('link_001')
    ET.SubElement(root.find('.//body'), 'inertial', pos='.01 -.02 .03', mass=str(mass*2),
                  quat=' '.join(map(str, principal.iquat)),
                  diaginertia=' '.join(map(str, principal.inertia*2)))
    ET.ElementTree(root).write(path)
    bad = audit(path, density=density)['links'][0]['references']['mesh']
    assert bad['mass_relative_error'] == pytest.approx(1, abs=1e-6)
    assert bad['inertia_relative_frobenius_error'] == pytest.approx(1, abs=1e-6)


def test_cad_inertia_mm_to_si_and_parallel_axis():
    shape = cq.Workplane().box(20, 30, 40).translate((100, -200, 300)).val()
    density = 1200.
    mass = shape.Volume()*1e-9*density
    tensor = np.array(cq.Shape.matrixOfInertia(shape))*density*1e-15
    exact = np.diag(mass/12*np.array([.03**2+.04**2, .02**2+.04**2, .02**2+.03**2]))
    np.testing.assert_allclose(tensor, exact, rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(shift_inertia(tensor, 2., [1, 0, 0]), tensor+np.diag([0., 2., 2.]))


def test_two_cable_cad_reference_and_read_only(compound):
    content = compound.read_bytes()
    params = json.loads((compound.parent/'build_params.json').read_text())
    report = audit(compound, params=params, links=['link_001', 'link_002', 'link_021'])
    assert compound.read_bytes() == content
    for link in report['links']:
        assert abs(link['references']['cad']['mass_relative_error']) < 1e-6
        assert link['references']['cad']['inertia_relative_frobenius_error'] < 1e-6


@pytest.mark.parametrize('joint_type', ['hinge', 'ball'])
@pytest.mark.parametrize('anchor', ['generator', 'first-flexible'])
def test_update_gains_preserves_base_inertia_and_contacts(compound, tmp_path, joint_type, anchor):
    source = compound
    if joint_type == 'ball':
        root = ET.parse(compound).getroot()
        for mesh in root.findall('./asset/mesh'):
            mesh.set('file', str(compound.parent/mesh.get('file')))
        for joint in root.findall('.//worldbody//joint'):
            joint.set('type', 'ball')
            joint.set('limited', 'false')
            for attr in ('axis', 'range'):
                joint.attrib.pop(attr, None)
        source = tmp_path/'ball.xml'
        ET.ElementTree(root).write(source)
    before = source.read_bytes()
    report = update_gains(source, tmp_path/'relocated'/'new.xml', .4, .02, 1.05, anchor=anchor)
    model = mujoco.MjModel.from_xml_path(report['output'])
    original = mujoco.MjModel.from_xml_path(str(source))
    assert source.read_bytes() == before
    assert model.joint('j_001').stiffness[0] == 100
    assert model.dof_damping[0] == 50
    exponent_origin = 1 if anchor == 'generator' else 2
    for number in (2, 11, 21):
        joint = model.joint(f'j_{number:03d}')
        factor = 1.05**(-3*(number-exponent_origin))
        assert joint.stiffness[0] == pytest.approx(.4*factor)
        adr = model.jnt_dofadr[joint.id]
        np.testing.assert_allclose(model.dof_damping[adr:adr+(3 if joint_type == 'ball' else 1)], .02*factor)
    np.testing.assert_array_equal(model.body_mass, original.body_mass)
    np.testing.assert_array_equal(model.body_inertia, original.body_inertia)
    for field in ('ngeom', 'nq', 'nv', 'nu', 'ntendon'):
        assert getattr(model, field) == getattr(original, field)


def test_memory_only_edit_and_dense_contact_allocation(compound, tmp_path):
    target = tmp_path/'memory.xml'
    update_gains(compound, target, .2, .01, 1.03, arena_memory_mib=128)
    model = mujoco.MjModel.from_xml_path(str(target))
    original = mujoco.MjModel.from_xml_path(str(compound))
    assert model.narena == 128*2**20
    np.testing.assert_allclose(model.jnt_stiffness, original.jnt_stiffness, rtol=1e-13)
    np.testing.assert_allclose(model.dof_damping, original.dof_damping, rtol=1e-13)
    # Reproduced initial-pose, -10 N motor-0 run just before its old 14 MiB
    # arena overflow. A dense-contact fixture, not a physically settled pose.
    data = mujoco.MjData(model)
    data.qpos[:] = json.loads((ROOT/'tests/data/dense_contact_pose.json').read_text())['qpos']
    mujoco.mj_forward(model, data)
    assert data.ncon > 4000 and data.nefc > 16000
    assert not np.any(data.warning.number)
    assert data.maxuse_arena < model.narena


def test_invalid_edit_does_not_publish(compound, tmp_path):
    for bad in (-1., float('nan')):
        with pytest.raises(ValueError):
            update_gains(compound, tmp_path/'bad.xml', bad, .01, 1.03)
        assert not (tmp_path/'bad.xml').exists()
    with pytest.raises(ValueError, match='original'):
        update_gains(compound, compound, .2, .01, 1.03)
    with pytest.raises(ValueError, match='not found'):
        update_gains(compound, tmp_path/'bad.xml', .2, .01, 1.03, exclude=['wrong_joint'])


def test_additional_exclusions_never_unprotect_base(compound, tmp_path):
    report = update_gains(compound, tmp_path/'gains.xml', .4, .02, 1.03, exclude=['j_010'])
    protected = {row['joint']: row for row in report['joints'] if row['protected']}
    assert set(protected) == {'j_001', 'j_010'}
    for row in protected.values():
        assert row['old_stiffness'] == row['new_stiffness']
        assert row['old_damping'] == row['new_damping']


def test_bounded_stress_reports_and_rejects_unclamped_targets(compound):
    model = mujoco.MjModel.from_xml_path(str(compound))
    report = stress_report(model, [0., 0.], seconds=.05, ramp_seconds=0.)
    assert report['status'] == 'passed'
    assert report['completed_steps'] == 500
    assert report['arena_allocated_bytes'] == 128*2**20
    with pytest.raises(ValueError, match='ctrlrange'):
        stress_report(model, [-100, 0])
