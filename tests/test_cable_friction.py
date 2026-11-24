"""Tests for spirob.cable_friction (stick-slip capstan cable friction)."""
from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from spirob.cable_friction import CableFriction, capstan_profile, stick_slip_chain


def _chain_xml(n_links: int = 5, offset: float = 0.006, length: float = 0.02) -> str:
    """Planar chain of hinged links, two holes per link, one cable each side."""
    bodies, close = [], []
    for i in range(n_links):
        pos = "0 0 0" if i == 0 else f"0 0 {length}"
        joint = "" if i == 0 else (
            '<joint name="j%d" type="hinge" axis="0 1 0" stiffness="0.05" damping="0.002" '
            'range="-0.5 0.5" limited="true"/>' % i)
        sites = "".join(
            f'<site name="c{c}_{i}_s{k}" pos="{sgn * offset} 0 {z}"/>'
            for c, sgn in ((0, -1), (1, 1)) for k, z in ((1, 0.004), (2, 0.016)))
        bodies.append(f'<body name="l{i}" pos="{pos}">{joint}'
                      f'<geom type="box" size="0.008 0.004 0.01" pos="0 0 0.01" contype="0" conaffinity="0"/>'
                      f'{sites}')
        close.append("</body>")
    tendons = "".join(
        f'<spatial name="cable_{c}">' + "".join(
            f'<site site="c{c}_{i}_s{k}"/>' for i in range(n_links - 1, -1, -1) for k in (2, 1)) + "</spatial>"
        for c in (0, 1))
    return (f'<mujoco><compiler angle="radian"/><option gravity="0 0 0" timestep="5e-4"/><worldbody>{"".join(bodies)}'
            f'{"".join(close)}</worldbody><tendon>{tendons}</tendon></mujoco>')


@pytest.fixture()
def model():
    m = mujoco.MjModel.from_xml_string(_chain_xml())
    return m, mujoco.MjData(m)


def test_route_is_ordered_base_to_tip(model):
    m, _ = model
    cf = CableFriction(m)
    r = cf.routes[0]
    depth = [m.site_bodyid[s] for s in r.site_ids]
    assert depth == sorted(depth)
    assert len(r.seg_dof) == 2 * (m.nbody - 2)
    assert np.all(r.seg_dof[0::2] >= 0) and np.all(r.seg_dof[1::2] < 0)


def test_zero_friction_equals_mujoco_tendon(model):
    m, d = model
    cf = CableFriction(m, mu=0.0)
    d.qpos[:] = np.linspace(0.1, 0.4, m.nq)
    mujoco.mj_forward(m, d)
    f = cf.generalized_force(d, [0.0, 3.0])
    J = d.ten_J.reshape(m.ntendon, m.nv)
    np.testing.assert_allclose(f, -3.0 * J[1], atol=1e-12)


def test_pull_from_rest_is_capstan(model):
    m, d = model
    cf = CableFriction(m, mu=0.3)
    d.qpos[:] = 0.3
    mujoco.mj_forward(m, d)
    t = cf.update(d, [0.0, 5.0])[1]
    alpha = cf.contact_angles(d, cf.routes[1])
    np.testing.assert_allclose(t, capstan_profile(5.0, alpha, 0.3), rtol=1e-12)
    assert np.all(np.diff(t) <= 1e-12)       # falls toward the tip


def test_hold_and_release_memory(model):
    m, d = model
    cf = CableFriction(m, mu=0.3)
    d.qpos[:] = 0.3
    mujoco.mj_forward(m, d)
    pulled = cf.update(d, [0.0, 5.0])[1].copy()
    held = cf.update(d, [0.0, 5.0])[1]
    np.testing.assert_allclose(held, pulled)            # holding changes nothing
    rel = cf.update(d, [0.0, 2.0])[1]
    alpha = cf.contact_angles(d, cf.routes[1])
    # release: tension may now rise toward the tip, never above the old value
    assert rel[-1] > 2.0 * math.exp(-0.3 * alpha.sum())
    assert np.all(rel <= pulled + 1e-12)
    assert np.all(rel <= 2.0 * np.exp(0.3 * np.cumsum(alpha)) + 1e-12)


def test_chain_bounds_property():
    rng = np.random.default_rng(1)
    for _ in range(50):
        K = 12
        alpha = rng.uniform(0, 0.6, K)
        t_prev = rng.uniform(0, 10, K)
        t0 = rng.uniform(0, 10)
        t = stick_slip_chain(t_prev, t0, alpha, 0.25)
        prev = np.r_[t0, t[:-1]]
        assert np.all(t >= prev * np.exp(-0.25 * alpha) - 1e-12)
        assert np.all(t <= prev * np.exp(0.25 * alpha) + 1e-12)


def test_correction_mode_matches_direct(model):
    m, d = model
    xml = _chain_xml().replace("</mujoco>", '<actuator><motor tendon="cable_1" gear="1"/></actuator></mujoco>')
    m2 = mujoco.MjModel.from_xml_string(xml)
    d2 = mujoco.MjData(m2)
    d2.qpos[:] = 0.25
    d2.ctrl[0] = -4.0
    mujoco.mj_forward(m2, d2)
    cf2 = CableFriction(m2, mu=0.2)
    cf2.apply(d2, correction=True)
    total = d2.qfrc_applied + d2.qfrc_actuator
    d.qpos[:] = 0.25
    mujoco.mj_forward(m, d)
    cf = CableFriction(m, mu=0.2)
    direct = cf.generalized_force(d, [0.0, 4.0])
    np.testing.assert_allclose(total, direct, atol=1e-10)


def test_friction_bends_base_first(model):
    """With friction the base joint bends more than the tip joint under the
    same small pull; without it the (equally stiff) joints bend equally."""
    m, d = model
    out = {}
    for mu in (0.0, 0.4):
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        cf = CableFriction(m, mu=mu)
        for k in range(4000):
            cf.apply(d, tension=[0.0, 2.0 * min(1.0, k / 2000)])
            mujoco.mj_step(m, d)
        out[mu] = np.abs(d.qpos.copy())
    assert np.ptp(out[0.0]) < 0.05 * out[0.0].mean()
    assert out[0.4][0] > 1.2 * out[0.4][-1]


def test_torch_batch_matches_numpy(model):
    torch = pytest.importorskip("torch")
    from spirob.cable_friction import TorchCableFriction
    m, d = model
    cf = CableFriction(m, mu=0.25)
    tcf = TorchCableFriction(m, mu=0.25)
    sites = tcf.site_ids()
    dofs = tcf.joint_dofs
    forces_np, forces_t = [], []
    for q, t1 in ((0.2, 3.0), (0.35, 3.0), (0.35, 1.0)):     # pull, bend further, release
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        forces_np.append(cf.generalized_force(d, [0.5, t1])[dofs])
        J = d.ten_J.reshape(m.ntendon, m.nv)
        tid = [r.tendon_id for r in cf.routes]
        xs = torch.tensor(d.site_xpos[sites][None], dtype=torch.float64).repeat(3, 1, 1, 1)
        ft = tcf(xs, torch.tensor([[0.5, t1]] * 3, dtype=torch.float64),
                 torch.tensor(J[tid][:, dofs][None], dtype=torch.float64).repeat(3, 1, 1))
        forces_t.append(ft[1].numpy())
    np.testing.assert_allclose(np.array(forces_t), np.array(forces_np), rtol=1e-9, atol=1e-12)
