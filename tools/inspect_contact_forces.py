"""Report solved contact locations and wrenches, optionally near a body-local point.

Import contact_report(model, data, ...) into a controller after mj_forward or
mj_step1/mj_step2 as appropriate. Coordinates and forces must belong to the same
pipeline state. The CLI calls mj_forward after its final integration step.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np


def contact_report(model, data, body_name=None, point_local_m=None, radius_m=None):
    """Forces acting ON selected bodies; moments about the stated local point.

    A point and radius select a spherical region in body axes. This sums solved
    point contacts, not a continuous pressure distribution. Contact IDs are
    temporary indices and must not be tracked as material points across steps.
    """
    if (point_local_m is None) != (radius_m is None):
        raise ValueError('Provide both a local point and a radius')
    if point_local_m is not None and body_name is None:
        raise ValueError('A local point requires --body')
    point = np.zeros(3) if point_local_m is None else np.asarray(point_local_m, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError('Local point must have three finite coordinates in metres')
    if radius_m is not None and (not math.isfinite(radius_m) or radius_m <= 0):
        raise ValueError('Radius must be finite and positive')
    if body_name is not None:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid <= 0:
            raise ValueError(f'No movable link/object body named {body_name}')
        selected = [bid]
    else:
        selected = [b for b in range(1, model.nbody) if model.body(b).name.startswith('link_')]
    rows = {b: [] for b in selected}
    forces = {b: np.zeros(3) for b in selected}
    moments = {b: np.zeros(3) for b in selected}
    for cid, contact in enumerate(data.contact):
        if contact.efc_address < 0 or np.any(contact.geom < 0):
            continue  # No solved rigid-geom contact (or a flex contact).
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, cid, wrench)
        # Contact frame rows are world-axis directions; normal goes geom0->1.
        frame = contact.frame.reshape(3, 3)
        force_on_1 = frame.T @ wrench[:3]
        torque_on_1 = frame.T @ wrench[3:]
        for side, gid in enumerate(contact.geom):
            bid = int(model.geom_bodyid[gid])
            if bid not in rows:
                continue
            rotation = data.xmat[bid].reshape(3, 3)
            local = rotation.T @ (contact.pos-data.xpos[bid])
            if radius_m is not None and np.linalg.norm(local-point) > radius_m:
                continue
            sign = 1 if side == 1 else -1
            force = sign*force_on_1
            torque_at_contact = sign*torque_on_1
            origin_world = data.xpos[bid]+rotation @ point
            moment = torque_at_contact+np.cross(contact.pos-origin_world, force)
            other = int(contact.geom[1-side])
            rows[bid].append(dict(contact_id=cid, geom=model.geom(gid).name,
                other_geom=model.geom(other).name or f'geom#{other}',
                other_body=model.body(model.geom_bodyid[other]).name or 'world',
                position_world_m=contact.pos.tolist(), position_body_m=local.tolist(),
                distance_m=float(contact.dist), normal_force_N=float(wrench[0]),
                force_on_body_world_N=force.tolist(), force_on_body_local_N=(rotation.T @ force).tolist(),
                torque_at_contact_world_Nm=torque_at_contact.tolist()))
            forces[bid] += force
            moments[bid] += moment
    bodies = []
    for bid in selected:
        rotation = data.xmat[bid].reshape(3, 3)
        bodies.append(dict(body=model.body(bid).name, contacts=rows[bid], count=len(rows[bid]),
            reference_point_body_m=point.tolist(), region_radius_m=radius_m,
            net_force_world_N=forces[bid].tolist(), net_force_body_N=(rotation.T @ forces[bid]).tolist(),
            net_moment_world_Nm=moments[bid].tolist(), net_moment_body_Nm=(rotation.T @ moments[bid]).tolist()))
    return dict(time_s=float(data.time), ncon=int(data.ncon), scalar_constraints=int(data.nefc),
        convention='Forces act ON each named body; moments about reference_point_body_m. SI units.',
        scope='Solved rigid-geom contacts only; not joint reactions, cable loads, or a pressure map. '
              'Contact indices are transient. Self-contact appears once for each participating selected body.',
        bodies=bodies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mjcf', required=True, help='Input standalone MuJoCo XML, including its associated mesh assets')
    parser.add_argument('--body', help='One body, e.g. link_002; default: all SpiRob links')
    parser.add_argument('--point-local-m', nargs=3, type=float, help='Region centre in the selected body frame: x y z in metres')
    parser.add_argument('--radius-m', type=float, help='Spherical force aggregation region radius, in metres')
    parser.add_argument('--state-npz', help='Optional NumPy archive with qpos, qvel, ctrl arrays')
    parser.add_argument('--seconds', type=float, default=0., help='Run from initial/loaded state before reporting')
    parser.add_argument('--controls', nargs='+', type=float, help='Target controls in XML actuator order')
    parser.add_argument('--ramp-seconds', type=float, default=0., help='Duration of a linear control ramp, in simulated seconds')
    parser.add_argument('--json', required=True, help='Write the numeric inspection report to this JSON file')
    args = parser.parse_args()
    try:
        model = mujoco.MjModel.from_xml_path(str(Path(args.mjcf).resolve()))
        data = mujoco.MjData(model)
        if args.state_npz:
            with np.load(args.state_npz, allow_pickle=False) as state:
                for key in ('qpos', 'qvel', 'ctrl'):
                    if key not in state or state[key].shape != getattr(data, key).shape or not np.all(np.isfinite(state[key])):
                        raise ValueError(f'State requires finite {key} with shape {getattr(data, key).shape}')
                    getattr(data, key)[:] = state[key]
        target = data.ctrl.copy() if args.controls is None else np.asarray(args.controls)
        if target.shape != (model.nu,) or not np.all(np.isfinite(target)):
            raise ValueError(f'Expected {model.nu} finite controls')
        limited = model.actuator_ctrllimited.astype(bool)
        if np.any(target[limited] < model.actuator_ctrlrange[limited, 0]) or np.any(target[limited] > model.actuator_ctrlrange[limited, 1]):
            raise ValueError('Controls exceed the XML actuator ctrlrange')
        if not math.isfinite(args.seconds) or not math.isfinite(args.ramp_seconds) or not 0 <= args.ramp_seconds <= args.seconds:
            raise ValueError('Require 0 <= ramp-seconds <= seconds')
        steps = math.ceil(args.seconds/model.opt.timestep)
        if steps > 2_000_000:
            raise ValueError('Limit is two million steps; shorten --seconds')
        # Validate selection before running an expensive simulation.
        contact_report(model, data, args.body, args.point_local_m, args.radius_m)
        start_ctrl = data.ctrl.copy()
        for i in range(steps):
            fraction = 1. if args.ramp_seconds == 0 else min((i+1)*model.opt.timestep/args.ramp_seconds, 1.)
            data.ctrl[:] = start_ctrl+(target-start_ctrl)*fraction
            mujoco.mj_step(model, data)
            if np.any(data.warning.number) or not np.all(np.isfinite(data.qpos)):
                raise ValueError('Simulation warning/nonfinite state; no force report written')
        data.ctrl[:] = target
        mujoco.mj_forward(model, data)
        if np.any(data.warning.number) or not np.all(np.isfinite(data.qacc)):
            raise ValueError('Invalid final dynamics; no force report written')
        report = contact_report(model, data, args.body, args.point_local_m, args.radius_m)
        dest = Path(args.json); dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2)+'\n')
        for body in report['bodies']:
            print(f"{body['body']}: {body['count']} contacts; net world force {body['net_force_world_N']} N")
        print(f'Saved {dest}')
    except (ValueError, OSError, mujoco.FatalError) as exc:
        parser.exit(1, f'Contact inspection failed: {exc}\n')


if __name__ == '__main__':
    main()
