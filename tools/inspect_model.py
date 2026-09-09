"""Audit link frames; optionally inspect the model without advancing physics.

MuJoCo recentres meshes and rotates them to their principal inertia axes. Its
Geom frame display therefore need not align along a robot. Body frames describe
the authored link/joint convention. Recover the authored mesh frame as well so
an actual placement error cannot be dismissed as compiler recentering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import mujoco
import numpy as np


def _rotation(quat):
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, quat)
    return result.reshape(3, 3)


def frame_report(model, data):
    """Report rest-pose link alignment and undo the mesh compiler transform."""
    rows = []
    reference = None
    for body_id in range(1, model.nbody):
        name = model.body(body_id).name
        if not re.fullmatch(r'link_\d+', name):
            continue
        body_rotation = data.xmat[body_id].reshape(3, 3)
        if reference is None:
            reference = body_rotation
        for geom_id in range(model.body_geomadr[body_id],
                             model.body_geomadr[body_id] + model.body_geomnum[body_id]):
            if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh_id = model.geom_dataid[geom_id]
            authored_rotation = _rotation(model.geom_quat[geom_id]) @ _rotation(
                model.mesh_quat[mesh_id]).T
            authored_position = model.geom_pos[geom_id] - authored_rotation @ model.mesh_pos[mesh_id]
            rows.append({
                'link': name,
                'body_axes_max_error_vs_base': float(np.max(np.abs(body_rotation - reference))),
                'authored_mesh_axes_max_error_vs_body': float(np.max(np.abs(authored_rotation - np.eye(3)))),
                'authored_mesh_origin_error_m': float(np.linalg.norm(authored_position)),
                'compiled_geom_quat_wxyz': model.geom_quat[geom_id].tolist(),
            })
    if not rows:
        raise ValueError('No link_NNN mesh geoms found in this SpiRob model')
    aligned = all(row['body_axes_max_error_vs_base'] < 1e-9
                  and row['authored_mesh_axes_max_error_vs_body'] < 1e-9
                  and row['authored_mesh_origin_error_m'] < 1e-9 for row in rows)
    return {'links_checked': len(rows), 'rest_frames_aligned': aligned, 'links': rows}


def inspect_view(model, data):
    """Display body axes at a frozen pose; joint sliders still update the pose."""
    import mujoco.viewer
    print('Inspection view: body frames, no time integration. Use joint sliders to inspect poses.')
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.02)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mjcf', required=True, help='Generated SpiRob XML')
    parser.add_argument('--view', action='store_true', help='Open the body-frame inspection viewer')
    parser.add_argument('--json', action='store_true', help='Print the full per-link audit as JSON')
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(Path(args.mjcf).resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    report = frame_report(model, data)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Links checked: {report['links_checked']}")
        print(f"Rest frames aligned: {report['rest_frames_aligned']}")
        if not report['rest_frames_aligned']:
            print('Unexpected authored/body frame placement; rerun with --json for per-link details.')
    if args.view:
        inspect_view(model, data)
    return 0 if report['rest_frames_aligned'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
