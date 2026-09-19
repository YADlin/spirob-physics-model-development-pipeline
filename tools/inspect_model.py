"""Audit link frames; optionally inspect the model without advancing physics.

MuJoCo recentres meshes and rotates them to their principal inertia axes. Its
Geom frame display therefore need not align along a robot. Report those actual
axes separately from the authored mesh and body axes. Optionally align the
compiled Geom frames, display them, and save the aligned model as MJB.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spirob.mujoco_frames import aligned_geom_model

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
                'geom_axes_max_error_vs_body': float(np.max(np.abs(
                    data.geom_xmat[geom_id].reshape(3, 3) - body_rotation))),
                'compiled_geom_quat_wxyz': model.geom_quat[geom_id].tolist(),
            })
    if not rows:
        raise ValueError('No link_NNN mesh geoms found in this SpiRob model')
    aligned = all(row['body_axes_max_error_vs_base'] < 1e-9
                  and row['authored_mesh_axes_max_error_vs_body'] < 1e-9
                  and row['authored_mesh_origin_error_m'] < 1e-9 for row in rows)
    geom_aligned = all(row['geom_axes_max_error_vs_body'] < 1e-9 for row in rows)
    return {'links_checked': len({row['link'] for row in rows}), 'mesh_geoms_checked': len(rows), 'authored_rest_frames_aligned': aligned,
            'geom_axes_aligned_with_bodies': geom_aligned,
            'rest_frames_aligned': aligned and geom_aligned, 'links': rows}


def inspect_view(model, data, frames='geom'):
    """Display the selected real frames at a frozen pose; sliders update joints."""
    import mujoco.viewer
    print(f'Inspection view: {frames} frames, no time integration. Use joint sliders to inspect poses.')
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.opt.frame = (mujoco.mjtFrame.mjFRAME_GEOM if frames == 'geom'
                                else mujoco.mjtFrame.mjFRAME_BODY)
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.02)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mjcf', '--model', dest='model', required=True, help='Generated SpiRob XML or MJB')
    parser.add_argument('--view', action='store_true', help='Open the frozen inspection viewer')
    parser.add_argument('--frames', choices=['geom', 'body'], default='geom', help='Coordinate frames to display in the inspection viewer')
    parser.add_argument('--align-geom-frames', action='store_true',
                        help='Align actual compiled Geom axes with body axes, preserving mesh surfaces')
    parser.add_argument('--save-mjb', type=Path, help='Save the inspected compiled model as .mjb')
    parser.add_argument('--json', action='store_true', help='Print the full per-link audit as JSON')
    args = parser.parse_args()
    path = Path(args.model).resolve()
    if args.save_mjb and args.save_mjb.suffix.lower() != '.mjb':
        parser.error('--save-mjb must end in .mjb (XML recompilation resets geom axes)')
    if args.save_mjb and args.save_mjb.resolve() == path:
        parser.error('--save-mjb must differ from the input path')
    model = (mujoco.MjModel.from_binary_path(str(path)) if path.suffix.lower() == '.mjb'
             else mujoco.MjModel.from_xml_path(str(path)))
    if args.align_geom_frames:
        model = aligned_geom_model(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    report = frame_report(model, data)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Links checked: {report['links_checked']}")
        print(f"Authored mesh/body rest frames aligned: {report['authored_rest_frames_aligned']}")
        print(f"Actual Geom axes aligned with bodies: {report['geom_axes_aligned_with_bodies']}")
        if not report['geom_axes_aligned_with_bodies']:
            print('Use --align-geom-frames to align the compiled Geom axes; --json shows each link.')
    if args.save_mjb:
        args.save_mjb.parent.mkdir(parents=True, exist_ok=True)
        mujoco.mj_saveModel(model, str(args.save_mjb.resolve()), None)
        if not args.json:
            print(f'Saved compiled model: {args.save_mjb.resolve()}')
    if args.view:
        inspect_view(model, data, args.frames)
    return 0 if report['rest_frames_aligned'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
