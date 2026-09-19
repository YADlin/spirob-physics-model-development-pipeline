"""Inspect the delivered XML's CAD/proxy fit at a frozen pose; no frame rebasing."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def collision_report(xml_path, model):
    root = ET.parse(xml_path).getroot()
    bodies = [b for b in root.findall('.//body') if b.get('name', '').startswith('link_')]
    proxies = [g for g in root.findall('.//geom') if g.get('name', '').startswith('collision_')]
    assets = root.findall('./asset/mesh')
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'target')
    per_body = {}
    for body in bodies:
        bid = model.body(body.get('name')).id
        ids = range(model.body_geomadr[bid], model.body_geomadr[bid]+model.body_geomnum[bid])
        per_body[body.get('name')] = sum(bool(model.geom_contype[g] or model.geom_conaffinity[g]) for g in ids)
    return dict(links=len(bodies), mesh_assets=len(assets),
                target_site_retained=target_id >= 0,
                target_marker_visible=bool(target_id >= 0 and model.site_rgba[target_id, 3] > 0),
                source_stl_files=sorted({m.get('file') for m in assets if m.get('file')}),
                collision_primitives=len(proxies),
                contact_geoms_per_link=per_body,
                total_link_contact_geoms=sum(per_body.values()),
                inline_convex_meshes=sum(m.get('vertex') is not None for m in assets),
                nativeccd=not bool(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD),
                multiccd=bool(model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_MULTICCD),
                explicit_inertias=len([b for b in bodies if b.find('inertial') is not None]),
                inertia_from_geom=root.find('compiler').get('inertiafromgeom'),
                all_proxies_zero_mass=all(float(g.get('mass', '-1')) == 0 for g in proxies),
                total_robot_mass_kg=float(sum(model.body(b.get('name')).mass[0] for b in bodies)),
                arena_memory_bytes=int(model.narena),
                max_contact_margin_m=float(np.max(model.geom_margin)))


def stress_report(model, controls, seconds=10., ramp_seconds=2.):
    """Run a bounded native test, stopping at the first warning or fatal error."""
    target = np.asarray(controls, dtype=float)
    if target.shape != (model.nu,) or not np.all(np.isfinite(target)):
        raise ValueError(f'Provide {model.nu} finite actuator controls')
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(ramp_seconds) or not 0 <= ramp_seconds <= seconds:
        raise ValueError('Require seconds > 0 and 0 <= ramp-seconds <= seconds')
    limited = model.actuator_ctrllimited.astype(bool)
    if np.any(target[limited] < model.actuator_ctrlrange[limited, 0]) or np.any(target[limited] > model.actuator_ctrlrange[limited, 1]):
        raise ValueError('Controls exceed the XML actuator ctrlrange')
    steps = math.ceil(seconds/model.opt.timestep)
    if steps > 2_000_000:
        raise ValueError('This check is limited to two million steps; shorten --seconds')
    data = mujoco.MjData(model)
    peak_contacts = peak_constraints = peak_arena = 0
    status, error, completed = 'passed', None, 0
    attempted_step_start = 0.
    warning_counts = np.zeros(len(data.warning), dtype=int)
    start = time.perf_counter()
    try:
        for i in range(steps):
            attempted_step_start = float(data.time)
            fraction = 1. if ramp_seconds == 0 else min((i+1)*model.opt.timestep/ramp_seconds, 1.)
            data.ctrl[:] = target*fraction
            mujoco.mj_step(model, data)
            completed = i+1
            peak_contacts = max(peak_contacts, data.ncon)
            peak_constraints = max(peak_constraints, data.nefc)
            peak_arena = max(peak_arena, int(data.maxuse_arena))
            warning_counts = np.maximum(warning_counts, data.warning.number)
            if np.any(warning_counts) or not all(np.all(np.isfinite(a)) for a in (data.qpos, data.qvel, data.qacc)):
                status, error = 'failed', 'MuJoCo warning or nonfinite state; stopped instead of continuing after reset/disabled constraints'
                break
    except mujoco.FatalError as exc:
        status, error = 'failed', str(exc)
        peak_contacts = max(peak_contacts, data.ncon)
        peak_constraints = max(peak_constraints, data.nefc)
        peak_arena = max(peak_arena, int(data.maxuse_arena))
        warning_counts = np.maximum(warning_counts, data.warning.number)
    warnings = {mujoco.mjtWarning(i).name: int(n) for i, n in enumerate(warning_counts) if n}
    return dict(status=status, error=error, controls=target.tolist(), requested_seconds=seconds,
                ramp_seconds=ramp_seconds, data_time_after_test_s=float(data.time), completed_steps=completed,
                attempted_step_start_time_s=attempted_step_start,
                nominal_completed_duration_s=completed*float(model.opt.timestep),
                timestep_s=float(model.opt.timestep), wall_seconds=time.perf_counter()-start,
                peak_contacts=peak_contacts, peak_scalar_constraints=peak_constraints,
                arena_allocated_bytes=int(model.narena), peak_arena_bytes=peak_arena,
                peak_arena_fraction=peak_arena/model.narena, warnings=warnings,
                scope='This control sequence from the initial pose only; not a proof of stability for all actuation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mjcf', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--view', action='store_true')
    mode.add_argument('--stress', action='store_true', help='Bounded headless actuator/contact/memory test')
    parser.add_argument('--controls', nargs='+', type=float)
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--ramp-seconds', type=float, default=2.)
    parser.add_argument('--json', help='Write the inspection report to this path')
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(Path(args.mjcf).resolve()))
    report = collision_report(args.mjcf, model)
    if args.stress:
        if args.controls is None:
            parser.error('--stress requires --controls, in actuator order')
        try:
            report['stress'] = stress_report(model, args.controls, args.seconds, args.ramp_seconds)
        except ValueError as exc:
            parser.error(str(exc))
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2)+'\n')
    if args.view:
        from mujoco import viewer as mjviewer
        data = mujoco.MjData(model)
        model.geom_rgba[model.geom_group == 1, 3] = 0.2
        model.geom_rgba[model.geom_group == 3] = [0.95, 0.45, 0.08, 0.7]
        mujoco.mj_forward(model, data)
        print('Blue: CAD. Orange: collision shapes. Toggle groups 1/3 to compare; joint sliders change pose.')
        with mjviewer.launch_passive(model, data) as viewer:
            with viewer.lock():
                viewer.opt.geomgroup[1] = 1
                viewer.opt.geomgroup[3] = 1
                viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
                viewer.cam.lookat[:] = np.mean(data.xpos[1:], axis=0)
                viewer.cam.distance = 0.45
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -10
            while viewer.is_running():
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.02)
    return 2 if args.stress and report['stress']['status'] != 'passed' else 0


if __name__ == '__main__':
    sys.exit(main())
