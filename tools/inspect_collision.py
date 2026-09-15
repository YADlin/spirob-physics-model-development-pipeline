"""Inspect the delivered XML's CAD/proxy fit at a frozen pose; no frame rebasing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def collision_report(xml_path, model):
    root = ET.parse(xml_path).getroot()
    bodies = [b for b in root.findall('.//body') if b.get('name', '').startswith('link_')]
    proxies = [g for g in root.findall('.//geom') if g.get('name', '').startswith('collision_')]
    assets = root.findall('./asset/mesh')
    return dict(links=len(bodies), mesh_assets=len(assets),
                source_stl_files=sorted({m.get('file') for m in assets}),
                collision_primitives=len(proxies),
                explicit_inertias=len([b for b in bodies if b.find('inertial') is not None]),
                inertia_from_geom=root.find('compiler').get('inertiafromgeom'),
                all_proxies_zero_mass=all(float(g.get('mass', '-1')) == 0 for g in proxies),
                total_robot_mass_kg=float(sum(model.body(b.get('name')).mass[0] for b in bodies)),
                max_contact_margin_m=float(np.max(model.geom_margin)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mjcf', required=True)
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--json', help='Write the inspection report to this path')
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(Path(args.mjcf).resolve()))
    report = collision_report(args.mjcf, model)
    print(json.dumps(report, indent=2))
    if args.json:
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


if __name__ == '__main__':
    main()
