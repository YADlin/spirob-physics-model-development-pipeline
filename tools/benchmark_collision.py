"""Compare collider counts on the SAME CAD and three controlled contact patches.

The plane, box and link-pair probes isolate one selected link. These are static
contact/constraint counts at a small prescribed overlap, not stability tests.
Use inspect_collision.py --stress separately for bounded dynamic checks.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.inspect_section import compiled_surface
from tools.inspect_collision import collision_report


def isolated_patch(xml_path, body_name, kind, penetration_m=0.00005):
    """Compile two opposing +Y/-Y faces, or one +Y face against a plane."""
    path = Path(xml_path).resolve()
    original_model = mujoco.MjModel.from_xml_path(str(path))
    vertices, _ = compiled_surface(original_model, body_name)
    upper, lower = vertices[:, 1].max()/1000, vertices[:, 1].min()/1000
    source = ET.parse(path).getroot()
    root = ET.Element('mujoco', model='collision_patch_probe')
    for tag in ('compiler', 'option', 'size', 'default', 'asset'):
        element = source.find(tag)
        if element is not None:
            root.append(copy.deepcopy(element))
    compiler = root.find('compiler')
    meshdir = compiler.attrib.pop('meshdir', '')
    for asset in root.findall('./asset/mesh'):
        if asset.get('file'):
            asset.set('file', str((path.parent/meshdir/asset.get('file')).resolve()))
    source_body = source.find(f'.//body[@name="{body_name}"]')
    if source_body is None or source_body.find('inertial') is None:
        raise ValueError('Requires a generated body with explicit inertia')
    world = ET.SubElement(root, 'worldbody')
    for index in range(2 if kind == 'pair' else 1):
        y = index*(upper-lower-penetration_m)
        body = ET.SubElement(world, 'body', name=f'probe_link_{index}', pos=f'0 {y} 0')
        ET.SubElement(body, 'freejoint')
        for element in source_body:
            if element.tag in ('geom', 'inertial'):
                element = copy.deepcopy(element)
                if element.get('name'):
                    element.set('name', element.get('name')+f'_probe{index}')
                body.append(element)
    if kind == 'plane':
        ET.SubElement(world, 'geom', name='probe_plane', type='plane', size='.1 .1 .01',
                      pos=f'0 {upper-penetration_m} 0', quat='.7071067811865476 .7071067811865476 0 0', margin='0')
    elif kind == 'box':
        ET.SubElement(world, 'geom', name='probe_box', type='box', size='.1 .01 .1',
                      pos=f'0 {upper-penetration_m+.01} 0', margin='0')
    elif kind != 'pair':
        raise ValueError('Probe kind must be plane, box or pair')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def benchmark(baseline, candidate, body_name='link_002'):
    models = [mujoco.MjModel.from_xml_path(str(Path(p).resolve())) for p in (baseline, candidate)]
    # A different shape or different inertia is not a controlled comparison.
    for attr in ('nbody', 'nq', 'nv'):
        if getattr(models[0], attr) != getattr(models[1], attr):
            raise ValueError('Comparison requires the same robot topology')
    for bid in range(1, models[0].nbody):
        name = models[0].body(bid).name
        a, _ = compiled_surface(models[0], name); b, _ = compiled_surface(models[1], name)
        if a.shape != b.shape or not np.allclose(a, b, rtol=1e-10, atol=1e-7):
            raise ValueError(f'{name}: comparison requires identical CAD meshes')
    for attr in ('body_mass', 'body_ipos', 'body_inertia', 'body_iquat', 'body_pos', 'body_quat'):
        if not np.allclose(getattr(models[0], attr), getattr(models[1], attr), rtol=1e-12, atol=1e-15):
            raise ValueError(f'Comparison requires identical {attr}')
    report = dict(body=body_name, penetration_m=.00005,
                  scope='Controlled static patches using identical CAD/inertias; not a dynamic stress test', cases={})
    for label, path, model in zip(('compound', 'candidate'), (baseline, candidate), models):
        entry = dict(model=collision_report(path, model), probes={})
        for kind in ('plane', 'box', 'pair'):
            probe = isolated_patch(path, body_name, kind)
            data = mujoco.MjData(probe)
            mujoco.mj_forward(probe, data)
            entry['probes'][kind] = dict(contacts=int(data.ncon), scalar_constraints=int(data.nefc),
                arena_bytes=int(data.maxuse_arena), warnings=data.warning.number.tolist())
        report['cases'][label] = entry
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, help='Baseline XML to compare, normally the compound model')
    parser.add_argument('--candidate', required=True, help='Candidate XML to compare, normally the convex model')
    parser.add_argument('--body', default='link_002', help='Link body name to inspect, for example link_002')
    parser.add_argument('--json', required=True, help='Write the numeric inspection report to this JSON file')
    args = parser.parse_args()
    try:
        report = benchmark(args.baseline, args.candidate, args.body)
        dest = Path(args.json); dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2)+'\n')
        for label, entry in report['cases'].items():
            print(f"{label}: {entry['model']['total_link_contact_geoms']} robot contact geoms; {entry['probes']}")
        print(f'Saved {dest}')
    except (ValueError, OSError, mujoco.FatalError) as exc:
        parser.exit(1, f'Benchmark failed: {exc}\n')


if __name__ == '__main__':
    main()
