"""Add cable-centred touch sensing regions without adding collision geometry.

Each spherical region reports the sum of normal contact forces on its parent
body inside the region. Overlapping regions can count the same contact; they
are not independent force components or a calibrated tactile skin.
"""
import argparse
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from spirob.xml_tools import read_standalone, write_validated


def add_touch_sensors(source, output, *, links=None, offset_m=.0018, radius_m=None, overwrite=False):
    if not math.isfinite(offset_m) or offset_m < 0:
        raise ValueError('offset must be finite and nonnegative')
    if radius_m is not None and (not math.isfinite(radius_m) or radius_m <= 0):
        raise ValueError('radius must be finite and positive')
    tree = read_standalone(source)
    root = tree.getroot()
    original = mujoco.MjModel.from_xml_path(str(Path(source).resolve()))
    sensor = root.find('sensor')
    if sensor is None:
        sensor = ET.SubElement(root, 'sensor')
    existing = {e.get('name') for e in root.iter() if e.get('name')}
    available, count = set(), 0
    for body in root.findall('.//worldbody//body'):
        match = re.fullmatch(r'link_(\d+)', body.get('name', ''))
        if not match:
            continue
        number = int(match[1]); available.add(number)
        if links is not None and number not in links:
            continue
        for start in list(body.findall('site')):
            match = re.fullmatch(r'c(\d+)_(\d+)_s1', start.get('name', ''))
            if not match:
                continue
            cable = int(match[1])
            end = body.find(f'site[@name="c{cable}_{number:03d}_s2"]')
            if end is None:
                raise ValueError(f'Missing distal route site for link {number}, cable {cable}')
            p, q = (np.fromstring(s.get('pos', '0 0 0'), sep=' ') for s in (start, end))
            centre = (p+q)/2
            radial = np.linalg.norm(centre[:2])
            if radial:
                centre[:2] *= (radial+offset_m)/radial
            radius = radius_m if radius_m is not None else np.linalg.norm(q-p)/2
            if radius <= 0:
                raise ValueError('Degenerate cable segment; supply a positive --radius-mm')
            site_name, sensor_name = f'cs_{number:03d}_c{cable}', f'touch_{number:03d}_c{cable}'
            if site_name in existing or sensor_name in existing:
                raise ValueError(f'{sensor_name} already exists; use the original XML as input')
            ET.SubElement(body, 'site', name=site_name, type='sphere',
                          pos=' '.join(format(x, '.17g') for x in centre),
                          size=format(radius, '.17g'), rgba='0.1 0.65 0.55 0.18', group='4')
            ET.SubElement(sensor, 'touch', name=sensor_name, site=site_name)
            count += 1
    if links is not None and not set(links) <= available:
        raise ValueError(f'Unknown link indices: {sorted(set(links)-available)}')
    if not count:
        raise ValueError('No SpiRob cable site pairs found')

    def verify(model):
        for field in ('body_mass','body_ipos','body_iquat','body_inertia','jnt_stiffness','dof_damping',
                      'geom_type','geom_size','geom_contype','geom_conaffinity'):
            np.testing.assert_array_equal(getattr(model,field), getattr(original,field))
        assert model.nsensor == original.nsensor + count
        for i in range(original.nsensor):
            assert model.sensor(i).name == original.sensor(i).name
    write_validated(tree, source, output, verify=verify, overwrite=overwrite)
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mjcf', required=True, help='Standalone input XML; its existing sensors are preserved')
    p.add_argument('--out', required=True, help='Separate output XML; asset paths are rebased automatically')
    p.add_argument('--links', type=int, nargs='+', help='Link numbers to instrument; omitted means every link')
    p.add_argument('--offset-mm', type=float, default=1.8, help='Radial outward offset from each cable midpoint')
    p.add_argument('--radius-mm', type=float, help='Sensing sphere radius; omitted uses half the cable-segment length')
    p.add_argument('--overwrite', action='store_true', help='Replace an existing output, never the source XML')
    a=p.parse_args()
    try:
        count=add_touch_sensors(a.mjcf,a.out,links=a.links,offset_m=a.offset_mm/1000,
                               radius_m=None if a.radius_mm is None else a.radius_mm/1000,overwrite=a.overwrite)
    except (ValueError,OSError) as exc:
        p.error(str(exc))
    print(f'Added {count} touch sensors; existing sensors, inertias and collision geometry preserved: {a.out}')


if __name__ == '__main__':
    main()
