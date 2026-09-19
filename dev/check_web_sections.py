"""Check sampled browser slice boundaries against OpenCascade solids.

Run: uv run python dev/check_web_sections.py (Node required).
Tests base, complete and tip links at three slice stations, including a zero
notch profile. These probes validate the preview, not a manufacturing tolerance.
"""
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd

from spirob.csv_io import generate_cable_sites_csv_zrot_from_P
from spirob.geometry import from_params
from spirob.pipeline.csv2geom_nlobe import build_unit_inputs
from spirob.pipeline.cad_export import _simulation_element_mm

ROOT = Path(__file__).resolve().parents[1]


def main():
    cases = [json.loads((ROOT/'examples'/name).read_text()) for name in
             ['params-two-cable-hex.json', 'params-three-cable.json', 'params-four-cable.json']]
    cases[-1]['notch_factor'] = 0
    script = """import {derive,section} from './site/geometry.mjs';import fs from 'node:fs';
console.log(JSON.stringify(JSON.parse(fs.readFileSync(0,'utf8')).map(p=>{const g=derive(p);return [0,1,g.units.length-1].flatMap(i=>[.25,.5,.75].map(f=>({i,...section(p,g,g.units[i],f,64)})));})));"""
    values = json.loads(subprocess.run(['node', '--input-type=module', '-e', script],
        input=json.dumps(cases), capture_output=True, text=True, check=True, cwd=ROOT).stdout)
    count = 0
    with tempfile.TemporaryDirectory() as td:
        for p, slices in zip(cases, values):
            geometry = from_params(p)
            csv = Path(td)/'geometry.csv'
            generate_cable_sites_csv_zrot_from_P(geometry.inverted_quads(), p['n_cables'], csv)
            units = build_unit_inputs(pd.read_csv(csv), geometry)
            shapes = {i: _simulation_element_mm(units[i], p) for i in [0, 1, len(units)-1]}
            for sl in slices:
                shape = shapes[sl['i']]
                for point in sl['points']:
                    position = np.array([*point, sl['z']])*1000
                    radial = np.array([*point, 0.]); radial /= np.linalg.norm(radial)
                    for delta, expected in [(-.003, True), (.003, False)]:
                        actual = shape.isInside(tuple(position+delta*radial), 1e-7)
                        assert actual == expected, (p['n_cables'], sl['i'], sl['z'], point, delta, actual)
                        count += 1
    print(f'{count} inside/outside CAD probes passed around browser sections (±0.003 mm).')


if __name__ == '__main__':
    main()
