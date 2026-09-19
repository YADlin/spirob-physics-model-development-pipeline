"""Check browser equations against the canonical Python geometry (requires Node)."""
import copy
import json
from pathlib import Path
import subprocess
import numpy as np
from spirob.geometry import from_params
from spirob.sections import section_dimensions

ROOT=Path(__file__).resolve().parents[1]

def check():
    cases=[]
    for path in sorted((ROOT/'examples').glob('params*.json')):
        p=json.loads(path.read_text());cases.append(p)
        for changes in ({'terminal_unit_policy':'whole_units'}, {'phi_deg':9.2,'terminal_unit_policy':'whole_units'},
                        {'L':.3,'Delta_theta_deg':24.,'terminal_unit_policy':'whole_units'}):
            q=copy.deepcopy(p);q.update(changes);cases.append(q)
        if p['n_cables']==2:
            q=copy.deepcopy(p);q['base_thickness_m']=.018;cases.append(q)
            q=copy.deepcopy(p);q['thickness_profile']='stepped';cases.append(q)
    script="""import {derive,section} from './site/geometry.mjs';
import fs from 'node:fs';
const cases=JSON.parse(fs.readFileSync(0,'utf8'));
console.log(JSON.stringify(cases.map(p=>{const g=derive(p);return {...g,slices:g.units.map(u=>section(p,g,u,.5,256))};})));"""
    result=subprocess.run(['node','--input-type=module','-e',script],input=json.dumps(cases),text=True,
                          capture_output=True,cwd=ROOT,check=True)
    values=json.loads(result.stdout)
    for p,v in zip(cases,values):
        g=from_params(p)
        for key,expected in [('a',g.spiral.a_m),('b',g.spiral.b),('q',g.spiral.q0_rad),
                             ('beta',g.spiral.beta_nominal),('length',g.lengths.discrete_chord_length_m),
                             ('effective',g.lengths.effective_continuous_length_m)]:
            np.testing.assert_allclose(v[key],expected,rtol=2e-12,atol=1e-13,err_msg=key)
        np.testing.assert_allclose(v['quads'],g.inverted_quads(),rtol=0,atol=2e-13)
        for u,gu in zip(v['units'],g.units):
            for c in range(p['n_cables']):
                expected=[pt.routed_m for pt in g.tendon_path(c).points if pt.unit_index_base_to_tip==gu.index_base_to_tip]
                np.testing.assert_allclose(u['cable'][c],expected,atol=2e-13,rtol=0)
        if p['n_cables']==2:
            dimensions=section_dimensions(p,g)
            for u,d in zip(v['units'],dimensions['links']):
                np.testing.assert_allclose([u['t0'],u['t1']],[d['proximal_centre_thickness_m'],d['distal_centre_thickness_m']],rtol=0,atol=2e-13)
    print(f'{len(cases)} browser/Python configurations agree: spiral, quads, lengths, every cable route, thickness endpoints.')
    return cases,values

if __name__=='__main__':check()
