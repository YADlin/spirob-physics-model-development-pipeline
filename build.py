"""Build CSV -> simulation meshes -> MJCF, with optional fabrication CAD.

Every stage uses this interpreter and a fresh temporary output directory. Failed
builds leave the previous outputs in place. Simulation names and physics remain
owned by the existing generators.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def run_step(argv, desc, cwd):
    print(f'\n{desc}', flush=True)
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    subprocess.run([sys.executable, *map(str, argv)], cwd=cwd, env=env, check=True)


def verify_meshes(directory, geometry):
    import struct
    expected = {u.link_name+'.stl' for u in geometry.units}
    actual = {p.name for p in Path(directory).glob('*.stl')}
    if actual != expected:
        raise ValueError(f'Incomplete mesh set: missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}')
    for name in expected:
        data = (Path(directory)/name).read_bytes()
        if len(data) < 84:
            raise ValueError(f'{name}: incomplete binary STL')
        n = struct.unpack_from('<I', data, 80)[0]
        if n == 0 or len(data) != 84+50*n:
            raise ValueError(f'{name}: invalid binary STL triangle count')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--params',default='params.json')
    p.add_argument('--output-dir',default='.',help='Destination for generated outputs')
    p.add_argument('--noclean',action='store_true',help='Compatibility flag; builds always use fresh staging')
    p.add_argument('--no-preview',action='store_true')
    section = p.add_mutually_exclusive_group()
    section.add_argument('--plain',action='store_true')
    section.add_argument('--nlobe',action='store_true',help='Compatibility alias: use the n_cables cross-section')
    physics = p.add_mutually_exclusive_group()
    for name in ('safe','fast','high'): physics.add_argument('--'+name,action='store_true')
    p.add_argument('--cad',action='store_true')
    p.add_argument('--fuse-cad',action='store_true')
    p.add_argument('--cad-profile',choices=['fabrication','simulation'],default='fabrication')
    p.add_argument('--flat-thickness-m',type=float); p.add_argument('--flat-edge-ratio',type=float)
    p.add_argument('--neck-width-mm',type=float,default=1.0)
    p.add_argument('--cable-hole-diameter-mm',type=float,default=0.0)
    a=p.parse_args()
    if a.fuse_cad and not a.cad: p.error('--fuse-cad requires --cad')
    params_path = Path(a.params).resolve()
    params = json.loads(params_path.read_text(encoding='utf-8'))
    from spirob_csv_generator import validate_params
    from spirob.geometry import from_params
    validate_params(params)
    geometry = from_params(params)
    if not a.no_preview:
        preview=[ROOT/'preview.py','--params',params_path]
        if not a.plain and params['n_cables']>=3: preview.append('--nlobe')
        run_step(preview,'Geometry preview',ROOT)
    output = Path(a.output_dir).resolve(); output.mkdir(parents=True,exist_ok=True)
    # Stage alongside the destination so publishing uses same-filesystem renames.
    with tempfile.TemporaryDirectory(prefix='.spirob-build-',dir=output.parent) as td:
        stage=Path(td)
        run_step([ROOT/'spirob_csv_generator.py','--params',params_path,'--yes'],'Generate CSV',stage)
        csv=Path('Geom_Data_CSV/Spirob_geom_data.csv')
        mesh=[ROOT/'csv2geom_nlobe.py','--in',csv,'--params',params_path]
        if a.plain: mesh.append('--plain')
        run_step(mesh,'Generate simulation meshes',stage)
        verify_meshes(stage/'meshes',geometry)
        xml=[ROOT/'csv2xml.py','--in',csv,'--out','spirob_physics_model.xml',
             '--params',params_path,'--tendon-shift',params['tendon_inward_shift'],
             '--phi-deg',params['phi_deg']]
        if params['n_cables']==2 and not a.plain: xml.append('--hinge')
        for name in ('safe','fast','high'):
            if getattr(a,name): xml.append('--'+name)
        run_step(xml,'Generate MJCF',stage)
        import mujoco
        model=mujoco.MjModel.from_xml_path(str(stage/'spirob_physics_model.xml'))
        if model.ntendon != params['n_cables'] or model.nu != params['n_cables']:
            raise ValueError('Compiled MJCF cable/actuator count mismatch')
        if a.cad:
            cad=[ROOT/'cad_export.py','--in',csv,'--params',params_path,'--profile',a.cad_profile,
                 '--neck-width-mm',a.neck_width_mm,'--cable-hole-diameter-mm',a.cable_hole_diameter_mm]
            if a.plain: cad.append('--plain')
            if a.fuse_cad: cad.append('--fuse')
            for key in ('flat_thickness_m','flat_edge_ratio'):
                if getattr(a,key) is not None: cad.extend(['--'+key.replace('_','-'),getattr(a,key)])
            run_step(cad,'Export CAD in millimetres',stage)
        # Stages validated. Roll back replacements if publishing itself fails.
        backup=stage/'previous'; backup.mkdir()
        published=[]; moved=[]
        names=['Geom_Data_CSV','meshes','spirob_physics_model.xml']+(['cad'] if a.cad else [])
        try:
            for name in names:
                dest=output/name
                if dest.exists(): dest.rename(backup/name); moved.append(name)
                (stage/name).rename(dest); published.append(name)
        except Exception:
            for name in reversed(published):
                dest=output/name
                if dest.is_dir(): shutil.rmtree(dest)
                else: dest.unlink()
            for name in moved: (backup/name).rename(output/name)
            raise
    print(f'Build completed and MJCF compiled: {output}')

if __name__=='__main__':
    try: main()
    except (ValueError,OSError,subprocess.CalledProcessError) as exc:
        print(f'Build failed: {exc}',file=sys.stderr); sys.exit(1)
