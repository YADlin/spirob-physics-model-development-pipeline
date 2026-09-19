"""Build CSV -> simulation meshes -> MJCF, with optional fabrication CAD.

Every stage uses this interpreter and a fresh temporary output directory. Failed
builds leave the previous outputs in place. Simulation names and physics remain
owned by the existing generators.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def run_step(argv, desc, cwd):
    print(f'\n{desc}', flush=True)
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''))
    subprocess.run([sys.executable, *map(str, argv)], cwd=cwd, env=env, check=True)


def verify_meshes(directory, geometry, layout='individual'):
    import struct
    from spirob.mesh_assets import mesh_assets
    expected = {asset.filename for asset in mesh_assets(geometry, layout)}
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--params',default='params.json', help='Input geometry, dynamics and build settings JSON; CLI values override JSON')
    p.add_argument('--output-dir',default='build/spirob',help='Destination containing XML, meshes, reports and optional CAD')
    p.add_argument('--noclean',action='store_true',help='Compatibility flag; builds always use fresh staging')
    p.add_argument('--no-preview',action='store_true', help='Skip the desktop approval preview regardless of show_preview in JSON')
    p.add_argument('--timestep', type=float, help='Simulation timestep in seconds; overrides the preset/params')
    from spirob.sections import section_arguments, resolve_section_params
    section_arguments(p)
    p.add_argument('--mesh-layout',choices=['shared','individual'],default=None,
                   help='Reuse one complete-link STL plus a partial-base STL (default: shared)')
    p.add_argument('--collision-mode',choices=['mesh','capsule','compound','convex'],
                   help='convex: one hull per flat or n-lobe link; compound: legacy two-cable boxes/cylinders')
    p.add_argument('--collision-corner-radius-ratio',type=float,default=None,
                   help='Compound corner radius / link half-width (default: 0.04)')
    p.add_argument('--collision-margin-m',type=float,
                   help='Contact margin in metres; default 0 for compound/convex, otherwise the selected preset')
    p.add_argument('--arena-memory-mib',type=int,
                   help='Native MuJoCo arena MiB; default 128 for compound, otherwise compiler default')
    p.add_argument('--align-geom-frames',action=argparse.BooleanOptionalAction,default=None,
                   help='Also export spirob_aligned.mjb with actual Geom axes aligned to bodies (MuJoCo 3.3.5)')
    section = p.add_mutually_exclusive_group()
    section.add_argument('--plain',action='store_true', default=None, help='Use a revolved circular section; bypass flat and n-lobe cutters')
    section.add_argument('--nlobe',action='store_true',help='Compatibility alias: use the n_cables cross-section')
    physics = p.add_mutually_exclusive_group()
    for name, help_text in [('safe','Capsule preset with conservative solver settings; explicit collision-mode wins'), ('fast','Faster, softer preset for exploratory runs'), ('high','Higher solver accuracy preset; does not guarantee stability')]:
        physics.add_argument('--'+name, action='store_true', help=help_text)
    p.add_argument('--cad',action=argparse.BooleanOptionalAction, default=None, help='Export whole-robot STEP and millimetre STL with a CAD validation report')
    p.add_argument('--iges',action=argparse.BooleanOptionalAction, default=None, help='Also export an IGES surface model; requires --cad or build.cad=true')
    p.add_argument('--fuse-cad',action=argparse.BooleanOptionalAction, default=None, help='Fuse simulation CAD solids; fabrication is already fused; requires CAD')
    p.add_argument('--cad-profile',choices=['fabrication','simulation'],default=None, help='fabrication adds a central ligament and optional channels; simulation assembles link CAD')
    p.add_argument('--flat-thickness-m',type=float, help='Legacy stepped fabrication lens: full constant thickness in metres')
    p.add_argument('--flat-edge-ratio',type=float, help='Legacy stepped fabrication lens: edge / centre thickness, in (0,1]')
    p.add_argument('--neck-width-mm',type=float, help='Fabrication ligament X width (two cables) or cylinder diameter (n cables), in mm; default 1')
    p.add_argument('--cable-hole-diameter-mm',type=float, help='Fabrication channel diameter in mm; default 0 leaves the CAD undrilled')
    a=p.parse_args(argv)
    params_path = Path(a.params).resolve()
    from spirob.parameters import normalize_params
    original = normalize_params(json.loads(params_path.read_text(encoding='utf-8')))
    settings = dict(original.get('build', {}))
    defaults = dict(mesh_layout='shared', collision_mode=None, collision_corner_radius_ratio=.04,
                    collision_margin_m=None, arena_memory_mib=None, align_geom_frames=False, plain=False,
                    cad=False, iges=False, fuse_cad=False, cad_profile='fabrication',
                    flat_thickness_m=None, flat_edge_ratio=None, neck_width_mm=1., cable_hole_diameter_mm=0.)
    for key, fallback in defaults.items():
        if getattr(a, key) is None: setattr(a, key, settings.get(key, fallback))
    if a.nlobe: a.plain = False
    preset = next((name for name in ('safe','fast','high') if getattr(a,name)), settings.get('physics_preset','default'))
    for name in ('safe','fast','high'): setattr(a, name, preset == name)
    if (a.fuse_cad or a.iges) and not a.cad: p.error('--fuse-cad / --iges require CAD export')
    params = resolve_section_params(original,
                                    hex_section=a.hex_section, hex_edge_ratio=a.hex_edge_ratio, base_thickness_mm=a.base_thickness_mm, thickness_profile=a.thickness_profile, plain=a.plain)
    params['build'] = {key:getattr(a,key) for key in defaults if getattr(a,key) is not None}
    params['build']['physics_preset'] = preset
    if a.collision_mode is None:
        params['build']['collision_mode'] = 'capsule' if a.safe else 'mesh'
    params['post_gen'] = dict(params.get('post_gen', {}))
    params['post_gen'].pop('target_site_pos', None)
    if a.timestep is not None:
        if not math.isfinite(a.timestep) or a.timestep <= 0: p.error('--timestep must be finite and positive')
        params['post_gen'] = dict(params.get('post_gen', {}), timestep=a.timestep)
    if params.get('flat_section') == 'hex':
        if a.collision_mode == 'compound':
            p.error('--hex-section compound colliders are not implemented; use --collision-mode convex')
        if a.flat_thickness_m is not None or a.flat_edge_ratio is not None:
            p.error('--hex-section uses --base-thickness-mm and --hex-edge-ratio')
    from spirob.pipeline.spirob_csv_generator import validate_params
    from spirob.geometry import from_params
    validate_params(params)
    geometry = from_params(params)
    dimensions = None
    if params['n_cables'] == 2 and not a.plain:
        params.setdefault('thickness_profile', 'linear')
        if params['thickness_profile'] == 'linear' and a.collision_mode == 'compound':
            p.error('Linear-taper compound colliders are not implemented; use --collision-mode convex')
        from spirob.sections import section_dimensions
        dimensions = section_dimensions(params, geometry)
        print(f"Base centre thickness: {dimensions['base']['centre_thickness_m']*1000:.6f} mm; "
              f"tip: {dimensions['tip']['centre_thickness_m']*1000:.6f} mm "
              f"({dimensions['mode']}, {dimensions['thickness_profile']})", flush=True)
    if a.collision_mode == 'compound' and (a.plain or params['n_cables'] != 2):
        p.error('--collision-mode compound requires n_cables=2 without --plain')
    if a.collision_mode == 'convex' and a.plain:
        p.error('--collision-mode convex supports flat and n-lobe sections; omit --plain')
    if a.collision_mode == 'convex' and params['n_cables'] >= 3:
        print('Convex collision: one hull per n-lobe link; concave notches are bridged. '
              'Use tools/inspect_collision_surface.py to measure the surface difference.', flush=True)
    output = Path(a.output_dir).resolve(); output.mkdir(parents=True,exist_ok=True)
    # Stage alongside the destination so publishing uses same-filesystem renames.
    with tempfile.TemporaryDirectory(prefix='.spirob-build-',dir=output.parent) as td:
        stage=Path(td)
        params_path=stage/'build_params.json'
        params_path.write_text(json.dumps(params, indent=2)+'\n', encoding='utf-8')
        if dimensions is not None:
            (stage/'section_dimensions.json').write_text(json.dumps(dimensions, indent=2)+'\n', encoding='utf-8')
        if params.get('show_preview', False) and not a.no_preview:
            preview=[ROOT/'tools/preview.py','--params',params_path]
            if not a.plain and params['n_cables']>=3: preview.append('--nlobe')
            run_step(preview,'Geometry preview',ROOT)
        run_step([ROOT/'spirob/pipeline/spirob_csv_generator.py','--params',params_path,'--yes'],'Generate CSV',stage)
        csv=Path('Geom_Data_CSV/Spirob_geom_data.csv')
        mesh=[ROOT/'spirob/pipeline/csv2geom_nlobe.py','--in',csv,'--params',params_path,'--mesh-layout',a.mesh_layout]
        if a.plain: mesh.append('--plain')
        run_step(mesh,'Generate simulation meshes',stage)
        verify_meshes(stage/'meshes',geometry,a.mesh_layout)
        xml=[ROOT/'spirob/pipeline/csv2xml.py','--in',csv,'--out','spirob_physics_model.xml',
             '--params',params_path,'--tendon-shift',params['tendon_inward_shift'],
             '--phi-deg',params['phi_deg'],'--mesh-layout',a.mesh_layout,
             '--collision-corner-radius-ratio',a.collision_corner_radius_ratio]
        if a.collision_mode: xml.extend(['--collision-mode',a.collision_mode])

        if a.collision_margin_m is not None: xml.extend(['--collision-margin-m',a.collision_margin_m])
        if a.arena_memory_mib is not None: xml.extend(['--arena-memory-mib',a.arena_memory_mib])
        if a.plain: xml.append('--plain')
        if params['n_cables']==2 and not a.plain: xml.append('--hinge')
        for name in ('safe','fast','high'):
            if getattr(a,name): xml.append('--'+name)
        run_step(xml,'Generate MJCF',stage)
        import mujoco
        model=mujoco.MjModel.from_xml_path(str(stage/'spirob_physics_model.xml'))
        if model.ntendon != params['n_cables'] or model.nu != params['n_cables']:
            raise ValueError('Compiled MJCF cable/actuator count mismatch')
        from tools.inspect_collision import collision_report
        report = collision_report(stage/'spirob_physics_model.xml', model)
        (stage/'collision_summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        if a.align_geom_frames:
            from spirob.mujoco_frames import aligned_geom_model
            model = aligned_geom_model(model)
            mujoco.mj_saveModel(model, str(stage/'spirob_aligned.mjb'), None)
        if a.cad:
            cad=[ROOT/'spirob/pipeline/cad_export.py','--in',csv,'--params',params_path,'--profile',a.cad_profile,
                 '--neck-width-mm',a.neck_width_mm,'--cable-hole-diameter-mm',a.cable_hole_diameter_mm]
            if a.plain: cad.append('--plain')
            if a.fuse_cad: cad.append('--fuse')
            if a.iges: cad.append('--iges')
            for key in ('flat_thickness_m','flat_edge_ratio'):
                if getattr(a,key) is not None: cad.extend(['--'+key.replace('_','-'),getattr(a,key)])
            run_step(cad,'Export CAD in millimetres',stage)
        # Stages validated. Roll back replacements if publishing itself fails.
        backup=stage/'previous'; backup.mkdir()
        published=[]; moved=[]
        names=['Geom_Data_CSV','meshes','spirob_physics_model.xml','build_params.json','section_dimensions.json','collision_summary.json','cad']
        # Always retire a previous MJB when rebuilding: it must never describe
        # an older robot than the XML/meshes in this output directory.
        names.append('spirob_aligned.mjb')
        try:
            for name in names:
                dest=output/name
                if dest.exists(): dest.rename(backup/name); moved.append(name)
                if (stage/name).exists():
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
