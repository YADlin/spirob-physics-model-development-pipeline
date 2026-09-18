"""Measure axial thickness from the compiled mesh, and plot a YZ side view.

This samples transverse mesh cuts; it does not infer thickness from XML geom
frames, bounding boxes, or input parameters. Parameters provide the expected
taper and axial positions for comparison. All reported dimensions are mm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spirob.geometry import from_params
from spirob.sections import linear_thickness_law, thickness_at_z, resolve_section_params
from tools.inspect_section import compiled_surface


def measure(model, geometry):
    rows = []
    for unit in geometry.units:
        vertices, faces = compiled_surface(model, unit.link_name)
        surface = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        length = (unit.slit_reference_m[2]-unit.local_frame_origin_m[2])*1000
        stations = np.linspace(.01, .99, 9)*length
        thicknesses = []
        for z in stations:
            cut = trimesh.intersections.mesh_plane(surface, [0,0,1], [0,0,z])
            if len(cut) == 0:
                raise ValueError(f'{unit.link_name}: empty mesh cut at z={z:g} mm; check matching params')
            thicknesses.append(float(np.ptp(cut[:,:,1])))
        slope, intercept = np.polyfit(stations, thicknesses, 1)
        residual = np.max(np.abs(np.asarray(thicknesses)-(slope*stations+intercept)))
        start = (unit.local_frame_origin_m[2]-geometry.units[0].local_frame_origin_m[2])*1000
        rows.append(dict(link=unit.link_name, z_from_mount_mm=(stations+start).tolist(),
                         measured_thickness_mm=thicknesses, start_z_mm=start, end_z_mm=start+length,
                         proximal_thickness_mm=float(intercept), distal_thickness_mm=float(intercept+slope*length),
                         slope_mm_per_mm=float(slope), max_line_fit_residual_mm=float(residual)))
    return rows


def inspect(xml_path, params, out, *, comparison=None, show=False, tolerance_mm=1e-4):
    params = resolve_section_params(params)
    if params['n_cables'] != 2:
        raise ValueError('Axial thickness inspection requires two cables')
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError('tolerance_mm must be positive and finite')
    geometry = from_params(params)
    model = mujoco.MjModel.from_xml_path(str(Path(xml_path).resolve()))
    rows = measure(model, geometry)
    law = linear_thickness_law(params, geometry)
    for row in rows:
        expected = np.array([thickness_at_z(law, z/1000+law['z_base_m'])*1000 for z in row['z_from_mount_mm']])
        row['max_expected_error_mm'] = float(np.max(np.abs(np.array(row['measured_thickness_mm'])-expected)))
    gaps = [b['proximal_thickness_mm']-a['distal_thickness_mm'] for a,b in zip(rows,rows[1:])]
    max_error = max(r['max_expected_error_mm'] for r in rows)
    max_gap = max(map(abs,gaps),default=0.)
    report = dict(mjcf=str(xml_path), units='mm', frame='unbent robot; local Z measured from base mount',
                  measurement='9 interior mesh cuts per link; endpoint values extrapolated from measured linear fit',
                  tolerance_mm=tolerance_mm, max_expected_error_mm=max_error,
                  max_joint_thickness_jump_mm=max_gap, joint_thickness_jumps_mm=gaps,
                  passed=bool(max_error <= tolerance_mm and max_gap <= tolerance_mm), links=rows)
    if not show:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2 if comparison else 1, 1, figsize=(12,6 if comparison else 3.8), squeeze=False,
                             layout='constrained')
    datasets = [(rows,'Linear thickness — measured compiled mesh')]
    if comparison:
        previous = measure(mujoco.MjModel.from_xml_path(str(Path(comparison).resolve())),geometry)
        datasets.insert(0,(previous,'Previous stepped thickness — measured compiled mesh'))
    for ax,(records,title) in zip(axes.flat,datasets):
        for i, row in enumerate(records):
            z = [row['start_z_mm'],row['end_z_mm']]
            t = np.array([row['proximal_thickness_mm'],row['distal_thickness_mm']])/2
            ax.fill_between(z,-t,t,color='#a5c8e4' if i%2 else '#d6e5f1',edgecolor='#32658a',linewidth=.6)
        z = np.linspace(0,rows[-1]['end_z_mm'],100)
        t = np.array([thickness_at_z(law,v/1000+law['z_base_m'])*500 for v in z])
        ax.plot(z,t,'--',color='#c63f32',lw=1,label='Requested linear envelope')
        ax.plot(z,-t,'--',color='#c63f32',lw=1)
        ax.set_aspect('equal',adjustable='box')
        ax.set_xlabel('Z from base mounting plane (mm)');ax.set_ylabel('Y (mm)')
        ax.set_title(title,loc='left',fontsize=11);ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper right')
    fig.suptitle('Two-cable SpiRob · centreline thickness along the robot',fontsize=14)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=180)
    if show:plt.show()
    plt.close(fig)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mjcf',required=True)
    parser.add_argument('--params',required=True,help='Matching build_params.json')
    parser.add_argument('--comparison-mjcf',help='Optional previous XML with the same XZ geometry')
    parser.add_argument('--out',default='thickness_taper.png')
    parser.add_argument('--json',help='Save measured thicknesses and continuity errors')
    parser.add_argument('--tolerance-mm',type=float,default=1e-4)
    parser.add_argument('--show',action='store_true')
    args=parser.parse_args()
    try:
        report=inspect(args.mjcf,json.loads(Path(args.params).read_text()),args.out,
                       comparison=args.comparison_mjcf,show=args.show,tolerance_mm=args.tolerance_mm)
        if args.json:
            path=Path(args.json);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(report,indent=2)+'\n')
        print(f"Maximum thickness error: {report['max_expected_error_mm']:.6g} mm")
        print(f"Maximum joint thickness jump: {report['max_joint_thickness_jump_mm']:.6g} mm")
        print(f"{'PASS' if report['passed'] else 'FAIL'}; saved {args.out}")
        if not report['passed']:parser.exit(1)
    except (ValueError,OSError) as exc:
        parser.exit(1,f'Taper inspection failed: {exc}\n')


if __name__ == '__main__':
    main()
