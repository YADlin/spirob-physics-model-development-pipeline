"""Save actual compiled mesh projections in each link's body axes, in mm.

The XY view looks along the robot's local longitudinal Z axis. It is an end-on
projection, not a slice at a chosen Z station. MuJoCo's principal-axis mesh
rotation is undone before plotting; displayed geom frames cannot skew the view.
No simulation steps or XML edits are performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_inertia import rotation


def compiled_surface(model, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 1:
        raise ValueError(f'No link body named {name}')
    ids = [g for g in range(model.body_geomadr[bid], model.body_geomadr[bid]+model.body_geomnum[bid])
           if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_group[g] == 1]
    if len(ids) != 1:
        raise ValueError(f'{name}: expected one group-1 CAD mesh; found {len(ids)}')
    gid = ids[0]
    mid = model.geom_dataid[gid]
    start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    vertices = model.mesh_vert[start:start+count].astype(float) @ rotation(model.geom_quat[gid]).T + model.geom_pos[gid]
    start, count = model.mesh_faceadr[mid], model.mesh_facenum[mid]
    return vertices*1000, model.mesh_face[start:start+count].copy()


def projection(vertices, faces, axes):
    from shapely import Polygon, union_all
    triangles = [Polygon(vertices[face][:, axes]) for face in faces]
    outline = union_all([p for p in triangles if p.area > 1e-12]).simplify(1e-6, preserve_topology=True)
    if outline.geom_type != 'Polygon':
        raise ValueError('Projected link surface is disconnected or empty')
    return outline


def inspect(xml_path, out, links=None, show=False):
    if not show:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    model = mujoco.MjModel.from_xml_path(str(Path(xml_path).resolve()))
    names = [model.body(i).name for i in range(1, model.nbody)
             if re.fullmatch(r'link_\d+', model.body(i).name)]
    if not names:
        raise ValueError('No link_NNN bodies in XML')
    names = list(dict.fromkeys(links or [names[0], names[min(1, len(names)-1)], names[-1]]))
    if len(names) > 12:
        raise ValueError('Select at most 12 links per image')
    fig = plt.figure(figsize=(12, 3.3*len(names)), layout='constrained')
    report = dict(mjcf=str(Path(xml_path).resolve()), timestep_s=float(model.opt.timestep),
                  units='mm', frame='local body axes; joint origin at zero',
                  view='XY is the full end-on projection along local Z, not a planar slice', links=[])
    for row, name in enumerate(names):
        vertices, faces = compiled_surface(model, name)
        record = dict(body=name, bounds_mm=[vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()])
        for column, (axes, labels, title) in enumerate([((0, 1), ('X', 'Y'), 'End-on: look along Z'),
                                                       ((0, 2), ('X', 'Z'), 'Side: look along Y')]):
            ax = fig.add_subplot(len(names), 3, row*3+column+1)
            poly = projection(vertices, faces, axes)
            points = np.asarray(poly.exterior.coords)
            ax.fill(points[:, 0], points[:, 1], facecolor='#b9d3e8', edgecolor='#235b84', linewidth=1.5)
            for hole in poly.interiors:
                pts = np.asarray(hole.coords)
                ax.fill(pts[:, 0], pts[:, 1], color='white')
            ax.axhline(0, color='#666666', linewidth=.5, linestyle=':')
            ax.axvline(0, color='#666666', linewidth=.5, linestyle=':')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel(f'{labels[0]} (mm)'); ax.set_ylabel(f'{labels[1]} (mm)')
            ax.set_title(f'{name} · {title}', fontsize=10)
            ax.margins(.15)
            if column == 0:
                record.update(end_on_vertices_mm=points[:-1].tolist(), end_on_sides=len(points)-1,
                              end_on_area_mm2=float(poly.area))
        ax = fig.add_subplot(len(names), 3, row*3+3, projection='3d')
        ax.add_collection3d(Poly3DCollection(vertices[faces], facecolor='#9dbfd9', edgecolor='#497595', linewidth=.25))
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi-lo)
        ax.view_init(elev=24, azim=-55)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'{name} · surface (mm)', fontsize=10)
        report['links'].append(record)
    fig.suptitle('Compiled SpiRob mesh · body coordinates', fontsize=15)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    if show: plt.show()
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mjcf', required=True)
    parser.add_argument('--links', nargs='+', help='Default: base, first complete link, tip')
    parser.add_argument('--out', default='cross_sections.png')
    parser.add_argument('--json', help='Optional machine-readable outline coordinates')
    parser.add_argument('--show', action='store_true', help='Also open the Matplotlib window')
    args = parser.parse_args()
    try:
        report = inspect(args.mjcf, args.out, args.links, args.show)
        if args.json:
            dest = Path(args.json); dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        for link in report['links']:
            print(f"{link['body']}: {link['end_on_sides']} end-on sides; area {link['end_on_area_mm2']:.6g} mm^2")
        print(f'Saved {args.out}; timestep {report["timestep_s"]:g} s')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Inspection failed: {exc}\n')


if __name__ == '__main__':
    main()
