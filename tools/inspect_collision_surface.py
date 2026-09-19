"""Measure and plot the one-hull collider against the compiled CAD surface.

Distances are sampled in BOTH directions (not a certified Hausdorff bound).
Units are mm in each link's body axes; visual and collision transforms are
undone independently. Dependencies are installed with uv sync --locked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.inspect_section import compiled_surface
from tools.audit_inertia import rotation


def collider_surface(model, name):
    bid = model.body(name).id
    ids = [g for g in range(model.body_geomadr[bid], model.body_geomadr[bid]+model.body_geomnum[bid])
           if model.geom_group[g] == 3 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    if len(ids) != 1:
        raise ValueError(f'{name}: requires exactly one group-3 convex mesh collider')
    gid = ids[0]; mid = model.geom_dataid[gid]
    start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    vertices = model.mesh_vert[start:start+count].astype(float) @ rotation(model.geom_quat[gid]).T + model.geom_pos[gid]
    start, count = model.mesh_faceadr[mid], model.mesh_facenum[mid]
    return vertices*1000, model.mesh_face[start:start+count].copy()


def surface_samples(mesh, subdivisions=4):
    # Barycentric grid covers vertices, edges and triangle interiors; no RNG.
    weights = np.array([[i, j, subdivisions-i-j] for i in range(subdivisions+1)
                        for j in range(subdivisions+1-i)], dtype=float)/subdivisions
    return np.unique(np.einsum('ij,tjk->tik', weights, mesh.triangles).reshape(-1, 3), axis=0)


def inspection_mesh(vertices, faces):
    """Omit zero-area seam triangles for surface topology checks only.

    Some revolved n-lobe meshes retain collapsed seam faces after MuJoCo's
    vertex deduplication. They have exactly zero area. Do not fill holes,
    decimate surfaces, or change any positive-area triangle or model asset.
    """
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    keep = mesh.area_faces > 0
    removed = int(np.count_nonzero(~keep))
    if removed:
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
    return mesh, removed


def max_surface_distance(surface, samples):
    import trimesh
    maximum = 0.
    for start in range(0, len(samples), 1000):
        _, distance, _ = trimesh.proximity.closest_point(surface, samples[start:start+1000])
        maximum = max(maximum, float(np.max(distance)))
    return maximum


def inspect(xml_path, links, out=None):
    import trimesh
    from scipy.spatial import ConvexHull
    model = mujoco.MjModel.from_xml_path(str(Path(xml_path).resolve()))
    names = [model.body(i).name for i in range(1, model.nbody) if model.body(i).name.startswith('link_')]
    names = list(dict.fromkeys(links or [names[0], names[min(1, len(names)-1)], names[-1]]))
    if not names or len(names) > 12:
        raise ValueError('Select between 1 and 12 links')
    report = dict(units='mm', method='Triangle barycentric grid, 4 subdivisions per edge, both directions; '
                  'sampled deviations, not a certified global bound or printed-part measurement', links=[])
    meshes = []
    for name in names:
        v, f = compiled_surface(model, name)
        cv, cf = collider_surface(model, name)
        cad, cad_degenerate = inspection_mesh(v, f)
        collider, collider_degenerate = inspection_mesh(cv, cf)
        if not cad.is_volume or not collider.is_volume:
            raise ValueError(f'{name}: expected closed, outward-oriented solids')
        equations = ConvexHull(cv).equations
        outside = max(0., float(np.max(v @ equations[:, :3].T+equations[:, 3])))
        a = surface_samples(cad); b = surface_samples(collider)
        report['links'].append(dict(body=name, cad_volume_mm3=float(cad.volume),
            cad_zero_area_faces_omitted=cad_degenerate, collider_zero_area_faces_omitted=collider_degenerate,
            collider_volume_mm3=float(collider.volume), excess_volume_percent=100*(collider.volume/cad.volume-1),
            cad_outside_hull_mm=outside, cad_to_collider_sampled_max_mm=max_surface_distance(collider, a),
            collider_to_cad_sampled_max_mm=max_surface_distance(cad, b),
            cad_samples=len(a), collider_samples=len(b), collider_vertices=len(cv)))
        meshes.append((cad, collider))
    if out:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(names), 2, figsize=(10, 3.2*len(names)), squeeze=False, layout='constrained')
        for row, (name, (cad, collider)) in enumerate(zip(names, meshes)):
            zlo, zhi = cad.bounds[:, 2]
            # Z cuts reveal the thickness/hex surface, unlike end-on projection.
            for col, fraction in enumerate((.35, .65)):
                z = zlo+(zhi-zlo)*fraction
                ax = axes[row, col]
                for solid, color, label, style in [(cad, '#24669d', 'CAD', '-'), (collider, '#d16716', 'Collision hull', '--')]:
                    segments = trimesh.intersections.mesh_plane(solid, [0, 0, 1], [0, 0, z])
                    for i, segment in enumerate(segments):
                        ax.plot(segment[:, 0], segment[:, 1], style, color=color, lw=1.4, label=label if i == 0 else None)
                ax.set_aspect('equal'); ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
                ax.set_title(f'{name}: transverse cut at Z={z:.3f} mm'); ax.legend(fontsize=8)
        fig.suptitle('CAD and one-geom collision envelope · body coordinates')
        dest = Path(out); dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=170); plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mjcf', required=True, help='Input standalone MuJoCo XML, including its associated mesh assets')
    parser.add_argument('--links', nargs='+', help='Default: base, largest complete link, tip')
    parser.add_argument('--out', default='collision_surface.png', help='Output file for the generated plot or model')
    parser.add_argument('--json', required=True, help='Write the numeric inspection report to this JSON file')
    args = parser.parse_args()
    try:
        report = inspect(args.mjcf, args.links, args.out)
        dest = Path(args.json); dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2)+'\n')
        for row in report['links']:
            print(f"{row['body']}: outward excess {row['excess_volume_percent']:.5f}%; "
                  f"sampled surface gap {max(row['cad_to_collider_sampled_max_mm'], row['collider_to_cad_sampled_max_mm']):.6f} mm")
        print(f'Saved {dest} and {args.out}')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Surface inspection failed: {exc}\n')


if __name__ == '__main__':
    main()
