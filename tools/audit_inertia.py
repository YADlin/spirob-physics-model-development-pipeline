"""Compare compiled link inertia with independent mesh/CAD volume integrals.

All tensors use axes parallel to the containing link's body frame. Report both
each reference's own COM tensor and its tensor shifted to MuJoCo's COM. CAD and
mesh values assume uniform stated density; they are not physical measurements.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def rotation(quaternion):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def body_inertia(model, bid):
    matrix = rotation(model.body_iquat[bid])
    return matrix @ np.diag(model.body_inertia[bid]) @ matrix.T


def shift_inertia(tensor_com, mass, displacement):
    r = np.asarray(displacement, dtype=float)
    return tensor_com + mass*(np.dot(r, r)*np.eye(3)-np.outer(r, r))


def mesh_reference(model, bid, density):
    import trimesh
    ids = [g for g in range(model.body_geomadr[bid], model.body_geomadr[bid]+model.body_geomnum[bid])
           if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_group[g] == 1]
    if len(ids) != 1:
        raise ValueError(f'{model.body(bid).name}: expected one group-1 CAD mesh; found {len(ids)}')
    gid = ids[0]
    mid = model.geom_dataid[gid]
    va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
    # Undo compiler centring/principal-axis rotation through the compiled geom
    # transform. Vertex coordinates are now in the containing body's axes.
    vertices = model.mesh_vert[va:va+vn].astype(float) @ rotation(model.geom_quat[gid]).T + model.geom_pos[gid]
    mesh = trimesh.Trimesh(vertices=vertices, faces=model.mesh_face[fa:fa+fn], process=True)
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError(f'{model.body(bid).name}: triangle surface is not a closed, consistently oriented positive volume')
    mesh.density = density
    props = mesh.mass_properties
    return dict(mass=float(props.mass), com=np.asarray(props.center_mass), tensor=np.asarray(props.inertia),
                method='Independent signed triangle-volume integration (trimesh); compiled surface vertices in body axes')


def cad_references(params, names, density, plain=False):
    import cadquery as cq
    import pandas as pd
    from spirob.geometry import from_params
    from csv2geom_nlobe import build_unit_inputs
    from cad_export import _simulation_element_mm
    geometry = from_params(params)
    records = []
    for i, quad in enumerate(geometry.inverted_quads()):
        row = {'elem': i+1}
        for prefix, (x, z) in zip(('joint_s1', 'joint_s2', 'c0_s2', 'c0_s1'), quad):
            row.update({prefix+'_x': x, prefix+'_y': 0., prefix+'_z': z})
        records.append(row)
    units = build_unit_inputs(pd.DataFrame(records), geometry)
    missing = set(names)-{unit.link_name for unit in units}
    if missing:
        raise ValueError(f'Parameters have no matching CAD links: {sorted(missing)}')
    results = {}
    for unit in units:
        if unit.link_name not in names:
            continue
        # CAD exporter works in mm and CSV coordinates. Shift into local body
        # coordinates before integration, avoiding subtraction of large origins.
        shape = _simulation_element_mm(unit, params, plain).translate(tuple(-1000*v for v in unit.origin_m))
        results[unit.link_name] = dict(mass=float(shape.Volume())*1e-9*density,
                                      com=np.array(shape.Center().toTuple())*1e-3,
                                      tensor=np.array(cq.Shape.matrixOfInertia(shape))*density*1e-15,
                                      method='OpenCascade solid-volume integration; simulation CAD profile, uniform density')
    return results


def compare_reference(reference, mass, com, tensor):
    common = shift_inertia(reference['tensor'], reference['mass'], reference['com']-com)
    scale = float(np.linalg.norm(common))
    return dict(method=reference['method'], mass_kg=reference['mass'], com_in_body_m=reference['com'].tolist(),
                inertia_about_own_com_body_axes_kg_m2=reference['tensor'].tolist(),
                inertia_about_mujoco_com_body_axes_kg_m2=common.tolist(),
                mass_difference_mujoco_minus_reference_kg=mass-reference['mass'],
                mass_relative_error=(mass-reference['mass'])/reference['mass'],
                com_distance_m=float(np.linalg.norm(reference['com']-com)),
                inertia_relative_frobenius_error=float(np.linalg.norm(tensor-common))/scale,
                tensor_difference_mujoco_minus_reference_at_mujoco_com_kg_m2=(tensor-common).tolist())


def audit(xml_path, *, params=None, links=None, density=1200., plain=False, reference_xml=None):
    if not math.isfinite(density) or density <= 0:
        raise ValueError('density must be positive and finite, in kg/m^3')
    model = mujoco.MjModel.from_xml_path(str(Path(xml_path).resolve()))
    names = [model.body(i).name for i in range(1, model.nbody)
             if re.fullmatch(r'link_\d+', model.body(i).name)]
    if not names or (links is not None and not set(links) <= set(names)):
        raise ValueError('Requested link names must exist and follow link_NNN')
    if links:
        names = [name for name in names if name in links]
    cad = cad_references(params, names, density, plain) if params is not None else {}
    previous = mujoco.MjModel.from_xml_path(str(Path(reference_xml).resolve())) if reference_xml else None
    rows = []
    for name in names:
        bid = model.body(name).id
        mass, com, tensor = float(model.body_mass[bid]), model.body_ipos[bid], body_inertia(model, bid)
        row = dict(link=name, mujoco=dict(mass_kg=mass, com_in_body_m=com.tolist(),
                   inertia_about_com_body_axes_kg_m2=tensor.tolist(),
                   principal_moments_kg_m2=model.body_inertia[bid].tolist(),
                   principal_frame_quaternion_wxyz=model.body_iquat[bid].tolist()), references={})
        try:
            row['references']['mesh'] = compare_reference(mesh_reference(model, bid, density), mass, com, tensor)
        except ValueError as exc:
            row['mesh_reference_error'] = str(exc)
        if name in cad:
            row['references']['cad'] = compare_reference(cad[name], mass, com, tensor)
        if previous is not None:
            old = previous.body(name)
            ref = dict(mass=float(old.mass[0]), com=old.ipos, tensor=body_inertia(previous, old.id),
                       method='Compiled comparison XML; assumes matching link-body frame convention')
            row['references']['previous_xml'] = compare_reference(ref, mass, com, tensor)
        rows.append(row)
    return dict(xml=str(Path(xml_path).resolve()), density_kg_m3=density, mujoco_version=mujoco.__version__,
                frame='Axes parallel to each link body. Each own-COM tensor and each tensor shifted to MuJoCo COM are reported separately.',
                interpretation='CAD/mesh reference assumes a homogeneous solid at the stated density, not measured hardware inertia. No model properties are modified.',
                links=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mjcf', required=True)
    parser.add_argument('--params', help='Matching geometry parameters enable the independent CAD-solid reference')
    parser.add_argument('--density', type=float, default=1200., help='Reference density kg/m^3 (default 1200)')
    parser.add_argument('--links', nargs='+', help='Link names, e.g. link_001 link_002 link_021; default all')
    parser.add_argument('--plain', action='store_true', help='Use the plain revolved CAD reference')
    parser.add_argument('--reference-mjcf', help='Optional older XML for before/after inertia comparison')
    parser.add_argument('--json', help='Save full tensors and errors to JSON')
    args = parser.parse_args()
    try:
        params = json.loads(Path(args.params).read_text()) if args.params else None
        report = audit(args.mjcf, params=params, links=args.links, density=args.density,
                       plain=args.plain, reference_xml=args.reference_mjcf)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'Error: {exc}\n')
    print(report['frame'])
    print(report['interpretation'])
    for row in report['links']:
        mj = row['mujoco']
        print(f"\n{row['link']}: MuJoCo mass={mj['mass_kg']:.9g} kg; COM in body (m)={mj['com_in_body_m']}")
        print('MuJoCo inertia about COM, body-parallel axes (kg m^2):')
        print(np.array2string(np.array(mj['inertia_about_com_body_axes_kg_m2']), precision=8, suppress_small=False))
        if row.get('mesh_reference_error'):
            print('MESH REFERENCE UNAVAILABLE: '+row['mesh_reference_error'])
        for kind, ref in row['references'].items():
            print(f"{kind}: mass={ref['mass_kg']:.9g} kg; COM={ref['com_in_body_m']}")
            print('Reference inertia shifted to MuJoCo COM, body-parallel axes (kg m^2):')
            print(np.array2string(np.array(ref['inertia_about_mujoco_com_body_axes_kg_m2']), precision=8, suppress_small=False))
            print(f"  mass error={ref['mass_relative_error']:.6%}, COM separation={ref['com_distance_m']:.6g} m, "
                  f"tensor relative error={ref['inertia_relative_frobenius_error']:.6%}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2)+'\n')
    return 2 if any('mesh_reference_error' in row for row in report['links']) else 0


if __name__ == '__main__':
    sys.exit(main())
