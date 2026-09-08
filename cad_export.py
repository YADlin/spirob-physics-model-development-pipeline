"""Whole SpiRob CAD export, in millimetres for CAD applications and slicers.

Independent implementation using this repository's canonical geometry. Feature
inspiration: https://github.com/ZhanchiWang/Open-Spiral-Robots; no source copied.
Simulation assembly preserves the link geometry. Fabrication adds a finite
central flexure and optionally drills the canonical tendon paths. Its dimensions
are design choices, not an identification of the existing MJCF dynamics.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class CadExportResult:
    step_path: str
    stl_path: str
    n_elements: int
    fused: bool
    report_path: str


def _positive(name, value, allow_zero=False):
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def _lens_element_mm(unit, thickness_mm, edge_ratio):
    """Preserve all four XZ quad vertices, including the slanted slit faces.

    Each half is a closed hexahedron. Triangular faces avoid non-planar loft
    end wires (the centre and outer edge have different Z coordinates).
    """
    import cadquery as cq
    points = unit.profile_xyz
    footprint = [(points[i][0]*1000, points[i][2]*1000) for i in (0, 3, 2, 1)]
    vertices = [cq.Vector(0 if i in (0, 3) else x,
                          sign*thickness_mm/2*(1 if i in (0, 3) else edge_ratio), z)
                for sign in (-1, 1) for i, (x, z) in enumerate(footprint)]
    centre = sum(vertices, cq.Vector()) / 8
    faces = []
    for quad in ((0,1,2,3), (4,5,6,7), (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7)):
        for ids in ((quad[0],quad[1],quad[2]), (quad[0],quad[2],quad[3])):
            a, b, c = [vertices[i] for i in ids]
            if (b-a).cross(c-a).dot((a+b+c)/3-centre) < 0:
                b, c = c, b
            faces.append(cq.Face.makeFromWires(cq.Wire.makePolygon([a,b,c], close=True)))
    half = cq.Solid.makeSolid(cq.Shell.makeShell(faces)).fix()
    return half.fuse(half.mirror('YZ')).clean()


def _simulation_element_mm(unit, params, plain=False):
    from csv2geom_nlobe import build_flat_element, make_profile_from_points, revolve_profile, add_nlobe_cut
    n = params['n_cables']
    if n == 2 and not plain:
        shape = build_flat_element(unit.row, params.get('flat_thickness_ratio', .3))
        shape = shape.translate((0, 0, unit.origin_m[2]))
    else:
        shape = revolve_profile(make_profile_from_points(unit.profile_xyz))
        # Match the simulation's local construction before returning to CSV frame.
        origin = unit.origin_m
        shape = shape.translate(tuple(-v for v in origin))
        if not plain:
            shape = add_nlobe_cut(shape, n, unit.outer_radius_m, unit.height_z_m,
                                  params['phi_deg']/2, params.get('nlobe_t', .5),
                                  params.get('notch_factor', .25))
        shape = shape.translate(origin)
    return shape.val().scale(1000)


def build_cad(units, geometry, params, *, profile='fabrication', plain=False,
              flat_thickness_m=None, flat_edge_ratio=None, neck_width_mm=1.0,
              cable_hole_diameter_mm=0.0, fuse=False):
    import cadquery as cq
    if profile not in ('fabrication', 'simulation'):
        raise ValueError('profile must be fabrication or simulation')
    edge = float(params.get('flat_edge_ratio', .25) if flat_edge_ratio is None else flat_edge_ratio)
    if not math.isfinite(edge) or not 0 < edge <= 1:
        raise ValueError('flat_edge_ratio must be in (0, 1]')
    thickness = _positive('flat_thickness_m', flat_thickness_m if flat_thickness_m is not None
                          else params.get('flat_thickness_ratio', .3)*2*max(u.outer_radius_m for u in units))*1000
    neck = _positive('neck_width_mm', neck_width_mm)
    hole = _positive('cable_hole_diameter_mm', cable_hole_diameter_mm, True)
    n = geometry.inputs.n_cables
    if neck >= geometry.lengths.realized_tip_width_m*1000:
        raise ValueError('neck_width_mm must be smaller than the realized tip width')
    if profile == 'simulation' and (hole or flat_thickness_m is not None or flat_edge_ratio is not None):
        raise ValueError('Manufacturing thickness/hole overrides require profile=fabrication')
    shapes = []
    for unit in units:
        shape = (_lens_element_mm(unit, thickness, edge)
                 if profile == 'fabrication' and n == 2 and not plain
                 else _simulation_element_mm(unit, params, plain))
        if not shape.isValid() or shape.Volume() <= 0:
            raise ValueError(f'{unit.link_name}: invalid element solid')
        shapes.append(shape)
    z0 = geometry.units[0].local_frame_origin_m[2]*1000
    z1 = geometry.units[-1].slit_reference_m[2]*1000
    if profile == 'fabrication':
        # A finite central ligament makes the zero-width simulation hinges a
        # connected physical design. Fuse in mm to avoid metre-scale OCC tolerances.
        if n == 2 and not plain:
            core = cq.Workplane('XY').box(neck, thickness, z1-z0, centered=(True,True,False)).translate((0,0,z0)).val()
        else:
            core = cq.Solid.makeCylinder(neck/2, z1-z0, cq.Vector(0,0,z0))
        combined = core.fuse(*shapes).clean()
    elif fuse:
        combined = shapes[0].fuse(*shapes[1:]).clean()
    else:
        combined = cq.Compound.makeCompound(shapes)
    before_volume = combined.Volume()
    if hole:
        for path in geometry.tendon_paths:
            pts = [cq.Vector(*(round(v*1000, 9) for v in p.routed_m)) for p in path.points]
            d0 = (pts[1]-pts[0]).normalized()
            d1 = (pts[-1]-pts[-2]).normalized()
            extension = z1-z0
            pts = [pts[0]-d0*extension] + pts + [pts[-1]+d1*extension]
            wire = cq.Wire.makePolygon(pts, close=False)
            tool = cq.Workplane(cq.Plane(origin=pts[0], normal=d0)).circle(hole/2).sweep(
                wire, isFrenet=True, transition='round').val()
            if not tool.isValid(): raise ValueError(f'Cable {path.cable_index}: invalid hole tool')
            combined = combined.cut(tool).clean()
        if before_volume-combined.Volume() <= 1e-6:
            raise ValueError('Cable paths removed no material; inspect routing')
    if not combined.isValid() or combined.Volume() <= 0:
        raise ValueError('CAD boolean operations produced invalid geometry')
    solid_count = len(combined.Solids())
    if profile == 'fabrication' and solid_count != 1:
        raise ValueError(f'Fabrication model has {solid_count} disconnected solids; adjust neck/hole dimensions')
    return combined, {'profile': profile, 'solid_count': solid_count, 'valid': True,
                      'neck_width_mm': neck if profile == 'fabrication' else None,
                      'cable_hole_diameter_mm': hole,
                      'flat_centre_thickness_mm': thickness if n == 2 and profile == 'fabrication' else None,
                      'flat_edge_ratio': edge if n == 2 and profile == 'fabrication' else None,
                      'removed_hole_volume_mm3': before_volume-combined.Volume()}


def process_cad(csv_file, params, *, outdir='cad', prefix='spirob', fuse=False,
                plain=False, stl_tolerance=1e-4, flat_thickness_m=None,
                flat_edge_ratio=None, geometry=None, profile='fabrication',
                neck_width_mm=1.0, cable_hole_diameter_mm=0.0):
    import cadquery as cq
    import pandas as pd
    from spirob.geometry import from_params
    from csv2geom_nlobe import build_unit_inputs
    from spirob_csv_generator import validate_params
    validate_params(params)
    if not prefix or Path(prefix).name != prefix or prefix in ('.', '..'):
        raise ValueError('prefix must be a filename, without directory components')
    _positive('stl_tolerance', stl_tolerance)
    geometry = geometry or from_params(params)
    units = build_unit_inputs(pd.read_csv(csv_file), geometry)
    shape, report = build_cad(units, geometry, params, profile=profile, plain=plain,
                             flat_thickness_m=flat_thickness_m, flat_edge_ratio=flat_edge_ratio,
                             neck_width_mm=neck_width_mm, cable_hole_diameter_mm=cable_hole_diameter_mm,
                             fuse=fuse)
    # Origin at the base centre; CSV frame +Z points base -> tip. No post_gen pose.
    shape = shape.translate((0, 0, -geometry.units[0].local_frame_origin_m[2]*1000))
    dest = Path(outdir); dest.mkdir(parents=True, exist_ok=True)
    step, stl, manifest = (dest/f'{prefix}{ext}' for ext in ('.step', '.stl', '_cad_report.json'))
    cq.exporters.export(shape, str(step))  # mm geometry and STEP's mm declaration agree
    shape.exportStl(str(stl), tolerance=stl_tolerance*1000, relative=False)
    import trimesh
    mesh = trimesh.load(str(stl), force='mesh')
    # Remove duplicate/zero-area tessellation faces and weld numerically identical
    # vertices. This is not hole filling; a remaining open mesh fails the export.
    mesh.merge_vertices(digits_vertex=7)  # 1e-7 mm, below OCC's modelling tolerance
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if profile == 'fabrication' and not mesh.is_volume:
        raise ValueError('Fabrication STL is not a closed oriented volume')
    mesh.export(str(stl))
    check_mesh = trimesh.load(str(stl), force='mesh')
    if profile == 'fabrication' and not check_mesh.is_volume:
        raise ValueError('Fabrication STL round-trip validation failed')
    report.update({'stl_watertight':bool(check_mesh.is_watertight),
                   'stl_oriented_volume':bool(check_mesh.is_volume),
                   'stl_vertex_merge_decimal_places_mm':7})
    restored = cq.importers.importStep(str(step)).val()
    bb = restored.BoundingBox()
    if not restored.isValid() or len(restored.Solids()) != report['solid_count']:
        raise ValueError('STEP round-trip validation failed')
    report.update({'units':'mm', 'stl_import_units':'mm', 'n_elements':len(units),
                   'cadquery_version':cq.__version__, 'volume_mm3':restored.Volume(),
                   'bounds_mm':[[bb.xmin,bb.ymin,bb.zmin],[bb.xmax,bb.ymax,bb.zmax]],
                   'lengths':asdict(geometry.lengths), 'params':params,
                   'csv_sha256':hashlib.sha256(Path(csv_file).read_bytes()).hexdigest(),
                   'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (step,stl)},
                   'frame':'base centre at z=0; +Z toward tip; post_gen pose not applied',
                   'notes':['Fabrication flexure and lens dimensions require mechanical calibration.',
                            'Cable holes are optional; zero diameter leaves undrilled geometry.',
                            'Single valid solid is a topology check, not a print/process qualification.']})
    manifest.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f'CAD: {step}, {stl}; {len(units)} elements, {report["solid_count"]} solid(s), millimetres')
    return CadExportResult(str(step),str(stl),len(units),profile=='fabrication' or fuse,str(manifest))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--in',dest='input',default='Geom_Data_CSV/Spirob_geom_data.csv')
    p.add_argument('--params',default='params.json')
    p.add_argument('--outdir',default='cad'); p.add_argument('--prefix',default='spirob')
    p.add_argument('--profile',choices=['fabrication','simulation'],default='fabrication')
    p.add_argument('--fuse',action='store_true'); p.add_argument('--plain',action='store_true')
    p.add_argument('--flat-thickness-m',type=float); p.add_argument('--flat-edge-ratio',type=float)
    p.add_argument('--neck-width-mm',type=float,default=1.0)
    p.add_argument('--cable-hole-diameter-mm',type=float,default=0.0)
    a=p.parse_args()
    process_cad(a.input,json.loads(Path(a.params).read_text(encoding='utf-8')),outdir=a.outdir,
                prefix=a.prefix,fuse=a.fuse,plain=a.plain,profile=a.profile,
                flat_thickness_m=a.flat_thickness_m,flat_edge_ratio=a.flat_edge_ratio,
                neck_width_mm=a.neck_width_mm,cable_hole_diameter_mm=a.cable_hole_diameter_mm)

if __name__=='__main__': main()
