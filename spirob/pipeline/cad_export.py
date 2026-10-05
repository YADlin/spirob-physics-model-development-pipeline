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
    iges_path: str | None = None


def _positive(name, value, allow_zero=False):
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def _lens_element_mm(unit, thickness_mm, edge_ratio):
    from spirob.sections import lens_solid_mm
    return lens_solid_mm(unit.profile_xyz, thickness_mm, edge_ratio)


def _simulation_element_mm(unit, params, plain=False):
    from spirob.sections import resolve_section_params
    params = resolve_section_params(params, plain=plain)
    from spirob.pipeline.csv2geom_nlobe import build_flat_element, make_profile_from_points, revolve_profile, add_nlobe_cut
    n = params['n_cables']
    if n == 2 and not plain:
        from spirob.sections import resolve_flat_thickness_ratio, axial_thickness_law, thickness_at_z
        law = axial_thickness_law(params) if params.get('thickness_profile', 'linear') != 'stepped' else None
        endpoints = tuple(thickness_at_z(law, unit.profile_xyz[j][2]) for j in (0,1)) if law else None
        shape = build_flat_element(unit.row, resolve_flat_thickness_ratio(params),
                                   hex_edge_ratio=params.get('hex_edge_ratio') if params.get('flat_section') == 'hex' else None,
                                   endpoint_thicknesses_m=endpoints)
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
              flat_thickness_m=None, flat_edge_ratio=None, neck_width_mm=None, elastic_core_percent=None,
              cable_hole_diameter_mm=0.0, fuse=False):
    import cadquery as cq
    from spirob.sections import resolve_section_params
    params = resolve_section_params(params, plain=plain)
    hex_section = params.get('flat_section') == 'hex'
    n = geometry.inputs.n_cables
    from spirob.sections import resolve_flat_thickness_ratio, axial_thickness_law, thickness_at_z
    ratio = resolve_flat_thickness_ratio(params, geometry) if n == 2 and not plain else .3
    continuous = n == 2 and not plain and params.get('thickness_profile', 'linear') != 'stepped'
    law = axial_thickness_law(params, geometry) if continuous else None
    scaled_flat = n == 2 and not plain and (continuous or hex_section or 'base_thickness_m' in params or 'flat_thickness_ratio' not in params)
    if scaled_flat and (flat_thickness_m is not None or flat_edge_ratio is not None):
        raise ValueError('Use --base-thickness-mm and --hex-edge-ratio with this section, not legacy manufacturing-only flat overrides')
    if profile not in ('fabrication', 'simulation'):
        raise ValueError('profile must be fabrication or simulation')
    edge = float(params.get('flat_edge_ratio', .25) if flat_edge_ratio is None else flat_edge_ratio)
    if not math.isfinite(edge) or not 0 < edge <= 1:
        raise ValueError('flat_edge_ratio must be in (0, 1]')
    thickness = _positive('flat_thickness_m', flat_thickness_m if flat_thickness_m is not None
                          else ratio*2*max(u.outer_radius_m for u in units))*1000
    from spirob.core import resolve_core_percent, core_dimensions, reference_width_at
    if elastic_core_percent is None and neck_width_mm is None:
        elastic_core_percent = params.get('build', {}).get('elastic_core_percent')
        neck_width_mm = params.get('build', {}).get('neck_width_mm')
    percent = resolve_core_percent(geometry, elastic_core_percent, neck_width_mm)
    core_report = core_dimensions(geometry, percent)
    width_law = core_report['width_law']
    def neck_at(z_mm):
        return percent/100 * reference_width_at(width_law, z_mm/1000)*1000
    hole = _positive('cable_hole_diameter_mm', cable_hole_diameter_mm, True)
    n = geometry.inputs.n_cables
    if profile == 'simulation' and (hole or flat_thickness_m is not None or flat_edge_ratio is not None):
        raise ValueError('Manufacturing thickness/hole overrides require profile=fabrication')
    from spirob.mesh_assets import mesh_assets
    assets = mesh_assets(geometry, 'shared')
    templates = {}
    shapes = []
    for index, unit in enumerate(units):
        if profile == 'fabrication' and n == 2 and not plain and not scaled_flat:
            # Legacy constant-thickness manufacturing lens is not uniformly similar.
            shape = _lens_element_mm(unit, thickness, edge)
        elif n != 2 or plain or params.get('thickness_profile') == 'constant':
            # Constant Y cannot use uniform XYZ scaling of a template.
            # OCC's curved n-lobe booleans show small volume differences when
            # scaled after construction. Preserve individual CAD construction
            # until that numerical contract is separately established.
            shape = _simulation_element_mm(unit, params, plain)
        else:
            asset = assets[index]
            if asset.source_index not in templates:
                source = units[asset.source_index]
                templates[asset.source_index] = _simulation_element_mm(source, params, plain).translate(
                    tuple(-v*1000 for v in source.origin_m))
            template = templates[asset.source_index]
            shape = (template if asset.scale == 1 else template.scale(asset.scale)).translate(
                tuple(v*1000 for v in unit.origin_m))
        if not shape.isValid() or shape.Volume() <= 0:
            raise ValueError(f'{unit.link_name}: invalid element solid')
        shapes.append(shape)
    z0 = geometry.units[0].local_frame_origin_m[2]*1000
    z1 = geometry.units[-1].slit_reference_m[2]*1000
    if profile == 'fabrication':
        # A finite central ligament makes the zero-width simulation hinges a
        # connected physical design. Fuse in mm to avoid metre-scale OCC tolerances.
        if scaled_flat:
            # The ligament must cover the entire shared centreline at each
            # hinge, including the ridge tips. A rectangular inset leaves
            # zero-thickness touching edges outside it (non-manifold STL).
            # Use a narrow six-sided loft following each link's ridge instead.
            nodes = [(u.profile_xyz[0][2]*1000, u.outer_radius_m*1000) for u in units]
            nodes.append((units[-1].profile_xyz[1][2]*1000, units[-1].outer_radius_m*1000))
            wires = []
            for z, radius in nodes:
                neck = neck_at(z)
                h = thickness_at_z(law,z/1000)*500 if law else radius*ratio
                he = h*(1-(1-params.get('hex_edge_ratio', 1.))*neck/(2*radius))
                xy = [(-neck/2, -he), (0, -h), (neck/2, -he),
                      (neck/2, he), (0, h), (-neck/2, he)]
                if not hex_section:
                    xy = [(-neck/2,-h), (neck/2,-h), (neck/2,h), (-neck/2,h)]
                wires.append(cq.Wire.makePolygon([cq.Vector(x,y,z) for x,y in xy], close=True))
            core = cq.Solid.makeLoft(wires, ruled=True)
        elif n == 2 and not plain:
            wires = [cq.Wire.makePolygon([cq.Vector(x,y,z) for x,y in
                     [(-neck_at(z)/2,-thickness/2),(neck_at(z)/2,-thickness/2),
                      (neck_at(z)/2,thickness/2),(-neck_at(z)/2,thickness/2)]], close=True) for z in (z0,z1)]
            core = cq.Solid.makeLoft(wires, ruled=True)
        else:
            core = cq.Solid.makeCone(neck_at(z0)/2, neck_at(z1)/2, z1-z0, cq.Vector(0,0,z0))
        combined = core.fuse(*shapes).clean()
    elif fuse:
        combined = shapes[0].fuse(*shapes[1:]).clean()
    else:
        combined = cq.Compound.makeCompound(shapes)
    before_volume = combined.Volume()
    from spirob.fabrication_routes import cable_hole_axes, cylinder_for_axis
    axes = cable_hole_axes(geometry) if hole else []
    for axis in axes:
        # One straight cylinder: no rounded sweep joints, self-intersections,
        # spherical/toric transition patches, or long remote extensions.
        tool = cylinder_for_axis(axis, hole)
        combined = combined.cut(tool).clean()
        if not combined.intersect(tool).Volume() <= 1e-6:
            raise ValueError(f"Cable {axis['cable_index']}: material obstructs the bore")
    if hole and before_volume-combined.Volume() <= 1e-6:
        raise ValueError('Cable paths removed no material; inspect routing')
    if not combined.isValid() or combined.Volume() <= 0:
        raise ValueError('CAD boolean operations produced invalid geometry')
    solid_count = len(combined.Solids())
    if profile == 'fabrication' and solid_count != 1:
        raise ValueError(f'Fabrication model has {solid_count} disconnected solids; adjust neck/hole dimensions')
    section_name = ('plain' if plain else f'{n}-lobe' if n >= 3 else 'hex' if hex_section
                    else 'fabrication_lens' if profile == 'fabrication' and not scaled_flat else 'rectangular')
    return combined, {'profile': profile, 'flat_section': section_name,
                      'thickness_profile': params.get('thickness_profile','linear') if n == 2 and not plain else None,
                      'hex_edge_ratio': params.get('hex_edge_ratio') if hex_section else None,
                      'thickness_scales_with_link': (profile == 'simulation' or scaled_flat) if n == 2 and not plain else None, 'solid_count': solid_count, 'valid': True, 'unique_link_solids_built': len(templates) if templates else len(units),
                      'elastic_core': core_report if profile == 'fabrication' else None,
                      'cable_hole_diameter_mm': hole,
                      'cable_hole_route': 'straight between first and last simulation anchors, extended to end planes',
                      'cable_hole_axes_csv_frame': axes,
                      'flat_centre_thickness_mm': thickness if n == 2 and profile == 'fabrication' else None,
                      'flat_edge_ratio': (params['hex_edge_ratio'] if hex_section else 1. if scaled_flat else edge) if n == 2 and profile == 'fabrication' else None,
                      'removed_hole_volume_mm3': before_volume-combined.Volume()}


def process_cad(csv_file, params, *, outdir='cad', prefix='spirob', fuse=False,
                plain=False, stl_tolerance=1e-4, flat_thickness_m=None,
                flat_edge_ratio=None, geometry=None, profile='fabrication',
                neck_width_mm=None, elastic_core_percent=None, cable_hole_diameter_mm=0.0, iges=False):
    import cadquery as cq
    import pandas as pd
    from spirob.geometry import from_params
    from spirob.pipeline.csv2geom_nlobe import build_unit_inputs
    from spirob.pipeline.spirob_csv_generator import validate_params
    from spirob.sections import resolve_section_params
    params = resolve_section_params(params, plain=plain)
    validate_params(params)
    if not prefix or Path(prefix).name != prefix or prefix in ('.', '..'):
        raise ValueError('prefix must be a filename, without directory components')
    _positive('stl_tolerance', stl_tolerance)
    geometry = geometry or from_params(params)
    units = build_unit_inputs(pd.read_csv(csv_file), geometry)
    shape, report = build_cad(units, geometry, params, profile=profile, plain=plain,
                             flat_thickness_m=flat_thickness_m, flat_edge_ratio=flat_edge_ratio,
                             neck_width_mm=neck_width_mm, elastic_core_percent=elastic_core_percent, cable_hole_diameter_mm=cable_hole_diameter_mm,
                             fuse=fuse)
    # Origin at the base centre; CSV frame +Z points base -> tip. No post_gen pose.
    shape = shape.translate((0, 0, -geometry.units[0].local_frame_origin_m[2]*1000))
    dest = Path(outdir); dest.mkdir(parents=True, exist_ok=True)
    step, stl, manifest = (dest/f'{prefix}{ext}' for ext in ('.step', '.stl', '_cad_report.json'))
    cq.exporters.export(shape, str(step))  # mm geometry and STEP's mm declaration agree
    # Resolve small bores accurately enough that STL facets do not close them.
    mesh_tolerance_mm = min(stl_tolerance*1000, cable_hole_diameter_mm*.005) if cable_hole_diameter_mm else stl_tolerance*1000
    shape.exportStl(str(stl), tolerance=mesh_tolerance_mm, relative=False)
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
    if profile == 'fabrication' and len(check_mesh.split(only_watertight=False)) != 1:
        raise ValueError('Fabrication STL contains detached components')
    if cable_hole_diameter_mm:
        import numpy as np
        from spirob.fabrication_routes import cable_hole_axes
        for axis in cable_hole_axes(geometry):
            a=np.array(axis['base_m'])*1000; b=np.array(axis['tip_m'])*1000
            a[2]-=geometry.units[0].local_frame_origin_m[2]*1000
            b[2]-=geometry.units[0].local_frame_origin_m[2]*1000
            d=(b-a)/np.linalg.norm(b-a)
            u=np.cross(d,[0,1,0]);u/=np.linalg.norm(u);v=np.cross(d,u)
            origins=[a-d*cable_hole_diameter_mm*2]
            for r in (.25,.475):
                origins.extend(a-d*cable_hole_diameter_mm*2+cable_hole_diameter_mm*r*(u*np.cos(t)+v*np.sin(t)) for t in np.linspace(0,2*np.pi,16,endpoint=False))
            hits,_,_=check_mesh.ray.intersects_location(np.array(origins),np.tile(d,(len(origins),1)))
            if len(hits): raise ValueError('STL contains material obstructing a sampled cable-bore probe')
        report['stl_bore_check']={'method':'33 axial rays per bore through centre and rings up to 95% diameter','passed':True}
    report['stl_tolerance_mm']=mesh_tolerance_mm
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
    if cable_hole_diameter_mm:
        from spirob.fabrication_routes import cable_hole_axes, cylinder_for_axis
        residuals=[]
        for axis in cable_hole_axes(geometry):
            probe=cylinder_for_axis(axis, cable_hole_diameter_mm*.999,
                                    geometry.units[0].local_frame_origin_m[2])
            residual=restored.intersect(probe).Volume()
            if residual > 1e-6:
                raise ValueError('STEP round-trip has material inside a cable bore')
            residuals.append(residual)
        report['bore_check']={'method':'full-length 99.9%-diameter cylinder intersection',
                              'residual_volumes_mm3':residuals,'tolerance_mm3':1e-6}
    iges_path = None
    if iges:
        from OCP.IGESControl import IGESControl_Writer, IGESControl_Reader
        from OCP.IFSelect import IFSelect_RetDone
        writer = IGESControl_Writer('MM', 1)  # BRep mode preserves trimmed face topology.
        iges_path = dest / f'{prefix}.iges'
        if not writer.AddShape(shape.wrapped) or not writer.Write(str(iges_path)):
            raise ValueError('IGES export failed')
        reader = IGESControl_Reader()
        if reader.ReadFile(str(iges_path)) != IFSelect_RetDone or reader.TransferRoots() < 1:
            raise ValueError('IGES round-trip import failed')
        restored_iges = cq.Shape.cast(reader.OneShape())
        ib = restored_iges.BoundingBox()
        if max(abs(x-y) for x,y in zip((ib.xmin,ib.ymin,ib.zmin,ib.xmax,ib.ymax,ib.zmax),(bb.xmin,bb.ymin,bb.zmin,bb.xmax,bb.ymax,bb.zmax))) > .01:
            raise ValueError('IGES round-trip dimensions differ by more than 0.01 mm')
        if len(restored_iges.Faces()) != len(shape.Faces()):
            raise ValueError('IGES round-trip changed the trimmed face count')
        if len(restored_iges.Solids()) != len(shape.Solids()):
            raise ValueError('IGES round-trip changed the solid count')
        if cable_hole_diameter_mm:
            for axis in cable_hole_axes(geometry):
                probe=cylinder_for_axis(axis,cable_hole_diameter_mm*.999,
                                        geometry.units[0].local_frame_origin_m[2])
                if restored_iges.intersect(probe).Volume() > 1e-6:
                    raise ValueError('IGES round-trip has material inside a cable bore')
        report['iges'] = {'units':'mm', 'representation':'trimmed BRep', 'round_trip_bounds_tolerance_mm':.01,
                          'round_trip_solid_count':len(restored_iges.Solids()),'round_trip_face_count':len(restored_iges.Faces()),
                          'bore_probe_passed':True if cable_hole_diameter_mm else None}
        report['files'][iges_path.name] = hashlib.sha256(iges_path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f'CAD: {step}, {stl}; {len(units)} elements, {report["solid_count"]} solid(s), millimetres')
    return CadExportResult(str(step),str(stl),len(units),profile=='fabrication' or fuse,str(manifest),str(iges_path) if iges_path else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--in',dest='input',default='Geom_Data_CSV/Spirob_geom_data.csv', help='Input canonical geometry CSV')
    p.add_argument('--params',default='params.json', help='Input SpiRob parameters JSON')
    p.add_argument('--outdir',default='cad', help='Destination directory for CAD exports and their validation report'); p.add_argument('--prefix',default='spirob', help='Output filename stem, without directory components')
    p.add_argument('--profile',choices=['fabrication','simulation'],default='fabrication', help='fabrication adds flexures/channels; simulation assembles rigid link surfaces')
    p.add_argument('--fuse',action='store_true', help='Fuse the simulation assembly when possible')
    p.add_argument('--plain',action='store_true', help='Use a circular revolved section')
    p.add_argument('--iges',action='store_true', help='Also export trimmed IGES BRep in mm and validate round-trip dimensions')
    p.add_argument('--flat-thickness-m',type=float, help='Legacy fabrication lens thickness in metres; current sections use base-thickness-mm'); p.add_argument('--flat-edge-ratio',type=float, help='Legacy fabrication lens edge / centre thickness in (0,1]')
    p.add_argument('--neck-width-mm',type=float, help='Legacy base core width/diameter in mm; converted to a tapered width percentage')
    p.add_argument('--elastic-core-percent',type=float, help='Core X width (2 cables) or diameter (n cables) as percent of local reference width; default 5')
    p.add_argument('--cable-hole-diameter-mm',type=float,default=0.0, help='Fabrication cable channel diameter in mm; zero means no drilling')
    from spirob.sections import section_arguments, resolve_section_params
    section_arguments(p)
    a=p.parse_args()
    params = resolve_section_params(json.loads(Path(a.params).read_text(encoding='utf-8')),
                                    hex_section=a.hex_section, hex_edge_ratio=a.hex_edge_ratio, base_thickness_mm=a.base_thickness_mm, thickness_profile=a.thickness_profile, plain=a.plain)
    process_cad(a.input,params,outdir=a.outdir,
                prefix=a.prefix,fuse=a.fuse,plain=a.plain,profile=a.profile,
                flat_thickness_m=a.flat_thickness_m,flat_edge_ratio=a.flat_edge_ratio,
                neck_width_mm=a.neck_width_mm,elastic_core_percent=a.elastic_core_percent,cable_hole_diameter_mm=a.cable_hole_diameter_mm,iges=a.iges)

if __name__=='__main__': main()
