"""Optional transverse sections, independently constructed from our XZ profile.

The six-sided section has vertices (0, +/-H), (+/-R, +/-e*H).
It preserves the existing axial slit faces and the partial base's flat mount.
"""
from __future__ import annotations

import math


def section_arguments(parser):
    parser.add_argument('--base-thickness-mm', metavar='MM|auto',
                        help='Two-cable base centre thickness in mm; auto makes thickness equal width at every link')
    parser.add_argument('--hex-section', action='store_true',
                        help='Two-cable six-sided XY section with centre ridges; opt-in')
    parser.add_argument('--hex-edge-ratio', type=float,
                        help='Edge thickness / centre thickness, strictly between 0 and 1 (default: 0.75)')


def resolve_section_params(params, *, hex_section=False, hex_edge_ratio=None,
                           base_thickness_mm=None, plain=False):
    """Return a copy, preserving legacy defaults unless explicitly selected."""
    result = dict(params)
    if base_thickness_mm is not None:
        if result.get('n_cables') != 2 or plain:
            raise ValueError('--base-thickness-mm requires n_cables=2 without --plain')
        if isinstance(base_thickness_mm, str) and base_thickness_mm.strip().lower() == 'auto':
            result['base_thickness_m'] = None
        else:
            if isinstance(base_thickness_mm, bool):
                raise ValueError('base thickness must be a positive finite number in mm, or auto')
            try:
                result['base_thickness_m'] = float(base_thickness_mm)/1000
            except (TypeError, ValueError):
                raise ValueError('base thickness must be a positive finite number in mm, or auto') from None
        result.pop('flat_thickness_ratio', None)
    if 'base_thickness_m' in result:
        if 'flat_thickness_ratio' in result:
            raise ValueError('Specify base_thickness_m OR legacy flat_thickness_ratio, not both')
        value = result['base_thickness_m']
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('base_thickness_m must be positive and finite, or null for thickness=width')
            if result.get('n_cables') != 2 or plain:
                raise ValueError('base_thickness_m applies only to the two-cable flat section')
    section = 'hex' if hex_section else result.get('flat_section', 'rectangular')
    if section not in ('rectangular', 'hex'):
        raise ValueError('flat_section must be rectangular or hex')
    if section != 'hex':
        if hex_edge_ratio is not None or 'hex_edge_ratio' in result:
            raise ValueError('hex_edge_ratio requires --hex-section or flat_section="hex"')
        return result
    if result.get('n_cables') != 2 or plain:
        raise ValueError('--hex-section requires n_cables=2 without --plain')
    edge = result.get('hex_edge_ratio', .75) if hex_edge_ratio is None else hex_edge_ratio
    if isinstance(edge, bool) or not isinstance(edge, (int, float)) or not math.isfinite(edge) or not 0 < edge < 1:
        raise ValueError('hex_edge_ratio must be finite and strictly between 0 and 1')
    thickness = result.get('flat_thickness_ratio', 1.)
    if isinstance(thickness, bool) or not isinstance(thickness, (int, float)) or not math.isfinite(thickness) or thickness <= 0:
        raise ValueError('flat_thickness_ratio must be finite and positive')
    result.update(flat_section='hex', hex_edge_ratio=float(edge))
    return result


def resolve_flat_thickness_ratio(params, geometry=None):
    """Internal scale factor; the user sets the largest link's centre thickness.

    Explicit base thickness stays fixed if geometry changes. Null/missing
    thickness selects T_i=W_i. Old files with a ratio retain that ratio.
    """
    params = resolve_section_params(params)
    if params.get('n_cables') != 2:
        raise ValueError('Flat thickness is defined only for two cables')
    if 'base_thickness_m' not in params:
        return float(params.get('flat_thickness_ratio', 1.))
    if params['base_thickness_m'] is None:
        return 1.
    if geometry is None:
        from spirob.geometry import from_params
        geometry = from_params(params)
    return float(params['base_thickness_m'])/geometry.units[0].realized_width_m


def section_dimensions(params, geometry=None):
    """Resolved dimensions in metres, including the partial base if present."""
    if geometry is None:
        from spirob.geometry import from_params
        geometry = from_params(params)
    ratio = resolve_flat_thickness_ratio(params, geometry)
    params = resolve_section_params(params)
    edge = params['hex_edge_ratio'] if params.get('flat_section') == 'hex' else 1.
    links = [dict(link=u.link_name, width_m=u.realized_width_m,
                  centre_thickness_m=u.realized_width_m*ratio,
                  edge_thickness_m=u.realized_width_m*ratio*edge) for u in geometry.units]
    return dict(reference='largest base link, link_001; centre thickness along local Y',
                mode=('absolute_base' if params.get('base_thickness_m') is not None else
                      'legacy_ratio' if 'flat_thickness_ratio' in params else 'thickness_equals_width'),
                resolved_thickness_to_width_ratio=ratio, base=links[0], tip=links[-1], links=links)


def lens_solid_mm(profile_xyz, thickness_mm, edge_ratio):
    """Closed symmetric solid in CSV coordinates, millimetres.

    XZ profile coordinates arrive in metres. Each mirrored half is a closed
    hexahedron. Triangles handle the generally non-planar axial end faces.
    This is the repository's existing fabrication lens construction, shared
    with the new simulation section; no upstream implementation is used.
    """
    import cadquery as cq
    footprint = [(profile_xyz[i][0]*1000, profile_xyz[i][2]*1000) for i in (0, 3, 2, 1)]
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
