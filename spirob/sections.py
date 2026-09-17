"""Optional transverse sections, independently constructed from our XZ profile.

The six-sided section has vertices (0, +/-H), (+/-R, +/-e*H).
It preserves the existing axial slit faces and the partial base's flat mount.
"""
from __future__ import annotations

import math


def section_arguments(parser):
    parser.add_argument('--hex-section', action='store_true',
                        help='Two-cable six-sided XY section with centre ridges; opt-in')
    parser.add_argument('--hex-edge-ratio', type=float,
                        help='Edge thickness / centre thickness, strictly between 0 and 1 (default: 0.75)')


def resolve_section_params(params, *, hex_section=False, hex_edge_ratio=None, plain=False):
    """Return a copy, preserving legacy defaults unless explicitly selected."""
    result = dict(params)
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
    thickness = result.get('flat_thickness_ratio', .3)
    if isinstance(thickness, bool) or not isinstance(thickness, (int, float)) or not math.isfinite(thickness) or thickness <= 0:
        raise ValueError('flat_thickness_ratio must be finite and positive')
    result.update(flat_section='hex', hex_edge_ratio=float(edge))
    return result


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
