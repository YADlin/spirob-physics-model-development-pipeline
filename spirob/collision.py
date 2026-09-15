"""Massless native-primitive colliders for the flat, two-cable CAD profile.

The XZ outline is rounded *inwards*: inset the convex polygon by r, then
cover its dilation by r with edge boxes, corner cylinders, and interior
boxes. Cylinder axes run along Y, preserving the CAD's flat front/back.
MuJoCo treats these as overlapping geoms, not a Boolean-fused solid.
"""
from __future__ import annotations

import math
import numpy as np


def flat_outline(geometry, index):
    """Return the unshifted CAD outline in the link frame, in metres."""
    q = geometry.inverted_quads()[index]
    q = q - q[0]
    polygon = np.array([q[0], q[3], q[2], q[1],
                        [-q[2, 0], q[2, 1]], [-q[3, 0], q[3, 1]]])
    # Remove collinear vertices (the base has a single flat mounting face).
    keep = []
    for i, p in enumerate(polygon):
        u = p - polygon[i-1]
        v = polygon[(i+1) % len(polygon)] - p
        if abs(u[0]*v[1] - u[1]*v[0]) > 1e-12*np.linalg.norm(u)*np.linalg.norm(v):
            keep.append(p)
    polygon = np.array(keep)
    area2 = np.sum(polygon[:, 0]*np.roll(polygon[:, 1], -1)
                   - polygon[:, 1]*np.roll(polygon[:, 0], -1))
    return polygon if area2 > 0 else polygon[::-1]


def rounded_flat_primitives(geometry, index, thickness_ratio, radius_ratio=0.04):
    """Return XML-ready shape attributes; contact/mass policy is set by caller.

    Radius is relative to the link's half-width. Every primitive is contained
    in the original extrusion. Straight outline edges remain tangent; only
    corners are rounded off. This changes the contact approximation, not CAD.
    """
    if geometry.inputs.n_cables != 2:
        raise ValueError('Compound box/cylinder collision currently supports the two-cable flat section only')
    if not (math.isfinite(thickness_ratio) and thickness_ratio > 0):
        raise ValueError('flat_thickness_ratio must be positive and finite')
    if not (math.isfinite(radius_ratio) and 0.005 <= radius_ratio <= 0.2):
        raise ValueError('collision corner radius ratio must be between 0.005 and 0.2')
    p = flat_outline(geometry, index)
    half_width = float(np.max(np.abs(p[:, 0])))
    half_y = half_width * thickness_ratio  # Matches build_flat_element extrusion.
    r = half_width * radius_ratio
    edges = np.roll(p, -1, axis=0) - p
    tangents = edges / np.linalg.norm(edges, axis=1)[:, None]
    normals = np.column_stack([tangents[:, 1], -tangents[:, 0]])
    bounds = np.einsum('ij,ij->i', normals, p) - r
    inset = np.array([np.linalg.solve(normals[[i-1, i]], bounds[[i-1, i]])
                      for i in range(len(p))])
    if np.any(inset @ normals.T > bounds + 1e-12):
        raise ValueError('Corner radius is too large for this partial link; reduce --collision-corner-radius-ratio')
    fmt = lambda values: ' '.join(format(float(v), '.17g') for v in values)
    shapes = []
    for i, u in enumerate(inset):
        v = inset[(i+1) % len(inset)]
        delta = v-u
        length = np.linalg.norm(delta)
        if length <= 1e-12:
            raise ValueError('Corner radius collapses an outline edge; reduce --collision-corner-radius-ratio')
        angle = -math.atan2(delta[1], delta[0])
        shapes.append(dict(type='box', pos=fmt([(u[0]+v[0])/2, 0, (u[1]+v[1])/2]),
                           size=fmt([length/2, half_y, r]),
                           quat=fmt([math.cos(angle/2), 0, math.sin(angle/2), 0])))
        shapes.append(dict(type='cylinder', fromto=fmt([u[0], -half_y, u[1], u[0], half_y, u[1]]),
                           size=fmt([r])))

    # Fill the inset's interior. Edge bands cover the staircase error, bounded
    # by max_slope * strip_width <= r. No exposed internal steps or holes.
    nonvertical = abs(normals[:, 1]) > 1e-12
    max_slope = np.max(abs(normals[nonvertical, 0] / normals[nonvertical, 1]))
    xmin, xmax = inset[:, 0].min(), inset[:, 0].max()
    count = max(1, math.ceil(1.02*(xmax-xmin)*max_slope/r))
    cuts = np.linspace(xmin, xmax, count+1)
    lower = normals[:, 1] < -1e-12
    upper = normals[:, 1] > 1e-12
    for left, right in zip(cuts[:-1], cuts[1:]):
        # Small overlap prevents floating-point cracks at shared box faces.
        overlap = 0.01*(right-left)
        left, right = max(xmin, left-overlap), min(xmax, right+overlap)
        zlo = max(np.max((bounds[lower]-normals[lower, 0]*x)/normals[lower, 1])
                  for x in (left, right))
        zhi = min(np.min((bounds[upper]-normals[upper, 0]*x)/normals[upper, 1])
                  for x in (left, right))
        if zhi <= zlo:
            raise ValueError('Partial link is too thin for this collider; reduce the corner radius')
        shapes.append(dict(type='box', pos=fmt([(left+right)/2, 0, (zlo+zhi)/2]),
                           size=fmt([(right-left)/2, half_y, (zhi-zlo)/2])))
    return shapes
