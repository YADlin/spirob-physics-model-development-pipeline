"""Source-STL reuse for geometrically similar complete SpiRob links."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeshAsset:
    link_name: str
    filename: str
    scale: float
    source_index: int


def mesh_assets(geometry, layout='shared'):
    """Keep link asset names while sharing one complete-link STL on disk.

    Scale belongs to the MJCF mesh asset, not to a mesh geom. MuJoCo therefore
    still compiles a separate scaled asset for each link size. Partial units
    always retain their own STL. Reject loss of similarity rather than silently
    approximating a changed geometry with the template.
    """
    if layout not in ('shared', 'individual'):
        raise ValueError('mesh layout must be shared or individual')
    units = geometry.units
    reference = next((i for i, u in enumerate(units) if not u.is_partial), None)
    quads = geometry.inverted_quads()
    result = []
    for i, unit in enumerate(units):
        if layout == 'individual' or unit.is_partial:
            result.append(MeshAsset(unit.link_name, unit.link_name + '.stl', 1., i))
            continue
        scale = unit.realized_width_m / units[reference].realized_width_m
        local = quads[i] - quads[i][0]
        template = quads[reference] - quads[reference][0]
        if not np.allclose(local, scale * template, atol=1e-10, rtol=1e-10):
            raise ValueError(f'{unit.link_name}: complete profile is not a scaled copy; use individual meshes')
        result.append(MeshAsset(unit.link_name, 'link_template.stl', scale, reference))
    return result
