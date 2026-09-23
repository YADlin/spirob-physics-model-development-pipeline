"""Fabrication core proportional to the continuous straightened width envelope.

This is a geometric design law, not an identified material model. Width is
linear in straight axial position; complete-link stations form a geometric
sequence. The finite partial base is handled using its actual axial extent.
"""
from __future__ import annotations
import math

DEFAULT_CORE_PERCENT = 5.0


def reference_width_law(geometry):
    s = geometry.spiral
    angle = geometry.inputs.delta_theta_rad
    beta_minus_one = math.expm1(s.b * angle)
    beta = 1 + beta_minus_one
    radius = .5 * s.a_m * (s.E + 1)
    chord = radius * math.hypot(beta * math.cos(angle)-1, beta * math.sin(angle))
    z0 = geometry.units[0].local_frame_origin_m[2]
    z1 = geometry.units[-1].slit_reference_m[2]
    apex = z1 + chord / beta_minus_one
    width = geometry.units[0].realized_width_m
    return dict(z_base_m=z0, z_tip_m=z1, virtual_apex_z_m=apex,
                base_width_m=width, slope_m_per_m=-width/(apex-z0))


def reference_width_at(law, z):
    return law['base_width_m'] + law['slope_m_per_m'] * (z-law['z_base_m'])


def core_dimensions(geometry, percent=DEFAULT_CORE_PERCENT):
    if isinstance(percent, bool) or not isinstance(percent, (int, float)) or not math.isfinite(percent) or not 0 < percent < 100:
        raise ValueError('elastic_core_percent must be finite and strictly between 0 and 100')
    law = reference_width_law(geometry)
    stations = [u.local_frame_origin_m[2] for u in geometry.units] + [law['z_tip_m']]
    return dict(elastic_core_percent=float(percent), reference='continuous straightened pre-notch width envelope',
                axial_law='linear', width_law=law,
                stations=[dict(z_from_base_mm=(z-law['z_base_m'])*1000,
                               reference_width_mm=reference_width_at(law,z)*1000,
                               core_width_mm=reference_width_at(law,z)*percent*10) for z in stations])


def resolve_core_percent(geometry, percent=None, neck_width_mm=None):
    """Old millimetre input now anchors the tapered core at the base only."""
    if percent is not None and neck_width_mm is not None:
        raise ValueError('Choose elastic_core_percent or legacy neck_width_mm, not both')
    if neck_width_mm is not None:
        if isinstance(neck_width_mm, bool) or not math.isfinite(neck_width_mm) or neck_width_mm <= 0:
            raise ValueError('neck_width_mm must be finite and positive')
        percent = 100 * neck_width_mm / (geometry.units[0].realized_width_m*1000)
    percent = DEFAULT_CORE_PERCENT if percent is None else percent
    core_dimensions(geometry, percent)  # validate all entry points, including direct CAD calls
    return float(percent)
