"""Canonical SpiRob geometry layer.

This package holds the single authoritative geometric description of a SpiRob.
It depends only on the Python standard library and NumPy — never on
matplotlib, MuJoCo, CadQuery, Qt or VTK — so it can be imported by headless
tooling, the CSV/XML pipeline, the preview, and a future CAD exporter alike.
"""

from .geometry import (  # noqa: F401
    TerminalUnitPolicy,
    UserInputs,
    SpiralParameters,
    LengthReport,
    UnitRecord,
    TendonPoint,
    TendonPath,
    BaseFrame,
    SpiRobGeometry,
    Tolerances,
    build_geometry,
)

__all__ = [
    "TerminalUnitPolicy",
    "UserInputs",
    "SpiralParameters",
    "LengthReport",
    "UnitRecord",
    "TendonPoint",
    "TendonPath",
    "BaseFrame",
    "SpiRobGeometry",
    "Tolerances",
    "build_geometry",
]
