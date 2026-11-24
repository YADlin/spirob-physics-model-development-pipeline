"""Design family for the taper-selection study.

Every design is a two-cable SpiRob described by the four inputs of this
repository's generator: continuous length L, nominal tip width d_tip, full
taper angle phi and discretisation step Delta_theta. Models are produced by the
repository's own ``build.py`` so that the study exercises the real pipeline.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
BUILD_ROOT = STUDY / "_build"          # generated models (git-ignored)
TEMPLATE = REPO / "examples" / "params-two-cable.json"


@dataclass(frozen=True)
class Design:
    name: str
    L: float              # continuous centre-spiral length, m
    d_tip: float          # nominal tip width, m
    phi_deg: float        # full included taper angle, deg
    dtheta_deg: float     # discretisation step, deg
    policy: str = "exact_requested_length"
    group: str = "sweep"
    note: str = ""

    @property
    def build_dir(self) -> Path:
        return BUILD_ROOT / self.name


# Current robot used in spirob-mjlab (L, tip width, taper and step copied from
# its generator parameters; 21 links).
CURRENT = Design("current_6p3", 0.22628, 0.007139, 6.3, 30.78, group="reference",
                 note="robot currently used in spirob-mjlab")

# Main sweep: only the taper changes. Length and tip width are those of the
# current robot so that the comparison isolates the taper angle.
SWEEP_PHI = (4.0, 5.0, 6.3, 8.0, 10.0, 12.0, 15.0)
SWEEP = tuple(
    Design(f"phi{str(p).replace('.', 'p')}", 0.22628, 0.007139, p, 30.0)
    for p in SWEEP_PHI
)

# Validation set 1: the paper's Fig. 2D-F robots (L = 25 cm, tip 5.5 mm).
PAPER = tuple(
    Design(f"paper_phi{int(p)}", 0.25, 0.0055, p, 30.0, group="paper",
           note="Wang et al. 2025, Fig. 2D-F")
    for p in (5.0, 10.0, 15.0)
)

# Validation set 2: the authors' published print file
# (Open-Spiral-Robots/fabrication/example/spi_rob_3dprinting.3mf): b = 0.1764,
# a = 4.95 mm, Delta_theta = 30 deg, 24 units, scaled by 0.5164 in the slicer.
# Its continuous length and tip width are derived in analytic.osr_print_design().
OSR_PRINT_NAME = "osr_print_phi10"


# Length study: the taper candidates at four lengths, tip width held at the
# current robot's 7.139 mm (the tip is set by the cable holes and the printer).
LENGTHS = (0.22628, 0.30, 0.375, 0.45)
LENGTH_PHI = (6.3, 8.0, 10.0)


def length_design(L: float, phi: float) -> Design:
    if abs(L - 0.22628) < 1e-9:
        return sweep_by_phi()[phi]
    # whole units: the exact-length policy can leave a base stub too short for
    # its joint face at these lengths; the assembled length is measured anyway
    return Design(f"L{int(round(L * 1000))}_phi{str(phi).replace('.', 'p')}", L, 0.007139, phi, 30.0,
                  policy="whole_units", group="length")


def length_grid() -> list[Design]:
    return [length_design(L, p) for L in LENGTHS for p in LENGTH_PHI]


def sweep_by_phi() -> dict[float, Design]:
    return {d.phi_deg: d for d in SWEEP}


def params_for(design: Design) -> dict:
    p = json.loads(TEMPLATE.read_text())
    p["L"] = design.L
    p["d_tip"] = design.d_tip
    p["phi_deg"] = design.phi_deg
    p["Delta_theta_deg"] = design.dtheta_deg
    p["terminal_unit_policy"] = design.policy
    p["post_gen"]["robot_pos"] = [0, 0, design.L]
    p["_comment"] = f"taper-selection study design {design.name}"
    return p


def build(design: Design, force: bool = False) -> Path:
    """Run the repository pipeline for one design; return the output dir."""
    out = design.build_dir
    xml = out / "spirob_physics_model.xml"
    if xml.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    pfile = BUILD_ROOT / f"{design.name}.params.json"
    pfile.write_text(json.dumps(params_for(design), indent=2))
    cmd = [sys.executable, str(REPO / "build.py"), "--params", str(pfile),
           "--no-preview", "--output-dir", str(out)]
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"build failed for {design.name}:\n{res.stdout}\n{res.stderr}")
    return out


def describe(design: Design) -> dict:
    return asdict(design)
