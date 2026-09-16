"""Update the SpiRob exponential joint gains, preserving the protected base."""
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
from spirob.xml_tools import read_standalone, set_arena_memory, write_validated


def update_gains(source, output, stiffness, damping, beta, *, anchor='generator',
                 exclude=('j_001',), arena_memory_mib=None, overwrite=False):
    for name, value in [('stiffness', stiffness), ('damping', damping), ('beta', beta)]:
        if not math.isfinite(value) or value < 0 or (name == 'beta' and value == 0):
            raise ValueError(f'{name} must be finite and {"positive" if name == "beta" else "nonnegative"}')
    if anchor not in ('generator', 'first-flexible'):
        raise ValueError('anchor must be generator or first-flexible')
    # Additional exclusions must never accidentally unprotect the base.
    exclude = set(exclude) | {'j_001'}
    original = mujoco.MjModel.from_xml_path(str(Path(source).resolve()))
    tree = read_standalone(source)
    joints = sorted([(int(j.get('name')[2:]), j) for j in tree.findall('.//worldbody//joint')
                     if re.fullmatch(r'j_\d+', j.get('name', ''))], key=lambda pair: pair[0])
    if not joints:
        raise ValueError('No j_NNN joints found')
    names = {j.get('name') for _, j in joints}
    if not set(exclude) <= names:
        raise ValueError(f'Excluded joint names not found: {sorted(set(exclude)-names)}')
    selected = [(n, j) for n, j in joints if j.get('name') not in exclude]
    if not selected:
        raise ValueError('No flexible joints selected')
    origin = 1 if anchor == 'generator' else selected[0][0]
    expected_k, expected_d = original.jnt_stiffness.copy(), original.dof_damping.copy()
    rows = []
    for number, joint in joints:
        name = joint.get('name')
        jid = original.joint(name).id
        if original.jnt_type[jid] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_BALL):
            raise ValueError(f'{name}: only rotational hinge/ball joints are supported')
        adr = original.jnt_dofadr[jid]
        ndof = 3 if original.jnt_type[jid] == mujoco.mjtJoint.mjJNT_BALL else 1
        old_k, old_d = float(original.jnt_stiffness[jid]), original.dof_damping[adr:adr+ndof].tolist()
        protected = name in exclude
        if not protected:
            try:
                factor = beta ** (-3*(number-origin))
            except OverflowError as exc:
                raise ValueError('The requested decay law overflows') from exc
            k, d = stiffness*factor, damping*factor
            if not math.isfinite(k+d):
                raise ValueError('The requested gains are not finite')
            joint.set('stiffness', format(k, '.17g'))
            joint.set('damping', format(d, '.17g'))
            # Disable a possible inherited automatic spring/damper specification.
            if any(j.get('springdamper') is not None for j in tree.findall('.//joint')):
                joint.set('springdamper', '0 0')
            expected_k[jid] = k
            expected_d[adr:adr+ndof] = d
        rows.append(dict(joint=name, protected=protected, exponent=3*(number-origin),
                         old_stiffness=old_k, new_stiffness=float(expected_k[jid]),
                         old_damping=old_d, new_damping=expected_d[adr:adr+ndof].tolist()))
    if arena_memory_mib is not None:
        set_arena_memory(tree.getroot(), arena_memory_mib)

    def verify(model):
        for field in ('body_mass', 'body_ipos', 'body_iquat', 'body_inertia', 'body_pos', 'body_quat',
                      'jnt_pos', 'jnt_axis', 'jnt_range', 'site_pos', 'actuator_ctrlrange',
                      'actuator_gear', 'geom_size', 'geom_pos', 'geom_quat', 'geom_contype', 'geom_conaffinity'):
            np.testing.assert_allclose(getattr(model, field), getattr(original, field), rtol=1e-13, atol=1e-16)
        np.testing.assert_allclose(model.jnt_stiffness, expected_k, rtol=1e-14)
        np.testing.assert_allclose(model.dof_damping, expected_d, rtol=1e-14)

    model = write_validated(tree, source, output, verify=verify, overwrite=overwrite)
    return dict(source=str(Path(source).resolve()), output=str(Path(output).resolve()),
                anchor=anchor, beta=beta, stiffness_coefficient=stiffness, damping_coefficient=damping,
                law=f'gain(j_NNN) = coefficient / beta**(3*(NNN-{origin})); excluded joints unchanged',
                units=dict(stiffness='N m/rad', damping='N m s/rad'),
                arena_memory_bytes=int(model.narena), verified_inertias_and_geometry_unchanged=True, joints=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mjcf', required=True)
    parser.add_argument('--out', required=True, help='New XML; keep the original for comparison')
    parser.add_argument('--stiffness', required=True, type=float, help='Base coefficient, N m/rad')
    parser.add_argument('--damping', required=True, type=float, help='Base coefficient, N m s/rad')
    parser.add_argument('--beta', required=True, type=float, help='Existing generator decay parameter, e.g. 1.03')
    parser.add_argument('--anchor', choices=['generator', 'first-flexible'], default='generator',
                        help='generator preserves indexing; first-flexible anchors the first unprotected joint (normally j_002)')
    parser.add_argument('--exclude-joints', nargs='+', default=[],
                        help='Additional protected joints; j_001 is always preserved')
    parser.add_argument('--arena-memory-mib', type=int, help='Optional native MuJoCo arena allocation, e.g. 128')
    parser.add_argument('--json', help='Write the before/after report')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing output, never the input')
    args = parser.parse_args()
    try:
        report = update_gains(args.mjcf, args.out, args.stiffness, args.damping, args.beta,
                              anchor=args.anchor, exclude=args.exclude_joints,
                              arena_memory_mib=args.arena_memory_mib, overwrite=args.overwrite)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Error: {exc}\n')
    print(report['law'])
    print('joint       stiffness: old -> new       damping: old -> new')
    for row in report['joints']:
        print(f"{row['joint']:10s} {row['old_stiffness']:.8g} -> {row['new_stiffness']:.8g}    "
              f"{row['old_damping'][0]:.8g} -> {row['new_damping'][0]:.8g}"
              + ('  [protected]' if row['protected'] else ''))
    print(f"Saved and compiled: {report['output']}\nArena: {report['arena_memory_bytes']/2**20:g} MiB")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
