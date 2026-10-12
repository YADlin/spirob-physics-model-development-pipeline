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


def update_gains(source, output, stiffness=None, damping=None, beta=None, *, k2=None, rho_k=None, d2=None, rho_d=None, reference_joint=None, anchor='generator',
                 exclude=('j_001',), arena_memory_mib=None, overwrite=False):
    if anchor not in ('generator', 'first-flexible'):
        raise ValueError('anchor must be generator or first-flexible')
    if stiffness is not None and k2 is not None:
        raise ValueError('Choose --stiffness or --k2, not both')
    if damping is not None and d2 is not None:
        raise ValueError('Choose --damping or --d2, not both')
    if beta is not None and rho_k is not None:
        raise ValueError('Choose --beta or --rho-k, not both')
    if stiffness is None and k2 is None:
        raise ValueError('Supply stiffness or k2')
    if beta is None and rho_k is None:
        raise ValueError('Supply beta or rho_k')
    for name, value in [('stiffness', stiffness), ('damping', damping), ('beta', beta),
                        ('k2', k2), ('rho_k', rho_k), ('d2', d2), ('rho_d', rho_d)]:
        if value is not None and (isinstance(value, bool) or not math.isfinite(value)
                or value < 0 or (name in ('beta','rho_k','rho_d') and value == 0)):
            raise ValueError(f'{name} must be finite, with nonnegative gains and positive ratios')
    if (k2 is not None or d2 is not None) and reference_joint not in (None, 'j_002'):
        raise ValueError('k2/d2 always refer to j_002')
    origin = 2 if k2 is not None or anchor == 'first-flexible' else 1
    if reference_joint is not None:
        if not re.fullmatch(r'j_\d+', reference_joint):
            raise ValueError('Reference must be a named j_NNN joint')
        origin = int(reference_joint[2:])
    rk = rho_k if rho_k is not None else beta ** -3
    # Legacy explicit damping + beta keeps its historical shared law.
    # Canonical stiffness never supplies damping decay implicitly.
    if (damping is not None or d2 is not None) and rho_d is None and beta is None:
        raise ValueError('Damping updates require --rho-d (or legacy --beta)')
    if rho_d is not None and damping is None and d2 is None:
        raise ValueError('--rho-d requires a damping amplitude')
    rd = rho_d if rho_d is not None else (beta ** -3 if beta is not None else None)
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
    if f'j_{origin:03d}' not in names:
        raise ValueError('Reference joint not found; specify --reference-joint explicitly')
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
                factor = rk ** (number-origin)
                k = k2 * rk ** (number-2) if k2 is not None else stiffness*factor
                d = (d2 * rd ** (number-2) if d2 is not None else
                     damping * rd ** (number-origin) if damping is not None else None)
            except OverflowError as exc:
                raise ValueError('The requested decay law overflows') from exc
            if not math.isfinite(k) or (d is not None and not math.isfinite(d)):
                raise ValueError('The requested gains are not finite')
            joint.set('stiffness', format(k, '.17g'))
            if d is not None:
                joint.set('damping', format(d, '.17g'))
            # Disable a possible inherited automatic spring/damper specification.
            if any(j.get('springdamper') is not None for j in tree.findall('.//joint')):
                if d is None:
                    if len(set(old_d)) != 1:
                        raise ValueError(f'{name}: inherited anisotropic damping cannot be preserved as one XML joint value')
                    joint.set('damping', format(old_d[0], '.17g'))
                joint.set('springdamper', '0 0')
            expected_k[jid] = k
            if d is not None:
                expected_d[adr:adr+ndof] = d
        rows.append(dict(joint=name, protected=protected, index_offset=number-origin, exponent=3*(number-origin),
                         old_stiffness=old_k, new_stiffness=float(expected_k[jid]),
                         old_damping=old_d, new_damping=expected_d[adr:adr+ndof].tolist()))
    if arena_memory_mib is not None:
        set_arena_memory(tree.getroot(), arena_memory_mib)

    def verify(model):
        for field in ('body_mass', 'body_ipos', 'body_iquat', 'body_inertia', 'body_pos', 'body_quat',
                      'jnt_pos', 'jnt_axis', 'jnt_range', 'site_pos', 'actuator_ctrlrange',
                      'actuator_gear', 'actuator_gainprm', 'actuator_biasprm', 'actuator_dynprm',
                      'actuator_trnid', 'actuator_forcerange', 'actuator_actrange', 'geom_size', 'geom_pos', 'geom_quat', 'geom_contype', 'geom_conaffinity'):
            np.testing.assert_allclose(getattr(model, field), getattr(original, field), rtol=1e-13, atol=1e-16)
        np.testing.assert_allclose(model.jnt_stiffness, expected_k, rtol=1e-14)
        np.testing.assert_allclose(model.dof_damping, expected_d, rtol=1e-14)

    model = write_validated(tree, source, output, verify=verify, overwrite=overwrite)
    return dict(source=str(Path(source).resolve()), output=str(Path(output).resolve()),
                k2=k2 if k2 is not None else stiffness*rk**(2-origin), d2=d2 if d2 is not None else (damping*rd**(2-origin) if damping is not None else None),
                anchor=anchor, reference_joint=f'j_{origin:03d}', rho_k=rk, rho_d=rd, beta_j=rk**(-1/3), damping_preserved=damping is None and d2 is None, beta=beta, stiffness_coefficient=stiffness, damping_coefficient=damping,
                law=f'k(j_NNN) = reference stiffness * rho_k**(NNN-{origin}); damping independently specified or preserved; exclusions never shift the reference',
                units=dict(stiffness='N m/rad', damping='N m s/rad'),
                arena_memory_bytes=int(model.narena), verified_inertias_and_geometry_unchanged=True, joints=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mjcf', required=True, help='Input standalone MuJoCo XML, including its associated mesh assets')
    parser.add_argument('--out', required=True, help='New XML; keep the original for comparison')
    parser.add_argument('--stiffness', type=float, help='Legacy stiffness coefficient at --reference-joint or selected anchor, N m/rad')
    parser.add_argument('--damping', type=float, help='Optional legacy damping coefficient; omission preserves compiled damping')
    parser.add_argument('--beta', type=float, help='Legacy shared decay beta; equivalent rho = beta^(-3)')
    parser.add_argument('--k2', type=float, help='Actual stiffness at fixed reference j_002, N m/rad')
    parser.add_argument('--rho-k', type=float, help='Adjacent stiffness ratio, independent of geometry')
    parser.add_argument('--d2', type=float, help='Optional actual damping at j_002, N m s/rad')
    parser.add_argument('--rho-d', type=float, help='Independent adjacent damping ratio; not inferred from a static fit')
    parser.add_argument('--reference-joint', help='Explicit reference for legacy coefficients; exclusions never move it')
    parser.add_argument('--anchor', choices=['generator', 'first-flexible'], default='generator',
                        help='generator preserves indexing; first-flexible means fixed j_002, including when excluded')
    parser.add_argument('--exclude-joints', nargs='+', default=[],
                        help='Additional protected joints; j_001 is always preserved')
    parser.add_argument('--arena-memory-mib', type=int, help='Optional native MuJoCo arena allocation, e.g. 128')
    parser.add_argument('--json', help='Write the before/after report')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing output, never the input')
    args = parser.parse_args()
    try:
        report = update_gains(args.mjcf, args.out, args.stiffness, args.damping, args.beta,
                              anchor=args.anchor, exclude=args.exclude_joints, k2=args.k2, rho_k=args.rho_k,
                              d2=args.d2, rho_d=args.rho_d, reference_joint=args.reference_joint,
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
