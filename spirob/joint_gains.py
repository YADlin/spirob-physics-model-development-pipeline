"""Explicit j_002 experimental gains with backward-compatible legacy defaults."""
import math


def resolve_gains(post, count, stiffness=.2, damping=.01, beta=1.03):
    """Return per-joint gains; canonical fields override only their own channel.

    Legacy K0/D0 remain the mount fallback. Neither k2 nor rho_k changes it.
    Omitting damping fields retains the complete legacy damping law.
    """
    for key in ('joint_stiffness_base', 'joint_damping_base', 'first_joint_stiffness',
                'first_joint_damping', 'joint_beta', 'k2', 'rho_k', 'd2', 'rho_d'):
        value = post.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (key in ('joint_beta', 'rho_k', 'rho_d') and value == 0)):
            raise ValueError(f'post_gen.{key}: expected finite nonnegative gain or positive ratio')
    K = post.get('joint_stiffness_base', stiffness)
    D = post.get('joint_damping_base', damping)
    B = post.get('joint_beta', beta)
    legacy_ratio = B ** -3
    rk = post.get('rho_k', legacy_ratio)
    rd = post.get('rho_d', legacy_ratio)
    # Convert legacy anchors with the LEGACY ratio even when a fitted ratio is set.
    k2 = post.get('k2', K * legacy_ratio)
    d2 = post.get('d2', D * legacy_ratio)
    ks = [post.get('first_joint_stiffness', K)]
    ds = [post.get('first_joint_damping', D)]
    try:
        for i in range(2, count + 1):
            # Retain original floating-point evaluation for unmodified legacy inputs.
            ks.append(k2 * rk ** (i-2) if 'k2' in post or 'rho_k' in post else K / ((B**3)**(i-1)))
            ds.append(d2 * rd ** (i-2) if 'd2' in post or 'rho_d' in post else D / ((B**3)**(i-1)))
    except OverflowError as exc:
        raise ValueError('Joint gain law overflows') from exc
    if not all(math.isfinite(v) for v in ks + ds):
        raise ValueError('Joint gain law overflows')
    return dict(reference_joint='j_002', k2=k2, rho_k=rk, beta_j=rk**(-1/3),
                d2=d2, rho_d=rd, stiffness=ks[:count], damping=ds[:count])
