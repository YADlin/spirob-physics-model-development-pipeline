// j_002 is the fixed experimental reference; geometry taper is independent.
export function resolveGains(p, count) {
  const post = p.post_gen ?? {};
  const preset = {safe:[.2,.2],fast:[.01,.001],high:[.1,.02]}[p.build?.physics_preset] ?? [.2,.01];
  for (const key of ['joint_stiffness_base','joint_damping_base','first_joint_stiffness','first_joint_damping','joint_beta','k2','rho_k','d2','rho_d']) {
    const v = post[key];
    if (v !== undefined && (typeof v !== 'number' || !Number.isFinite(v) || v < 0 || (['joint_beta','rho_k','rho_d'].includes(key) && v === 0))) throw Error(`Invalid post_gen.${key}`);
  }
  const K=post.joint_stiffness_base ?? preset[0], D=post.joint_damping_base ?? preset[1];
  const B=post.joint_beta ?? 1.03, legacy=B**-3;
  const rho_k=post.rho_k ?? legacy, rho_d=post.rho_d ?? legacy;
  const k2=post.k2 ?? K*legacy, d2=post.d2 ?? D*legacy;
  const gains=Array.from({length:count},(_,i)=>({joint:`j_${String(i+1).padStart(3,'0')}`,
    k:i===0 ? (post.first_joint_stiffness ?? K) : k2*rho_k**(i-1),
    d:i===0 ? (post.first_joint_damping ?? D) : d2*rho_d**(i-1)}));
  if(gains.some(g=>!Number.isFinite(g.k)||!Number.isFinite(g.d))) throw Error('Joint gain law overflows');
  return {reference_joint:'j_002',k2,rho_k,beta_j:rho_k**(-1/3),d2,rho_d,gains};
}
