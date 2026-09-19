// Browser geometry in SI. Equations and discrete profiles are tested against
// spirob.geometry; CAD-only surface previews are sampled, not export meshes.
const pi = Math.PI;
const norm = (p) => Math.hypot(...p);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const rot = (p, a) => [
  Math.cos(a) * p[0] - Math.sin(a) * p[1],
  Math.sin(a) * p[0] + Math.cos(a) * p[1],
];
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
export function phiFromB(b) {
  const e = Math.exp(2 * pi * b);
  return 2 * Math.atan2(b * (e - 1), Math.sqrt(1 + b * b) * (e + 1));
}
export function derive(p) {
  for (const k of [
    "L",
    "d_tip",
    "phi_deg",
    "Delta_theta_deg",
    "tendon_inward_shift",
  ])
    if (typeof p[k] !== "number" || !Number.isFinite(p[k]))
      throw Error(`${k} must be a finite number.`);
  if (
    p.L <= 0 ||
    p.d_tip <= 0 ||
    p.phi_deg <= 0 ||
    p.phi_deg >= 45 ||
    p.Delta_theta_deg <= 0 ||
    p.Delta_theta_deg >= 180
  )
    throw Error("Use positive dimensions, 0 < φ < 45°, and 0 < Δθ < 180°.");
  if (!Number.isInteger(p.n_cables) || p.n_cables < 2)
    throw Error("Cable count must be an integer of at least 2.");
  if (p.tendon_inward_shift < 0 || p.tendon_inward_shift >= p.d_tip / 2)
    throw Error(
      "Cable inward shift must be nonnegative and smaller than half the nominal tip width.",
    );
  if (p.n_cables > 32)
    throw Error(
      "Interactive preview supports up to 32 cables. The Python generator has no 32-cable limit.",
    );
  let lo = 1e-6,
    hi = 2,
    b;
  const phi = (p.phi_deg * pi) / 180,
    dth = (p.Delta_theta_deg * pi) / 180;
  for (let i = 0; i < 120; i++) {
    b = (lo + hi) / 2;
    const f = phiFromB(b) - phi;
    if (Math.abs(f) < 1e-14 || hi - lo < 1e-14) break;
    if (f > 0) hi = b;
    else lo = b;
  }
  const E = Math.exp(2 * pi * b),
    a = p.d_tip / (E - 1),
    rc0 = 0.5 * a * (E + 1),
    A = (Math.sqrt(1 + b * b) / b) * rc0;
  const qRequested = Math.log1p(p.L / A) / b;
  let q = qRequested;
  const boundary =
    Math.abs(q / dth - Math.round(q / dth)) * dth <= 1e-9 &&
    Math.round(q / dth) >= 1;
  if (p.terminal_unit_policy === "whole_units" && !boundary)
    q = Math.ceil(q / dth) * dth;
  const n =
    Math.abs(q / dth - Math.round(q / dth)) * dth <= 1e-9
      ? Math.round(q / dth)
      : Math.ceil(q / dth);
  if (n > 400)
    throw Error(
      "This design exceeds the interactive limit of 400 links. Increase Δθ or build from the terminal.",
    );
  const theta = Array.from({ length: n + 1 }, (_, i) => Math.min(i * dth, q));
  const centres = theta.map((t) => [
    rc0 * Math.exp(b * t) * Math.cos(t),
    rc0 * Math.exp(b * t) * Math.sin(t),
  ]);
  const inner = theta.map((t) => [
    a * Math.exp(b * t) * Math.cos(t),
    a * Math.exp(b * t) * Math.sin(t),
  ]);
  const curled = Array.from({ length: n }, (_, i) =>
    [centres[i], centres[i + 1], inner[i + 1], inner[i]].map((v) => [...v]),
  );
  let cursor = [0, 0];
  const angles = [];
  const straight = curled.map((quad) => {
    const v = sub(quad[1], quad[0]);
    const angle = Math.atan2(v[0], v[1]);
    angles.push(angle);
    const result = quad.map((pt) =>
      rot(sub(pt, quad[0]), angle).map((v, i) => v + cursor[i]),
    );
    cursor = [...result[1]];
    return result;
  });
  const length = cursor[1],
    effective = A * Math.expm1(b * q),
    flip = p.terminal_unit_policy === "whole_units" ? effective : p.L;
  const quads = straight
    .map((quad) => [1, 0, 3, 2].map((i) => [quad[i][0], flip - quad[i][1]]))
    .reverse();
  const partial = theta[n] - theta[n - 1] < dth - 1e-9;
  if (partial) {
    const c0 = [rc0, 0],
      c1 = [
        rc0 * Math.exp(b * dth) * Math.cos(dth),
        rc0 * Math.exp(b * dth) * Math.sin(dth),
      ];
    const angle = Math.atan2(c1[0] - c0[0], c1[1] - c0[1]);
    const edge = rot([a - rc0, 0], angle),
      slope = edge[1] / Math.abs(edge[0]);
    const base = quads[0],
      r = Math.abs(base[2][0] - base[1][0]);
    if (base[1][1] - r * slope <= base[0][1] + 1e-12)
      throw Error(
        "Partial base is too short for its joint-facing slope. Choose “whole units”, or adjust L / Δθ.",
      );
    base[2][1] = base[1][1] - r * slope;
    base[3][1] = base[0][1];
    for (const [i, j] of [
      [2, 3],
      [3, 2],
    ]) {
      straight[n - 1][j] = [base[i][0], flip - base[i][1]];
      curled[n - 1][j] = rot(
        sub(straight[n - 1][j], straight[n - 1][0]),
        -angles[n - 1],
      ).map((v, k) => v + centres[n - 1][k]);
    }
  }
  const origin = quads[0][0][1],
    beta = Math.exp(b * dth),
    tipChord = rc0 * Math.hypot(beta * Math.cos(dth) - 1, beta * Math.sin(dth));
  const apex = quads[n - 1][1][1] + tipChord / Math.expm1(b * dth);
  const width = 2 * Math.abs(quads[0][2][0]);
  if ("base_thickness_m" in p && "flat_thickness_ratio" in p)
    throw Error(
      "Choose base_thickness_m or legacy flat_thickness_ratio, not both.",
    );
  const thickness = p.base_thickness_m ?? width * (p.flat_thickness_ratio ?? 1);
  if (!Number.isFinite(thickness) || thickness <= 0)
    throw Error("Base thickness must be positive, or auto.");
  const thicknessAt = (z) => (thickness * (apex - z)) / (apex - origin);
  const units = quads.map((quad, i) => {
    const z0 = quad[0][1],
      z1 = quad[1][1],
      radius = Math.abs(quad[2][0]);
    const r1 = Math.max(Math.abs(quad[3][0]) - p.tendon_inward_shift, 1e-6),
      r2 = Math.max(
        Math.abs(quad[3][0]) -
          p.tendon_inward_shift -
          (quad[2][1] - quad[3][1]) * Math.tan(phi / 2),
        1e-6,
      );
    const cable = Array.from({ length: p.n_cables }, (_, c) => {
      const angle = pi + (2 * pi * c) / p.n_cables;
      return [
        [r1 * Math.cos(angle), r1 * Math.sin(angle), quad[3][1]],
        [r2 * Math.cos(angle), r2 * Math.sin(angle), quad[2][1]],
      ];
    });
    const t0 =
        p.thickness_profile === "stepped"
          ? (thickness * 2 * radius) / width
          : thicknessAt(z0),
      t1 = p.thickness_profile === "stepped" ? t0 : thicknessAt(z1);
    return {
      index: i,
      name: `link_${String(i + 1).padStart(3, "0")}`,
      quad,
      z0,
      z1,
      radius,
      t0,
      t1,
      cable,
      partial: i === 0 && partial,
    };
  });
  const preset = { safe: [0.2, 0.2], fast: [0.01, 0.001], high: [0.1, 0.02] }[
    p.build?.physics_preset
  ] ?? [0.2, 0.01];
  const gainBeta = p.post_gen?.joint_beta ?? 1.03,
    K = p.post_gen?.joint_stiffness_base ?? preset[0],
    D = p.post_gen?.joint_damping_base ?? preset[1];
  const gains = units.map((u, i) => ({
    joint: `j_${String(i + 1).padStart(3, "0")}`,
    k:
      i === 0
        ? (p.post_gen?.first_joint_stiffness ?? K)
        : K / gainBeta ** (3 * i),
    d:
      i === 0
        ? (p.post_gen?.first_joint_damping ?? D)
        : D / gainBeta ** (3 * i),
  }));
  if (gains.some((v) => !Number.isFinite(v.k) || !Number.isFinite(v.d)))
    throw Error("Joint gain law overflows. Choose a gain decay closer to 1.");
  return {
    a,
    b,
    E,
    beta,
    q,
    qRequested,
    rc0,
    phi,
    dth,
    theta,
    centres,
    curled,
    quads,
    units,
    length,
    effective,
    origin,
    apex,
    width,
    thickness,
    tipThickness: units.at(-1).t1,
    partial,
    gains,
  };
}
export function radiusAt(unit, z) {
  const q = unit.quad,
    hits = [];
  for (let i = 0; i < q.length; i++) {
    const a = q[i],
      b = q[(i + 1) % q.length];
    if (Math.abs(b[1] - a[1]) < 1e-14) {
      if (Math.abs(z - a[1]) < 1e-12) hits.push(Math.abs(a[0]), Math.abs(b[0]));
    } else if (
      z >= Math.min(a[1], b[1]) - 1e-12 &&
      z <= Math.max(a[1], b[1]) + 1e-12
    )
      hits.push(Math.abs(a[0] + ((b[0] - a[0]) * (z - a[1])) / (b[1] - a[1])));
  }
  return Math.max(0, ...hits);
}
export function section(p, g, u, fraction = 0.5, samples = 128) {
  const z = u.z0 + (u.z1 - u.z0) * fraction,
    R = radiusAt(u, z),
    T = u.t0 + (u.t1 - u.t0) * fraction,
    n = p.n_cables;
  const result = {
    z,
    R,
    T,
    points: [],
    cables: [],
    notches: [],
    polygon: [],
    neck: p.build?.neck_width_mm ?? 1,
    hole: p.build?.cable_hole_diameter_mm ?? 0,
  };
  for (const [a, b] of u.cable) {
    const t = (z - a[2]) / (b[2] - a[2] || 1);
    result.cables.push([
      a[0] + clamp(t, 0, 1) * (b[0] - a[0]),
      a[1] + clamp(t, 0, 1) * (b[1] - a[1]),
    ]);
  }
  if (n === 2 && !p.build?.plain) {
    const e = p.flat_section === "hex" ? (p.hex_edge_ratio ?? 0.75) : 1,
      h = T / 2,
      he = h * (1 - ((1 - e) * R) / u.radius);
    result.points =
      e === 1
        ? [
            [-R, -h],
            [R, -h],
            [R, h],
            [-R, h],
          ]
        : [
            [-R, -he],
            [0, -h],
            [R, -he],
            [R, he],
            [0, h],
            [-R, he],
          ];
  } else if (p.build?.plain) {
    result.points = Array.from({ length: samples }, (_, i) => [
      R * Math.cos((i * 2 * pi) / samples),
      R * Math.sin((i * 2 * pi) / samples),
    ]);
  } else {
    const rp = u.radius * (1 + (p.nlobe_t ?? 0.5) * (1 / Math.cos(pi / n) - 1)),
      side = 2 * rp * Math.sin(pi / n),
      h = u.z1 - u.z0;
    const top = Math.max(0.001, 1 - (Math.tan(g.phi / 2) * 2 * h) / side);
    const bblo = Math.min(...u.quad.map((q) => q[1])),
      bbhi = Math.max(...u.quad.map((q) => q[1]));
    const cutterZ = (bblo + bbhi) / 2 - h,
      scale = 1 + ((top - 1) * (z - cutterZ)) / (2 * h),
      apothem = rp * scale * Math.cos(pi / n),
      notchR = (side / 2) * (p.notch_factor ?? 0.25);
    const verts = Array.from({ length: n }, (_, i) => [
      rp * Math.cos(pi + (i * 2 * pi) / n),
      rp * Math.sin(pi + (i * 2 * pi) / n),
    ]);
    result.polygon = verts.map((v) => v.map((x) => x * scale));
    result.notches = verts.map((v, i) => ({
      c: v.map((x, j) => (x + verts[(i + 1) % n][j]) / 2),
      r: notchR,
    }));
    result.rp = rp;
    result.notchR = notchR;
    result.points = Array.from({ length: samples }, (_, i) => {
      const angle = (2 * pi * i) / samples,
        dir = [Math.cos(angle), Math.sin(angle)];
      let r = R;
      for (let k = 0; k < n; k++) {
        const normal = pi + ((k + 0.5) * 2 * pi) / n,
          projection = Math.cos(angle - normal);
        if (projection > 0) r = Math.min(r, apothem / projection);
      }
      if (notchR > 0)
        for (const { c } of result.notches) {
          const proj = dir[0] * c[0] + dir[1] * c[1],
            disc = proj * proj - (c[0] * c[0] + c[1] * c[1] - notchR * notchR);
          if (disc > 0 && proj > 0) r = Math.min(r, proj - Math.sqrt(disc));
        }
      return dir.map((x) => x * Math.max(0, r));
    });
  }
  return result;
}
export function hull2(points) {
  const pts = points
    .map((p) => [...p])
    .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [],
    upper = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower.at(-2), lower.at(-1), p) <= 0)
      lower.pop();
    lower.push(p);
  }
  for (const p of pts.reverse()) {
    while (upper.length >= 2 && cross(upper.at(-2), upper.at(-1), p) <= 0)
      upper.pop();
    upper.push(p);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}
export function clearance(s) {
  // sampled XY clearance, not 3D print certification
  function distance(p, a, b) {
    const v = sub(b, a),
      w = sub(p, a),
      t = clamp(
        (v[0] * w[0] + v[1] * w[1]) / (v[0] * v[0] + v[1] * v[1] || 1),
        0,
        1,
      );
    return norm([w[0] - t * v[0], w[1] - t * v[1]]);
  }
  function inside(pt) {
    let odd = false;
    for (let i = 0, j = s.points.length - 1; i < s.points.length; j = i++) {
      const a = s.points[i],
        b = s.points[j];
      if (
        a[1] > pt[1] !== b[1] > pt[1] &&
        pt[0] < ((b[0] - a[0]) * (pt[1] - a[1])) / (b[1] - a[1]) + a[0]
      )
        odd = !odd;
    }
    return odd;
  }
  return s.cables.map((p) => {
    const d = Math.min(
      ...s.points.map((a, i) =>
        distance(p, a, s.points[(i + 1) % s.points.length]),
      ),
    );
    return (inside(p) ? d : -d) - s.hole / 2000;
  });
}
