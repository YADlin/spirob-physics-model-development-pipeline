import { derive, section, clearance, radiusAt } from "./geometry.mjs";
const $ = (id) => document.getElementById(id),
  pi = Math.PI,
  mm = (x) => x * 1000,
  fmt = (x, d = 2) =>
    Number(x).toLocaleString("en-US", {
      maximumFractionDigits: d,
      minimumFractionDigits: d,
    });
const escape = (s) =>
  String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const schema = await fetch("parameters.schema.json").then((r) => r.json());
const groups = {
  geometry: "01 · Overall geometry",
  section: "02 · Cross-section",
  fabrication: "03 · Elastic core & channels",
  dynamics: "04 · Joint dynamics",
  placement: "05 · World placement",
  build: "06 · Build options",
};
const presets = {
  2: "two",
  "2hex": "two-cable-hex",
  3: "three",
  4: "four",
  6: "six",
};
let params = {},
  geom,
  activeKey = "L",
  local = false,
  valid = false,
  activeJob = null;
let stored = null;
try {
  stored = localStorage.getItem("spirob-designer-v1");
} catch {}
let camera = { yaw: 0.6, pitch: 0.6, zoom: 1 },
  faces = [],
  lines3 = [];
const get = (obj, path) => path.split(".").reduce((v, k) => v?.[k], obj);
const set = (obj, path, value) => {
  const keys = path.split(".");
  let p = obj;
  keys.slice(0, -1).forEach((k) => (p = p[k] ??= {}));
  p[keys.at(-1)] = value;
};
const fields = [];
for (const [key, field] of Object.entries(schema.properties)) {
  if (field.properties)
    for (const [sub, f] of Object.entries(field.properties)) {
      if (!f["x-hidden"]) fields.push([`${key}.${sub}`, f]);
    }
  else if (!field["x-hidden"]) fields.push([key, field]);
}
function clean(p) {
  p = structuredClone(p);
  if (p.post_gen && "target_site_pos" in p.post_gen) {
    delete p.post_gen.target_site_pos;
    notice(
      "The obsolete target_site_pos was removed. New models contain no target site.",
    );
  }
  return p;
}
function available(f) {
  return f["x-cables"] === "two"
    ? params.n_cables === 2
    : f["x-cables"] === "hex"
      ? params.n_cables === 2 && params.flat_section === "hex"
      : f["x-cables"] === "many"
        ? params.n_cables >= 3
        : true;
}
function focusField(key, f) {
  activeKey = key;
  $("insight-title").textContent = f.title;
  $("insight-text").textContent = f.description;
}
function form() {
  const open = new Set(
    [...$("form").querySelectorAll("details[open]")].map(
      (e) => e.dataset.group,
    ),
  );
  $("form").replaceChildren();
  for (const [group, title] of Object.entries(groups)) {
    const detail = document.createElement("details");
    detail.dataset.group = group;
    detail.open = open.size ? open.has(group) : group === "geometry";
    const summary = document.createElement("summary");
    summary.textContent = title;
    detail.append(summary);
    for (const [key, f] of fields.filter(
      ([, f]) => f["x-group"] === group && available(f),
    )) {
      const item = document.createElement("div");
      item.className = "field";
      const header = document.createElement("div");
      header.className = "field-header";
      const label = document.createElement("label");
      label.htmlFor = `param-${key}`;
      label.textContent = f.title;
      const unit = document.createElement("span");
      unit.className = "unit";
      unit.textContent = f["x-unit"] === "m" ? "mm" : f["x-unit"];
      header.append(label, unit);
      item.append(header);
      const types = Array.isArray(f.type) ? f.type : [f.type],
        nullable = types.includes("null");
      let value = get(params, key);
      if (
        key === "base_thickness_m" &&
        value === undefined &&
        params.flat_thickness_ratio !== undefined
      ) {
        try {
          value = derive(params).thickness;
        } catch {}
      }
      const omitted = key.startsWith("post_gen.") && value === undefined;
      if (omitted) {
        const msg = document.createElement("p");
        msg.className = "field-help";
        msg.textContent =
          "Omitted: generator/preset default. Set an explicit value to override.";
        item.append(msg);
      }
      const factor = f["x-unit"] === "m" ? 1000 : 1;
      let control;
      const update = (v) => {
        set(params, key, v);
        if (key === "base_thickness_m") delete params.flat_thickness_ratio;
        if (key === "n_cables") {
          if (v !== 2)
            for (const k of [
              "base_thickness_m",
              "flat_thickness_ratio",
              "flat_section",
              "hex_edge_ratio",
              "thickness_profile",
            ])
              delete params[k];
          else {
            params.base_thickness_m = null;
            params.flat_section = "rectangular";
            params.thickness_profile = "linear";
          }
          form();
        }
        if (key === "flat_section") {
          if (v !== "hex") delete params.hex_edge_ratio;
          else params.hex_edge_ratio ??= 0.75;
          form();
        }
        render();
      };
      if (f.enum) {
        control = document.createElement("select");
        for (const v of f.enum) {
          const o = document.createElement("option");
          o.value = v;
          o.textContent = v.replaceAll("_", " ");
          control.append(o);
        }
        control.value =
          value ??
          (key === "build.collision_mode"
            ? params.build?.physics_preset === "safe"
              ? "capsule"
              : "mesh"
            : f.default);
        control.onchange = () => update(control.value);
      } else if (types.includes("boolean")) {
        control = document.createElement("input");
        control.type = "checkbox";
        control.checked = value ?? f.default;
        control.onchange = () => update(control.checked);
      } else if (types.includes("array")) {
        control = document.createElement("div");
        control.className = "vector";
        control.style.gridTemplateColumns = `repeat(${f.maxItems},1fr)`;
        const values = value ?? f.default;
        for (let i = 0; i < f.maxItems; i++) {
          const input = document.createElement("input");
          input.type = "number";
          input.step = "any";
          input.value = Number((values[i] * factor).toPrecision(12));
          input.setAttribute(
            "aria-label",
            `${f.title} ${f.maxItems === 4 ? ["w", "x", "y", "z"][i] : ["x", "y", "z"][i]}`,
          );
          input.oninput = () => {
            const a = [...(get(params, key) ?? f.default)];
            a[i] = input.value === "" ? NaN : Number(input.value) / factor;
            update(a);
          };
          control.append(input);
        }
      } else {
        control = document.createElement("input");
        control.type = "number";
        control.step = types.includes("integer") ? "1" : "any";
        const resolved =
          value === undefined ? (omitted ? null : f.default) : value;
        control.value =
          resolved == null ? "" : Number((resolved * factor).toPrecision(12));
        control[key === "n_cables" ? "onchange" : "oninput"] = () =>
          update(control.value === "" ? NaN : Number(control.value) / factor);
        if (f.minimum !== undefined) control.min = f.minimum * factor;
        if (f.maximum !== undefined) control.max = f.maximum * factor;
        if (nullable) {
          const row = document.createElement("div");
          row.className = "auto-row";
          const auto = document.createElement("input");
          auto.type = "checkbox";
          auto.checked = resolved == null;
          auto.setAttribute("aria-label", `${f.title} automatic`);
          control.disabled = auto.checked;
          const autoLabel = document.createElement("label");
          autoLabel.append(auto, " Auto");
          auto.onchange = () => {
            control.disabled = auto.checked;
            if (auto.checked) update(null);
            else {
              control.value =
                key === "base_thickness_m"
                  ? fmt(geom?.thickness * 1000 || 20)
                  : key.includes("memory")
                    ? 128
                    : 0;
              update(Number(control.value) / factor);
            }
          };
          row.append(control, autoLabel);
          item.append(row);
        }
      }
      control.id = `param-${key}`;
      if (!control.parentNode) item.append(control);
      item.addEventListener("focusin", () => focusField(key, f));
      item.addEventListener("pointerenter", () => focusField(key, f));
      const help = document.createElement("p");
      help.className = "field-help";
      help.textContent = f.description;
      item.append(help);
      const jsonKey = document.createElement("span");
      jsonKey.className = "json-key";
      jsonKey.textContent = key;
      item.append(jsonKey);
      detail.append(item);
    }
    $("form").append(detail);
  }
}
function validate(value, s, path = "params") {
  const types = Array.isArray(s.type) ? s.type : [s.type],
    type =
      value === null
        ? "null"
        : Array.isArray(value)
          ? "array"
          : typeof value === "number" && Number.isInteger(value)
            ? "integer"
            : typeof value;
  if (
    !types.includes(type) &&
    !(type === "integer" && types.includes("number"))
  )
    throw Error(`${path}: expected ${types.join(" or ")}.`);
  if (type === "number" || type === "integer") {
    if (!Number.isFinite(value)) throw Error(`${path}: enter a finite number.`);
    for (const [key, ok] of [
      ["minimum", (v) => value >= v],
      ["maximum", (v) => value <= v],
      ["exclusiveMinimum", (v) => value > v],
      ["exclusiveMaximum", (v) => value < v],
    ])
      if (key in s && !ok(s[key])) throw Error(`${path}: ${key} is ${s[key]}.`);
  }
  if (s.enum && !s.enum.includes(value))
    throw Error(`${path}: choose ${s.enum.join(", ")}.`);
  if (type === "object") {
    for (const k of s.required ?? [])
      if (!(k in value)) throw Error(`${path}.${k} is required.`);
    for (const [k, v] of Object.entries(value)) {
      if (!s.properties[k]) throw Error(`${path}.${k}: unknown parameter.`);
      validate(v, s.properties[k], `${path}.${k}`);
    }
  }
  if (type === "array") {
    if (value.length < s.minItems || value.length > s.maxItems)
      throw Error(`${path}: expected ${s.minItems} values.`);
    value.forEach((v, i) => validate(v, s.items, `${path}[${i}]`));
  }
}
function configurationChecks() {
  if (
    params.post_gen?.robot_quat &&
    !params.post_gen.robot_quat.some((v) => v !== 0)
  )
    throw Error("Root quaternion must be nonzero.");
  const b = params.build ?? {};
  if (
    (params.n_cables !== 2 || b.plain) &&
    (params.base_thickness_m != null ||
      params.flat_section === "hex" ||
      "thickness_profile" in params)
  )
    throw Error(
      "Base thickness, hex section and thickness_profile require two cables without plain revolve.",
    );
  if ("hex_edge_ratio" in params && params.flat_section !== "hex")
    throw Error("hex_edge_ratio requires flat_section=hex.");
  if (
    params.taper_angle_deg !== undefined &&
    Math.abs(params.taper_angle_deg - params.phi_deg) > 1e-9
  )
    throw Error("taper_angle_deg and phi_deg must agree.");
  if ((b.iges || b.fuse_cad) && !b.cad)
    throw Error(
      "Enable “Export fabrication CAD” before selecting IGES or CAD fusion.",
    );
  if (b.plain && (b.collision_mode ?? "mesh") === "convex")
    throw Error(
      "Plain revolve does not support convex mode. Select mesh or capsule, or disable plain.",
    );
  if (
    b.collision_mode === "compound" &&
    (params.n_cables !== 2 ||
      params.flat_section === "hex" ||
      (params.thickness_profile ?? "linear") !== "stepped" ||
      b.plain)
  )
    throw Error(
      "Legacy compound requires two cables, rectangular section and stepped thickness. Use convex for the current shapes.",
    );
  if (b.cad && b.cad_profile === "simulation" && b.cable_hole_diameter_mm > 0)
    throw Error("Cable channels require the fabrication CAD profile.");
}
function notice(msg) {
  $("notice").hidden = !msg;
  $("notice").textContent = msg;
}
const line = (x1, y1, x2, y2, cls = "dim") =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="${cls}"/>`;
const txt = (x, y, text, cls = "", anchor = "start") =>
  `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${escape(text)}</text>`;
const poly = (points, cls = "material", extra = "") =>
  `<polygon points="${points.map((p) => p.join(",")).join(" ")}" class="${cls}" ${extra}/>`;
const path = (points, cls = "cable-line") =>
  `<polyline points="${points.map((p) => p.join(",")).join(" ")}" class="${cls}"/>`;
function dimension(
  x1,
  y1,
  x2,
  y2,
  label,
  tx = (x1 + x2) / 2,
  ty = (y1 + y2) / 2 - 7,
) {
  return (
    line(x1, y1, x2, y2) +
    line(x1 - 3, y1 - 4, x1 + 3, y1 + 4) +
    line(x2 - 3, y2 - 4, x2 + 3, y2 + 4) +
    txt(tx, ty, label, "dim-text", "middle")
  );
}
function drawProfiles() {
  const g = geom,
    index = +$("link").value - 1,
    selected = g.units[index],
    scale = 660 / g.length,
    x = (z) => 65 + (z - g.origin) * scale;
  const maxW = Math.max(g.width, params.n_cables === 2 ? g.thickness : g.width),
    sY = Math.min(scale, 95 / maxW),
    front = 92,
    side = 245;
  let svg =
    line(45, front, 756, front, "axis") + line(45, side, 756, side, "axis");
  svg +=
    txt(24, 26, "WIDTH · XZ", "dim-text") +
    txt(
      24,
      186,
      params.n_cables === 2 ? "THICKNESS · YZ" : "SECOND SIDE · YZ",
      "dim-text",
    );
  for (const u of g.units) {
    const outline = [
      u.quad[0],
      u.quad[3],
      u.quad[2],
      u.quad[1],
      [-u.quad[2][0], u.quad[2][1]],
      [-u.quad[3][0], u.quad[3][1]],
    ];
    if (params.n_cables === 2 || params.build?.plain)
      svg += poly(
        outline.map(([r, z]) => [x(z), front - r * sY]),
        u.index === index ? "selected" : "material",
        `data-link="${u.index + 1}" style="cursor:pointer"`,
      );
    if (params.n_cables === 2) {
      svg += poly(
        [
          [x(u.z0), side - (u.t0 * sY) / 2],
          [x(u.z1), side - (u.t1 * sY) / 2],
          [x(u.z1), side + (u.t1 * sY) / 2],
          [x(u.z0), side + (u.t0 * sY) / 2],
        ],
        u.index === index ? "selected" : "material",
      );
    } else {
      const fs = [0.0001, 0.08, 0.15, 0.3, 0.5, 0.7, 0.85, 0.92, 0.9999],
        top = [],
        bottom = [],
        xtop = [],
        xbottom = [];
      for (const f of fs) {
        const s = section(params, g, u, f, 96),
          ys = s.points.map((v) => v[1]),
          xs = s.points.map((v) => v[0]);
        top.push([x(s.z), side - Math.max(...ys) * sY]);
        bottom.push([x(s.z), side - Math.min(...ys) * sY]);
        xtop.push([x(s.z), front - Math.max(...xs) * sY]);
        xbottom.push([x(s.z), front - Math.min(...xs) * sY]);
      }
      svg += poly(
        top.concat(bottom.reverse()),
        u.index === index ? "selected" : "material",
      );
      if (!params.build?.plain)
        svg += poly(
          xtop.concat(xbottom.reverse()),
          u.index === index ? "selected" : "material",
          `data-link="${u.index + 1}" style="cursor:pointer"`,
        );
    }
  }
  if ($("routes").checked)
    for (let c = 0; c < params.n_cables; c++) {
      const pts = g.units.flatMap((u) => u.cable[c]);
      svg += path(pts.map((v) => [x(v[2]), front - v[0] * sY]));
      svg += path(pts.map((v) => [x(v[2]), side - v[1] * sY]));
    }
  if ($("core").checked) {
    const neck = (params.build?.neck_width_mm ?? 1) / 1000;
    svg += poly(
      [
        [65, front - (neck * sY) / 2],
        [725, front - (neck * sY) / 2],
        [725, front + (neck * sY) / 2],
        [65, front + (neck * sY) / 2],
      ],
      "core-shape",
    );
    svg += txt(
      400,
      front - 9,
      `${params.n_cables === 2 ? "Elastic core width" : "Elastic core diameter"} = ${fmt(neck * 1000)} mm`,
      "dim-text",
      "middle",
    );
  }
  const sectionZ =
    selected.z0 + ((selected.z1 - selected.z0) * +$("station").value) / 100;
  svg += `<line x1="${x(sectionZ)}" y1="38" x2="${x(sectionZ)}" y2="295" stroke="#d7912c" stroke-dasharray="4 4"/>`;
  if ($("dimensions").checked) {
    svg += dimension(
      65,
      151,
      725,
      151,
      `Assembled Z length = ${fmt(mm(g.length))} mm`,
    );
    svg += dimension(
      65,
      325,
      65 + params.L * scale,
      325,
      `L (requested arc) = ${fmt(mm(params.L))} mm`,
    );
    svg += txt(
      73,
      38,
      `${params.n_cables === 2 ? "Wbase" : "Wref"} = ${fmt(mm(g.width))} mm`,
      "dim-text",
    );
    svg += txt(
      731,
      72,
      `${params.n_cables === 2 ? "Wtip" : "Wref,tip"} ${fmt(mm(g.units.at(-1).radius * 2))}`,
      "dim-text",
      "end",
    );
    svg += txt(
      70,
      199,
      params.n_cables === 2
        ? `Tbase = ${fmt(mm(g.thickness))} mm`
        : `φ = ${fmt(params.phi_deg)}° (continuous taper)`,
      "dim-text",
    );
    if (params.n_cables === 2)
      svg += txt(
        731,
        222,
        `Ttip = ${fmt(mm(g.tipThickness))} mm`,
        "dim-text",
        "end",
      );
    svg += txt(745, 354, "Base → tip, Z [mm]", "", "end");
  }
  $("profiles").innerHTML = svg;
  $("profiles")
    .querySelectorAll("[data-link]")
    .forEach(
      (e) =>
        (e.onclick = () => {
          $("link").value = e.dataset.link;
          render(false);
        }),
    );
  $("profile-note").textContent =
    `${g.partial ? "Partial base with flat mount." : "Complete base unit."} φ = ${fmt(params.phi_deg)}° applies to the continuous taper. ${params.n_cables === 2 ? "YZ shows the centre-thickness envelope; XZ shows the joint slits." : "Both side profiles sample the cut n-lobe surface. Wbase is the revolved reference width."}`;
}
function drawSection() {
  const u = geom.units[+$("link").value - 1],
    s = section(params, geom, u, +$("station").value / 100),
    bound = Math.max(...s.points.flat().map(Math.abs), s.rp ?? 0, 0.0001),
    k = 122 / bound,
    X = (x) => 220 + x * k,
    Y = (y) => 175 - y * k;
  let svg =
    line(60, 175, 380, 175, "axis") +
    line(220, 32, 220, 318, "axis") +
    txt(390, 178, "+X") +
    txt(225, 27, "+Y");
  if ($("construction").checked && s.polygon.length) {
    svg += poly(
      s.polygon.map(([x, y]) => [X(x), Y(y)]),
      "",
      `fill="none" stroke="#bfcbd1" stroke-dasharray="5 4"`,
    );
    for (const n of s.notches)
      svg += `<circle cx="${X(n.c[0])}" cy="${Y(n.c[1])}" r="${n.r * k}" fill="none" stroke="#c79755" stroke-dasharray="3 3"/>`;
  }
  svg += poly(s.points.map(([x, y]) => [X(x), Y(y)]));
  if ($("core").checked) {
    const half = s.neck / 2000;
    if (params.n_cables === 2)
      svg += poly(
        [
          [-half, -s.T / 2],
          [half, -s.T / 2],
          [half, s.T / 2],
          [-half, s.T / 2],
        ].map(([x, y]) => [X(x), Y(y)]),
        "core-shape",
      );
    else
      svg += `<circle cx="220" cy="175" r="${half * k}" class="core-shape"/>`;
  }
  if ($("routes").checked)
    s.cables.forEach(([x, y], i) => {
      svg +=
        `<circle cx="${X(x)}" cy="${Y(y)}" r="${s.hole ? Math.max(1, (s.hole * k) / 2000) : 3}" fill="white" stroke="#148aac" stroke-width="1.5"/>` +
        txt(X(x) + 7, Y(y) - 6, `c${i}`);
    });
  if ($("dimensions").checked) {
    const xs = s.points.map((p) => p[0]),
      ys = s.points.map((p) => p[1]),
      left = Math.min(...xs),
      right = Math.max(...xs),
      low = Math.min(...ys),
      high = Math.max(...ys);
    svg += dimension(
      X(left),
      323,
      X(right),
      323,
      `W = ${fmt(mm(right - left))} mm`,
    );
    svg += txt(
      19,
      42,
      params.n_cables === 2
        ? `Tcentre = ${fmt(mm(s.T))} mm`
        : `t = ${fmt(params.nlobe_t ?? 0.5)} · notch = ${fmt(params.notch_factor ?? 0.25)}`,
      "dim-text",
    );
    svg += txt(
      19,
      61,
      params.n_cables === 2 && params.flat_section === "hex"
        ? `Tedge / Tcentre = ${fmt(params.hex_edge_ratio ?? 0.75)}`
        : `Y span = ${fmt(mm(high - low))} mm`,
    );
    if (s.hole) svg += txt(19, 350, `Cable Ø ${fmt(s.hole)} mm`, "dim-text");
    if (s.notchR)
      svg += txt(
        419,
        350,
        `Notch radius ${fmt(mm(s.notchR))} mm`,
        "dim-text",
        "end",
      );
  }
  if ($("core").checked)
    svg += txt(
      220,
      303,
      `${params.n_cables === 2 ? "Core width" : "Core diameter"} = ${fmt(s.neck)} mm`,
      "dim-text",
      "middle",
    );
  $("section").innerHTML = svg;
  const gap = Math.min(...clearance(s));
  $("section-location").textContent =
    `${u.name} · ${$("station").value}% along link · Z = ${fmt(mm(s.z - geom.origin))} mm`;
  $("section-note").textContent =
    `Cable inward shift: ${fmt(mm(params.tendon_inward_shift))} mm. ${s.hole ? "Sampled XY wall clearance around hole" : "Sampled XY cable-centre clearance"}: ${fmt(mm(gap), 3)} mm${gap < 0 ? " — route/channel lies outside this section" : ""}. This is not a full 3D clearance check.`;
}
function drawPolar() {
  const g = geom,
    rmax = g.a * g.E * Math.exp(g.b * g.q),
    k = 145 / rmax,
    cx = 278,
    cy = 187,
    to = (r, t) => [cx + r * Math.cos(t) * k, cy - r * Math.sin(t) * k];
  let svg = "";
  for (let i = 1; i <= 3; i++) {
    const r = (rmax * i) / 3;
    svg +=
      `<circle cx="${cx}" cy="${cy}" r="${r * k}" fill="none" stroke="#e1e8e9"/>` +
      txt(cx + 4, cy - r * k + 12, `${fmt(mm(r), 1)} mm`);
  }
  for (let a = 0; a < 360; a += 45) {
    const t = (a * pi) / 180,
      pt = to(rmax * 1.08, t);
    svg +=
      line(cx, cy, ...to(rmax, t), "axis") +
      txt(pt[0], pt[1] + 4, `${a}°`, "", "middle");
  }
  const N = 450,
    inner = [],
    outer = [],
    centre = [];
  for (let i = 0; i <= N; i++) {
    const t = (i * g.q) / N;
    inner.push(to(g.a * Math.exp(g.b * t), t));
    outer.push(to(g.a * g.E * Math.exp(g.b * t), t));
    centre.push(to(g.rc0 * Math.exp(g.b * t), t));
  }
  svg +=
    poly(
      inner.concat([...outer].reverse()),
      "",
      `fill="#cbe3e1" fill-opacity=".55" stroke="none"`,
    ) +
    path(inner, "cable-line") +
    path(outer, "cable-line") +
    path(centre, "dim");
  const selected = g.units.length - +$("link").value;
  g.curled.forEach((q, i) => {
    if (i === selected)
      svg += poly(
        q.map(([x, y]) => [cx + x * k, cy - y * k]),
        "selected",
      );
  });
  for (const t of g.theta) {
    const a = to(g.a * Math.exp(g.b * t), t),
      b = to(g.rc0 * Math.exp(g.b * t), t);
    svg += line(...a, ...b, "dim");
  }
  if ($("dimensions").checked) {
    const r = rmax * 0.45,
      start = to(r, 0),
      end = to(r, g.dth);
    svg +=
      `<path d="M ${start.join(" ")} A ${r * k} ${r * k} 0 0 0 ${end.join(" ")}" fill="none" stroke="#d7912c" stroke-width="2"/>` +
      line(cx, cy, ...to(r * 1.1, 0), "dim") +
      line(cx, cy, ...to(r * 1.1, g.dth), "dim");
    svg += txt(18, 29, `Δθ = ${fmt(params.Delta_theta_deg)}°`, "dim-text");
    svg += txt(18, 48, `q₀ = ${fmt((g.q * 180) / pi)}°`);
    const tip = to(g.rc0, 0),
      base = to(g.rc0 * Math.exp(g.b * g.q), g.q);
    svg += txt(tip[0] + 7, tip[1] - 8, "Tip · θ = 0", "dim-text");
    svg += txt(base[0] + 8, base[1] - 8, "Base", "dim-text");
    svg +=
      `<circle cx="${cx}" cy="${cy}" r="2.5" fill="#183348"/>` +
      txt(cx - 7, cy + 15, "Pole", "", "end");
  }
  $("polar").innerHTML = svg;
}
function drawGains() {
  const first = $("include-base").checked ? 0 : 1,
    values = geom.gains.slice(first);
  let svg = "";
  for (let panel = 0; panel < 2; panel++) {
    const key = panel ? "d" : "k",
      left = panel ? 423 : 58,
      width = 280,
      max = Math.max(...values.map((r) => r[key]), 1e-12),
      top = 40,
      h = 160;
    for (let j = 0; j <= 4; j++) {
      const y = top + (h * j) / 4;
      svg +=
        line(left, y, left + width, y, "axis") +
        txt(left - 8, y + 4, (max * (1 - j / 4)).toPrecision(3), "", "end");
    }
    const pts = values.map((v, i) => [
      left + (i * width) / Math.max(values.length - 1, 1),
      top + h * (1 - v[key] / max),
    ]);
    svg +=
      path(pts, "cable-line") +
      txt(
        left,
        21,
        panel ? "Damping [N·m·s/rad]" : "Stiffness [N·m/rad]",
        "dim-text",
      );
    const step = Math.max(1, Math.ceil(values.length / 6));
    values.forEach((v, i) => {
      if (i % step === 0 || i === values.length - 1)
        svg += txt(pts[i][0], 220, String(i + first + 1), "", "middle");
    });
    svg += txt(
      left + width / 2,
      246,
      "Joint number (base → tip)",
      "",
      "middle",
    );
  }
  $("gains").innerHTML = svg;
  $("gain-note").textContent =
    `Protected base: K₁ = ${fmt(geom.gains[0].k, 3)}, D₁ = ${fmt(geom.gains[0].d, 3)}. ${(params.post_gen?.joint_beta ?? 1.03) === 1 ? "Flexible-joint gains are constant." : "Flexible-joint gains follow the exponential law."} Linear Y thickness does not imply linear stiffness.`;
}
function make3D() {
  faces = [];
  lines3 = [];
  const g = geom;
  for (const u of g.units) {
    const q = u.quad,
      fs = [
        0.0001,
        0.9999,
        (q[3][1] - u.z0) / (u.z1 - u.z0),
        (q[2][1] - u.z0) / (u.z1 - u.z0),
      ]
        .filter((v) => v >= 0 && v <= 1)
        .sort((a, b) => a - b);
    const rings = fs.map((f) => {
      const s = section(params, g, u, f, 64);
      return s.points.map(([x, y]) => [x, y, s.z - g.origin - g.length / 2]);
    });
    for (let j = 0; j < rings.length - 1; j++)
      for (let i = 0; i < rings[j].length; i++) {
        const k = (i + 1) % rings[j].length;
        faces.push({
          pts: [rings[j][i], rings[j][k], rings[j + 1][k], rings[j + 1][i]],
          index: u.index,
        });
      }
    faces.push(
      { pts: [...rings[0]].reverse(), index: u.index },
      { pts: rings.at(-1), index: u.index },
    );
  }
  if ($("routes").checked)
    for (let c = 0; c < params.n_cables; c++)
      lines3.push(
        g.units.flatMap((u) =>
          u.cable[c].map((p) => [p[0], p[1], p[2] - g.origin - g.length / 2]),
        ),
      );
}
function draw3D() {
  if (!geom) return;
  const canvas = $("three"),
    rect = canvas.getBoundingClientRect(),
    dpr = Math.min(devicePixelRatio || 1, 2),
    w = rect.width,
    h = rect.height;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const c = canvas.getContext("2d");
  c.scale(dpr, dpr);
  c.clearRect(0, 0, w, h);
  const transform = ([x, y, z]) => {
    const xx = Math.cos(camera.yaw) * x - Math.sin(camera.yaw) * y,
      yy = Math.sin(camera.yaw) * x + Math.cos(camera.yaw) * y;
    return [
      Math.cos(camera.pitch) * z - Math.sin(camera.pitch) * yy,
      xx,
      Math.sin(camera.pitch) * z + Math.cos(camera.pitch) * yy,
    ];
  };
  const transformed = faces.map((f) => ({ ...f, p: f.pts.map(transform) })),
    all = transformed.flatMap((f) => f.p),
    xs = all.map((p) => p[0]),
    ys = all.map((p) => p[1]),
    spanX =
      xs.reduce((a, b) => Math.max(a, b), -Infinity) -
      xs.reduce((a, b) => Math.min(a, b), Infinity),
    spanY =
      ys.reduce((a, b) => Math.max(a, b), -Infinity) -
      ys.reduce((a, b) => Math.min(a, b), Infinity),
    scale =
      Math.min(
        (w - 60) / Math.max(spanX, 0.001),
        (h - 55) / Math.max(spanY, 0.001),
      ) * camera.zoom;
  transformed.sort(
    (a, b) =>
      a.p.reduce((s, p) => s + p[2], 0) / a.p.length -
      b.p.reduce((s, p) => s + p[2], 0) / b.p.length,
  );
  for (const f of transformed) {
    c.beginPath();
    f.p.forEach((p, i) => {
      const x = w / 2 + p[0] * scale,
        y = h / 2 - p[1] * scale;
      if (i) c.lineTo(x, y);
      else c.moveTo(x, y);
    });
    c.closePath();
    const a = f.p[0],
      b = f.p[1],
      d = f.p[2];
    const v = b.map((x, i) => x - a[i]),
      u = d.map((x, i) => x - a[i]),
      n = [
        v[1] * u[2] - v[2] * u[1],
        v[2] * u[0] - v[0] * u[2],
        v[0] * u[1] - v[1] * u[0],
      ],
      mag = Math.hypot(...n) || 1,
      shade =
        0.45 + 0.5 * Math.abs((0.3 * n[0] + 0.5 * n[1] + 0.8 * n[2]) / mag);
    const selected = f.index === +$("link").value - 1;
    c.fillStyle = selected
      ? `rgb(${Math.round(45 + 55 * shade)},${Math.round(128 + 75 * shade)},${Math.round(134 + 72 * shade)})`
      : `rgb(${Math.round(60 + 100 * shade)},${Math.round(101 + 105 * shade)},${Math.round(126 + 97 * shade)})`;
    c.fill();
    if (params.n_cables === 2) {
      c.lineWidth = 0.3;
      c.strokeStyle = "#27587150";
      c.stroke();
    }
  }
  c.strokeStyle = "#148aac";
  c.lineWidth = 1;
  for (const line of lines3) {
    c.beginPath();
    line
      .map(transform)
      .forEach((p, i) =>
        i
          ? c.lineTo(w / 2 + p[0] * scale, h / 2 - p[1] * scale)
          : c.moveTo(w / 2 + p[0] * scale, h / 2 - p[1] * scale),
      );
    c.stroke();
  }
  c.font = "11px system-ui";
  c.fillStyle = "#647683";
  c.fillText("Local geometry · world pose is applied only in XML", 16, h - 16);
  for (const [label, z] of [
    ["Base", -geom.length / 2],
    ["Tip", geom.length / 2],
  ]) {
    const p = transform([0, 0, z]);
    c.textAlign = "center";
    c.fillText(
      label,
      Math.max(20, Math.min(w - 20, w / 2 + p[0] * scale)),
      Math.max(18, Math.min(h - 32, h / 2 - p[1] * scale + 30)),
    );
  }
  c.textAlign = "left";
}
function render(rebuild3 = true) {
  try {
    validate(params, schema);
    configurationChecks();
    geom = derive(params);
    valid = true;
    $("errors").hidden = true;
    const n = geom.units.length;
    $("link").max = n;
    if (+$("link").value > n) $("link").value = n;
    $("link-value").textContent = String($("link").value).padStart(3, "0");
    $("station-value").textContent = $("station").value + "%";
    $("metrics").innerHTML = [
      ["Requested arc L", fmt(mm(params.L)), "mm", "Before segmentation"],
      [
        "Assembled length",
        fmt(mm(geom.length)),
        "mm",
        `${fmt((geom.length / geom.effective - 1) * 100)}% arc-to-chord difference`,
      ],
      [
        "Links",
        n,
        "",
        geom.partial ? "Includes one partial base" : "All links complete",
      ],
      ["Base width", fmt(mm(geom.width)), "mm", "Revolved / XZ reference"],
      [
        "Geometric scale βg",
        fmt(geom.beta, 4),
        "",
        "Adjacent complete-link ratio",
      ],
    ]
      .map(
        ([a, b, c, d]) =>
          `<div class="metric"><small>${a}</small><strong>${b}<span>${c}</span></strong><div class="sub">${d}</div></div>`,
      )
      .join("");
    $("section-tag").textContent = params.build?.plain
      ? "Circular revolve"
      : params.n_cables === 2
        ? `${params.flat_section ?? "rectangular"} · ${params.thickness_profile ?? "linear"}`
        : `${params.n_cables}-lobe`;
    drawProfiles();
    drawSection();
    drawPolar();
    drawGains();
    if (rebuild3) make3D();
    draw3D();
    $("constants").innerHTML = [
      ["a", `${fmt(mm(geom.a), 6)} mm`],
      ["b", fmt(geom.b, 8)],
      ["E", fmt(geom.E, 6)],
      ["q₀ requested", `${fmt(geom.qRequested, 6)} rad`],
      ["q₀ effective", `${fmt(geom.q, 6)} rad`],
      ["βg = exp(bΔθ)", fmt(geom.beta, 6)],
      ["βⱼ (gain law)", fmt(params.post_gen?.joint_beta ?? 1.03, 6)],
    ]
      .map(([a, b]) => `<div><span>${a}</span><strong>${b}</strong></div>`)
      .join("");
    $("json").value = JSON.stringify(params, null, 2) + "\n";
    $("command").textContent =
      "uv run python build.py --params params.json \\\n  --no-preview --output-dir build/my-spirob";
    try {
      localStorage.setItem("spirob-designer-v1", JSON.stringify(params));
    } catch {}
  } catch (e) {
    valid = false;
    $("errors").textContent =
      e.message + " The last valid preview is shown; export/build is disabled.";
    $("errors").hidden = false;
  }
  $("download").disabled = !valid;
  $("copy-json").disabled = !valid;
  $("generate").disabled = !valid || !!activeJob;
}
async function loadPreset() {
  const id = $("preset").value,
    name =
      id === "2hex"
        ? "params-two-cable-hex.json"
        : `params-${presets[id]}-cable.json`;
  params = clean(await fetch(`presets/${name}`).then((r) => r.json()));
  notice("");
  form();
  render();
}
function download(name, text, type = "application/json") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function copy(text, button) {
  try {
    await navigator.clipboard.writeText(text);
    const old = button.textContent;
    button.textContent = "Copied";
    setTimeout(() => (button.textContent = old), 1500);
  } catch {
    notice("Clipboard access is unavailable. Select and copy the text below.");
  }
}
$("download").onclick = () =>
  download("params.json", JSON.stringify(params, null, 2) + "\n");
$("copy-json").onclick = (e) => copy(JSON.stringify(params, null, 2), e.target);
$("copy-command").onclick = (e) => copy($("command").textContent, e.target);
$("apply-json").onclick = () => {
  try {
    const p = clean(JSON.parse($("json").value));
    validate(p, schema);
    params = p;
    form();
    render();
  } catch (e) {
    notice(e.message);
  }
};
$("import").onclick = () => $("file").click();
$("file").onchange = async () => {
  try {
    const file = $("file").files[0];
    if (!file) return;
    if (file.size > 100000) throw Error("Parameter file exceeds 100 kB.");
    const p = clean(JSON.parse(await file.text()));
    validate(p, schema);
    params = p;
    form();
    render();
  } catch (e) {
    notice(e.message);
  } finally {
    $("file").value = "";
  }
};
$("preset").onchange = loadPreset;
$("reset").onclick = loadPreset;
for (const id of [
  "dimensions",
  "routes",
  "core",
  "link",
  "station",
  "construction",
  "include-base",
])
  $(id).oninput = () => render(id === "routes");
let drag = null;
$("three").onpointerdown = (e) => {
  drag = { x: e.clientX, y: e.clientY, yaw: camera.yaw, pitch: camera.pitch };
  $("three").setPointerCapture(e.pointerId);
};
$("three").onpointermove = (e) => {
  if (drag) {
    camera.yaw = drag.yaw + (e.clientY - drag.y) * 0.01;
    camera.pitch = drag.pitch + (e.clientX - drag.x) * 0.01;
    draw3D();
  }
};
$("three").onpointerup = () => (drag = null);
$("three").onpointercancel = () => (drag = null);
$("three").addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    camera.zoom = Math.max(
      0.5,
      Math.min(3, camera.zoom * Math.exp(-e.deltaY * 0.001)),
    );
    draw3D();
  },
  { passive: false },
);
const resetCamera = () => {
  camera = { yaw: 0.6, pitch: 0.6, zoom: 1 };
  draw3D();
};
$("three").ondblclick = resetCamera;
document.querySelectorAll("[data-camera]").forEach(
  (b) =>
    (b.onclick = () => {
      if (b.dataset.camera === "end")
        camera = { yaw: 0, pitch: pi / 2, zoom: 1 };
      else if (b.dataset.camera === "front")
        camera = { yaw: 0, pitch: 0, zoom: 1 };
      else camera = { yaw: 0.6, pitch: 0.6, zoom: 1 };
      draw3D();
    }),
);
new ResizeObserver(() => draw3D()).observe($("three"));
$("generate").onclick = async () => {
  if (!local || !valid || activeJob) return;
  $("generate").disabled = true;
  $("model-download").hidden = true;
  try {
    const response = await fetch("/api/build", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    const result = await response.json();
    if (!response.ok) throw Error(result.error);
    activeJob = result.id;
    $("build-log").textContent = "Building from the current configuration…";
    const poll = async () => {
      try {
        const r = await fetch(`/api/jobs/${activeJob}`);
        const job = await r.json();
        if (!r.ok) throw Error(job.error);
        $("build-log").textContent = job.log || job.status;
        if (job.status === "running") {
          setTimeout(poll, 1000);
          return;
        }
        if (job.status === "completed") {
          const a = $("model-download");
          a.href = job.download;
          a.download = "spirob-model.zip";
          a.hidden = false;
          a.click();
        }
        activeJob = null;
        $("generate").disabled = !valid;
      } catch (e) {
        activeJob = null;
        $("generate").disabled = !valid;
        $("build-log").textContent = e.message;
      }
    };
    setTimeout(poll, 500);
  } catch (e) {
    activeJob = null;
    $("generate").disabled = !valid;
    $("build-log").textContent = e.message;
  }
};
await loadPreset();
try {
  const saved = stored;
  if (saved) {
    const p = clean(JSON.parse(saved));
    validate(p, schema);
    params = p;
    form();
    render();
  }
} catch {}
if (["127.0.0.1", "localhost"].includes(location.hostname))
  try {
    const r = await fetch("/api/status");
    if (r.ok) {
      const s = await r.json();
      local = s.builder === true;
      if (local) {
        $("mode").textContent = "Local builder connected";
        $("local-actions").hidden = false;
        $("remote-actions").hidden = true;
        $("build-title").textContent = "Generate XML, meshes & CAD";
        $("build-description").textContent =
          "The local Python generator validates the XML and CAD, then downloads a ZIP. Enable CAD and IGES in “Elastic core & channels” for those files.";
      }
    }
  } catch {}
window.spirob = {
  get params() {
    return structuredClone(params);
  },
  get geometry() {
    return geom;
  },
  get valid() {
    return valid;
  },
};
