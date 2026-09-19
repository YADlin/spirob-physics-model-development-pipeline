// Browser interaction and download checks. No Node code is shipped to the site.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { readFile, mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { createServer } from "node:http";
const { chromium } = await import(
  process.env.SPIROB_PLAYWRIGHT_MODULE || "playwright"
);
const root = resolve(import.meta.dirname, "../.."),
  python = process.env.SPIROB_PYTHON || resolve(root, ".venv/bin/python");
const server = spawn(
  python,
  [
    "tools/designer.py",
    "--no-browser",
    "--port",
    "8765",
    "--output-dir",
    "build/browser tests",
  ],
  { cwd: root, stdio: ["ignore", "pipe", "pipe"] },
);
await new Promise((yes, no) => {
  server.stdout.once("data", yes);
  server.once("exit", (c) => no(Error(`Builder exited ${c}`)));
});
let browser, staticServer;
const errors = [];
try {
  browser = await chromium.launch({
    headless: true,
    ...(process.env.SPIROB_BROWSER_EXECUTABLE
      ? { executablePath: process.env.SPIROB_BROWSER_EXECUTABLE }
      : {}),
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  const context = await browser.newContext({
      viewport: { width: 1600, height: 1150 },
    }),
    page = await context.newPage();
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("http://127.0.0.1:8765");
  await page.waitForFunction(() => window.spirob?.valid);
  assert.equal(
    await page.locator("#mode").textContent(),
    "Local builder connected",
  );
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
    true,
  );
  const initial = await page.evaluate(() => window.spirob.geometry.b);
  await page.locator("#param-phi_deg").fill("8");
  await page.waitForFunction(
    (b) => window.spirob?.valid && window.spirob.geometry.b !== b,
    initial,
  );
  await page.locator("#param-L").fill("-1");
  assert.equal(await page.locator("#download").isDisabled(), true);
  assert.equal(await page.locator("#errors").isVisible(), true);
  await page.selectOption("#preset", "2hex");
  await page.waitForFunction(
    () => window.spirob?.valid && window.spirob.params.n_cables === 2,
  );
  await page.locator('details[data-group="section"] summary').click();
  await page
    .getByRole("checkbox", { name: "Base centre thickness automatic" })
    .uncheck();
  await page.locator("#param-base_thickness_m").fill("20");
  await page.waitForFunction(() => window.spirob.geometry.thickness === 0.02);
  await page.locator("#param-hex_edge_ratio").fill(".6");
  await page.waitForFunction(() => window.spirob.params.hex_edge_ratio === 0.6);
  const old = await page.locator("#section").innerHTML();
  await page.locator("#link").fill("10");
  await page.locator("#link").dispatchEvent("input");
  assert.notEqual(await page.locator("#section").innerHTML(), old);
  const downloadWait = page.waitForEvent("download");
  await page.locator("#download").click();
  const d = await downloadWait;
  const data = JSON.parse(await readFile(await d.path(), "utf8"));
  assert.equal(data.base_thickness_m, 0.02);
  assert.equal(data.hex_edge_ratio, 0.6);
  assert.equal(data.build.collision_mode, "convex");
  assert.equal("target_site_pos" in data.post_gen, false);
  await page.reload();
  await page.waitForFunction(
    () =>
      window.spirob?.params.base_thickness_m === 0.02 && window.spirob.valid,
  );
  await page.selectOption("#preset", "4");
  await page.waitForFunction(
    () => window.spirob?.valid && window.spirob.params.n_cables === 4,
  );
  assert.equal(await page.locator("#param-base_thickness_m").count(), 0);
  const p = { ...data, n_cables: 3 };
  for (const k of [
    "base_thickness_m",
    "flat_section",
    "hex_edge_ratio",
    "thickness_profile",
  ])
    delete p[k];
  p.notch_factor = 0;
  p.post_gen.target_site_pos = [1, 2, 3];
  await page
    .locator("#file")
    .setInputFiles({
      name: "params.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(p)),
    });
  await page.waitForFunction(
    () =>
      window.spirob?.valid &&
      window.spirob.params.n_cables === 3 &&
      window.spirob.params.notch_factor === 0,
  );
  assert.equal(
    await page.evaluate(
      () => "target_site_pos" in window.spirob.params.post_gen,
    ),
    false,
  );
  // Browser build really creates an archive; a short whole-unit robot keeps CI bounded.
  p.L = 0.025;
  p.terminal_unit_policy = "whole_units";
  delete p.post_gen.target_site_pos;
  p.build.cad = true;
  p.build.iges = true;
  p.build.neck_width_mm = 0.6;
  await page
    .locator("#file")
    .setInputFiles({
      name: "params.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(p)),
    });
  await page.waitForFunction(
    () => window.spirob?.valid && window.spirob.params.L === 0.025,
  );
  const zipWait = page.waitForEvent("download", { timeout: 180000 });
  await page.locator("#generate").click();
  const zip = await zipWait;
  await mkdir(resolve(root, "build/verification"), { recursive: true });
  await zip.saveAs(resolve(root, "build/verification/browser-model.zip"));
  assert.equal(
    await page
      .locator("#build-log")
      .textContent()
      .then((t) => t.includes("Build completed")),
    true,
  );
  await page.selectOption("#preset", "3");
  await page.waitForFunction(
    () => window.spirob?.valid && window.spirob.params.L === 0.22628,
  );
  await page.evaluate(() => scrollTo(0, 0));
  await page.screenshot({
    path: resolve(root, "docs/figures/designer-desktop.png"),
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: resolve(root, "build/verification/designer-mobile.png"),
    fullPage: true,
  });
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
    true,
  );
  // Serve only static files under a project subpath, as GitHub Pages does.
  staticServer = createServer(async (req, res) => {
    const prefix = "/spirob-project/";
    if (!req.url.startsWith(prefix)) {
      res.writeHead(404);
      res.end();
      return;
    }
    const name = req.url.slice(prefix.length) || "index.html";
    const path = resolve(root, "site", name);
    if (!path.startsWith(resolve(root, "site") + "/")) {
      res.writeHead(404);
      res.end();
      return;
    }
    try {
      const data = await readFile(path);
      res.setHeader(
        "Content-Type",
        name.endsWith(".mjs")
          ? "text/javascript"
          : name.endsWith(".json")
            ? "application/json"
            : name.endsWith(".css")
              ? "text/css"
              : "text/html",
      );
      res.end(data);
    } catch {
      res.writeHead(404);
      res.end();
    }
  });
  await new Promise((yes) => staticServer.listen(0, "127.0.0.1", yes));
  const staticContext = await browser.newContext({
    viewport: { width: 1600, height: 1250 },
  });
  const staticPage = await staticContext.newPage();
  staticPage.on("pageerror", (e) => errors.push(e.message));
  await staticPage.goto(
    `http://127.0.0.1:${staticServer.address().port}/spirob-project/`,
  );
  await staticPage.waitForFunction(() => window.spirob?.valid);
  assert.equal(await staticPage.locator("#remote-actions").isVisible(), true);
  assert.equal(await staticPage.locator("#local-actions").isVisible(), false);
  await staticPage.selectOption("#preset", "2hex");
  await staticPage.waitForFunction(
    () => window.spirob?.valid && window.spirob.params.n_cables === 2,
  );
  await staticPage.screenshot({
    path: resolve(root, "build/verification/designer-static.png"),
  });
  assert.deepEqual(errors, []);
  console.log(
    "Browser checks passed: controls, validation, section selection, JSON import/export, persistence, obsolete target migration, mobile layout, static project subpath, and actual XML/STL/STEP/IGES ZIP download.",
  );
} finally {
  await browser?.close();
  staticServer?.close();
  server.kill();
}
