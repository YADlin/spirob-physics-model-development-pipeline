// Exercise a report download on static hosting, without a Python model builder.
import assert from 'node:assert/strict';
import {spawn,execFileSync} from 'node:child_process';
import {mkdir,readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {chromium} from 'playwright';
const root=resolve(import.meta.dirname,'../..'), python=process.env.SPIROB_PYTHON||resolve(root,'.venv/bin/python');
const server=spawn(python,['-u','-m','http.server','8879','--bind','127.0.0.1'],{cwd:root,stdio:['ignore','pipe','pipe']});
await new Promise((ok,no)=>{server.stdout.once('data',ok);server.once('exit',c=>no(Error(`Server exited ${c}`)));});
let browser;
try{
 browser=await chromium.launch({headless:true,...(process.env.SPIROB_BROWSER_EXECUTABLE?{executablePath:process.env.SPIROB_BROWSER_EXECUTABLE}:{}),args:['--no-sandbox','--disable-dev-shm-usage']});
 const page=await browser.newPage({viewport:{width:1500,height:1100}}), errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:8879/site/');
 await page.waitForFunction(()=>window.spirob?.valid);
 await page.selectOption('#preset','2');
 await page.waitForFunction(()=>window.spirob?.valid && window.spirob.params.n_cables===2);
 await page.locator('details[data-group="dynamics"] summary').click();
 const originalGains=await page.evaluate(()=>window.spirob.geometry.gains);
 await page.locator('#convert-joint2').click();
 const converted=await page.evaluate(()=>({p:window.spirob.params.post_gen,g:window.spirob.geometry.gains}));
 assert.ok(Math.abs(converted.p.k2-.2/1.03**3)<1e-14);
 for(let i=0;i<originalGains.length;i++) {
   assert.ok(Math.abs(originalGains[i].k-converted.g[i].k)<1e-13);
   assert.ok(Math.abs(originalGains[i].d-converted.g[i].d)<1e-13);
 }
 await page.locator('[id="param-post_gen.k2"]').fill('0.35');
 await page.locator('[id="param-post_gen.rho_k"]').fill('0.8');
 const fitted=await page.evaluate(()=>window.spirob.geometry.gains);
 assert.ok(Math.abs(fitted[2].k-.35*.8)<1e-14);
 assert.equal(fitted[0].k,originalGains[0].k);
 for(let i=0;i<originalGains.length;i++) assert.equal(fitted[i].d,converted.g[i].d);
 await page.locator('[id="param-post_gen.k2"]').fill('');
 assert.ok(await page.evaluate(()=>window.spirob.valid && !('k2' in window.spirob.params.post_gen)));
 await page.locator('[id="param-post_gen.k2"]').fill('0.35');
 await page.locator('[id="param-post_gen.d2"]').fill('0.015');
 await page.locator('[id="param-post_gen.rho_d"]').fill('0.96');
 assert.ok(await page.evaluate(()=>Math.abs(window.spirob.geometry.gains[2].d-.015*.96)<1e-14));
 assert.ok((await page.locator('#gain-note').textContent()).includes('Reference j_002'));
 await page.locator('details[data-group="section"] summary').click();
 await page.selectOption('#param-thickness_profile','constant');
 await page.getByRole('checkbox',{name:'Base centre thickness automatic'}).uncheck();
 await page.locator('#param-base_thickness_m').fill('3');
 await page.waitForFunction(()=>window.spirob.valid && window.spirob.geometry.units.every(u=>u.t0===.003 && u.t1===.003));
 await page.locator('details[data-group="fabrication"] summary').click();
 await page.getByLabel('Elastic core / local width',{exact:true}).fill('8');
 await page.getByLabel('Cable channel diameter',{exact:true}).fill('1.6');
 await page.locator('#routes').uncheck();
 assert.equal(await page.locator('#section [clip-path="url(#hole-clip)"]').count(),2);
 await page.locator('[data-camera="end"]').click();
 const drilled=await page.locator('#three').evaluate(c=>c.toDataURL());
 await page.locator('#holes').uncheck();
 const undrilled=await page.locator('#three').evaluate(c=>c.toDataURL());
 assert.notEqual(drilled,undrilled);
 assert.equal(await page.locator('#section [clip-path="url(#hole-clip)"]').count(),0);
 await page.locator('#holes').check();

 await page.locator('#link').fill('3');await page.locator('#link').dispatchEvent('input');
 await page.locator('#station').fill('65');await page.locator('#station').dispatchEvent('input');
 const params=await page.evaluate(()=>window.spirob.params);
 const folder=resolve(root,'build/report-browser');await mkdir(folder,{recursive:true});
 await page.locator('#three').screenshot({path:resolve(folder,'bore-end-view.png')});
 await page.locator('#section').screenshot({path:resolve(folder,'bore-section.png')});
 const waiting=page.waitForEvent('download');await page.locator('#download-report').click();
 const download=await waiting;assert.equal(download.suggestedFilename(),'spirob-design-report.zip');
 const zip=resolve(folder,'report.zip');await download.saveAs(zip);
 execFileSync(python,['-c',`import zipfile,sys,hashlib,json,xml.etree.ElementTree as E
from pathlib import Path
p=Path(sys.argv[1])
with zipfile.ZipFile(p) as z:
 assert z.testzip() is None
 raw=z.read('params.json');t=z.read('report.md').decode()
 assert hashlib.sha256(raw).hexdigest() in t
 assert 'Fixed reference j_002' in t and 'rho_k' in t and 'rho_d' in t
 assert 'DESIGN PREVIEW' in t and 'Selected link: 3; section station: 65%' in t
 assert json.loads(raw)['build']['elastic_core_percent']==8
 assert json.loads(raw)['thickness_profile']=='constant'
 assert json.loads(raw)['base_thickness_m']==.003
 assert len([n for n in z.namelist() if n.endswith('.svg')])==4
 for n in z.namelist():
  if n.endswith('.svg'): E.fromstring(z.read(n))
 assert z.read('figures/three.png').startswith(b'\\x89PNG')
 z.extractall(p.parent/'extracted')
`,zip]);
 assert.deepEqual(JSON.parse(await readFile(resolve(folder,'extracted/params.json'),'utf8')),params);
 await page.goto('http://127.0.0.1:8879/build/report-browser/extracted/report.html');
 await page.screenshot({path:resolve(folder,'report.png'),fullPage:true});
 await page.emulateMedia({media:'print'});
 await page.pdf({path:resolve(folder,'report.pdf'),format:'A4',printBackground:true});
 await page.goto('http://127.0.0.1:8879/site/');await page.waitForFunction(()=>window.spirob?.valid);
 await page.locator('#param-L').fill('-1');assert.equal(await page.locator('#download-report').isDisabled(),true);
 await page.setViewportSize({width:390,height:844});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
 assert.deepEqual(errors,[]);
 console.log('Report browser checks passed: joint-2 conversion, independent damping, clearing optional gains, report convention, static project subpath, constant-thickness UI/preview, visible bore openings and toggle, ZIP/CRC, exact params/hash, selected link/station, SVG/PNG, printable PDF, invalid input and mobile width.');
}finally{await browser?.close();server.kill();}
