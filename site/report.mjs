// Portable Markdown + exact rendered figures. No CDN or server dependency.
import {coreWidthAt} from "./geometry.mjs";
const enc = new TextEncoder();
export function zipFiles(files) {
  const parts=[], directory=[]; let offset=0;
  const header=(n)=>new Uint8Array(n), put=(a,o,v)=>{new DataView(a.buffer).setUint32(o,v,true);};
  for (const [name, input] of Object.entries(files)) {
    const data=typeof input==='string'?enc.encode(input):input, filename=enc.encode(name);
    let crc=0xffffffff;
    for(const b of data){crc^=b;for(let i=0;i<8;i++)crc=(crc>>>1)^((crc&1)?0xedb88320:0);}
    crc=(crc^0xffffffff)>>>0;
    const local=header(30), central=header(46);
    put(local,0,0x04034b50);local[4]=20;local[12]=0x21;
    put(local,14,crc);put(local,18,data.length);put(local,22,data.length);local[26]=filename.length;
    put(central,0,0x02014b50);central[4]=20;central[6]=20;central[14]=0x21;
    put(central,16,crc);put(central,20,data.length);put(central,24,data.length);central[28]=filename.length;put(central,42,offset);
    parts.push(local,filename,data);directory.push(central,filename);offset+=30+filename.length+data.length;
  }
  const end=header(22), count=Object.keys(files).length;
  put(end,0,0x06054b50);end[8]=count;end[10]=count;
  put(end,12,directory.reduce((n,a)=>n+a.length,0));put(end,16,offset);
  return new Blob([...parts,...directory,end],{type:'application/zip'});
}
const escape=(s)=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function svgSnapshot(id){
  const src=document.getElementById(id), copy=src.cloneNode(true);
  copy.setAttribute('xmlns','http://www.w3.org/2000/svg');
  copy.style.background='white';
  const originals=[src,...src.querySelectorAll('*')], clones=[copy,...copy.querySelectorAll('*')];
  originals.forEach((el,i)=>{
    const style=getComputedStyle(el);
    for(const key of ['fill','stroke','stroke-width','stroke-dasharray','opacity','font-family','font-size','font-weight','text-anchor','dominant-baseline'])
      clones[i].style.setProperty(key,style.getPropertyValue(key));
  });
  return new XMLSerializer().serializeToString(copy);
}
export async function designReport(params,g,schema) {
  const p=structuredClone(params), json=JSON.stringify(p,null,2)+'\n', when=new Date().toISOString();
  const snapshot=Object.fromEntries(['profiles','section','polar','gains'].map(id=>[id,svgSnapshot(id)]));
  const imageURL=document.getElementById('three').toDataURL('image/png');
  const selected=document.getElementById('link').value, station=document.getElementById('station').value;
  const includeBase=document.getElementById('include-base').checked;
  const captions=['profile-note','section-note','gain-note'].map(id=>document.getElementById(id).textContent);
  const digest=await crypto.subtle.digest('SHA-256',enc.encode(json));
  const hash=Array.from(new Uint8Array(digest),b=>b.toString(16).padStart(2,'0')).join('');
  const metrics=[['Requested arc L (mm)',p.L*1000],['Effective arc (mm)',g.effective*1000],['Assembled chord length (mm)',g.length*1000],['Links',g.units.length],['Partial base',g.partial],['Base reference width (mm)',g.width*1000],['a (mm)',g.a*1000],['b',g.b],['E',g.E],['q requested (rad)',g.qRequested],['q effective (rad)',g.q],['Geometric beta',g.beta],['Gain beta',p.post_gen?.joint_beta??1.03],['Core percent',p.build?.elastic_core_percent??5],['Core base (mm)',coreWidthAt(p,g,g.units[0].z0)*1000],['Core tip (mm)',coreWidthAt(p,g,g.units.at(-1).z1)*1000]];
  const views=[['profiles','Length and side profiles'],['section','Cross-section'],['polar','Spiral in polar coordinates'],['gains','Joint stiffness and damping']];
  const files={'params.json':json};
  const notes=['DESIGN PREVIEW: this snapshot does not certify a completed CAD/MJCF build. The report inside a generated model ZIP records the completed build.',
    `Created (UTC): ${when}. Source: ${location.href}. Selected link: ${selected}; section station: ${station}%. Figures preserve the current labels, overlays and view settings.`,
    `Protected base included in gain plot: ${includeBase}. Complete gain values are in joint_gains.json.`,
    'Spiral: r_inner(theta) = a exp(b theta); r_outer(theta) = a E exp(b theta); E = exp(2 pi b). Theta runs from tip (0) to base (q).',
    'Elastic core: c(z) = (p/100) Wref(z). Width/diameter is linear in axial distance and geometric at complete-link stations. Two-cable Y thickness follows the selected thickness profile.',
    'Joint law: K_i = K0 / beta_j^(3(i-1)), D_i = D0 / beta_j^(3(i-1)); base overrides apply. These are modelling assumptions, not material calibration.',
    'The 3D figure is a sampled preview with bore openings; interior bore walls are omitted. Manufacturing bores are straight; blue simulation routing sites are separate. Fabrication holes/core and simulation mass/contact geometry are distinct. CAD export units are mm; simulation geometry uses metres.',
    ...captions];
  let md='# SpiRob design report\n\n'+notes.join('\n\n')+`\n\nParameters SHA-256 (params.json bytes): \`${hash}\`\n\n## Derived dimensions and constants\n\n| Quantity | Value |\n|---|---|\n`+metrics.map(([k,v])=>`| ${k} | ${v} |`).join('\n');
  let body='<h1>SpiRob design report</h1>'+notes.map(n=>`<p>${escape(n)}</p>`).join('')+`<p>Parameters SHA-256: <code>${hash}</code></p><h2>Derived dimensions and constants</h2><table>`+metrics.map(([k,v])=>`<tr><th>${escape(k)}</th><td>${escape(v)}</td></tr>`).join('')+'</table>';
  const rows=[];
  function fields(obj,spec,prefix='') {
    for(const [key,value] of Object.entries(obj)) {
      const item=spec.properties?.[key]??{};
      if(value && typeof value==='object' && !Array.isArray(value))fields(value,item,prefix+key+'.');
      else rows.push([prefix+key,item.title??key,JSON.stringify(value),item['x-unit']??'']);
    }
  }
  fields(p,schema);
  md+='\n\n## Selected parameters (native JSON units)\n\n| Parameter | Meaning | Value | Unit |\n|---|---|---|---|\n'+rows.map(row=>'| '+row.map(x=>String(x).replaceAll('|','\\|')).join(' | ')+' |').join('\n');
  body+='<h2>Selected parameters (native JSON units)</h2><table><tr><th>Parameter</th><th>Meaning</th><th>Value</th><th>Unit</th></tr>'+rows.map(row=>'<tr>'+row.map(x=>'<td>'+escape(x)+'</td>').join('')+'</tr>').join('')+'</table>';
  for(const [id,title] of views){const svg=snapshot[id];files[`figures/${id}.svg`]=svg;md+=`\n\n## ${title}\n\n![${title}](figures/${id}.svg)`;body+=`<section><h2>${title}</h2>${svg}</section>`;}
  const url=imageURL;
  files['figures/three.png']=Uint8Array.from(atob(url.split(',')[1]),c=>c.charCodeAt(0));
  md+='\n\n## 3D preview\n\n![3D preview](figures/three.png)\n\n## Selected parameters (JSON)\n\n```json\n'+json+'```\n';
  body+=`<section><h2>3D preview</h2><img src="${url}" alt="3D preview"></section><h2>Selected parameters (JSON)</h2><pre>${escape(json)}</pre>`;
  files['joint_gains.json']=JSON.stringify(g.gains,null,2)+'\n';
  files['report.md']=md;
  files['report.html']='<!doctype html><html lang="en"><meta charset="utf-8"><title>SpiRob design report</title><style>body{max-width:1000px;margin:32px auto;padding:0 20px;font:15px system-ui;color:#20343e}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:7px;text-align:left}svg,img{width:100%;height:auto}pre{white-space:pre-wrap}code{overflow-wrap:anywhere}section{break-inside:avoid}@media print{button{display:none}body{margin:0}}</style><button onclick="window.print()">Print / Save as PDF</button>'+body+'</html>';
  return zipFiles(files);
}
