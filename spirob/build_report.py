"""Portable build record, produced after all outputs have passed build checks.

Uses the actual compiled simulation meshes and joint gains. No Node/browser is
required. Drawings are dimensioned previews, not manufacturing inspection.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np


def write_build_report(stage, params, geometry, model):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Circle
    import trimesh
    from tools.inspect_section import compiled_surface
    from spirob.core import core_dimensions

    stage=Path(stage); out=stage/'design_report'; figs=out/'figures'; figs.mkdir(parents=True)
    raw=(stage/'build_params.json').read_bytes()
    (out/'params.json').write_bytes(raw)
    digest=hashlib.sha256(raw).hexdigest()
    root=Path(__file__).resolve().parents[1]
    try:
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,stderr=subprocess.DEVNULL,text=True).strip()
        dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,stderr=subprocess.DEVNULL,text=True).strip()
        if dirty: revision+=' (modified working tree)'
    except (OSError, subprocess.CalledProcessError): revision='unavailable (source archive/install)'
    core=core_dimensions(geometry,params['build']['elastic_core_percent'])
    zbase=core['width_law']['z_base_m']*1000
    surfaces=[]
    for unit in geometry.units:
        vertices,faces=compiled_surface(model,unit.link_name)
        surfaces.append((vertices,faces))
    records=[]
    for j in range(model.njnt):
        start=model.jnt_dofadr[j]
        end=model.jnt_dofadr[j+1] if j+1<model.njnt else model.nv
        records.append(dict(joint=model.joint(j).name,stiffness=float(model.jnt_stiffness[j]),
                            damping_per_dof=model.dof_damping[start:end].tolist()))
    (out/'joint_gains.json').write_text(json.dumps(records,indent=2)+'\n')
    titles=[]
    def save(fig,name,title):
        fig.savefig(figs/(name+'.svg'),bbox_inches='tight');plt.close(fig);titles.append((name,title))
    with plt.rc_context({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(2,1,figsize=(12,6),layout='constrained')
        for (v,f),unit in zip(surfaces,geometry.units):
            v=v.copy();v[:,2]+=unit.local_frame_origin_m[2]*1000-zbase
            for ax,xy in zip(axes,[(2,0),(2,1)]):
                ax.add_collection(PolyCollection(v[f][:,:,xy],facecolor='#c7dfdf',edgecolor='none',rasterized=True))
        stations=core['stations'];z=[s['z_from_base_mm'] for s in stations];c=np.array([s['core_width_mm'] for s in stations])
        axes[0].fill_between(z,-c/2,c/2,color='#cc8b23',label='Core width / diameter envelope')
        for ax,title,axis in zip(axes,['Front profile XZ','Side profile YZ'],['X','Y']):
            ax.autoscale();ax.set_aspect('equal');ax.set_ylabel(axis+' (mm)');ax.set_xlabel('Distance from base (mm)');ax.set_title(title)
        axes[0].legend(fontsize=8,loc='upper right')
        fig.suptitle(f"Requested arc L = {params['L']*1000:g} mm | Assembled = {geometry.lengths.discrete_chord_length_m*1000:.3f} mm\n"
                     f"Taper φ = {params['phi_deg']:g}° | Δθ = {params['Delta_theta_deg']:g}° | Core = {core['elastic_core_percent']:g}%\n"
                     f"Core base {c[0]:.4f} mm → tip {c[-1]:.4f} mm",fontsize=12)
        save(fig,'profiles','Simulation mesh side profiles; gold = reference core width envelope')

        indices=sorted(set([0,len(surfaces)//2,len(surfaces)-1]))
        fig,axes=plt.subplots(1,len(indices),figsize=(12,4),layout='constrained',squeeze=False)
        for ax,i in zip(axes.flat,indices):
            v,f=surfaces[i];u=geometry.units[i]
            height=(u.slit_reference_m[2]-u.local_frame_origin_m[2])*1000
            cut=trimesh.Trimesh(v,f,process=False).section(plane_origin=[0,0,height/2],plane_normal=[0,0,1])
            if cut is not None:
                for loop in cut.discrete:ax.fill(loop[:,0],loop[:,1],color='#c7dfdf',ec='#246e73',lw=.8)
            width=(stations[i]['core_width_mm']+stations[i+1]['core_width_mm'])/2
            if params['n_cables']==2 and not params['build']['plain']:
                # A dimensional indicator, not an approximation of hex core shoulders.
                ax.axvspan(-width/2,width/2,color='#cc8b23',alpha=.45,label=f'Core X width {width:.3f} mm')
            else:ax.add_patch(Circle((0,0),width/2,color='#cc8b23',alpha=.7,label=f'Core Ø {width:.3f} mm'))
            ax.autoscale();ax.set_aspect('equal');ax.set_title(f'{u.link_name} · 50% of link');ax.set_xlabel('X (mm)');ax.set_ylabel('Y (mm)');ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,-.18))
        save(fig,'sections','Actual simulation mesh mid-sections; gold indicates core X width or diameter, not drilled fabrication channels')

        s=geometry.spiral;theta=np.linspace(0,s.q0_rad,600)
        fig=plt.figure(figsize=(7,6),layout='constrained');ax=fig.add_subplot(projection='polar')
        for mult,label in [(1,'Inner'),((s.E+1)/2,'Centre'),(s.E,'Outer')]:
            ax.plot(theta,s.a_m*mult*np.exp(s.b*theta)*1000,label=label)
        ax.set_title('Logarithmic spiral · radius in mm\nθ = 0 at tip; θ = q at base');ax.legend(loc='lower left')
        save(fig,'polar','Spiral construction in polar coordinates')

        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for ax,key,label in [(axes[0],'stiffness','Stiffness (N·m/rad)'),(axes[1],'damping_per_dof','Damping (N·m·s/rad)')]:
            values=[r[key] if key=='stiffness' else r[key][0] for r in records]
            # Keep the protected base from flattening the rest of the graph.
            ax.plot(range(2,len(values)+1),values[1:],'o-',color='#246e73')
            ax.set_xlabel('Joint index (base → tip)');ax.set_ylabel(label)
            ax.set_title(f'Protected base value: {values[0]:.6g}\nBase omitted from plot; all values in joint_gains.json',fontsize=9)
        save(fig,'gains','Actual compiled joint gains; damping plot uses the first DOF of each joint')

        fig,ax=plt.subplots(figsize=(11,5),layout='constrained');triangles=[]
        from scipy.spatial.transform import Rotation
        camera=Rotation.from_euler('xyz',[35,15,-5],degrees=True).as_matrix()
        for (v,f),u in zip(surfaces,geometry.units):
            v=v.copy();v[:,2]+=u.local_frame_origin_m[2]*1000-zbase
            triangles.append((v[:,[2,0,1]]@camera.T)[f])
        tri=np.concatenate(triangles);tri=tri[np.argsort(tri[:,:,2].mean(axis=1))]
        normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-15)
        colors=np.array([.15,.48,.5])*(.5+.45*np.abs(normal[:,2]))[:,None]
        ax.add_collection(PolyCollection(tri[:,:,:2],facecolors=colors,edgecolor='none',rasterized=True));ax.autoscale();ax.set_aspect('equal');ax.set_axis_off()
        ax.set_title(f"{params['n_cables']} cables · {len(surfaces)} links · simulation surfaces in straight configuration")
        save(fig,'three','3D view of actual compiled simulation surfaces (fabrication core and holes not shown)')

    metrics={'Requested arc L (mm)':params['L']*1000,'Effective arc (mm)':geometry.lengths.effective_continuous_length_m*1000,
             'Assembled chord length (mm)':geometry.lengths.discrete_chord_length_m*1000,'Links':len(surfaces),
             'Partial base':geometry.lengths.has_partial_unit,'a (mm)':s.a_m*1000,'b':s.b,'E':s.E,
             'q requested (rad)':s.q0_requested_rad,'q effective (rad)':s.q0_rad,'Geometric beta':s.beta_nominal,
             'Core percent':core['elastic_core_percent'],'Core base (mm)':c[0],'Core tip (mm)':c[-1],
             'Compiled timestep (s)':float(model.opt.timestep),'Cables / actuators':int(model.nu),
             'CAD exported':params['build']['cad'],'CAD profile':params['build']['cad_profile'],'IGES exported':params['build']['iges']}
    if params['n_cables']==2 and not params['build']['plain']:
        from spirob.sections import section_dimensions
        dimensions=section_dimensions(params,geometry)
        metrics['Base centre thickness (mm)']=dimensions['base']['centre_thickness_m']*1000
        metrics['Tip centre thickness (mm)']=dimensions['tip']['centre_thickness_m']*1000
    from spirob.parameters import SCHEMA_PATH
    schema=json.loads(SCHEMA_PATH.read_text())
    selected=[]
    def fields(obj, spec, prefix=''):
        for key,value in obj.items():
            item=spec.get('properties',{}).get(key,{})
            if isinstance(value,dict): fields(value,item,prefix+key+'.')
            else: selected.append((prefix+key,item.get('title',key),json.dumps(value,ensure_ascii=False),item.get('x-unit','')))
    fields(params,schema)
    notes=[f'Completed build record. Created (UTC): {datetime.now(timezone.utc).isoformat()}.',f'Generator revision: {revision}.',
           'Parameters below are the resolved build inputs, including command-line overrides. params.json is byte-identical to build_params.json.',
           'r_inner(θ) = a exp(bθ); r_outer(θ) = a E exp(bθ); E = exp(2πb). θ increases from tip to base.',
           'Core c(z) = (p/100) Wref(z): linear with axial distance, geometric at complete-link stations. Two-cable Y follows the selected thickness law.',
           'K_i = K0 / beta_j^(3(i−1)); D_i = D0 / beta_j^(3(i−1)); protected base overrides apply. These are assumptions, not material calibration.',
           'Drawings use compiled simulation meshes; core dimensions are reference overlays. Fabrication channels/core are not simulation mass/contact geometry. CAD is in mm; simulation geometry is in metres.',
           'The report records the files at build time. Later edits with joint/sensor tools do not update this record; regenerate the build for a new record.']
    if params.get('thickness_profile') == 'constant':
        notes.append('Constant Y thickness breaks uniform geometric similarity. The existing cubic gain-decay assumption must be calibrated independently for this design.')
    manifest={p.relative_to(stage).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(stage.rglob('*')) if p.is_file() and not p.is_relative_to(out)}
    (out/'output_checksums.json').write_text(json.dumps(manifest,indent=2)+'\n')
    md='# SpiRob build report\n\n'+'\n\n'.join(notes)+f'\n\nParameters SHA-256: `{digest}`\n\n## Dimensions and derived constants\n\n| Quantity | Value |\n|---|---|\n'
    md+='\n'.join(f'| {k} | {v} |' for k,v in metrics.items())
    body='<h1>SpiRob build report</h1>'+''.join('<p>'+html.escape(n)+'</p>' for n in notes)+f'<p>Parameters SHA-256: <code>{digest}</code></p><table>'
    body+=''.join(f'<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>' for k,v in metrics.items())+'</table>'
    md+='\n\n## Selected parameters (native JSON units)\n\n| Parameter | Meaning | Value | Unit |\n|---|---|---|---|\n'
    md+='\n'.join('| '+' | '.join(str(x).replace('|','\\|').replace('\n',' ') for x in row)+' |' for row in selected)
    body+='<h2>Selected parameters (native JSON units)</h2><table><tr><th>Parameter</th><th>Meaning</th><th>Value</th><th>Unit</th></tr>'
    body+=''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in selected)+'</table>'
    for name,title in titles:
        md+=f'\n\n## {title}\n\n![{title}](figures/{name}.svg)'
        svg=(figs/(name+'.svg')).read_text();svg=svg[svg.index('<svg'):]
        body+=f'<section><h2>{html.escape(title)}</h2>{svg}</section>'
    md+='\n\n## Resolved parameters\n\n```json\n'+raw.decode()+'```\n\nSee output_checksums.json for SHA-256 hashes of generated files, and joint_gains.json for compiled gains.\n'
    body+='<h2>Resolved parameters</h2><pre>'+html.escape(raw.decode())+'</pre>'
    (out/'report.md').write_text(md,encoding='utf-8')
    (out/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>SpiRob build report</title><style>body{max-width:1000px;margin:32px auto;padding:0 20px;font:15px system-ui;color:#20343e}td,th{border:1px solid #ccc;padding:7px;text-align:left}svg{width:100%;height:auto}pre{white-space:pre-wrap}code{overflow-wrap:anywhere}section{break-inside:avoid}@media print{button{display:none}body{margin:0}}</style><button onclick="window.print()">Print / Save as PDF</button>'+body+'</html>',encoding='utf-8')
    with zipfile.ZipFile(stage/'design_report.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.rglob('*')):
            if p.is_file():archive.write(p,p.relative_to(out))
