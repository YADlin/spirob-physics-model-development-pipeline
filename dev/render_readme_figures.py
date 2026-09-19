"""Regenerate README figures from actual meshes and the tested geometry equations.

Run from the repository root: uv run python dev/render_readme_figures.py
Missing example models are built under build/documentation. Requires Node for
sampling the same section functions used by the browser; no browser is needed.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.collections import PolyCollection
from scipy.spatial.transform import Rotation
import mujoco
import numpy as np

from spirob.geometry import from_params
from spirob.sections import section_dimensions
from tools.inspect_section import compiled_surface

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/figures'
COLORS=['#146d70','#db8645','#626cad']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'#ffffff',
                     'axes.labelcolor':'#364b54','text.color':'#20343e','savefig.facecolor':'white'})


def samples(params):
    script="""import {derive,section} from './site/geometry.mjs';import fs from 'node:fs';
console.log(JSON.stringify(JSON.parse(fs.readFileSync(0,'utf8')).map(p=>{const g=derive(p);return {g,s:section(p,g,g.units[1],.5,256)};})));"""
    return json.loads(subprocess.run(['node','--input-type=module','-e',script],cwd=ROOT,
        input=json.dumps(params),capture_output=True,text=True,check=True).stdout)


def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/name,dpi=170,bbox_inches='tight');plt.close(fig)
    print(OUT/name)


def figure_models(models,params,values):
    fig=plt.figure(figsize=(14,9),layout='constrained')
    grid=fig.add_gridspec(3,2,width_ratios=[4,1])
    for row,(folder,p,value,color) in enumerate(zip(models,params,values,COLORS)):
        ax=fig.add_subplot(grid[row,0])
        m=mujoco.MjModel.from_xml_path(str(folder/'spirob_physics_model.xml'))
        g=from_params(p);triangles=[]
        camera=Rotation.from_euler('xyz',[35,15,-5],degrees=True).as_matrix()
        for u in g.units:
            v,f=compiled_surface(m,u.link_name)
            v+=np.asarray(u.local_frame_origin_m)*1000
            v[:,2]-=g.units[0].local_frame_origin_m[2]*1000
            triangles.append((v[:,[2,0,1]]@camera.T)[f])
        triangles=np.concatenate(triangles)
        order=np.argsort(triangles[:,:,2].mean(axis=1));triangles=triangles[order]
        normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-15)
        light=np.array([-.2,.7,1.]);light/=np.linalg.norm(light)
        shade=.45+.5*np.maximum(0,normal@light)
        facecolors=np.clip(np.asarray(matplotlib.colors.to_rgb(color))*shade[:,None],0,1)
        ax.add_collection(PolyCollection(triangles[:,:,:2],facecolors=facecolors,edgecolors='none',antialiased=False))
        ax.autoscale();ax.set_aspect('equal');ax.margins(x=.03,y=.2);ax.set_axis_off()
        length=g.lengths.discrete_chord_length_m*1000
        label='2 cables · hex, linear thickness' if row==0 else f'{p["n_cables"]} cables · notched lobes'
        ax.set_title(f'{label}     |     {len(g.units)} links, {length:.2f} mm assembled',loc='left',fontsize=12,pad=18)
        ax.text(.02,-.04,'BASE',transform=ax.transAxes,fontsize=9,color='#60747b')
        ax.text(.97,-.04,'TIP',transform=ax.transAxes,fontsize=9,color='#60747b',ha='right')
        cross=fig.add_subplot(grid[row,1]);xy=np.asarray(value['s']['points'])*1000
        cross.add_patch(Polygon(xy,fc=color,ec='white',lw=.5))
        for c in value['s']['cables']:cross.plot(*np.asarray(c)*1000,'o',color='#e7aa43',ms=5)
        cross.set_aspect('equal');cross.autoscale();cross.margins(.2)
        cross.set_xlabel('X (mm)');cross.set_ylabel('Y (mm)');cross.set_title('Link 002 · mid-slice',fontsize=10)
    fig.suptitle('Generated simulation surfaces · 2, 3 and 4 cables',fontsize=19)
    fig.supxlabel('Actual compiled mesh surfaces in the straight, local robot frame. Orange points are routing sites projected to the slice.',fontsize=10)
    save(fig,'spirob-2-3-4.png')


def dimension(ax,a,b,text,color='#3b5968',offset=(0,0)):
    ax.annotate('',b,a,arrowprops=dict(arrowstyle='<->',color=color,lw=1))
    mid=(np.asarray(a)+b)/2+offset
    ax.text(*mid,text,ha='center',va='center',fontsize=9,color=color,
            bbox=dict(facecolor='white',edgecolor='none',pad=2,alpha=.9))


def figure_geometry(params,values):
    p=params[0];g=from_params(p);v=values[0]['g'];d=section_dimensions(p,g)
    fig=plt.figure(figsize=(14,7.4),layout='constrained');grid=fig.add_gridspec(1,3,width_ratios=[1,1,1.6])
    front=fig.add_subplot(grid[0,0]);side=fig.add_subplot(grid[0,1]);polar=fig.add_subplot(grid[0,2],projection='polar')
    zbase=g.units[0].local_frame_origin_m[2]*1000
    for i,quad in enumerate(g.inverted_quads()):
        q=np.asarray(quad)*1000;q[:,1]-=zbase
        pts=q[[0,3,2,1]]
        pts=np.concatenate([pts,pts[-2:0:-1]*[-1,1]])
        front.add_patch(Polygon(pts,fc='#b6d7d4',ec='#146d70',lw=.65))
        t=d['links'][i];z0=t['z_start_m']*1000-zbase;z1=t['z_end_m']*1000-zbase
        h0=t['proximal_centre_thickness_m']*500;h1=t['distal_centre_thickness_m']*500
        side.add_patch(Polygon([[-h0,z0],[h0,z0],[h1,z1],[-h1,z1]],fc='#dfe5f5',ec='#626cad',lw=.65))
    length=v['length']*1000;w=v['width']*1000;thick=v['thickness']*1000
    front.autoscale();front.set_xlim(-w*.95,w*.95);front.set_ylim(-15,length+15)
    dimension(front,(-w*.8,0),(-w*.8,length),f'Assembled\n{length:.2f} mm',offset=(-2,0))
    dimension(front,(-w/2,-8),(w/2,-8),f'Base width\n{w:.2f} mm')
    front.text(0,length+8,f'Requested L = {p["L"]*1000:g} mm',ha='center',fontsize=9)
    front.set_title('Front profile · XZ');front.set_xlabel('X (mm)');front.set_ylabel('Axial position from base (mm)');front.set_aspect('equal')
    side.set_xlim(-thick*.8,thick*.8);side.set_ylim(-15,length+15)
    dimension(side,(-thick/2,-8),(thick/2,-8),f'Tbase = {thick:.2f} mm')
    side.text(0,length+8,f'Ttip = {v["tipThickness"]*1000:.2f} mm',ha='center',fontsize=9)
    side.set_title('Thickness profile · YZ');side.set_xlabel('Y (mm)');side.set_aspect('equal')
    side.text(.5,.48,'One linear law\nthrough every link',transform=side.transAxes,ha='center',fontsize=9,bbox=dict(fc='white',ec='none',alpha=.85))
    theta=np.linspace(0,v['q'],900)
    for label,r,color in [('inner',v['a']*np.exp(v['b']*theta),'#626cad'),('centre',v['rc0']*np.exp(v['b']*theta),'#146d70'),('outer',v['a']*v['E']*np.exp(v['b']*theta),'#db8645')]:
        polar.plot(theta,r*1000,color=color,label=label,lw=1.8)
    for quad in v['curled']:
        q=np.asarray(quad);angles=np.unwrap(np.arctan2(q[:,1],q[:,0]));r=np.linalg.norm(q,axis=1)*1000
        polar.plot(np.r_[angles,angles[0]],np.r_[r,r[0]],color='#56686c',alpha=.4,lw=.65)
    radius=max(v['a']*v['E']*np.exp(v['b']*theta))*1000
    polar.plot(np.linspace(0,v['dth'],50),np.full(50,radius*.94),color='#20343e',lw=2)
    polar.text(v['dth']/2,radius*1.10,f'Δθ = {p["Delta_theta_deg"]}°',ha='center',fontsize=9)
    polar.set_title(f'Curled construction · polar radius in mm\nIncluded taper φ = {p["phi_deg"]}°',pad=15,fontsize=11)
    polar.legend(loc='lower center',bbox_to_anchor=(.5,-.2),ncol=3,frameon=False)
    fig.suptitle('Length, taper and segmentation are different controls',fontsize=18,y=1.04)
    fig.supxlabel('L is the continuous centre-spiral arc, not the straight chord sum. Tip width and thickness are derived after segmentation.',fontsize=10)
    save(fig,'geometry-parameters.png')


def figure_sections(params):
    cases=[]
    for edge in [.35,.75,.95]:
        p=copy.deepcopy(params[0]);p['hex_edge_ratio']=edge;cases.append(p)
    for notch in [0,.2,.4]:
        p=copy.deepcopy(params[1]);p['notch_factor']=notch;cases.append(p)
    for fill in [0,.5,1]:
        p=copy.deepcopy(params[2]);p['nlobe_t']=fill;cases.append(p)
    values=samples(cases)
    fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
    for i,(p,value,ax) in enumerate(zip(cases,values,axes.flat)):
        s=value['s'];xy=np.asarray(s['points'])*1000
        ax.add_patch(Polygon(xy,fc=['#c4dedb','#f2d8bd','#d3d9f0'][i//3],ec=COLORS[i//3],lw=2))
        if s['polygon']:
            poly=np.asarray(s['polygon'])*1000;ax.plot(*np.vstack([poly,poly[0]]).T,'--',color='#9aa7ac',lw=.8)
        for c in s['cables']:ax.plot(*np.asarray(c)*1000,'o',color='#a66023',ms=4)
        ax.set_aspect('equal');ax.autoscale();ax.margins(.1);ax.set_xlabel('X (mm)');ax.set_ylabel('Y (mm)')
        title=f'2 cables · edge / centre = {p["hex_edge_ratio"]}' if i<3 else f'3 cables · notch ratio = {p["notch_factor"]}' if i<6 else f'4 cables · polygon fill t = {p["nlobe_t"]}'
        ax.set_title(title,fontsize=11)
    for row in axes:
        bounds=[max(abs(v) for v in (*ax.get_xlim(),*ax.get_ylim())) for ax in row]
        for ax in row:ax.set(xlim=(-max(bounds),max(bounds)),ylim=(-max(bounds),max(bounds)))
    fig.suptitle('How cross-section controls change the same link',fontsize=18)
    fig.supxlabel('Each row varies one input only. Slices are at link 002 midpoint; dimensions in mm. Dashed outlines show the polygon cutter.',fontsize=10)
    save(fig,'section-parameters.png')


def figure_fabrication(params):
    cases=[]
    for source in params[:2]:
        p=copy.deepcopy(source);p['build'].update(neck_width_mm=2,cable_hole_diameter_mm=.8,cad=True);cases.append(p)
    values=samples(cases)
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
    for p,value,ax in zip(cases,values,axes):
        s=value['s'];xy=np.asarray(s['points'])*1000
        ax.add_patch(Polygon(xy,fc='#dcebea',ec='#146d70',lw=1.6))
        extent=max(abs(xy.ravel()));neck=p['build']['neck_width_mm']
        if p['n_cables']==2:
            ax.add_patch(Rectangle((-neck/2,-s['T']*500),neck,s['T']*1000,fc='#d4ae6b',alpha=.8))
            text='Elastic core X width = 2 mm'
        else:
            ax.add_patch(Circle((0,0),neck/2,fc='#d4ae6b',alpha=.8));text='Elastic core diameter = 2 mm'
        ax.annotate(text,(0,0),(0,extent*1.25),ha='center',arrowprops=dict(arrowstyle='->',color='#936d38'),fontsize=10)
        for c in s['cables']:ax.add_patch(Circle(np.asarray(c)*1000,.4,fc='white',ec='#a66023',lw=1.5))
        c=np.asarray(s['cables'][0])*1000
        ax.annotate('Cable hole Ø 0.8 mm',c,(-extent, -extent*1.25),ha='left',fontsize=10,arrowprops=dict(arrowstyle='->',color='#a66023'))
        ax.set_aspect('equal');ax.set_xlim(-extent*1.35,extent*1.35);ax.set_ylim(-extent*1.55,extent*1.5)
        ax.set_xlabel('X (mm)');ax.set_ylabel('Y (mm)');ax.set_title(f'{p["n_cables"]} cables · fabrication overlay')
    fig.suptitle('The available elastic dimension is the central core',fontsize=18)
    fig.supxlabel('Mid-slice illustration. The core joins links in fabrication CAD; holes follow the routed path. These additions do not recalibrate simulation inertia.',fontsize=10)
    save(fig,'fabrication-parameters.png')


def figure_gains():
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');i=np.arange(2,22)
    for ax,base,label,unit in zip(axes,[.2,.01],['Stiffness K','Damping D'],['N·m/rad','N·m·s/rad']):
        for beta,color in zip([1,1.03,1.08],COLORS):
            ax.plot(i,base/beta**(3*(i-1)),color=color,marker='o',ms=3,label=f'βⱼ = {beta:g}')
        ax.set(xlabel='Joint number i · base → tip',ylabel=f'{label} ({unit})',xticks=[2,5,10,15,21])
        ax.grid(alpha=.18);ax.legend(frameon=False);ax.set_title(f'{label}₀ = {base:g} {unit}')
    fig.suptitle('Assumed gain decay · coefficient / βⱼ³⁽ⁱ⁻¹⁾',fontsize=17)
    fig.supxlabel('The example overrides j_001 to K = 100 and D = 50; it is excluded from these plots. Gains are assumptions, not material identification.',fontsize=10)
    save(fig,'joint-gain-law.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs=3,type=Path,help='Existing two-hex, three- and four-cable output directories')
    args=parser.parse_args()
    files=['params-two-cable-hex.json','params-three-cable.json','params-four-cable.json']
    params=[json.loads((ROOT/'examples'/name).read_text()) for name in files]
    models=args.models or [ROOT/'build/documentation'/str(n) for n in [2,3,4]]
    for p,name,folder in zip(params,files,models):
        if not (folder/'spirob_physics_model.xml').exists():
            subprocess.run([sys.executable,str(ROOT/'build.py'),'--params',str(ROOT/'examples'/name),
                            '--no-preview','--output-dir',str(folder)],cwd=ROOT,check=True)
    values=samples(params)
    figure_models(models,params,values);figure_geometry(params,values)
    figure_sections(params);figure_fabrication(params);figure_gains()

if __name__=='__main__':main()
