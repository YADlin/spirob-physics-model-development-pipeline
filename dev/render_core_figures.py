"""Render measured fabrication sections and coordinate-dependent taper laws.

Run after building build/core-two, build/core-three and build/core-four with CAD.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import trimesh
from spirob.geometry import from_params
from spirob.core import core_dimensions, reference_width_at

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/figures'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})


def main():
    fig=plt.figure(figsize=(14,12))
    fig.subplots_adjust(left=.07,right=.97,bottom=.08,top=.91,wspace=.28,hspace=.5)
    for row,(n,name) in enumerate([(2,'two'),(3,'three'),(4,'four')]):
        folder=ROOT/f'build/core-{name}'
        mesh=trimesh.load(folder/'cad/spirob.stl',force='mesh')
        p=json.loads((folder/'build_params.json').read_text());g=from_params(p)
        report=core_dimensions(g,p['build']['elastic_core_percent']);law=report['width_law']
        from matplotlib.collections import PolyCollection
        from matplotlib.patches import Polygon
        ax=fig.add_subplot(3,2,2*row+1)
        ax.add_collection(PolyCollection(mesh.triangles[:,:,[2,0]],facecolors='#afcdd0',edgecolors='none'))
        length=mesh.bounds[1,2]-mesh.bounds[0,2]
        base,tip=report['stations'][0]['core_width_mm'],report['stations'][-1]['core_width_mm']
        ax.add_patch(Polygon([[0,-base/2],[length,-tip/2],[length,tip/2],[0,base/2]],color='#d79b3f'))
        ax.set_xlim(-3,length+3);ax.set_ylim(-18,18);ax.set_xlabel('Z from base (mm)');ax.set_ylabel('X (mm)')
        ax.set_title(f'{n} cables · XZ projection of exported fabrication STL\nGold overlay: continuous tapered core width')
        ax.text(3,16,f'Base c = {base:.4f} mm',fontsize=9,va='top')
        ax.text(length-3,-16,f'Tip c = {tip:.4f} mm',fontsize=9,ha='right',va='bottom')
        ax.grid(alpha=.15)
        ax=fig.add_subplot(3,2,2*row+2)
        for i,color in zip([1,10,20],['#a66410','#00888b','#3e53ac']):
            station=report['stations'][i];z=station['z_from_base_mm']
            lines=trimesh.intersections.mesh_plane(mesh,[0,0,1],[0,0,z])
            for segment in lines:ax.plot(segment[:,0],segment[:,1],color=color,lw=1.2)
            ax.plot([],[],color=color,label=f'Joint {i+1:02d}: z={z:.1f} mm, c={station["core_width_mm"]:.3f} mm')
        ax.set_aspect('auto' if n==2 else 'equal');ax.set_xlabel('X (mm; horizontally expanded)' if n==2 else 'X (mm)');ax.set_ylabel('Y (mm)');ax.legend(loc='upper right',fontsize=7)
        ax.set_title('Actual STL sections at joint planes\nFinite connection = tapered core')
        ax.grid(alpha=.2)
    fig.suptitle('Fabrication core tapers at 5% of local reference width',fontsize=18)
    fig.supxlabel('Standard examples: base 1.5544 mm → tip 0.3462 mm. STEP, STL and IGES share the same fused CAD solid.',fontsize=11)
    fig.savefig(OUT/'elastic-core-cad.png',dpi=170);plt.close(fig)
    p=json.loads((ROOT/'examples/params-three-cable.json').read_text());g=from_params(p);r=core_dimensions(g);law=r['width_law']
    z=np.linspace(law['z_base_m'],law['z_tip_m'],400);x=(z-law['z_base_m'])*1000
    widths=np.array([reference_width_at(law,v) for v in z])*50
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    axes[0,0].plot(x,widths,color='#00888b',lw=2);axes[0,0].scatter([s['z_from_base_mm'] for s in r['stations']],[s['core_width_mm'] for s in r['stations']],s=12,color='#ba7d24')
    axes[0,0].set(title='Core diameter vs axial distance: LINEAR',xlabel='Distance from base (mm)',ylabel='Core diameter (mm)')
    complete=[u for u in g.units if not u.is_partial];i=np.arange(len(complete));w=np.array([reference_width_at(law,u.local_frame_origin_m[2])*50 for u in complete])
    axes[0,1].plot(i,w,'o-',color='#00888b',ms=4);axes[0,1].set(title='Complete-link stations: EXPONENTIAL',xlabel='Complete-link index (0 = first complete link)',ylabel='Core diameter (mm)')
    axes[1,0].plot(x,(widths/widths[0])**2,color='#946622',lw=2);axes[1,0].set(title='Circular core area vs axial distance: QUADRATIC',xlabel='Distance from base (mm)',ylabel='Area / base area')
    j=np.arange(2,len(g.units)+1)
    for beta,label,color in [(p['post_gen']['joint_beta'],'Retained βj = 1.03','#3e53ac'),(g.spiral.beta_nominal,f'Geometric βg = {g.spiral.beta_nominal:.5f}','#00888b')]:
        axes[1,1].plot(j,beta**(-3*(j-2)),label=label,color=color,lw=2)
    axes[1,1].set(title='Joint gain vs link number: EXPONENTIAL',xlabel='Joint number (protected j_001 omitted)',ylabel='K / K₂ or D / D₂');axes[1,1].legend(fontsize=9)
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Linear in distance does not mean linear in link index',fontsize=17)
    fig.supxlabel('Unequal link lengths reconcile both descriptions. Cubic stiffness scaling assumes similar sections and constant material properties; damping requires an additional model.',fontsize=9)
    fig.savefig(OUT/'elastic-core-laws.png',dpi=170);plt.close(fig)

if __name__=='__main__':main()
