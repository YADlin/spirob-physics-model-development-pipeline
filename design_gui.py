"""Desktop design tool for the canonical SpiRob pipeline (Tk + Matplotlib).

Feature inspiration: OpenSpiRobs; implemented with this repository's geometry.
Worker threads only enqueue messages. Tk widgets are accessed by the main thread.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parent
_FIELDS=[('L','Continuous length (m)'),('d_tip','Nominal tip width (m)'),
         ('phi_deg','Full taper angle (deg)'),('Delta_theta_deg','Segment angle (deg)'),
         ('n_cables','Cables'),('tendon_inward_shift','Cable inward shift (m)'),
         ('nlobe_t','n-lobe t'),('notch_factor','Notch factor'),
         ('flat_thickness_ratio','Flat thickness ratio')]


def collect_params(original, values):
    from spirob_csv_generator import validate_params
    from spirob.geometry import from_params
    p=dict(original)
    for key,raw in values.items():
        p[key]=int(raw) if key=='n_cables' else float(raw)
    validate_params(p); from_params(p)
    return p


def build_command(params_path, output_dir, *, cad=False, profile='fabrication',
                  neck_width_mm=1, cable_hole_diameter_mm=0):
    cmd=[sys.executable,str(ROOT/'build.py'),'--no-preview','--params',str(Path(params_path).resolve()),
         '--output-dir',str(Path(output_dir).resolve())]
    if cad:
        cmd += ['--cad','--cad-profile',profile,'--neck-width-mm',str(neck_width_mm),
                '--cable-hole-diameter-mm',str(cable_hole_diameter_mm)]
    return cmd


class DesignApp:
    def __init__(self, root, params_path, output_dir):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.root=root; self.params_path=Path(params_path).resolve()
        self.params=json.loads(self.params_path.read_text(encoding='utf-8'))
        self.messages=queue.Queue(); self.busy=False; self.process=None
        self.values={}; self.buttons=[]
        root.title('SpiRob Design Tool'); root.geometry('1240x850')
        controls=ttk.Frame(root,padding=10); controls.pack(side='left',fill='y')
        display=ttk.Frame(root,padding=10); display.pack(side='right',fill='both',expand=True)
        ttk.Label(controls,text='Geometry',font=('',13,'bold')).pack(anchor='w')
        form=ttk.Frame(controls); form.pack(fill='x')
        for i,(key,label) in enumerate(_FIELDS):
            ttk.Label(form,text=label).grid(row=i,column=0,sticky='w',pady=2)
            value=tk.StringVar(value=str(self.params.get(key,''))); self.values[key]=value
            ttk.Entry(form,textvariable=value,width=14).grid(row=i,column=1)
        self.output=tk.StringVar(value=str(Path(output_dir).resolve()))
        ttk.Label(controls,text='Output folder').pack(anchor='w',pady=(8,0))
        ttk.Entry(controls,textvariable=self.output,width=38).pack(fill='x')
        self.add_button(controls,'Choose output folder',self.choose_output)
        self.profile=tk.StringVar(value='fabrication')
        ttk.Label(controls,text='CAD profile').pack(anchor='w',pady=(8,0))
        ttk.Combobox(controls,textvariable=self.profile,values=['fabrication','simulation'],state='readonly').pack(fill='x')
        self.neck=tk.StringVar(value='1'); self.hole=tk.StringVar(value='0')
        self.count=tk.StringVar(value='6'); self.radius=tk.StringVar(value='0.12')
        extra=ttk.Frame(controls); extra.pack(fill='x',pady=6)
        for i,(label,var) in enumerate([('Flexure width (mm)',self.neck),('Cable hole diameter (mm; 0=off)',self.hole),('Array robot count',self.count),('Array radius (m)',self.radius)]):
            ttk.Label(extra,text=label).grid(row=i,column=0,sticky='w')
            ttk.Entry(extra,textvariable=var,width=10).grid(row=i,column=1)
        for label,fn in [('Update geometry preview',self.preview),('Save parameters',self.save),
                         ('Build simulation model',lambda:self.build(False)),
                         ('Build + export STEP/STL',lambda:self.build(True)),
                         ('Preview exported CAD',self.preview_cad),('Split fabrication file',self.split),
                         ('Generate robot array',self.array),('Open MuJoCo viewer',self.viewer)]:
            self.add_button(controls,label,fn)
        ttk.Label(controls,text='CAD/STL export: millimetres\nSimulation meshes: metres\nFabrication dimensions require calibration.',wraplength=310).pack(anchor='w',pady=8)
        self.figure=Figure(figsize=(7,5),dpi=100)
        self.canvas=FigureCanvasTkAgg(self.figure,master=display)
        self.canvas.get_tk_widget().pack(fill='both',expand=True)
        self.log=tk.Text(display,height=12,wrap='word'); self.log.pack(fill='x')
        root.after(100,self.poll)
        self.preview()

    def add_button(self,parent,label,fn):
        from tkinter import ttk
        b=ttk.Button(parent,text=label,command=fn); b.pack(fill='x',pady=2); self.buttons.append(b)

    def note(self,message):
        self.log.insert('end',message+'\n'); self.log.see('end')

    def collect(self): return collect_params(self.params,{k:v.get() for k,v in self.values.items()})

    def save(self):
        from tkinter import messagebox
        if self.busy: return False
        try:
            p=self.collect()
            self.params_path.write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8')
            self.params=p; self.note(f'Saved {self.params_path}')
            return True
        except Exception as e:
            messagebox.showerror('Invalid parameters',str(e)); return False

    def choose_output(self):
        from tkinter import filedialog
        path=filedialog.askdirectory(initialdir=self.output.get())
        if path:self.output.set(path)

    def preview(self):
        from spirob.geometry import from_params
        import preview as pv
        try:
            p=self.collect(); g=from_params(p)
            self.figure.clear(); ax=self.figure.add_subplot(121); sec=self.figure.add_subplot(122)
            for q in g.inverted_quads():
                for sign in (-1,1):
                    xs=[sign*pt[0]*1000 for pt in q]+[sign*q[0][0]*1000]
                    zs=[pt[1]*1000 for pt in q]+[q[0][1]*1000]
                    ax.plot(xs,zs,lw=.8,color='#007e99')
            ax.set_aspect('equal'); ax.set_title('Canonical segment profile'); ax.set_xlabel('x (mm)'); ax.set_ylabel('z (mm)')
            r=g.lengths.realized_root_width_m/2
            if p['n_cables']>=3: pv.draw_nlobe_section(sec,r,p['n_cables'],p,title='Simulation section')
            else: pv.draw_flat_section(sec,r,p,title='Simulation section',quad=g.inverted_quads()[0])
            self.figure.tight_layout(); self.canvas.draw_idle()
            lr=g.lengths
            self.note(f'{lr.n_units_total} elements; continuous {lr.requested_continuous_length_m*1000:.3f} mm; chord chain {lr.discrete_chord_length_m*1000:.3f} mm. Use exported CAD preview to inspect the fabrication lens/holes.')
        except Exception as e:self.note(f'Preview error: {e}')

    def run(self,cmd):
        if self.busy:return
        self.busy=True
        for b in self.buttons:b.configure(state='disabled')
        self.note('Running: '+' '.join(cmd))
        def worker():
            try:
                with subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                      text=True,encoding='utf-8',errors='replace',
                                      env=dict(os.environ,PYTHONIOENCODING='utf-8')) as proc:
                    self.process=proc
                    for line in proc.stdout:self.messages.put(('log',line.rstrip()))
                    code=proc.wait()
                    self.messages.put(('log',f'Completed, exit {code}'))
            except Exception as e:self.messages.put(('log',f'Failed: {e}'))
            finally:self.process=None; self.messages.put(('done',None))
        threading.Thread(target=worker,daemon=True).start()

    def poll(self):
        try:
            while True:
                kind,msg=self.messages.get_nowait()
                if kind=='log':self.note(msg)
                else:
                    self.busy=False
                    for b in self.buttons:b.configure(state='normal')
        except queue.Empty:pass
        self.root.after(100,self.poll)

    def build(self,cad):
        if not self.save():return
        self.run(build_command(self.params_path,self.output.get(),cad=cad,profile=self.profile.get(),
                               neck_width_mm=self.neck.get(),cable_hole_diameter_mm=self.hole.get()))

    def preview_cad(self):
        try:
            import trimesh
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            mesh=trimesh.load(Path(self.output.get())/'cad/spirob.stl',force='mesh')
            self.figure.clear(); ax=self.figure.add_subplot(111,projection='3d')
            ax.add_collection3d(Poly3DCollection(mesh.triangles,facecolor='#08a3b4',edgecolor='none',alpha=1))
            lo,hi=mesh.bounds
            ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1]); ax.set_zlim(lo[2],hi[2])
            ax.set_box_aspect(hi-lo); ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_zlabel('z (mm)')
            ax.set_title('Last exported CAD (drag to rotate)'); self.canvas.draw_idle()
        except Exception as e:self.note(f'CAD preview: {e}')

    def split(self):
        from tkinter import filedialog,simpledialog
        path=filedialog.askopenfilename(initialdir=str(Path(self.output.get())/'cad'),filetypes=[('CAD','*.step *.stp *.stl')])
        if not path:return
        span=simpledialog.askfloat('Split along Z','Maximum Z span per part (mm):',minvalue=.001,initialvalue=100)
        if span:self.run([sys.executable,str(ROOT/'fabrication/part_splitter.py'),path,'--max-span-mm',str(span),'--file-units','mm'])

    def array(self):
        self.run([sys.executable,str(ROOT/'tools/multi_array.py'),'--in',str(Path(self.output.get()).resolve()/'spirob_physics_model.xml'),
                  '--count',self.count.get(),'--radius-m',self.radius.get()])

    def viewer(self):
        self.run([sys.executable,'-m','mujoco.viewer','--mjcf='+str(Path(self.output.get()).resolve()/'spirob_physics_model.xml')])


def launch(params_path='params.json',output_dir='.'):
    import tkinter as tk
    root=tk.Tk(); DesignApp(root,params_path,output_dir); root.mainloop()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--params',default='params.json'); p.add_argument('--output-dir',default='.')
    a=p.parse_args(); launch(a.params,a.output_dir)
