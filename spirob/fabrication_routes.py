"""Straight manufacturing bores, distinct from calibrated MJCF routing sites."""
import numpy as np


def cable_hole_axes(geometry):
    z0=geometry.units[0].local_frame_origin_m[2]
    z1=geometry.units[-1].slit_reference_m[2]
    result=[]
    for path in geometry.tendon_paths:
        a=np.asarray(path.points[0].routed_m)
        b=np.asarray(path.points[-1].routed_m)
        slope=(b-a)/(b[2]-a[2])
        base=a+slope*(z0-a[2]);tip=a+slope*(z1-a[2])
        direction=slope/np.linalg.norm(slope)
        deviations=[float(np.linalg.norm(np.cross(np.asarray(p.routed_m)-base,direction))) for p in path.points]
        result.append(dict(cable_index=path.cable_index,base_m=base.tolist(),tip_m=tip.tolist(),
                           max_simulation_route_deviation_mm=max(deviations)*1000))
    return result


def cylinder_for_axis(axis, diameter_mm, z_origin_m=0., clearance_mm=None):
    import cadquery as cq
    a=np.asarray(axis['base_m'])*1000; b=np.asarray(axis['tip_m'])*1000
    a[2]-=z_origin_m*1000;b[2]-=z_origin_m*1000
    d=b-a;length=np.linalg.norm(d);d/=length
    extension=max(diameter_mm, .1) if clearance_mm is None else clearance_mm
    return cq.Solid.makeCylinder(diameter_mm/2,length+2*extension,
                                 cq.Vector(*(a-d*extension)),cq.Vector(*d))
