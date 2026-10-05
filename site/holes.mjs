// Manufacturing bores are straight cylinders. Simulation routes stay independent.
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=(a)=>a.map(v=>v/Math.hypot(...a));
export function boreAxes(g) {
  return g.units[0].cable.map((_,i)=>{
    const a=g.units[0].cable[i][0],b=g.units.at(-1).cable[i][1];
    const at=(z)=>a.map((v,j)=>v+(b[j]-v)*(z-a[2])/(b[2]-a[2]));
    return {base:at(g.origin),tip:at(g.origin+g.length)};
  });
}
export function boreEllipse(axis,radius,face,samples=48) {
  const raw=cross(sub(face[1],face[0]),sub(face[2],face[0]));
  if(Math.hypot(...raw)<1e-20)return [];
  const n=unit(raw),d=unit(sub(axis.tip,axis.base)),nd=dot(n,d);
  // Parallel wall: a plane/cylinder section is not an ellipse.
  if(Math.abs(nd)<1e-8)return [];
  const t=dot(n,sub(face[0],axis.base))/nd;
  const centre=axis.base.map((v,i)=>v+t*d[i]);
  const tangent=d.map((v,i)=>v-nd*n[i]);
  const e1=unit(Math.hypot(...tangent)<1e-8?sub(face[1],face[0]):tangent),e2=cross(n,e1);
  return Array.from({length:samples},(_,j)=>{
    const angle=2*Math.PI*j/samples;
    return centre.map((v,i)=>v+radius*(e1[i]*Math.cos(angle)/Math.abs(nd)+e2[i]*Math.sin(angle)));
  });
}
