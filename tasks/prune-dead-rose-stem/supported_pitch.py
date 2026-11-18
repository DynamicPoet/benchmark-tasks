"""Supported hanging pitch and finite open-tool placement, using ordinary geometry."""
import numpy as np


def supported_pose(planned_position,planned_rotation,material_point,observed_rotation):
    """Keep observed rotation about the closing axis; place the same material section."""
    P,R,C,A=map(np.asarray,(planned_position,planned_rotation,material_point,observed_rotation))
    relative=R.T@A
    angle=float(np.arctan2(relative[2,1]-relative[1,2],relative[1,1]+relative[2,2]))
    c,s=np.cos(angle),np.sin(angle)
    supported=R@np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
    section=R.T@(C-P)
    position=C-supported@section
    residual=float(np.arccos(np.clip((np.trace(supported.T@A)-1)/2,-1.,1.)))
    return position,supported,angle,residual


def segment_convex_gap(v, h, a, b, r):
    """Capsule-axis distance to a solid; None denotes >3mm AABB separation."""
    if np.any(v.max(0) < np.minimum(a, b) - r - 0.003) or np.any(v.min(0) > np.maximum(a, b) + r + 0.003):
        return None
    eq = h.equations
    d = b - a
    n = eq[:, :3]
    s0 = n @ a + eq[:, 3]
    sd = n @ d
    lo = max(0.0, np.max(-s0[sd < -1e-15] / sd[sd < -1e-15], initial=-np.inf))
    hi = min(1.0, np.min(-s0[sd > 1e-15] / sd[sd > 1e-15], initial=np.inf))
    if lo <= hi and (not np.any(s0[np.abs(sd) <= 1e-15] > 1e-12)):
        return -r
    tri = v[h.simplices]
    p = tri[:, 0]
    e = tri[:, 1] - p
    f = tri[:, 2] - p
    ee = np.sum(e * e, 1)
    ff = np.sum(f * f, 1)
    ef = np.sum(e * f, 1)
    den = ee * ff - ef * ef
    best = np.inf
    for x in (a, b):
        w = x - p
        we = np.sum(w * e, 1)
        wf = np.sum(w * f, 1)
        u = np.divide(we * ff - wf * ef, den, out=np.zeros_like(den), where=den > 1e-26)
        z = np.divide(wf * ee - we * ef, den, out=np.zeros_like(den), where=den > 1e-26)
        inside = (u >= -1e-10) & (z >= -1e-10) & (u + z <= 1 + 1e-10)
        inside &= den > 1e-26
        dist = np.sum((w - u[:, None] * e - z[:, None] * f) ** 2, 1)
        best = min(best, np.min(dist[inside], initial=np.inf))
    c = tri.reshape(-1, 3)
    q = np.roll(tri, -1, axis=1).reshape(-1, 3)
    ed = q - c
    dd = d @ d
    edge2 = np.sum(ed * ed, 1)
    w = a - c
    de = ed @ d
    dw = w @ d
    ew = np.sum(ed * w, 1)
    for t in (0.0, 1.0):
        uu = np.clip((ew + t * de) / edge2, 0, 1)
        delta = w + t * d - uu[:, None] * ed
        best = min(best, float(np.min(np.sum(delta * delta, 1))))
    for u in (0.0, 1.0):
        tt = np.clip((u * de - dw) / dd, 0, 1)
        delta = w + tt[:, None] * d - u * ed
        best = min(best, float(np.min(np.sum(delta * delta, 1))))
    det = dd * edge2 - de * de
    ok = det > 1e-24
    tt = np.zeros(len(c))
    uu = tt.copy()
    tt[ok] = (de[ok] * ew[ok] - edge2[ok] * dw[ok]) / det[ok]
    uu[ok] = (dd * ew[ok] - de[ok] * dw[ok]) / det[ok]
    ok &= (tt >= 0) & (tt <= 1) & (uu >= 0) & (uu <= 1)
    delta = w + tt[:, None] * d - uu[:, None] * ed
    best = min(best, np.min(np.sum(delta[ok] * delta[ok], 1), initial=np.inf))
    return float(np.sqrt(best) - r)


class OpenThroat:
    """Authored finite head/neck solids at the observed material pose."""
    def __init__(self,sa):
        from scipy.spatial import ConvexHull
        self.sa=sa;self.parts=[]
        for side in(-1,1):
            parts,boxes=sa._closed_half(side)
            for name,mesh,_ in parts:
                pieces=sa.convex_jaw_sections(mesh,side)if name in('blade','hook')else[]
                for k,piece in enumerate(pieces):
                    self.parts.append((side,f'{name}_{k}',piece.v,ConvexHull(piece.v)))
            if side==-1:
                for name,piece in sa.pivot_collision_sections(parts):
                    self.parts.append((side,name,piece.v,ConvexHull(piece.v)))
            for name,size,centre in boxes:
                if name.startswith('grip'):continue
                points=sa.box(size,center=centre).v
                self.parts.append((side,name,points,ConvexHull(points)))

    def __call__(self,state,position,rotation):
        pose=state.get('plant_body_poses',{}).get('stub_col')
        if pose is None:raise ValueError('Missing actual material collider pose')
        a=np.asarray(pose['position_m']);body_R=np.asarray(pose['rotation'])
        b=a+body_R@np.array([0.,0.,self.sa.L_STUB_COL])
        P,R=np.asarray(position),np.asarray(rotation)
        if not all(np.isfinite(v).all()for v in(a,b,P,R)):
            raise ValueError('Nonfinite finite-throat geometry')
        aa,bb=R.T@(a-P),R.T@(b-P)
        if 'jaw'not in state or 'fuse'not in state:
            raise ValueError('Missing actual tool opening or blade gauge')
        q=float(state['jaw']);fuse=float(state['fuse'])
        if not np.isfinite(q+fuse):raise ValueError('Nonfinite tool opening')
        minimum=.003;nearest='outside head bounds'
        for side,name,points,hull in self.parts:
            angle=-side*(self.sa.HALF-q/2)+(fuse if side==1 and name.startswith('blade')else 0.)
            c,s=np.cos(angle),np.sin(angle);turn=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
            d=segment_convex_gap(points,hull,turn.T@aa,turn.T@bb,self.sa.R0)
            if d is not None and d<minimum:minimum,nearest=float(d),name
        return {'gap_lower_bound_m':minimum,'part':nearest}
