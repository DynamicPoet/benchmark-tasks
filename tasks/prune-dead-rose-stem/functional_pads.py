"""Actual stock YAM collision faces used to observe a functional inner-pad contact."""
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull


class FunctionalPads:
    def __init__(self):
        from yam_arena import YAM_URDF
        from yam_arena.urdf import forward_kinematics,rpy_matrix
        source=Path(YAM_URDF);root=ET.parse(source).getroot()
        frames=forward_kinematics(source,{f'joint{i}':0. for i in range(1,9)})
        self.faces={}
        for name,sign in (('tip_left',1.),('tip_right',-1.)):
            link=root.find(f"link[@name='{name}']");pieces=[]
            axis=frames[name][:3,:3].T@frames['gripper'][:3,:3]@np.array([sign,0.,0.])
            for collision in link.findall('collision'):
                mesh=collision.find('geometry/mesh')
                if mesh is None:raise RuntimeError('The functional pad requires the actual YAM collision mesh')
                raw=(source.parent/mesh.get('filename')).read_bytes();n=struct.unpack_from('<I',raw,80)[0]
                points=np.frombuffer(raw,dtype=np.dtype([('normal','<3f4'),('vertices','<9f4'),('attr','<u2')]),count=n,offset=84)['vertices'].reshape(-1,3).astype(float)
                points*=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
                origin=collision.find('origin')
                if origin is not None:
                    points=points@rpy_matrix(np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).T+np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                equations=ConvexHull(points).equations
                inner=equations[:,:3]@axis>1e-6
                if not np.any(inner):raise RuntimeError('No finite inner pad surface resolved')
                pieces.append((equations,inner))
            self.faces[name]=pieces

    def contains_surface(self,name,local_point,allowance=.0005):
        """A real inward-facing finite convex surface, with the existing allowance."""
        p=np.asarray(local_point,float)
        for equations,inner in self.faces[name]:
            signed=equations[:,:3]@p+equations[:,3]
            if np.max(signed)<=allowance and np.min(np.abs(signed[inner]))<=allowance:
                return True
        return False
