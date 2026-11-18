"""Read-only proprioception and current native contacts for the scripted demo."""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from blade_contact import array


def rotation(q):
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def body_pose(asset,name,origin):
    i=asset.body_names.index(name)
    p=array(asset.data.body_link_pose_w)[0,i]
    return p[:3]-origin,rotation(p[3:7])


class MotionReader:
    """Copies current-step data; never sets a drive, state, force or material."""
    capacity=4096

    def __init__(self,base,sa):
        self.base,self.sa=base,sa
        from functional_pads import FunctionalPads
        self.pads=FunctionalPads()
        self.dt=float(base.cfg.sim.dt)
        self.robot_limits={};self.finger_limits={}
        self.sensors=[]
        for side in ('left','right'):
            robot=base.scene[f'{side}_robot'];view=robot.root_physx_view
            self.sensors.extend(view.link_paths[0])
            ids=[robot.joint_names.index(f'joint{i}')for i in range(1,7)]
            self.robot_limits[side]={name:array(getattr(view,method)())[0,ids].tolist()
                for name,method in (('limits','get_dof_limits'),('K','get_dof_stiffnesses'),
                  ('D','get_dof_dampings'),('effort','get_dof_max_forces'),('speed','get_dof_max_velocities'))}
            fingers=[robot.joint_names.index(n)for n in('joint7','joint8')]
            self.finger_limits[side]=array(view.get_dof_limits())[0,fingers].tolist()
        self.filters=[]
        for name in ('bush','secateurs','bin'):
            asset=base.scene[name]
            view=getattr(asset,'root_physx_view',None)
            if view is not None and hasattr(view,'link_paths'):
                self.filters.extend(view.link_paths[0])
            else:
                self.filters.extend(asset.root_physx_view.prim_paths)
        # The table is a static collider. Retain exact resolved collision paths,
        # rather than assuming that it has a rigid-body tensor view.
        import omni.usd
        from pxr import UsdPhysics
        stage=omni.usd.get_context().get_stage()
        self.table_paths=[str(p.GetPath()) for p in stage.Traverse()
                          if 'table' in str(p.GetPath()).lower().split('/')
                          and p.HasAPI(UsdPhysics.CollisionAPI)]
        self.filters.extend(self.table_paths)
        self.filters=list(dict.fromkeys(self.filters))
        sim=base.scene['secateurs']._physics_sim_view
        self.view=sim.create_rigid_contact_view(self.sensors,
            filter_patterns=[self.filters]*len(self.sensors),max_contact_data_count=self.capacity)
        if self.view.sensor_count!=len(self.sensors) or self.view.filter_count!=len(self.filters):
            raise RuntimeError('Demo motion contact inventory did not resolve')
        self.tool_sensors=list(base.scene['secateurs'].root_physx_view.link_paths[0])
        self.plant_filters=list(base.scene['bush'].root_physx_view.link_paths[0])
        self.tool_plant_view=sim.create_rigid_contact_view(self.tool_sensors,
            filter_patterns=[self.plant_filters]*len(self.tool_sensors),max_contact_data_count=self.capacity)
        if (self.tool_plant_view.sensor_count!=len(self.tool_sensors)
                or self.tool_plant_view.filter_count!=len(self.plant_filters)):
            raise RuntimeError('Complete tool/plant diagnostic inventory did not resolve')
        self.grips={}
        for side,name,angle in ((-1,'hook_half',sa.HALF),(1,'blade_half',-sa.HALF)):
            parts,_=sa._closed_half(side)
            mesh=next(m for n,m,_ in parts if n=='grip')
            c,s=np.cos(angle),np.sin(angle)
            R=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
            self.grips[name]=ConvexHull(mesh.v@R.T).equations

    def contacts(self,origin,view=None,sensors=None,filters=None):
        view=self.view if view is None else view
        sensors=self.sensors if sensors is None else sensors
        filters=self.filters if filters is None else filters
        force,point,normal,separation,count,start=map(array,view.get_contact_data(self.dt))
        matrix=array(view.get_contact_force_matrix(self.dt))
        net=array(view.get_net_contact_forces(self.dt))
        if int(count.sum())>=self.capacity or not np.isfinite(matrix).all():
            raise RuntimeError('Motion contact data is saturated or nonfinite')
        rows=[]
        for i,j in np.argwhere(count>0):
            a=int(start[i,j]);b=a+int(count[i,j])
            vectors=force[a:b].reshape(-1,1)*normal[a:b]
            if np.linalg.norm(vectors.sum(0)-matrix[i,j])>max(1e-6,1e-4*np.linalg.norm(matrix[i,j])):
                raise RuntimeError('Motion points disagree with current-step pair force')
            for k in range(a,b):
                rows.append({'sensor':sensors[i],'other':filters[j],
                    'point':(point[k]-origin).tolist(),
                    'force':(float(force[k,0])*normal[k]).tolist(),
                    'signed_normal_N':float(force[k,0]),'normal':normal[k].tolist(),
                    'separation':float(separation[k,0])})
        friction=None
        try:
            f,p,n,s=map(array,view.get_friction_data(self.dt))
            if int(n.sum())>=self.capacity:
                raise RuntimeError('Motion friction buffer is saturated')
            friction=[]
            for i,j in np.argwhere(n>0):
                for k in range(int(s[i,j]),int(s[i,j])+int(n[i,j])):
                    friction.append({'sensor':sensors[i],'other':filters[j],
                        'point':(p[k]-origin).tolist(),'force':f[k].tolist()})
        except Exception as exc:
            friction={'unknown':str(exc)}
        return rows,{'get_net_contact_forces_N':net.tolist(),
            'selected_normal_pair_matrix_N':matrix.tolist(),'friction_points':friction,
            'sensor_paths':sensors,'filter_paths':filters,'table_paths':self.table_paths,
            # Preserve this diagnostic difference without assigning it a
            # collider or treating it as a complete external-force balance.
            'net_minus_selected_normal_N':(net-matrix.sum(1)).tolist()}

    def sample(self,origin,target_a,target_b,handles,severed):
        base=self.base;tool=base.scene['secateurs']
        state={'robots':{},'target_a':np.asarray(target_a).tolist(),
            'target_b':np.asarray(target_b).tolist(),'severed':bool(severed)}
        for side in ('left','right'):
            robot=base.scene[f'{side}_robot'];names=list(robot.joint_names)
            q=array(robot.data.joint_pos)[0];qd=array(robot.data.joint_vel)[0]
            ids=[names.index(f'joint{i}') for i in range(1,7)]
            p,R=body_pose(robot,'gripper',origin)
            from yam_arena.constants import TCP_OFFSET
            p=p+R@np.asarray(TCP_OFFSET)
            item={'q':q[ids].tolist(),'qd':qd[ids].tolist(),'TCP':p.tolist(),'R':R.tolist(),
                'fingers':[float(q[names.index(n)])for n in('joint7','joint8')],
                'joint_names':names,'body_names':list(robot.body_names),**self.robot_limits[side]}
            item['finger_body_poses']={}
            for name in ('tip_left','tip_right'):
                fp,fR=body_pose(robot,name,origin)
                item['finger_body_poses'][name]={'position_m':fp.tolist(),'rotation':fR.tolist()}
            item['finger_limits']=self.finger_limits[side]
            view=robot.root_physx_view
            for method in ('get_dof_position_targets','get_dof_actuation_forces',
                           'get_dof_projected_joint_forces','get_link_incoming_joint_force'):
                try:item[method]=array(getattr(view,method)()).tolist()
                except Exception as exc:item[method]={'unknown':str(exc)}
            gravity=array(robot.data.gravity_compensation_forces)[0]
            if len(gravity)==len(names)+6:gravity=gravity[6:]
            item['gravity_Nm']=gravity[ids].tolist()
            state['robots'][side]=item
        po,ro=body_pose(tool,'hook_half',origin)
        q=array(tool.data.joint_pos)[0,tool.joint_names.index('pivot')]
        c,s=np.cos(q/2),np.sin(q/2)
        state.update(tool_o=po.tolist(),tool_R=(ro@np.array([[c,-s,0],[s,c,0],[0,0,1]])).tolist(),jaw=float(q),
                     fuse=float(array(tool.data.joint_pos)[0,tool.joint_names.index('fuse')]))
        good={};handle_bodies=set();handle_coordinates=[];handle_checks=[]
        hand=state['robots']['right'];hand_R=np.asarray(hand['R'])
        for row in handles:
            name=row['tool_body'].rsplit('/',1)[-1]
            if name not in self.grips:continue
            p,R=body_pose(tool,name,origin)
            local=R.T@(np.asarray(row['point_world_m'])-origin-p)
            eq=self.grips[name]
            finger=row['finger'].rsplit('/',1)[-1]
            sign=-1 if finger=='tip_left' else 1
            compression=sign*float(np.asarray(row['force_on_finger_N'])@hand_R[:,0])
            pad=hand_R.T@(np.asarray(row['point_world_m'])-origin-np.asarray(hand['TCP']))
            inner=hand['fingers'][0] if finger=='tip_left' else -hand['fingers'][1]
            fp=hand['finger_body_poses'][finger]
            point_local=np.asarray(fp['rotation']).T@(np.asarray(row['point_world_m'])-origin-fp['position_m'])
            finite_surface=self.pads.contains_surface(finger,point_local)
            on_pad=finite_surface and abs(pad[1])<=.018 and -.012<=pad[2]<=.014
            handle_checks.append({'finger':finger,'tool_body':name,'compression_N':compression,
                'actual_tip_local_point_m':point_local.tolist(),'within_functional_surface':finite_surface,
                'flat_plane_residual_m':float(pad[0]-inner),'TCP_local_point_m':pad.tolist()})
            if np.max(eq[:,:3]@local+eq[:,3])<=.0005 and on_pad and compression>.05:
                good[finger]=True;handle_bodies.add(name)
                handle_coordinates.append(float(pad[0]))
        state['both_handles']=all(good.get(n,False)for n in('tip_left','tip_right')) and len(handle_bodies)==2
        state['handle_pad_contact_span_m']=(max(handle_coordinates)-min(handle_coordinates))if handle_coordinates else None
        state['handle_contact_checks']=handle_checks
        state['actual_tool_to_TCP_midframe_m']=(np.asarray(state['tool_R']).T@(np.asarray(hand['TCP'])-po)).tolist()
        points,detail=self.contacts(origin)
        a,b=np.asarray(target_a),np.asarray(target_b);d=b-a;L=float(np.linalg.norm(d))
        fingers={};blocking=[];payload_support=[]
        for row in points:
            if '/LeftArm/' not in row['sensor'] or np.linalg.norm(row['force'])<.05:continue
            body=row['other'].rsplit('/',1)[-1];sensor=row['sensor'].rsplit('/',1)[-1]
            if sensor=='base':continue  # the fixed mounting body is not an approaching hand
            p=np.asarray(row['point']);u=float((p-a)@d/max(L*L,1e-15));dist=float(np.linalg.norm(p-(a+u*d)))
            on_target=body in ('cut_top','cut_brown','cut_green') and 0<=u<=1 and dist<=self.sa.R0+.0005
            # This is a planning contact domain, not a scoring or damage rule.
            along=u*L
            left=state['robots']['left'];LR=np.asarray(left['R'])
            local=LR.T@(p-np.asarray(left['TCP']))
            sign=-1 if sensor=='tip_left' else 1
            compression=sign*float(np.asarray(row['force'])@LR[:,0])
            inner=left['fingers'][0] if sensor=='tip_left' else -left['fingers'][1]
            functional=False
            if sensor in ('tip_left','tip_right'):
                fp=left['finger_body_poses'][sensor]
                tip_local=np.asarray(fp['rotation']).T@(p-fp['position_m'])
                functional=self.pads.contains_surface(sensor,tip_local)
            desired=(on_target and sensor in ('tip_left','tip_right') and compression>.05
                     and abs(local[1])<=.018 and -.012<=local[2]<=.014
                     and functional
                     and (severed or abs(along-.275)<=.025))
            if desired:fingers[sensor]=True
            elif severed and body in ('cut_top','cut_brown','cut_green') and sensor in ('tip_left','tip_right'):
                # A severed payload can roll onto another real fingertip face.
                # Preserve this contact without declaring a new acquired grasp.
                payload_support.append(row)
            else:blocking.append(row)
        state.update(payload_transport_contacts=payload_support,both_target_fingers=all(fingers.get(n,False)for n in('tip_left','tip_right')),
                     blocking_left=blocking,robot_contact_points=points,robot_contact_detail=detail)
        tool_points,tool_detail=self.contacts(origin,self.tool_plant_view,self.tool_sensors,self.plant_filters)
        state['tool_plant_contact_points']=tool_points
        state['tool_plant_contact_detail']=tool_detail
        state['wrong_tool_plant_contacts']=[row for row in tool_points
            if row['other'].rsplit('/',1)[-1]=='pot' and np.linalg.norm(row['force'])>.05]
        state['blocking_right']=[row for row in points
            if '/RightArm/'in row['sensor'] and row['sensor'].rsplit('/',1)[-1]!='base'
            and row['other']not in self.tool_sensors and np.linalg.norm(row['force'])>.05]
        state['plant_body_poses']={}
        for name in base.scene['bush'].body_names:
            p,R=body_pose(base.scene['bush'],name,origin)
            state['plant_body_poses'][name]={'position_m':p.tolist(),'rotation':R.tolist()}
        try:state['physics_counter']=int(base.sim.get_physics_step_count())
        except Exception as exc:state['physics_counter']={'unknown':str(exc)}
        return state
