"""Inspect Robots embodiment ``yam_floor_bay``: the standard station's bimanual YAM, beside a floor bay instead of a table.

Everything the policy sees and commands is the station's (``yam_arena``): the two arms at (0, +-0.305, 0.75), 0.61 m
apart, their joint interface, gains and limits, the calibrated top camera and the wrist cameras, 20 Hz control on
240 Hz physics. The difference is the room: there is no table, so a full-size object can stand on the floor in front
of the arms. Each arm stands on a steel pedestal from the floor to its mount, and the top camera on a floor mast.
The floor is level vinyl.

It subclasses ``yam_arena.inspect_embodiment.YamArenaEmbodiment`` and changes only the room it builds; the harness
itself is not modified. Run a task file on it with ``--embodiment yam_floor_bay -E task=task.py``; the scripted
policy ``yam_scripted`` plays the task's demo on it unchanged.
"""

from __future__ import annotations

import atexit
import os
from dataclasses import replace

from yam_arena import inspect_embodiment as adapter
from yam_arena import station, task_file
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, READY

FLOOR_FRICTION = (0.55, 0.55)  # vinyl sheet flooring: static, dynamic
PEDESTAL = (0.10, 0.10)  # steel column under each arm mount, x and y
PEDESTAL_FOOT = (0.24, 0.24, 0.015)
DOCS = """Simulated rig (Isaac Lab-Arena 0.3.1, PhysX): Dream's calibrated YAM station without its table. Both arms are
mounted 0.61 m apart at the station's height (0.76 m above the floor), each on a steel pedestal, left arm at +y, both
facing +x over a level vinyl floor. Each arm's base frame is its own mount. Cameras: top_cam looks down over the arms
from a mast behind them; left_cam and right_cam are on the wrists, looking out along the fingers."""


def floor_bay_room(top_camera) -> list:
    """Floor, lights, arm pedestals and the top camera's floor mast (no table)."""
    import isaaclab.sim as sim_utils
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_library import DomeLight
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg
    from yam_arena.scene import (
        STATION_DOME_INTENSITY,
        STATION_FLOOR_COLOR,
        STATION_LAMP_INTENSITY,
        STATION_LAMP_RADIUS,
        _box,
        _stand_part,
    )

    floor_z = station.FLOOR_Z
    vinyl = sim_utils.RigidBodyMaterialCfg(static_friction=FLOOR_FRICTION[0], dynamic_friction=FLOOR_FRICTION[1])
    floor = Object(
        name="floor",
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.CuboidCfg(
            size=(200.0, 200.0, 0.02),
            collision_props=PhysxCollisionPropertiesCfg(),
            physics_material=vinyl,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=STATION_FLOOR_COLOR, roughness=0.6),
        ),
        initial_pose=Pose(position_xyz=(0.0, 0.0, floor_z - 0.01)),
    )
    steel = (0.30, 0.31, 0.33)
    room = [
        floor,
        DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=STATION_DOME_INTENSITY,
                                                     color=station.linear_rgb(station.SKY))),
        Object(name="lamp", object_type=ObjectType.BASE,
               spawner_cfg=sim_utils.SphereLightCfg(intensity=STATION_LAMP_INTENSITY, radius=STATION_LAMP_RADIUS),
               initial_pose=Pose(position_xyz=station.LAMP_POS)),
    ]
    for side, (x, y, z) in station.BASES.items():
        top = z - 0.004  # the arm's base plate sits on the column
        room += [
            _box(f"{side}_pedestal_foot", PEDESTAL_FOOT, (x, y, floor_z + PEDESTAL_FOOT[2] / 2), steel),
            _box(f"{side}_pedestal", (*PEDESTAL, top - floor_z - PEDESTAL_FOOT[2]),
                 (x, y, (top + floor_z + PEDESTAL_FOOT[2]) / 2), steel),
        ]
    stand = station.camera_stand(top_camera)
    cx, cy, cz = top_camera.pos
    mast_top = cz + 0.02
    room += [
        _box("camera_floor_foot", (0.24, 0.24, 0.015), (cx - 0.07, cy, floor_z + 0.0075), steel),
        _box("camera_floor_mast", (0.03, 0.03, mast_top - floor_z - 0.015),
             (cx - 0.07, cy, (mast_top + floor_z + 0.015) / 2), steel),
        _stand_part("stand_arm", *stand["stand_arm"]),
        _stand_part("camera_housing", *stand["camera_housing"]),
    ]
    return room


class FloorBayEmbodiment(adapter.YamArenaEmbodiment):
    """``yam_arena`` with the station's arms and cameras over a floor bay. Requires ``-E task=...``."""

    def __init__(self, **kwargs) -> None:
        if not kwargs.get("task"):
            raise ValueError("yam_floor_bay needs a task file: -E task=task.py")
        super().__init__(**kwargs)
        if self.task.scene != "station":
            raise ValueError("yam_floor_bay uses the station's rig: the task file must set SCENE = 'station'")
        # The id keeps "-station-", so the yam_scripted policy uses the station's arm bases.
        eid = f"isaaclab-arena-0.3.1-yam-floor-bay-{self.rig.name}-task-{self.task.name}"
        docs = "\n\n".join(d for d in (adapter.JOINT_GUIDE, DOCS, self.task.docs) if d)
        self.info = replace(self.info, name="yam_floor_bay", environment_id=eid, docs=docs)
        task_file.register(eid, self.task)

    def _ensure_sim(self, goal) -> None:
        if self._env is not None:
            return
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        if adapter._APP is None:
            from isaaclab.app import AppLauncher

            options = {"headless": self.headless, "device": self.device, "enable_cameras": True}
            if not self.headless:
                options["visualizer"] = ["kit"]
                os.environ["DISPLAY"] = adapter.window_display(os.environ)
            adapter._APP = AppLauncher(**options).app
            atexit.register(adapter._close_app_at_exit)

        import torch
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from yam_arena import scene as yam_scene
        from yam_arena.embodiment import BimanualYamEmbodiment

        embodiment = BimanualYamEmbodiment(enable_cameras=True, wrist_cam_mount=self.wrist_cam_mount,
                                           contact_sensors=False, rig=self.rig)
        environment = IsaacLabArenaEnvironment(
            name=f"yam_floor_bay_{self.task.name}",
            scene=Scene([*floor_bay_room(self.rig.top_cam), *(yam_scene.task_object(o) for o in self.task.objects)]),
            embodiment=embodiment,
            env_cfg_callback=yam_scene.env_cfg_callback(self.info.control_hz),
        )
        self._env = ArenaEnvBuilder(environment, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False,
                                                                    device=self.device)).make_registered()
        self._base = self._env.unwrapped
        self._goal, self._torch = goal, torch
        self._free_objects = self.task.free_objects
        self._ready = torch.tensor([[*READY, 1.0, *READY, 1.0]], dtype=torch.float32, device=self._base.device)
        self._ids = {
            name: (self._base.scene[name].find_joints(ARM_JOINTS, preserve_order=True)[0],
                   self._base.scene[name].find_joints(FINGER_JOINTS, preserve_order=True)[0])
            for name in adapter.ARMS
        }


def yam_floor_bay(**kwargs) -> FloorBayEmbodiment:
    """Embodiment ``yam_floor_bay``: the station's arms over a floor bay; ``-E task=task.py`` is required."""
    return FloorBayEmbodiment(**kwargs)
