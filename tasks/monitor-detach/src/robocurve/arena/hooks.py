"""Extensions a yam_arena task file needs for the monitor-detach fixtures, without changing yam_arena.

yam_arena's task files spawn primitives (``Box``, ``Cylinder``, ``Sphere``) and Arena library assets, and
score once per control step. The monitor-detach task also needs:

- :class:`UsdFixture`: a body from a USD file written at spawn time, rigid (with a ``mass``) or an
  articulation with driven joints (the stand and its spring-return lever). It subclasses yam_arena's
  ``Box`` only so the task-file checks accept it.
- callbacks every physics substep (:func:`on_substep`: the fastener's wrench, the B1 grip fade), after
  every control step before the task file is scored (:func:`on_step`: telemetry), and before every reset
  (:func:`on_reset`).

yam_arena's ``scene`` and ``inspect_embodiment`` modules import Isaac Sim, which starts after the task
file is loaded, so they are patched as they are imported (a ``sys.meta_path`` finder). Pure Python at
import time.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from yam_arena.task_file import Box, ObjectState

__all__ = ["FixtureState", "UsdFixture", "env", "install", "on_reset", "on_step", "on_substep"]


@dataclass(frozen=True)
class UsdFixture(Box):
    """A body from the USD that ``build()`` writes (called once Isaac Sim runs; it returns the path).

    ``pos``/``rot_xyzw`` place the file's origin. With ``articulation`` the file's articulation keeps its
    own fixed base, and ``drives`` gives each joint (stiffness, damping) around a target of 0. Otherwise it
    is a rigid body of ``mass`` kg (kinematic without one). ``size`` is only nominal."""

    build: Callable[[], str] | None = None
    rot_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    articulation: bool = False
    drives: tuple[tuple[str, float, float], ...] = ()
    friction: tuple[float, float] = (1.0, 0.9)  # static, dynamic

    def _check(self) -> None:
        super()._check()
        if self.build is None:
            raise ValueError(f"{self.name}: a UsdFixture needs build()")


@dataclass(frozen=True)
class FixtureState(ObjectState):
    """A fixture's root pose and velocities (station world frame) and, for an articulation, its joints."""

    joints: dict = field(default_factory=dict)


_SPECS: dict[str, UsdFixture] = {}
_SUBSTEP: list[Callable[[Any], None]] = []
_STEP: list[Callable[[Any], None]] = []
_RESET: list[Callable[[Any], None]] = []
_ENV: dict[str, Any] = {"env": None}


def on_substep(fn: Callable[[Any], None]) -> Callable[[Any], None]:
    """Run ``fn(env)`` every physics substep, after the arms' commands and before the simulator steps."""
    _SUBSTEP.append(fn)
    return fn


def on_step(fn: Callable[[Any], None]) -> Callable[[Any], None]:
    """Run ``fn(env)`` after every control step, before the task file's success() and score()."""
    _STEP.append(fn)
    return fn


def on_reset(fn: Callable[[Any], None]) -> Callable[[Any], None]:
    """Run ``fn(env)`` before every reset (the env exists; the scene is about to be reset and settled)."""
    _RESET.append(fn)
    return fn


def env() -> Any:
    """The current Isaac Lab (gym) env, once yam_arena has built it."""
    return _ENV["env"]


def _np(x: Any) -> np.ndarray:
    """A ProxyArray, torch tensor or warp array as a float64 numpy array."""
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x.numpy() if hasattr(x, "numpy") else x, dtype=np.float64)


# -- spawning ---------------------------------------------------------------------------------------------


def _fixture_object(spec: UsdFixture) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    assert spec.build is not None
    usd = spec.build()
    material = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction[0], dynamic_friction=spec.friction[1])
    addon: dict[str, Any] = {"physics_material": material}
    if spec.articulation:
        addon["articulation_props"] = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
        )
    else:
        addon["rigid_props"] = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=8, kinematic_enabled=spec.mass is None
        )
        # Contacts start 5 mm out and rest at zero separation, as yam_arena sets its own graspable objects.
        addon["collision_props"] = sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0)
        if spec.mass is not None:
            addon["mass_props"] = sim_utils.MassPropertiesCfg(mass=spec.mass)

    class FixtureObject(Object):
        def _generate_articulation_cfg(self):  # noqa: ANN202 - Arena's signature
            cfg = super()._generate_articulation_cfg()
            cfg.init_state.joint_pos = {name: 0.0 for name, _, _ in spec.drives}
            cfg.actuators = {
                name: ImplicitActuatorCfg(joint_names_expr=[name], stiffness=k, damping=d)
                for name, k, d in spec.drives
            }
            return cfg

    obj = FixtureObject(
        name=spec.name,
        usd_path=usd,
        object_type=ObjectType.ARTICULATION if spec.articulation else ObjectType.RIGID,
        spawn_cfg_addon=addon,
    )
    obj.set_initial_pose(Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=tuple(spec.rot_xyzw)))
    return obj


def _fixture_state(environment: Any, spec: UsdFixture) -> FixtureState:
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    pose = _np(asset.data.root_link_pose_w)[0]
    vel = _np(asset.data.root_com_vel_w)[0]
    joints = {}
    if spec.articulation:
        q = _np(asset.data.joint_pos)[0]
        joints = {n: float(q[i]) for i, n in enumerate(asset.joint_names)}
    pos = pose[:3] - origin
    return FixtureState(
        pos=pos, quat_xyzw=pose[3:7], lin_vel=vel[:3], ang_vel=vel[3:6], low=pos, high=pos, joints=joints
    )


# -- patching yam_arena ---------------------------------------------------------------------------------------


def _patch_scene(scene: Any) -> None:
    if getattr(scene, "_robocurve_hooks", False):
        return
    task_object, object_states, make_env = scene.task_object, scene.object_states, scene.make_env

    def task_object_hook(spec: Any) -> Any:
        if isinstance(spec, UsdFixture):
            _SPECS[spec.name] = spec
            return _fixture_object(spec)
        return task_object(spec)

    def object_states_hook(environment: Any, names: Any) -> dict[str, ObjectState]:
        plain = [n for n in names if n not in _SPECS]
        out = object_states(environment, plain) if plain else {}
        out.update({n: _fixture_state(environment, _SPECS[n]) for n in names if n in _SPECS})
        return {n: out[n] for n in names}

    def make_env_hook(*args: Any, **kwargs: Any) -> Any:
        environment = make_env(*args, **kwargs)
        _ENV["env"] = environment
        manager = environment.unwrapped.action_manager
        apply_action = manager.apply_action

        def apply_action_hook() -> None:
            apply_action()
            for fn in _SUBSTEP:
                fn(environment)

        manager.apply_action = apply_action_hook
        return environment

    scene.task_object, scene.object_states, scene.make_env = task_object_hook, object_states_hook, make_env_hook
    scene._robocurve_hooks = True


def _patch_embodiment(module: Any) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_robocurve_hooks", False):
        return
    task_step, ensure_sim = cls._task_step, cls._ensure_sim

    # reset() starts with _ensure_sim() (which builds the env on first use), then resets and settles the
    # scene; hooking _ensure_sim runs the reset callbacks once per reset, with the env in hand.
    def ensure_sim_hook(self: Any, *a: Any, **k: Any) -> Any:
        out = ensure_sim(self, *a, **k)
        for fn in _RESET:
            fn(self._env)
        return out

    def task_step_hook(self: Any, *a: Any, **k: Any) -> Any:
        for fn in _STEP:
            fn(self._env)
        return task_step(self, *a, **k)

    cls._ensure_sim, cls._task_step = ensure_sim_hook, task_step_hook
    cls._robocurve_hooks = True


class _Finder(importlib.abc.MetaPathFinder):
    """Patches yam_arena's scene and embodiment modules as they are imported."""

    targets = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        if fullname not in self.targets:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        exec_module, patch = spec.loader.exec_module, self.targets[fullname]

        def exec_and_patch(module: Any) -> None:
            exec_module(module)
            patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


def install() -> None:
    """Hook yam_arena now for modules already imported, and on import for the rest. Idempotent."""
    for name, patch in _Finder.targets.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
