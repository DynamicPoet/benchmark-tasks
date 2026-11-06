"""Write the monitor stand as a two-link USD articulation: a fixed stand and a spring-return lever.

Needs the ``pxr`` USD bindings, which Isaac Sim provides; import this after the app has started.
"""

from __future__ import annotations

from pathlib import Path

from robocurve.sim.layout import Box, Layout, box_center, box_size

STAND_COLOR = (0.09, 0.09, 0.10)  # graphite, like most monitor stands; stands out from the grey worktop
LEVER_COLOR = (0.95, 0.45, 0.10)


def _add_box(stage, path: str, box: Box, origin: tuple[float, float, float], color) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    center = box_center(box)
    cube.AddTranslateOp().Set(Gf.Vec3d(*(c - o for c, o in zip(center, origin))))
    cube.AddScaleOp().Set(Gf.Vec3f(*box_size(box)))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def write_stand_usd(path: Path, layout: Layout, world_anchor: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Path:
    """Create the stand articulation USD at ``path`` (overwritten) in station-frame coordinates.

    The base is fixed to the world at ``world_anchor``, the world position its origin is spawned at (the
    station frame's origin; (0, 0, 0) when the station frame is the world).
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Stand")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    base = UsdGeom.Xform.Define(stage, "/Stand/base")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    UsdPhysics.MassAPI.Apply(base.GetPrim()).CreateMassAttr(5.0)
    for name, box in (("plate", layout.plate_box), ("neck", layout.neck_box), ("bracket", layout.bracket_box)):
        _add_box(stage, f"/Stand/base/{name}", box, (0.0, 0.0, 0.0), STAND_COLOR)

    lever_origin = box_center(layout.lever_box)
    lever = UsdGeom.Xform.Define(stage, "/Stand/lever")
    lever.AddTranslateOp().Set(Gf.Vec3d(*lever_origin))
    UsdPhysics.RigidBodyAPI.Apply(lever.GetPrim())
    UsdPhysics.MassAPI.Apply(lever.GetPrim()).CreateMassAttr(0.05)
    _add_box(stage, "/Stand/lever/geom", layout.lever_box, lever_origin, LEVER_COLOR)

    fixed = UsdPhysics.FixedJoint.Define(stage, "/Stand/root_joint")
    fixed.CreateBody1Rel().SetTargets([Sdf.Path("/Stand/base")])
    fixed.CreateLocalPos0Attr(Gf.Vec3f(*world_anchor))  # body0 is the world

    slider = UsdPhysics.PrismaticJoint.Define(stage, "/Stand/lever_joint")
    slider.CreateBody0Rel().SetTargets([Sdf.Path("/Stand/base")])
    slider.CreateBody1Rel().SetTargets([Sdf.Path("/Stand/lever")])
    slider.CreateAxisAttr("X")
    slider.CreateLowerLimitAttr(0.0)
    slider.CreateUpperLimitAttr(layout.lever_travel)
    slider.CreateLocalPos0Attr(Gf.Vec3f(*lever_origin))
    slider.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    slider.CreateLocalRot0Attr(Gf.Quatf(1.0))
    slider.CreateLocalRot1Attr(Gf.Quatf(1.0))
    drive = UsdPhysics.DriveAPI.Apply(slider.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(layout.lever_stiffness)
    drive.CreateDampingAttr(5.0)
    drive.CreateTargetPositionAttr(0.0)

    stage.GetRootLayer().Save()
    return path


PANEL_COLOR = (0.07, 0.07, 0.08)
SCREEN_COLOR = (0.10, 0.35, 0.75)


def write_monitor_usd(path: Path, layout: Layout) -> Path:
    """The monitor as a rigid-body USD of ``layout.panel_mass`` with its origin at the panel's centre: a black
    box that collides, and a blue screen face on its +x side (visual only), so its orientation shows."""
    from pxr import Gf, UsdGeom, UsdPhysics, Usd

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Monitor")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    # The mass goes in the file: Isaac Lab's spawner only changes the mass of a prim that has MassAPI.
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(layout.panel_mass)
    t, w, h = box_size(layout.panel_box)
    body = UsdGeom.Cube.Define(stage, "/Monitor/body")
    body.CreateSizeAttr(1.0)
    body.AddScaleOp().Set(Gf.Vec3f(t, w, h))
    body.CreateDisplayColorAttr([Gf.Vec3f(*PANEL_COLOR)])
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    screen = UsdGeom.Cube.Define(stage, "/Monitor/screen")
    screen.CreateSizeAttr(1.0)
    screen.AddTranslateOp().Set(Gf.Vec3d(t / 2 + 0.0011, 0.0, 0.0))
    screen.AddScaleOp().Set(Gf.Vec3f(0.002, w - 0.02, h - 0.02))
    screen.CreateDisplayColorAttr([Gf.Vec3f(*SCREEN_COLOR)])
    stage.GetRootLayer().Save()
    return path
