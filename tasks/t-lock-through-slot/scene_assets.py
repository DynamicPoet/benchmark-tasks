"""The scene's models, written as plain-text USD when the task file loads (no pxr needed).

A painted steel panel fixed to the table, with a vertical slot. A steel rod runs through the slot: on the robot's side it ends in an
orange T-handle, behind the panel in a T-shaped steel head that turns on the rod. A crank pin stands on the head's back near one
end. The rod and the head are an articulation with a fixed root: the rod slides along x, the head turns about the rod. Sizes in
metres; every height is above the table.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import usdkit as uk
from usdkit import Look, Mesh, box, collider, cylinder, join, write_usda

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

PANEL_X = 0.25  # the panel's middle
PANEL_T = 0.010
PANEL_W = 0.40
PANEL_H = 0.12
ROD_Z = 0.07
SLOT_W = 0.022
SLOT_Z = (0.022, 0.118)
ROD_R = 0.006

HEAD_GAP = 0.025  # from the panel's back to the head's front, at the start
HEAD_L, HEAD_W, HEAD_T = 0.08, 0.012, 0.012  # across, its other width, and along the rod
HEAD_X = PANEL_X + PANEL_T / 2 + HEAD_GAP + HEAD_T / 2
PIN_OFFSET = 0.032  # from the rod's axis
PIN_L = 0.05
PIN_W = 0.012
TRAVEL = 0.10
AT_STOP = 0.095
HEAD_IN_SLOT = 0.035
REST_TILT_DEG = 20.0  # at rest the head hangs this far off level, its pin end up, still far too wide for the slot
UPRIGHT_Q_DEG = 90.0 - REST_TILT_DEG  # how far the head turns from rest to upright
STOP_Q_DEG = UPRIGHT_Q_DEG + 1.0  # the head's stop, at upright

SLEEVE_X = (0.155, 0.235)  # the handle's orange stem
SLEEVE_R = 0.011
BAR_X = 0.147
BAR_L = 0.09
BAR_R = 0.008
FOOT = (0.05, 0.05, 0.006)

LOOKS = {
    "panel": Look(color=(0.16, 0.24, 0.30), roughness=0.45),
    "rivet": Look(color=(0.62, 0.63, 0.65), metallic=1.0, roughness=0.35),
    "steel": Look(color=(0.70, 0.71, 0.73), metallic=1.0, roughness=0.28),
    "handle": Look(color=(0.85, 0.14, 0.008), roughness=0.4),
    "head": Look(color=(0.72, 0.60, 0.18), metallic=1.0, roughness=0.32),
    "dark": Look(color=(0.03, 0.03, 0.035), roughness=0.5),
}


def _quat_wxyz_x(deg: float) -> tuple[float, float, float, float]:
    a = math.radians(deg) / 2
    return (math.cos(a), math.sin(a), 0.0, 0.0)


def write_panel() -> str:
    """Origin at the table, under the panel's middle. Fixed."""
    root = "Panel"
    t, w, h = PANEL_T, PANEL_W, PANEL_H
    z0, z1 = SLOT_Z
    side = (w - SLOT_W) / 2
    pieces = [
        ((t, side, h), (0.0, (SLOT_W + side) / 2, h / 2)),
        ((t, side, h), (0.0, -(SLOT_W + side) / 2, h / 2)),
        ((t, SLOT_W, z0), (0.0, 0.0, z0 / 2)),
        ((t, SLOT_W, h - z1), (0.0, 0.0, (h + z1) / 2)),
    ]
    shell = join(*(box(size, center=c) for size, c in pieces))
    feet = join(*(box(FOOT, center=(0.012, s * 0.17, FOOT[2] / 2), bevel=0.001) for s in (-1, 1)))
    gussets = join(*(box((0.03, 0.006, 0.05), center=(0.012 + 0.0, s * 0.17, 0.03), bevel=0.0005) for s in (-1, 1)))
    rivets = join(*(cylinder(0.005, 0.002, center=(-t / 2 - 0.0008, y, z), axis="x", segments=12)
                    for y in (-0.18, -0.09, 0.09, 0.18) for z in (0.02, h - 0.02)))
    parts = [("shell", shell, "panel"), ("feet", feet, "dark"), ("gussets", gussets, "dark"), ("rivets", rivets, "rivet")]
    colliders = [collider(f"p{k}", size, c, root) for k, (size, c) in enumerate(pieces)]
    return write_usda(str(ASSETS / "panel.usda"), root, parts, LOOKS, colliders, mass=1.0, friction=(0.5, 0.4))


def _rod_meshes() -> list[tuple[str, Mesh, str]]:
    """In the rod link's frame: origin at the head's middle."""
    x0 = BAR_X - HEAD_X
    rod = cylinder(ROD_R, HEAD_X - SLEEVE_X[1] + 0.002, center=((SLEEVE_X[1] - HEAD_X) / 2, 0.0, 0.0), axis="x", segments=16)
    sleeve = cylinder(SLEEVE_R, SLEEVE_X[1] - SLEEVE_X[0], center=((SLEEVE_X[0] + SLEEVE_X[1]) / 2 - HEAD_X, 0.0, 0.0), axis="x", segments=20, bevel=0.002)
    bar = cylinder(BAR_R, BAR_L, center=(x0, 0.0, 0.0), axis="y", segments=16, bevel=0.002)
    collar = cylinder(0.009, 0.006, center=(HEAD_T / 2 * -1 - 0.003, 0.0, 0.0), axis="x", segments=16)
    return [("rod", rod, "steel"), ("sleeve", sleeve, "handle"), ("bar", bar, "handle"), ("collar", collar, "steel")]


def _head_meshes() -> list[tuple[str, Mesh, str]]:
    """In the head link's frame: origin on the rod's axis, the bar across y."""
    bar = box((HEAD_T, HEAD_L, HEAD_W), bevel=0.001, round_xy=0.0)
    hub = cylinder(0.010, HEAD_T + 0.004, center=(0.0, 0.0, 0.0), axis="x", segments=18)
    return [("bar", bar, "head"), ("hub", hub, "steel")]


def roller_rest() -> tuple[float, float, float]:
    """The crank pin's middle at rest, relative to the head's middle: a free-spinning sleeve on the head's back, near its -y end."""
    a = math.radians(-REST_TILT_DEG)
    return (HEAD_T / 2 + PIN_L / 2, -PIN_OFFSET * math.cos(a), -PIN_OFFSET * math.sin(a))


def write_mechanism() -> str:
    root = "Lock"
    used = {"steel", "handle", "head"}
    looks = "".join(uk._material_usda(k, root, LOOKS[k]) for k in sorted(used))

    def cube(name: str, size, centre, patch: float = 0.0) -> str:
        extra = f"\n            float physxCollision:torsionalPatchRadius = {patch:.4f}\n            float physxCollision:minTorsionalPatchRadius = {0.8 * patch:.4f}" if patch else ""
        return f"""
        def Cube "{name}" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            rel material:binding:physics = </{root}/PhysicsMaterial>
            uniform token purpose = "guide"
            double size = 1
            float physxCollision:contactOffset = 0.003
            float physxCollision:restOffset = 0{extra}
            double3 xformOp:translate = ({centre[0]:.6f}, {centre[1]:.6f}, {centre[2]:.6f})
            float3 xformOp:scale = ({size[0]:.6f}, {size[1]:.6f}, {size[2]:.6f})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}"""

    rod_body = "".join([
        cube("rod_c", (HEAD_X - SLEEVE_X[1], 2 * ROD_R, 2 * ROD_R), ((SLEEVE_X[1] - HEAD_X) / 2, 0.0, 0.0)),
        cube("sleeve_c", (SLEEVE_X[1] - SLEEVE_X[0], 2 * SLEEVE_R, 2 * SLEEVE_R), ((SLEEVE_X[0] + SLEEVE_X[1]) / 2 - HEAD_X, 0.0, 0.0), 0.01),
        cube("bar_c", (2 * BAR_R, BAR_L, 2 * BAR_R), (BAR_X - HEAD_X, 0.0, 0.0)),
    ]) + "".join(uk._mesh_usda(*p, root, LOOKS[p[2]], 40.0, indent="        ") for p in _rod_meshes())
    head_body = cube("head_c", (HEAD_T, HEAD_L, HEAD_W), (0.0, 0.0, 0.0)) + "".join(uk._mesh_usda(*p, root, LOOKS[p[2]], 40.0, indent="        ") for p in _head_meshes())
    roller_mesh = cylinder(PIN_W / 2, PIN_L, axis="x", segments=16, bevel=0.0015)
    roller_body = f"""
        def Cylinder "roller_c" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            rel material:binding:physics = </{root}/PhysicsMaterial>
            uniform token purpose = "guide"
            uniform token axis = "X"
            double height = {PIN_L:.6f}
            double radius = {PIN_W / 2:.6f}
            float physxCollision:contactOffset = 0.003
            float physxCollision:restOffset = 0
        }}""" + uk._mesh_usda("sleeve", roller_mesh, "steel", root, LOOKS["steel"], 40.0, indent="        ")
    rr = roller_rest()
    pin_local = (HEAD_T / 2 + PIN_L / 2, -PIN_OFFSET, 0.0)

    def link(name: str, mass: float, inertia: float, body: str, turn_deg: float = 0.0, at=(0.0, 0.0, 0.0)) -> str:
        q = _quat_wxyz_x(turn_deg)
        return f"""
    def Xform "{name}" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    )
    {{
        float physics:mass = {mass:.4f}
        float3 physics:diagonalInertia = ({inertia:.3e}, {inertia:.3e}, {inertia:.3e})
        point3f physics:centerOfMass = (0, 0, 0)
        bool physxRigidBody:disableGravity = 1
        float physxRigidBody:maxDepenetrationVelocity = 1
        double3 xformOp:translate = ({at[0]:.7f}, {at[1]:.7f}, {at[2]:.7f})
        quatd xformOp:orient = ({q[0]:.7f}, {q[1]:.7f}, {q[2]:.7f}, {q[3]:.7f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
{body}
    }}
"""

    text = f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]
)
{{
    bool physxArticulation:enabledSelfCollisions = 0
    int physxArticulation:solverPositionIterationCount = 32
    int physxArticulation:solverVelocityIterationCount = 4

    def Material "PhysicsMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    )
    {{
        float physics:staticFriction = 0.7
        float physics:dynamicFriction = 0.6
        float physics:restitution = 0
    }}

    def Scope "Looks"
    {{{looks}
    }}

    def Xform "frame" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        float physics:mass = 0.5
        float3 physics:diagonalInertia = (1e-3, 1e-3, 1e-3)
        point3f physics:centerOfMass = (0, 0, 0)
    }}
{link("rod", 0.15, 3e-4, rod_body)}{link("head", 0.05, 3e-5, head_body, -REST_TILT_DEG)}{link("roller", 0.01, 2e-6, roller_body, -REST_TILT_DEG, rr)}
    def PhysicsFixedJoint "world_joint"
    {{
        rel physics:body1 = </{root}/frame>
    }}

    def PhysicsPrismaticJoint "rod_slide" (
        prepend apiSchemas = ["PhysicsDriveAPI:linear"]
    )
    {{
        uniform token drive:linear:physics:type = "force"
        float drive:linear:physics:stiffness = 0
        float drive:linear:physics:damping = 0
        rel physics:body0 = </{root}/frame>
        rel physics:body1 = </{root}/rod>
        uniform token physics:axis = "X"
        point3f physics:localPos0 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
        float physics:lowerLimit = {-TRAVEL:.5f}
        float physics:upperLimit = 0
    }}

    def PhysicsRevoluteJoint "head_turn" (
        prepend apiSchemas = ["PhysicsDriveAPI:angular"]
    )
    {{
        uniform token drive:angular:physics:type = "force"
        float drive:angular:physics:stiffness = 0
        float drive:angular:physics:damping = 0
        rel physics:body0 = </{root}/rod>
        rel physics:body1 = </{root}/head>
        uniform token physics:axis = "X"
        point3f physics:localPos0 = (0, 0, 0)
        quatf physics:localRot0 = {tuple(round(v, 7) for v in _quat_wxyz_x(-REST_TILT_DEG))}
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
        float physics:lowerLimit = {-STOP_Q_DEG:.3f}
        float physics:upperLimit = 8
    }}

    def PhysicsRevoluteJoint "pin_spin" (
        prepend apiSchemas = ["PhysicsDriveAPI:angular"]
    )
    {{
        uniform token drive:angular:physics:type = "force"
        float drive:angular:physics:stiffness = 0
        float drive:angular:physics:damping = 0
        rel physics:body0 = </{root}/head>
        rel physics:body1 = </{root}/roller>
        uniform token physics:axis = "X"
        point3f physics:localPos0 = ({pin_local[0]:.7f}, {pin_local[1]:.7f}, {pin_local[2]:.7f})
        quatf physics:localRot0 = (1, 0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
    }}
}}
"""
    os.makedirs(ASSETS, exist_ok=True)
    path = ASSETS / "lock.usda"
    path.write_text(text)
    return str(path)


def write_all() -> None:
    write_panel()
    write_mechanism()
