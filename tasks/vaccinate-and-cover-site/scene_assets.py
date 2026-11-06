"""The scene's models, written as plain-text USD when the task file loads (no pxr needed).

An injection-training arm on a cushion, with a printed target circle on its upper arm; a spring-loaded skin pad under the circle
(one sliding joint); a pre-filled injector pen in a holder; a gauze pad on a small raised tray; a sharps bin. Sizes in metres;
every height is above the table.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import usdkit as uk
from usdkit import Look, Mesh, box, collider, cylinder, join, lathe, prism, write_usda

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

TARGET_XY = (0.26, 0.05)
TARGET_R = 0.015
ARM_AXIS_Z = 0.052
ARM_R_AT_TARGET = 0.040
SKIN_TOP = ARM_AXIS_Z + ARM_R_AT_TARGET  # the arm's top at the target
PAD_HALF = 0.022
PAD_PROUD = 0.002  # the skin pad stands this far above the arm's collider, so that the gauze can press it
PAD_TOP = SKIN_TOP + PAD_PROUD
PAD_TRAVEL = 0.015
PAD_STIFFNESS = 450.0  # N/m

PEN_L = 0.14
PEN_W = 0.022  # the barrel's width
PEN_NOSE = (0.012, 0.012, 0.018)
PEN_MASS = 0.06
HOLDER_XY = (0.20, 0.165)
HOLDER = (0.06, 0.06, 0.04)  # x, y, height
HOLDER_SOCKET = 0.032  # width of its square socket
HOLDER_FLOOR = 0.010

GAUZE = (0.05, 0.05, 0.014)
GAUZE_MASS = 0.03
TRAY_XY = (0.24, -0.17)
TRAY = (0.036, 0.09, 0.03)  # narrower than the gauze along x, so that the fingertips pass beside the tray

BIN_XY = (0.19, 0.27)
BIN = (0.11, 0.11, 0.12)
BIN_WALL = 0.005

LOOKS = {
    "skin": Look(color=(0.82, 0.58, 0.45), roughness=0.55),
    "stump": Look(color=(0.35, 0.36, 0.38), roughness=0.5),
    "foam": Look(color=(0.14, 0.24, 0.42), roughness=0.95),
    "mark": Look(color=(0.02, 0.02, 0.03), roughness=0.5),
    "site": Look(color=(0.92, 0.80, 0.70), roughness=0.5),
    "blood": Look(color=(0.55, 0.01, 0.01), roughness=0.3),
    "pen_body": Look(color=(0.72, 0.74, 0.78), roughness=0.3),
    "pen_band": Look(color=(0.05, 0.20, 0.55), roughness=0.35),
    "pen_nose": Look(color=(0.85, 0.42, 0.04), roughness=0.4),
    "pen_top": Look(color=(0.03, 0.03, 0.035), roughness=0.4),
    "holder": Look(color=(0.72, 0.74, 0.75), roughness=0.4),
    "gauze": Look(color=(0.92, 0.92, 0.90), roughness=0.95),
    "tray": Look(color=(0.55, 0.56, 0.58), metallic=0.8, roughness=0.35),
    "bin": Look(color=(0.92, 0.72, 0.03), roughness=0.45),
    "bin_lid": Look(color=(0.70, 0.05, 0.04), roughness=0.4),
    "bin_dark": Look(color=(0.03, 0.03, 0.03), roughness=0.8),
}


def _rings(rings: list[tuple[np.ndarray, float]]) -> Mesh:
    n = len(rings[0][0])
    v = [np.c_[o, np.full(n, z)] for o, z in rings]
    v.append(np.array([[*rings[0][0].mean(axis=0), rings[0][1]], [*rings[-1][0].mean(axis=0), rings[-1][1]]]))
    v = np.vstack(v)
    bottom, top = len(rings) * n, len(rings) * n + 1
    f = []
    for k in range(len(rings) - 1):
        for i in range(n):
            j = (i + 1) % n
            a, b, e, g = k * n + i, k * n + j, (k + 1) * n + j, (k + 1) * n + i
            f += [(a, b, e), (a, e, g)]
    last = (len(rings) - 1) * n
    for i in range(n):
        j = (i + 1) % n
        f += [(bottom, j, i), (top, last + i, last + j)]
    return Mesh(v, np.array(f))


def _band(rings: list[tuple[np.ndarray, float]]) -> Mesh:
    """A surface through outlines of equal length, with no caps: the bin's lid has an opening."""
    n = len(rings[0][0])
    v = np.vstack([np.c_[o, np.full(n, z)] for o, z in rings])
    f = []
    for k in range(len(rings) - 1):
        for i in range(n):
            j = (i + 1) % n
            a, b, e, g = k * n + i, k * n + j, (k + 1) * n + j, (k + 1) * n + i
            f += [(a, b, e), (a, e, g)]
    return Mesh(v, np.array(f))


def _ellipsoid(centre, semi, segments: int = 20, rings: int = 10) -> Mesh:
    prof = [(float(np.sin(a)), float(-np.cos(a))) for a in np.linspace(0.0, np.pi, rings + 1)]
    m = lathe([(0.0 if abs(r) < 1e-6 else r, z) for r, z in prof], segments)
    return Mesh(m.v * np.asarray(semi, float) + np.asarray(centre, float), m.f)


def write_arm() -> str:
    """Origin under the target, on the table. The upper arm lies along y, the shoulder towards +y."""
    root = "Arm"
    prof = [(0.0, -0.135), (0.031, -0.135), (0.034, -0.128), (0.036, -0.09), (0.038, -0.04), (ARM_R_AT_TARGET, 0.0), (0.043, 0.04), (0.047, 0.08),
            (0.050, 0.115), (0.048, 0.145), (0.040, 0.168), (0.026, 0.182), (0.0, 0.188)]
    upper = lathe(prof, 40).turned(-90.0, "x").moved((0.0, 0.0, ARM_AXIS_Z))
    foam = box((0.13, 0.36, 0.016), center=(0.0, 0.025, 0.008), bevel=0.005, round_xy=0.02)
    # The elbow bends towards the robot: forearm along -x, then the hand, palm down
    elbow = (0.0, -0.105)
    fore = lathe([(0.0, 0.0), (0.033, 0.0), (0.034, 0.05), (0.031, 0.12), (0.025, 0.20), (0.024, 0.215), (0.0, 0.215)], 28).turned(-90.0, "y").moved((elbow[0], elbow[1], 0.046))
    joint = _ellipsoid((elbow[0], elbow[1], 0.048), (0.037, 0.037, 0.034))
    palm = _ellipsoid((-0.262, elbow[1], 0.036), (0.05, 0.036, 0.015))
    fingers = join(*(_ellipsoid((-0.325, elbow[1] + dy, 0.030), (0.032, 0.0075, 0.0075), 12, 6) for dy in (-0.024, -0.008, 0.008, 0.024)))
    thumb = _ellipsoid((-0.275, elbow[1] + 0.044, 0.032), (0.028, 0.009, 0.009), 12, 6)
    foam2 = box((0.33, 0.09, 0.012), center=(-0.19, elbow[1], 0.006), bevel=0.004, round_xy=0.02)
    # Marks sit just above the skin pad, which is proud of the arm by 2 mm
    disc = cylinder(TARGET_R + 0.004, 0.0006, center=(0.0, 0.0, PAD_TOP + 0.0003), segments=40)
    ring = lathe([(TARGET_R - 0.0016, PAD_TOP + 0.0006), (TARGET_R, PAD_TOP + 0.0006), (TARGET_R, PAD_TOP + 0.0012), (TARGET_R - 0.0016, PAD_TOP + 0.0012)], 40)
    ring = Mesh(ring.v, ring.f)
    parts: list[tuple] = [("upper", upper, "skin"), ("foam", foam, "foam"), ("foam2", foam2, "foam"), ("site", disc, "site"), ("ring", ring, "mark"), ("fore", fore, "skin"), ("elbow", joint, "skin"),
                          ("palm", palm, "skin"), ("fingers", fingers, "skin"), ("thumb", thumb, "skin")]
    for k, r in enumerate((0.003, 0.005, 0.0075, 0.010, 0.0125)):
        spot = cylinder(r, 0.0005, center=(0.0, 0.0, PAD_TOP + 0.0016 + 0.0001 * k), segments=24)
        parts.append((f"spot_{k}", spot, "blood", True))
    h, top = PAD_HALF + 0.002, SKIN_TOP
    wide = 0.036
    lo = 0.02
    hgt = top - lo
    side_top = SKIN_TOP - 0.008  # the arm curves away from the target: fingers holding the gauze pass beside it
    colliders = [
        collider("far", (2 * wide, 0.14, hgt), (0.0, h + 0.07, lo + hgt / 2), root),
        collider("near", (2 * wide, 0.11, hgt - 0.006), (0.0, -h - 0.055, lo + (hgt - 0.006) / 2), root),
        collider("front", (wide - h, 2 * h, side_top - lo), (-(wide + h) / 2, 0.0, (side_top + lo) / 2), root),
        collider("back", (wide - h, 2 * h, side_top - lo), ((wide + h) / 2, 0.0, (side_top + lo) / 2), root),
        collider("fore", (0.25, 0.06, 0.07), (-0.13, -0.105, 0.04), root),
        collider("under", (2 * wide, 2 * h, 0.012), (0.0, 0.0, 0.008), root),
    ]
    return write_usda(str(ASSETS / "arm.usda"), root, parts, LOOKS, colliders, mass=1.0, friction=(0.8, 0.7), crease=50.0)


def write_skin_pad() -> str:
    """One sliding joint with a spring, on a fixed root."""
    root = "SkinPad"
    used = {"site"}
    looks = "".join(uk._material_usda(k, root, LOOKS[k]) for k in sorted(used))
    top = cylinder(PAD_HALF - 0.001, 0.003, center=(0.0, 0.0, -0.0015), segments=32)
    mesh = uk._mesh_usda("top", top, "site", root, LOOKS["site"], 40.0, indent="        ")
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
        float physics:staticFriction = 0.8
        float physics:dynamicFriction = 0.7
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

    def Xform "pad" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    )
    {{
        float physics:mass = 0.02
        float3 physics:diagonalInertia = (1e-5, 1e-5, 1e-5)
        point3f physics:centerOfMass = (0, 0, 0)
        bool physxRigidBody:disableGravity = 1
        float physxRigidBody:maxDepenetrationVelocity = 1
        double3 xformOp:translate = (0, 0, {PAD_TOP:.6f})
        quatd xformOp:orient = (1, 0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]

        def Cube "pad_c" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            rel material:binding:physics = </{root}/PhysicsMaterial>
            uniform token purpose = "guide"
            double size = 1
            float physxCollision:contactOffset = 0.003
            float physxCollision:restOffset = 0
            double3 xformOp:translate = (0, 0, -0.01)
            float3 xformOp:scale = ({2 * PAD_HALF:.6f}, {2 * PAD_HALF:.6f}, 0.02)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
{mesh}
    }}

    def PhysicsFixedJoint "world_joint"
    {{
        rel physics:body1 = </{root}/frame>
    }}

    def PhysicsPrismaticJoint "skin" (
        prepend apiSchemas = ["PhysicsDriveAPI:linear"]
    )
    {{
        uniform token drive:linear:physics:type = "force"
        float drive:linear:physics:stiffness = 0
        float drive:linear:physics:damping = 0
        rel physics:body0 = </{root}/frame>
        rel physics:body1 = </{root}/pad>
        uniform token physics:axis = "Z"
        point3f physics:localPos0 = (0, 0, {PAD_TOP:.6f})
        quatf physics:localRot0 = (1, 0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
        float physics:lowerLimit = {-PAD_TRAVEL:.5f}
        float physics:upperLimit = 0
    }}
}}
"""
    os.makedirs(ASSETS, exist_ok=True)
    path = ASSETS / "skin_pad.usda"
    path.write_text(text)
    return str(path)


def write_pen() -> str:
    """Origin at the pen's middle. The tip is 7 cm below it."""
    root = "Pen"
    half = PEN_L / 2
    w = PEN_W / 2
    barrel = lathe([(0.0, -half + PEN_NOSE[2]), (w - 0.001, -half + PEN_NOSE[2]), (w, -half + PEN_NOSE[2] + 0.002), (w, half - 0.03), (w, half - 0.028), (0.0, half - 0.028)], 28)
    band = lathe([(0.0, -0.02), (w + 0.0004, -0.02), (w + 0.0004, 0.012), (0.0, 0.012)], 28)
    nose = lathe([(0.0, -half), (0.0055, -half), (0.0065, -half + 0.002), (0.0075, -half + PEN_NOSE[2]), (0.0, -half + PEN_NOSE[2])], 20)
    top = lathe([(0.0, half - 0.03), (w + 0.0004, half - 0.03), (w + 0.0004, half - 0.004), (w - 0.003, half), (0.0, half)], 28)
    parts = [("barrel", barrel, "pen_body"), ("band", band, "pen_band"), ("nose", nose, "pen_nose"), ("top", top, "pen_top")]
    colliders = [
        collider("barrel_c", (PEN_W, PEN_W, PEN_L - PEN_NOSE[2]), (0.0, 0.0, PEN_NOSE[2] / 2), root),
        collider("nose_c", PEN_NOSE, (0.0, 0.0, -half + PEN_NOSE[2] / 2), root),
    ]
    return write_usda(str(ASSETS / "pen.usda"), root, parts, LOOKS, colliders, mass=PEN_MASS, friction=(0.9, 0.8), crease=40.0)


def write_holder() -> str:
    root = "Holder"
    sx, sy, h = HOLDER
    s = HOLDER_SOCKET
    o = uk._rounded_outline
    block = _rings([(o(sx, sy, 0.006, 4), 0.0), (o(sx, sy, 0.006, 4), h - 0.002), (o(sx - 0.003, sy - 0.003, 0.005, 4), h),
                    (o(s + 0.003, s + 0.003, 0.002, 4), h), (o(s, s, 0.001, 4), h - 0.002), (o(s, s, 0.001, 4), HOLDER_FLOOR)])
    side = (sx - s) / 2
    colliders = [
        collider("floor", (sx, sy, HOLDER_FLOOR), (0.0, 0.0, HOLDER_FLOOR / 2), root),
        collider("xp", (side, sy, h), ((s + side) / 2, 0.0, h / 2), root),
        collider("xn", (side, sy, h), (-(s + side) / 2, 0.0, h / 2), root),
        collider("yp", (s, side, h), (0.0, (s + side) / 2, h / 2), root),
        collider("yn", (s, side, h), (0.0, -(s + side) / 2, h / 2), root),
    ]
    return write_usda(str(ASSETS / "holder.usda"), root, [("block", block, "holder")], LOOKS, colliders, mass=1.0, friction=(0.5, 0.4))


def write_gauze() -> str:
    root = "Gauze"
    sx, sy, sz = GAUZE
    pad = box(GAUZE, bevel=0.002, round_xy=0.004)
    return write_usda(str(ASSETS / "gauze.usda"), root, [("pad", pad, "gauze")], LOOKS, [collider("pad_c", GAUZE, (0.0, 0.0, 0.0), root)], mass=GAUZE_MASS,
                      friction=(0.9, 0.8), crease=60.0)


def write_tray() -> str:
    root = "Tray"
    sx, sy, h = TRAY
    leg = box((sx, sy, h), center=(0.0, 0.0, h / 2), bevel=0.002)
    return write_usda(str(ASSETS / "tray.usda"), root, [("leg", leg, "tray")], LOOKS, [collider("leg_c", TRAY, (0.0, 0.0, h / 2), root)], mass=1.0, friction=(0.5, 0.4))


def write_bin() -> str:
    root = "Bin"
    sx, sy, h = BIN
    w = BIN_WALL
    o = uk._rounded_outline
    shell = _rings([(o(sx, sy, 0.012, 4), 0.0), (o(sx, sy, 0.012, 4), h - 0.022), (o(sx - 2 * w, sy - 2 * w, 0.010, 4), h - 0.022), (o(sx - 2 * w, sy - 2 * w, 0.010, 4), 0.004)])
    inside = prism(o(sx - 2 * w - 0.001, sy - 2 * w - 0.001, 0.009, 4), 0.0045, 0.006)
    lid = _band([(o(sx + 0.003, sy + 0.003, 0.013, 4), h - 0.022), (o(sx + 0.003, sy + 0.003, 0.013, 4), h - 0.002), (o(sx + 0.001, sy + 0.001, 0.012, 4), h),
                  (o(sx - 2 * w, sy - 2 * w, 0.010, 4), h), (o(sx - 2 * w, sy - 2 * w, 0.010, 4), h - 0.022)])
    colliders = [
        collider("floor", (sx, sy, 0.004), (0.0, 0.0, 0.002), root),
        collider("xp", (w, sy, h), (sx / 2 - w / 2, 0.0, h / 2), root),
        collider("xn", (w, sy, h), (-sx / 2 + w / 2, 0.0, h / 2), root),
        collider("yp", (sx, w, h), (0.0, sy / 2 - w / 2, h / 2), root),
        collider("yn", (sx, w, h), (0.0, -sy / 2 + w / 2, h / 2), root),
    ]
    return write_usda(str(ASSETS / "bin.usda"), root, [("shell", shell, "bin"), ("lid", lid, "bin_lid"), ("inside", inside, "bin_dark")], LOOKS, colliders, mass=1.0, friction=(0.5, 0.4), crease=45.0)


def write_all() -> None:
    write_arm()
    write_skin_pad()
    write_pen()
    write_holder()
    write_gauze()
    write_tray()
    write_bin()
