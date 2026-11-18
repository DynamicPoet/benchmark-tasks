"""On the station table, in front of the robot: a potted rose bush with five live, leafy canes and one dead cane, all leaning out
over the pot's rim. The dead cane stands between two live canes that lean the same way; it is green from the soil up to just above
a small red bud and brown above, up to an old pruning cut. Bypass secateurs with red handles lie open on the table to the right of
the pot, and an open black garden tub stands to the left.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 1200 --instruction "<the task.json instruction>"
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import scene_assets as sa  # noqa: E402
import scenekit  # noqa: E402
import steptime  # noqa: E402
from motion_checks import points_at_rest, finger_pad_points, ends_over_rim, cutting_held, clear_of_tub  # noqa: E402
from pruning_checks import PruningState  # noqa: E402
from cut_checks import section_contact, segment_blade_distance  # noqa: E402
from blade_contact import BladeContacts, target_load, array as native_array  # noqa: E402
from scenekit import UsdBody, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP, TCP_OFFSET  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation
from pose_motion import PoseWaypoint, install as install_pose_motion  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

sa.write_all()
install_pose_motion(__file__,lambda policy:_make_motion_controller(policy))

_bud_mesh = sa.bud_mesh()
BUD_VERTICES = _bud_mesh.v
BUD_EDGES = np.unique(np.sort(np.concatenate([_bud_mesh.f[:, [0, 1]], _bud_mesh.f[:, [1, 2]],
                                              _bud_mesh.f[:, [2, 0]]]), axis=1), axis=0)
BUD_EDGES = BUD_EDGES[np.linalg.norm(BUD_VERTICES[BUD_EDGES[:, 0]] - BUD_VERTICES[BUD_EDGES[:, 1]], axis=1) > 1e-8]
_BLADE_MESH = next(mesh for name, mesh, _ in sa._closed_half(1)[0] if name == "blade")
BLADE_HALF_T = 0.0014      # the blade's half thickness about BLADE_Z (scene_assets builds it 2.8 mm thick)
SHEAR_STRENGTH = 8.944e6   # Pa, measured fixture shear strength; engineering reference, not a measured secateur peak
F_CUT = SHEAR_STRENGTH * math.pi * sa.R0 ** 2  # 142.25 N for the natural 4.5 mm stem grade
F_NICK = 10.0              # N: a blade pressing a live cane this hard cuts into it
PRESS_STEPS = 2            # control steps the cutting force must hold
CONTACT_GAP = 0.0005       # finite blade/cane clearance; a native target-specific resisting load is also mandatory
STUB_RESTORE_DIST = 0.08   # the stub's collider returns to its cut face once the pivot is this far from it

CHAIN = ("c_px_joint", "c_py_joint", "c_pz_joint", "c_rx_joint", "c_ry_joint", "c_rz_joint")
SLIDES = ("stub_green_slide", "stub_brown_slide", "stub_col_slide", "stub_bud_slide", "cut_brown_slide", "cut_green_slide",
          "cut_bud_slide")
CHAIN_GAINS = {"c_p": (3.0e4, 100.0), "c_r": (50.0, 1.0)}  # stiffness, damping: they hold the cutting in place until the cut
BUSH = UsdBody("bush", usd="assets/bush.usda", pos=(sa.POT_XY[0], sa.POT_XY[1], TOP), size=(0.24, 0.24, 0.33), rot_xyzw=(0.,0.,math.sin(sa.BUSH_YAW/2),math.cos(sa.BUSH_YAW/2)), articulation=True,
               joint_pos=((".*", 0.0),),
               joint_groups=(("stub_.*", 200.0, 5.0e4), ("c_p.*", CHAIN_GAINS["c_p"][1], CHAIN_GAINS["c_p"][0]),
                             ("c_r.*", CHAIN_GAINS["c_r"][1], CHAIN_GAINS["c_r"][0]), ("cut_.*", 50.0, 2.0e4)),
               joint_armature=1.0e-5, friction=(0.9, 0.8), solver_iters=32)
SECATEURS = UsdBody("secateurs", usd="assets/secateurs.usda", pos=(sa.SECATEURS_START[0], sa.SECATEURS_START[1], TOP + sa.TOOL_HALF_T + 0.0005),
                    size=(0.20, 0.08, 0.02), rot_xyzw=(0.,0.,math.sin(math.radians(-63)/2),math.cos(math.radians(-63)/2)), articulation=True, joint_pos=(("pivot", 0.0), ("fuse", 0.0)),
                    joint_groups=(("pivot", 0.02, sa.SPRING_K), ("fuse", sa.FUSE_D, sa.FUSE_K)),
                    joint_armature=1.0e-5, friction=(1.0, 0.9), solver_iters=32)
BIN = UsdBody("bin", usd="assets/bin.usda", pos=(sa.BIN_XY[0], sa.BIN_XY[1], TOP), size=(0.27, 0.27, 0.12), mass=0.6, friction=(0.6, 0.5))
OBJECTS = [BUSH, SECATEURS, BIN]

GOOD_CUT = (0.003, 0.015)  # a good cut leaves the stub's top this far above the bud's top
FALL_DROP = 0.03           # a free cutting dropping this far, away from the bin's opening, has fallen
HOLD_REACH = 0.004         # held: both finger-pad middles of one gripper within this of the cutting's surface ...
HOLD_GAP_EXTRA = 0.006     # ... and the gripper closed to at most the cutting's width plus this
POT_MOVED, POT_TILT_DEG = 0.02, 10.0
CLEAR = 0.05               # every fingertip this far from the secateurs, the canes and the cutting at the end (from their surfaces)
TOOL_HALF_W = 0.008        # the secateurs' blades and handles, as lines this far from their surface
REST_MOVE, REST_STEPS = 0.005, 20  # the cutting moves less than this over the last second
BUD_ATTACH_TOL = 0.003     # the original bud's body stays within 3 mm of its original place on the pot
PAD_MID = np.array([0.0, 0.0, TCP_OFFSET[2] + PAD_TIP_PAST_TCP - 0.012])  # a finger pad's middle, in the gripper frame
PAD_TIP = np.array([0.0, 0.0, TCP_OFFSET[2] + PAD_TIP_PAST_TCP])

_S: dict = {}


def _clear() -> None:
    _S.clear()
    _S.update(ready=False, cut=False, h=None, green_cut=False, fell=False, pot_moved=False, tool_lost=False, in_jaw={}, closed=False, drop_ref=None,
              history=[], held=False, in_bin=False, over_bin=False, tool_down=False, clear=False, t=0,
              pruning=PruningState(sa.BUD_TOP_S), stub_height=sa.TOP_S, bud_attached=True, bud_cut_separately=False, cut_events=[], full_conditions=[], last_pivot=None,
              press=0, press_target=None, press_history=[], stub_gap=False, resist=0.0,
              contacts=None, blade_contacts=[], handle_contacts=[], target_loads={}, gripper_limits={},
              motion_requested=False,motion_reader=None,motion=None,demo_control={},
              held_at_parting=None, unheld_at_parting=False)


def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _rotation(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _body(asset, name: str, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    i = asset.body_names.index(name)
    pose = _np(asset.data.body_link_pose_w)[0, i]
    return pose[:3] - origin - np.array([0.0, 0.0, TOP]), _rotation(pose[3:7])


def _pads(env, origin: np.ndarray) -> dict[str, tuple[np.ndarray, float]]:
    """Each arm's two finger-pad middles and two pad tips (station frame, heights above the table), and its finger gap."""
    base = env.unwrapped
    out = {}
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        i = robot.body_names.index("gripper")
        pos = _np(robot.data.body_pos_w)[0, i] - origin - np.array([0.0, 0.0, TOP])
        rot = _rotation(_np(robot.data.body_quat_w)[0, i])
        q7,q8 = scenekit.arm_joints(env, side, ["joint7","joint8"])
        out[side] = finger_pad_points(pos,rot,q7,q8,PAD_MID,PAD_TIP)
    return out


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    d = b - a
    t = float(np.clip(np.dot(p - a, d) / max(np.dot(d, d), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * d)))


def _write_joints(asset, values: dict[str, float]) -> None:
    import torch

    ids = [asset.joint_names.index(n) for n in values]
    pos = torch.tensor([list(values.values())], device=asset.device, dtype=torch.float32)
    asset.write_joint_position_to_sim_index(position=pos, joint_ids=ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(pos), joint_ids=ids)
    asset.set_joint_position_target_index(target=pos, joint_ids=ids)


def _chain_gains(asset, on: bool) -> None:
    import torch

    for prefix, (k, c) in CHAIN_GAINS.items():
        ids = [i for i, n in enumerate(asset.joint_names) if n.startswith(prefix)]
        stiff = torch.full((1, len(ids)), k if on else 0.0, device=asset.device, dtype=torch.float32)
        damp = torch.full((1, len(ids)), c if on else 1.0e-4, device=asset.device, dtype=torch.float32)
        asset.write_joint_stiffness_to_sim_index(stiffness=stiff, joint_ids=ids)
        asset.write_joint_damping_to_sim_index(damping=damp, joint_ids=ids)


def _spring(tool) -> None:
    """The secateurs' spring, preloaded against the open stop."""
    import torch

    i = tool.joint_names.index("pivot")
    tool.set_joint_position_target_index(target=torch.tensor([[-sa.SPRING_PRELOAD]], device=tool.device, dtype=torch.float32), joint_ids=[i])


def _restore(env) -> None:
    import torch
    base = env.unwrapped
    bush = base.scene["bush"]
    _write_joints(bush, {n: 0.0 for n in CHAIN + SLIDES})
    _chain_gains(bush, True)
    _spring(base.scene["secateurs"])
    # Task-local real-rig envelope; do not rely on the harness's 50 N default.
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        ids = [robot.joint_names.index(n) for n in ("joint7", "joint8")]
        robot.write_joint_effort_limit_to_sim_index(
            limits=torch.full((1, 2), 40., device=robot.device), joint_ids=ids)
        actual = native_array(robot.root_physx_view.get_dof_max_forces())[0, ids]
        if not np.allclose(actual, 40., atol=1e-4, rtol=0):
            raise RuntimeError("The native per-finger effort cap is not 40 N")
        _S["gripper_limits"][side] = actual.tolist()
    folder = os.environ.get("TASK_SCORE_DIR")
    if folder:
        import json
        tool = base.scene["secateurs"]
        view = tool.root_physx_view
        raw = {name: native_array(getattr(view, name)()).tolist()
               for name in ("get_masses", "get_coms", "get_inertias")}
        raw["body_names"] = list(tool.body_names)
        raw["expected"] = sa.secateurs_mass_properties()
        masses = np.asarray(raw["get_masses"])[0]
        if abs(float(masses.sum()) - .160) > 5e-6:
            raise RuntimeError("Native tool mass differs from the declared 160 g model")
        os.makedirs(folder, exist_ok=True)
        pathlib.Path(folder, "native-tool-properties.json").write_text(json.dumps(raw, indent=2) + "\n")
    _S["contacts"] = BladeContacts(base)


def _retained_plant(bush, origin: np.ndarray) -> tuple[float, bool]:
    """The visible stub's length and the original bud's attachment, from actual link poses."""
    pot, pot_rot = _body(bush, "pot", origin)
    base = pot + pot_rot @ sa.on_cane(0.0)
    axis = pot_rot @ sa.CANE_DIR
    ends = []
    for name, length in (("stub_green", sa.L_SG), ("stub_brown", sa.L_SB)):
        pos, rot = _body(bush, name, origin)
        ends.append(float(axis @ (pos + rot @ np.array([0.0, 0.0, length]) - base)))
    bud, _ = _body(bush, "stub_bud", origin)
    expected = pot + pot_rot @ sa.on_cane(sa.BUD_S)
    height = max(0.006, *ends)
    attached = float(np.linalg.norm(bud - expected)) <= BUD_ATTACH_TOL and height >= sa.BUD_TOP_S
    return height, attached


def _canes(bush, origin: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    """All remaining cut targets, including the retained stub and its original bud."""
    pos, rot = _body(bush, "pot", origin)
    out = []
    if not _S["cut"]:
        out.append(("dead", pos + rot @ sa.on_cane(0.0), pos + rot @ sa.on_cane(sa.TOP_S), 0.0))
    else:
        height, _ = _retained_plant(bush, origin)
        height = min(height, _S["pruning"].stub_length)
        out.append(("stub", pos + rot @ sa.on_cane(0.0), pos + rot @ sa.on_cane(height), 0.0))
    if not _S["pruning"].bud_damaged:
        bud, bud_rot = _body(bush, "stub_bud", origin)
        vertices = bud + BUD_VERTICES @ bud_rot.T
        # A blade plane intersects the actual bud mesh, regardless of its approach direction.
        for k, (a, b) in enumerate(BUD_EDGES):
            out.append((f"bud{k}", vertices[a], vertices[b], 0.0))
    for k, (b, t, _r) in enumerate(sa.GREEN_CANES):
        base = pos + rot @ np.array([b[0], b[1], sa.SOIL_Z])
        out.append((f"green{k}", base, pos + rot @ np.asarray(t, float), sa.GREEN_CUT_FROM - sa.SOIL_Z))
    return out


def _blade_frame(point, q):
    """A hook-frame plane point in the blade mesh's authored closed frame."""
    angle = q - sa.HALF
    c, sn = math.cos(angle), math.sin(angle)
    return np.array([c * point[0] + sn * point[1], -sn * point[0] + c * point[1]])


def _tool_joint(tool, name: str) -> float:
    return float(_np(tool.data.joint_pos)[0, tool.joint_names.index(name)]) if name in tool.joint_names else 0.0


def _gauge(tool) -> tuple[float, float, float]:
    """(handle angle, blade angle, torque the canes put on the blade against closing). The blade sits on its handle through
    a stiff spring: when a cane in the jaws resists, the blade lags its handle and the spring carries the torque."""
    q = _tool_joint(tool, "pivot")
    q_fuse = _tool_joint(tool, "fuse")
    return q, q + q_fuse, sa.FUSE_K * max(0.0, -q_fuse)


def _jaw_point(a: np.ndarray, b: np.ndarray, o: np.ndarray, r: np.ndarray, s_min: float = 0.0):
    """Where a cane meets the blade's plane, in the hook frame, and how far along the cane (None if it does not)."""
    n = r[:, 2]
    d = b - a
    length = float(np.linalg.norm(d))
    d = d / length
    if abs(float(n @ d)) < 0.3:
        # Nearly in the blade's plane: take the point of the cane nearest that plane's jaw region.
        s = np.linspace(s_min, length, 200)
        local = (a + s[:, None] * d - o) @ r
        k = int(np.argmin(np.abs(local[:, 2] - sa.BLADE_Z) + 0.2 * np.abs(np.hypot(local[:, 0], local[:, 1]) - 0.025)))
        return local[k], float(s[k]), abs(float(n @ d))
    s = (sa.BLADE_Z - float(n @ (a - o))) / float(n @ d)
    if not s_min <= s <= length:
        return None
    return r.T @ (a + s * d - o), float(s), abs(float(n @ d))


def _in_gap(local: np.ndarray, q_blade: float, radius: float) -> bool:
    """A cane's section centre lies between the blade's and the hook's edges, within the jaws' reach."""
    rho = math.hypot(local[0], local[1])
    if not sa.JAW_REACH[0] <= rho <= sa.JAW_REACH[1]:
        return False
    phi_mid = math.atan2(-local[0], local[1]) - q_blade / 2
    return abs(phi_mid) <= (sa.JAW_OPEN - q_blade) / 2 + math.asin(min(1.0, radius / rho)) * 0.5 + 0.03


def _blade_distance(a: np.ndarray, b: np.ndarray, o: np.ndarray, r: np.ndarray, q_blade: float, radius: float) -> float:
    """How far a cane's surface is from the blade's metal (negative if it reaches into it), from any approach."""
    ends = []
    for p in (a, b):
        local = r.T @ (p - o)
        ends.append(np.array([*_blade_frame(local, q_blade), local[2]]))
    return segment_blade_distance(ends[0], ends[1], _BLADE_MESH.v, _BLADE_MESH.f, sa.BLADE_Z, BLADE_HALF_T, radius + 0.01) - radius


def _check_cut(env, origin: np.ndarray) -> None:
    """The jaws collide with every cane. A cane in the jaws resists their closing; the blade's spring measures that force.
    The dead cane (or the stub) parts at the blade's plane once the blade presses it with its cutting force for two control
    steps; a live cane in the jaws counts as cut once the blade presses it with F_NICK. Never from remembered jaw entry."""
    base = env.unwrapped
    tool, bush = base.scene["secateurs"], base.scene["bush"]
    o, r = _body(tool, "hook_half", origin)
    q, q_blade, resist = _gauge(tool)
    _S["blade_contacts"] = _S["contacts"].sample()
    _S["handle_contacts"] = _S["contacts"].sample_handles()
    _S["target_loads"] = {}
    _S["closed"] = q_blade >= sa.JAW_OPEN - 0.01
    closing = _S["last_pivot"] is not None and q_blade > _S["last_pivot"] + 1e-6
    n = r[:, 2]
    bud_section = []
    pressed = None
    for name, a, b, s_min in _canes(bush, origin):
        if name.startswith("bud"):
            d = (b - a) / max(float(np.linalg.norm(b - a)), 1e-12)
            if abs(float(n @ d)) < 1e-8:
                continue
            s = (sa.BLADE_Z - float(n @ (a - o))) / float(n @ d)
            if 0.0 <= s <= float(np.linalg.norm(b - a)):
                bud_section.append(_blade_frame(r.T @ (a + s * d - o), q_blade))
            continue
        hit = _jaw_point(a, b, o, r, 0.0 if name.startswith("green") else s_min)
        if hit is None:
            continue
        local, s, cos_angle = hit
        if name.startswith("green"):
            radius = sa.GREEN_CANES[int(name[5:])][2]
        else:
            radius = sa.R1 if sa.BROWN_S <= s < sa.T_LOW_S else sa.R0
        rho = max(math.hypot(local[0], local[1]), sa.JAW_REACH[0])
        # A blunt spine obstruction is not a cut: the current material must be
        # in the actual blade/hook aperture, never an earlier remembered entry.
        if resist <= 0.0 or not _in_gap(local, q_blade, radius):
            continue
        if _blade_distance(a, b, o, r, q_blade, radius) > CONTACT_GAP:
            continue
        # The spring may also be loaded by a pot, a table or a neighbouring cane.
        # Only the intended blade/plant pair's actual point forces can cut it.
        shift = origin + np.array([0., 0., TOP])
        bodies = {"pot"} if name.startswith("green") else {"stub_col", "cut_green", "cut_brown", "cut_top"}
        load = target_load(_S["blade_contacts"], a + shift, b + shift, radius,
                           o + shift, r[:, 2], bodies)
        _S["target_loads"][name] = load
        lever = load["effective_lever_m"]
        force = min(load["normal_load_N"], resist / lever) if lever else 0.0
        if name.startswith("green"):
            if force >= F_NICK:
                _S["green_cut"] = True
                _S["cut_events"].append({"target": name, "height": None, "t": _S["t"] / 20.0, "contact": "blade_force",
                                         "force": force, "q": q_blade})
        elif force >= F_CUT:
            pressed = (name, s, local, cos_angle, force)
    if pressed:
        name, s, local, cos_angle, force = pressed
        if _S["press_target"] != name:
            _S["press"] = 0
            _S["press_history"] = []
        _S["press_target"] = name
        _S["press"] += 1
        _S["press_history"] = (_S["press_history"] + [{"t": _S["t"] / 20., "target": name,
            "blade_load_N": force, "gauge_moment_Nm": resist, "native": _S["target_loads"][name]}])[-PRESS_STEPS:]
    else:
        _S["press"] = 0
        _S["press_target"] = None
        _S["press_history"] = []
    if pressed and _S["press"] >= PRESS_STEPS:
        name, s, local, cos_angle, force = pressed
        gap = sa.kerf(cos_angle)
        _S["cut_events"].append({"target": name, "height": float(s), "t": _S["t"] / 20.0, "contact": "blade_force",
                                 "force": force, "point": (o + r @ local).tolist(), "q": q_blade, "kerf": gap,
                                 "native_target_load": _S["target_loads"][name],
                                 "sustained_load_samples": _S["press_history"].copy()})
        _S["pruning"].cut(s)
        bud_ok = not _S["pruning"].bud_damaged
        if name == "dead":
            values = sa.cut_joint_positions(s, gap)
            _S["h"] = min(max(s, sa.H_MIN_S), sa.T_LOW_S - 0.003)
            _S["cut"] = True
            values.update(sa.stub_joint_positions(s, bud_ok, gap))
            if _S["bud_cut_separately"]:
                values["cut_bud_slide"] = 0.0
            _write_joints(bush, values)
            _chain_gains(bush, False)
        else:
            _write_joints(bush, sa.stub_joint_positions(_S["pruning"].stub_length, bud_ok, gap))
        _S["stub_gap"] = True
        _S["press"] = 0
    if closing and not _S["pruning"].bud_damaged:
        contact = section_contact(bud_section, _BLADE_MESH.v, _BLADE_MESH.f, sa.JAW_REACH)
        if contact is not None:
            angle = q_blade - sa.HALF
            c, sn = math.cos(angle), math.sin(angle)
            local = np.array([c * contact[0] - sn * contact[1], sn * contact[0] + c * contact[1], sa.BLADE_Z])
            _S["cut_events"].append({"target": "bud", "height": None, "t": _S["t"] / 20.0,
                                     "contact": "blade_mesh_section", "point": (o + r @ local).tolist(), "q": q_blade})
            _S["pruning"].bud_damaged = True
            _S["bud_cut_separately"] = True
            _write_joints(bush, {"stub_bud_slide": sa.BUD_SINK})
    if _S["stub_gap"] and _S["cut"]:
        # Once the jaws are well away, the stub's collider reaches its visible cut face again.
        pot, pot_rot = _body(bush, "pot", origin)
        top = pot + pot_rot @ sa.on_cane(_S["pruning"].stub_length)
        if float(np.linalg.norm(o - top)) > STUB_RESTORE_DIST:
            _write_joints(bush, {"stub_col_slide": sa.stub_joint_positions(_S["pruning"].stub_length, True)["stub_col_slide"]})
            _S["stub_gap"] = False
    _S["resist"] = resist
    _S["last_pivot"] = q_blade


def _cutting(bush, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos, rot = _body(bush, "cut_top", origin)
    return pos + rot @ np.array([0.0, 0.0, _S["h"] - sa.T_LOW_S]), pos + rot @ np.array([0.0, 0.0, sa.L_T])


def _tool_points(tool, origin: np.ndarray) -> np.ndarray:
    """Points along each half of the secateurs: the pivot, the jaw's tip, the handle's middle and its end (station frame)."""
    pts = []
    for half, side in (("hook_half", -1.0), ("blade_half", 1.0)):
        o, r = _body(tool, half, origin)
        angle = -side * sa.HALF  # each half's turn from closed to open, built into its link
        c, s = math.cos(angle), math.sin(angle)
        for x, y in ((0.0, 0.0), (side * 0.004, 0.05), (-side * sa.HANDLE_X, -0.05), (-side * sa.HANDLE_X, -0.12)):
            pts.append(o + r @ np.array([c * x - s * y, s * x + c * y, 0.0]))
    return np.array(pts)


def _measure(env) -> None:
    base = env.unwrapped
    if not _S.get("ready"):
        _clear()
        _restore(env)
        _S["ready"] = True
    origin = _np(base.scene.env_origins[0])
    bush, tool, tub = base.scene["bush"], base.scene["secateurs"], base.scene["bin"]
    _S["t"] += 1
    steptime.tick()

    pot, pot_rot = _body(bush, "pot", origin)
    moved = float(np.linalg.norm(pot - np.array([sa.POT_XY[0], sa.POT_XY[1], 0.0])))
    tilt = math.degrees(math.acos(float(np.clip(pot_rot[2, 2], -1.0, 1.0))))
    _S["pot_moved"] = _S["pot_moved"] or moved > POT_MOVED or tilt > POT_TILT_DEG
    tool_o, tool_r = _body(tool, "hook_half", origin)
    _S["tool_lost"] = _S["tool_lost"] or tool_o[2] < -0.05 or abs(tool_o[0] - 0.36) > 0.55 or abs(tool_o[1]) > 0.65

    was_cut = _S["cut"]
    _check_cut(env, origin)
    stub_height, bud_attached = _retained_plant(bush, origin)
    _S["pruning"].observe_bud(bud_attached)
    _S.update(stub_height=stub_height, bud_attached=bud_attached)
    pads = _pads(env, origin)
    tub_pose = _np(tub.data.root_link_pose_w)[0]
    tub_pos = tub_pose[:3] - origin - np.array([0.0, 0.0, TOP])
    tub_rot = _rotation(tub_pose[3:7])
    tub_upright = tub_rot[2, 2] > math.cos(math.radians(30.0))  # a tipped tub's opening faces sideways
    tool_pts = _tool_points(tool, origin)

    cane_segments = []
    stub_top = stub_height if _S["cut"] else sa.TOP_S
    cane_segments.append((pot + pot_rot @ sa.on_cane(0.0), pot + pot_rot @ sa.on_cane(stub_top), sa.R0))
    for b, t, r in sa.GREEN_CANES:
        cane_segments.append((pot + pot_rot @ np.array([b[0], b[1], sa.SOIL_Z]), pot + pot_rot @ np.asarray(t, float), r))
    p0, p1, p2, p3 = tool_pts[0], tool_pts[1], tool_pts[2], tool_pts[3]  # pivot, jaw, handle middle, handle end (hook half)
    q0, q1, q2, q3 = tool_pts[4], tool_pts[5], tool_pts[6], tool_pts[7]  # ... and the blade half
    tool_segments = [(p0, p1, TOOL_HALF_W), (p0, p2, TOOL_HALF_W), (p2, p3, TOOL_HALF_W),
                     (q0, q1, TOOL_HALF_W), (q0, q2, TOOL_HALF_W), (q2, q3, TOOL_HALF_W)]

    near = []
    if _S["cut"]:
        bottom, top = _cutting(bush, origin)
        held, hold_detail = cutting_held(pads, bottom, top, sa.R0, HOLD_REACH, HOLD_GAP_EXTRA)
        if not was_cut:
            # The cutting parts from the stem at this step: a gripper must already hold it, or it has fallen.
            _S["held_at_parting"] = {"t": _S["t"] / 20.0, "held": held, "grippers": hold_detail}
            for event in reversed(_S["cut_events"]):
                if event["target"] == "dead":
                    event["held_at_parting"] = _S["held_at_parting"]
                    break
            if not held:
                _S["unheld_at_parting"] = True
                _S["fell"] = True
        mid = (bottom + top) / 2
        inner = sa.BIN_R_TOP - sa.BIN_WALL
        over_bin = tub_upright and ends_over_rim([mid], tub_pos, tub_rot, sa.BIN_H, inner)
        ends_over = tub_upright and ends_over_rim([bottom, top], tub_pos, tub_rot, sa.BIN_H, inner)
        low = min(bottom[2], top[2])
        in_bin = (not held) and ends_over and low < tub_pos[2] + sa.BIN_H
        ends = np.array([bottom[2], top[2]])
        if held or over_bin:
            _S["drop_ref"] = None
        else:
            ref = ends if _S["drop_ref"] is None else np.maximum(_S["drop_ref"], ends)
            _S["drop_ref"] = ref
            if float(np.max(ref - ends)) >= FALL_DROP:
                _S["fell"] = True
        _S["history"] = (_S["history"] + [np.concatenate([bottom, top])])[-(REST_STEPS + 1):]
        _S.update(held=held, in_bin=in_bin, over_bin=over_bin)
        near.append((bottom, top, sa.R0))

    tips = np.vstack([pts for pts, _ in pads.values()])
    gap = float("inf")
    for p in tips:
        for a, b, radius in cane_segments + near + tool_segments:
            gap = min(gap, _seg_dist(p, a, b) - radius)
    tilt_tool = math.degrees(math.acos(float(np.clip(abs(tool_r[2, 2]), -1.0, 1.0))))
    off_pot = all(math.hypot(p[0] - pot[0], p[1] - pot[1]) > sa.POT_R_TOP + 0.005 for p in tool_pts)
    off_tub = clear_of_tub(tool_pts, tub_pos, tub_rot, sa.BIN_H, sa.BIN_R_TOP + 0.005)  # not in, on or over the tub
    _S["tool_down"] = tool_o[2] < sa.TOOL_HALF_T + 0.012 and tilt_tool < 20.0 and off_pot and off_tub
    _S["clear"] = gap >= CLEAR
    _record_full_conditions()
    if _S.get('motion_requested'):
        if _S['motion_reader'] is None:
            from pruning_observation import MotionReader
            _S['motion_reader']=MotionReader(base,sa)
        if _S['cut']:
            a,b=_cutting(bush,origin)
        else:
            current,current_R=_body(bush,'cut_top',origin)
            a=current+current_R@np.array([0.,0.,-sa.T_LOW_S])
            b=current+current_R@np.array([0.,0.,sa.L_T])
        _S['motion']=_S['motion_reader'].sample(origin,a+[0,0,TOP],b+[0,0,TOP],
                                              _S['handle_contacts'],_S['cut'])
    _log(tool, bush, tool_o, tool_r, pot, pads, gap, env)


def _log(tool, bush, tool_o, tool_r, pot, pads, gap, env) -> None:
    """With TASK_SCORE_DIR set (the run scripts set it), one line of state every 5 control steps, for debugging."""
    folder = os.environ.get("TASK_SCORE_DIR")
    if not folder or _S["t"] % 5:
        return
    import json

    os.makedirs(folder, exist_ok=True)
    q = float(_np(tool.data.joint_pos)[0, tool.joint_names.index("pivot")])
    row = {"t": _S["t"] / 20.0, "jaw": round(q, 4), "tool": [round(float(v), 4) for v in tool_o], "tool_up": round(float(tool_r[2, 2]), 3),
           "pot": [round(float(v), 4) for v in pot], "gap": round(float(gap), 3),
           "grips": {k: round(g, 4) for k, (_, g) in pads.items()}, "tcp": {k: [round(float(v), 4) for v in p[:2].mean(axis=0)] for k, (p, _) in pads.items()},
           **{k: bool(_S[k]) for k in ("cut", "green_cut", "fell", "pot_moved", "tool_lost", "held", "in_bin", "over_bin", "tool_down", "clear")}}
    if _S["h"] is not None:
        row["h"] = round(_S["h"], 4)
        row["cutting"] = [round(float(v), 4) for v in _S["history"][-1]] if _S["history"] else None
    row.update(fuse=_tool_joint(tool, "fuse"), resisting_torque_Nm=_S["resist"], press_samples=_S["press"],
               native_blade_contacts=_S["blade_contacts"], native_finger_tool_contacts=_S["handle_contacts"],
               target_loads=_S["target_loads"], native_gripper_effort_limits_N=_S["gripper_limits"],
               stub_collider_retracted=_S["stub_gap"], held_at_parting=_S["held_at_parting"],
               unheld_at_parting=bool(_S["unheld_at_parting"]), tub_pose=[round(float(v), 5) for v in
               _np(env.unwrapped.scene["bin"].data.root_link_pose_w)[0]] if env is not None else None,
               stub_height=round(_S["stub_height"], 4), bud_attached=bool(_S["bud_attached"]),
               bud_damaged=bool(_S["pruning"].bud_damaged), cuts=_S["pruning"].cuts,
               first_cut=_S["pruning"].first_cut, cut_events=_S["cut_events"],
               full_condition_samples=len(_S["full_conditions"]), full_condition_good=sum(_S["full_conditions"]),
               tool_yaw=round(math.atan2(tool_r[1, 0], tool_r[0, 0]), 4), tool_rotation_world=tool_r.tolist(),
               actual_pad_points={k: p.tolist() for k, (p, _g) in pads.items()})
    row['demo_control']=_S.get('demo_control',{})
    row['motion_observation']=_S.get('motion')
    if env is not None:
        row['actual_robot_joints'] = {side: dict(zip(env.unwrapped.scene[f'{side}_robot'].joint_names,
            _np(env.unwrapped.scene[f'{side}_robot'].data.joint_pos)[0].tolist())) for side in ('left','right')}
    joints = _np(bush.data.joint_pos)[0]
    row["sleeve_joints"] = {name: round(float(joints[bush.joint_names.index(name)]), 6) for name in SLIDES}
    with open(os.path.join(folder, "state.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


@on_reset
def _on_reset(env) -> None:
    _clear()
    _restore(env)
    _S["ready"] = True


@on_step
def _on_step(env) -> None:
    _measure(env)


def _at_rest() -> bool:
    return points_at_rest(_S["history"], REST_MOVE, REST_STEPS)


def good_cut() -> bool:
    return _S["pruning"].credit(_S["stub_height"], _S["bud_attached"], GOOD_CUT) == 1.0


def _full_conditions_now() -> bool:
    """Instantaneous necessary conditions; motion uses the same final one-second window separately."""
    return bool(_S and _S["cut"] and not any(_S[k] for k in ("green_cut", "fell", "pot_moved", "tool_lost"))
                and good_cut() and _S["in_bin"] and _S["tool_down"] and _S["clear"])


def _record_full_conditions() -> None:
    """Called once by the control-step measurement hook, never by a score query."""
    _S["full_conditions"] = (_S["full_conditions"] + [_full_conditions_now()])[-(REST_STEPS + 1):]


def score(objects) -> float:
    """0 for a cut or detached original bud, a live cane cut, a fall (including a cutting no gripper holds as it parts), a moved
    pot, lost secateurs, or no cut. Otherwise 0.25 if the first cut or final stub is misplaced; 0.5 if both are good but the
    cutting is not in the bin (both ends over its opening, lowest point below the rim); 0.75 if it is in the bin; 1.0 also
    requires the secateurs down on the table clear of the pot and the tub, fingertips clear, and the cutting still."""
    if not _S or _S["green_cut"] or _S["fell"] or _S["pot_moved"] or _S["tool_lost"] or not _S["cut"]:
        return 0.0
    plant_credit = _S["pruning"].credit(_S["stub_height"], _S["bud_attached"], GOOD_CUT)
    if plant_credit < 1.0:
        return plant_credit
    if not _S["in_bin"]:
        return 0.5
    complete_window = len(_S["full_conditions"]) == REST_STEPS + 1 and all(_S["full_conditions"])
    return 1.0 if complete_window and _full_conditions_now() and _at_rest() else 0.75


def success(objects) -> bool:
    return score(objects) == 1.0


# -- the scripted demo -------------------------------------------------------------------------------------------------

YAW_RIGHT = math.pi / 2  # fingers along x, across the secateurs' handles
YAW_LEFT = -math.pi / 2  # fingers along x too, across the leaning cane
TCP_OVER_TOOL = 0.004    # side-face alignment; actual finite finger hulls clear the pickup table by 1.16 mm
GRIP_BACK = 0.090        # offline placement reference; actual post-opening hand/tool geometry is measured
PICKUP_BACK = 0.079      # ordinary pickup position, distinct from the later measured working grasp
CUT_RADIUS = 0.0246       # finite staggered-jaw entry, verified against the inclined material
TOOL_SAG = 0.002         # level grasp geometry: measured TCP-to-root 5.9 mm minus the 4 mm pickup alignment
CARRY_PRECLOSE = 0.0     # entry at the real open stop; carrying uses the real closed stop
ENTRY_CENTRE_X = 0.0024998164  # load the blade before the real handle stop, within the open throat
CUT_AT = sa.BUD_TOP_S + 0.009   # along the cane: 9 mm above the bud
HOLD_AT = 0.275          # the left hand holds the cane here, near its tip
PINCH_BELOW_TCP = 0.006  # the cane sits this far below the left TCP, between the pads
LEFT_OPEN = -0.025       # 50 mm opening clears the target and the surrounding live canes
DRAW_OUT = 0.025         # after the cut, the left hand first draws the cutting this far out along the cane
OUT_OF_BUSH = (0.30, 0.30, TOP + 0.20)  # ... then carries it out past the live canes' tips, toward the bin
LEFT_CLEAR = (0.30, 0.30, TOP + 0.20)
RIGHT_CLEAR = (0.14, -0.305, TOP + 0.16)


def _handle_outer_x(y_tool: float, closure: float = 0.) -> float:
    """Outer handle face for a symmetric planned squeeze (actual contacts are logged)."""
    angle = sa.HALF - closure / 2
    xo, c, s = sa.HANDLE_X + sa.HANDLE_W / 2, math.cos(angle), math.sin(angle)
    y_c = (y_tool - xo * s) / c
    return xo * c - y_c * s


def _cane_point(s: float) -> np.ndarray:
    """A point on the dead cane's axis, s along it from the soil, in the station frame (height above the table)."""
    return sa.BUSH_R @ sa.on_cane(s) + np.array([sa.POT_XY[0], sa.POT_XY[1], 0.0])


def _live_cane_point(cane, height: float) -> np.ndarray:
    """The point on a live cane's axis at a height above the table, in the station frame."""
    s = (height - sa.SOIL_Z) / cane.direction[2]
    return sa.BUSH_R @ cane.at(s) + np.array([sa.POT_XY[0], sa.POT_XY[1], 0.0])


def demo() -> dict[str, list[Waypoint]]:
    """Ordinary position demonstrations; failure releases the cutting outside the bin."""
    mode = os.environ.get("TASK_DEMO", "success")
    cut = _cane_point(sa.BROWN_S + 0.004 if mode == "repair" else sa.BROWN_S + 0.011 if mode == "brown" else CUT_AT)
    if mode == "green":  # careless aim: the jaws go round the live cane beside the dead one, at the same height
        cut = _live_cane_point(sa.LIVE_CANES[0], cut[2])
    held = _cane_point(HOLD_AT)
    px, py = sa.SECATEURS_START
    grip_y = py - GRIP_BACK
    hold = FINGER_OPEN  # spring opens the real jaws; the actual finger/tool support is logged
    z_table = TOP + sa.TOOL_HALF_T + 0.0005 + TCP_OVER_TOOL
    z_cut = TOP + cut[2] - sa.BLADE_Z + TCP_OVER_TOOL + TOOL_SAG
    cut_x, cut_y = cut[0], cut[1] - (CUT_RADIUS + 0.008 if mode == "green" else CUT_RADIUS) - GRIP_BACK
    left_z = TOP + held[2] + PINCH_BELOW_TCP
    left_R = sa.BUSH_R @ sa._rx(math.radians(-35.)) @ grip_rotation(YAW_LEFT)
    down = (0.20, -0.18)
    # The near-vertical blade plane is aligned with the leaning stem.
    # Hanging the tool head down keeps gravity chiefly along the handles.
    heading = math.radians(27.)
    c, sn = math.cos(heading), math.sin(heading)
    yaw_R = np.array([[c,-sn,0.],[sn,c,0.],[0.,0.,1.]])
    initial_tool_R = yaw_R @ np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
    hanging_R = np.column_stack((-sa.BUSH_R[:,0],np.array([0.,0.,-1.]),-sa.BUSH_R[:,1]))
    cut_tool_R = hanging_R @ sa._rx(math.radians(13.91310653875023))
    pickup_R = grip_rotation(heading)
    cut_R = cut_tool_R @ np.diag([1.,-1.,-1.])
    offset = np.array([0., -GRIP_BACK, TCP_OVER_TOOL + TOOL_SAG])
    def work_tcp(point):
        pivot = np.asarray(point) + np.array([0.,0.,TOP]) - cut_tool_R @ np.array([ENTRY_CENTRE_X,CUT_RADIUS,sa.BLADE_Z])
        return pivot + cut_tool_R @ offset
    target = work_tcp(cut)
    # Enter through the physical mouth without climbing into the adjacent cane.
    # The counterblade curls inward at R37; follow its open throat to the working section.
    entry_offset = -cut_tool_R@np.array([.0024998164-ENTRY_CENTRE_X,.066-CUT_RADIUS,0.])
    pre = target + entry_offset
    mouth_offset = -cut_tool_R@np.array([.004231304959-ENTRY_CENTRE_X,.037-CUT_RADIUS,0.])
    mouth = target + mouth_offset
    pickup = np.array([px,py,z_table]) - PICKUP_BACK*initial_tool_R[:,1]
    lift = pickup + np.array([0.,0.,TOP+.16-z_table])
    # Open at the clear high station, then descend to the mouth approach.
    free = pre.copy()
    def pose(dt,p,f,label,R,q):
        return PoseWaypoint(dt,tuple(p),f,label,0.,rotation=R,nominal_jaw=q)
    # A closed-tool transit may change the wrist branch in free space. Its
    # reference is an ordinary smooth joint motion, not a constrained TCP arc.
    from yam_arena import ik, station
    def endpoint(p,R,side,seed):
        q,pe,re=ik.solve(p,ik.matrix_to_quat_xyzw(R),seed,base_pos=station.BASES[side],
                         damping=.01,iters=350,tol=2e-6)
        if pe>.0001 or re>math.radians(.1):
            raise ValueError(f"Unreachable {side} transit endpoint: {pe*1000:.3f} mm, {math.degrees(re):.3f} deg")
        return q
    q_lift=endpoint(lift,pickup_R,'right',[.76,1.28,1.18,-1.47,0.,.29])
    q_free=endpoint(free,cut_R,'right',[0.623925477903757, 2.2370938558943103, 2.3455501795263776, -1.4358360752479864, -1.320438281472247, 1.821729228154476])
    def transit(dt,p,f,label,R,jaw,q0,q1):
        w=pose(dt,p,f,label,R,jaw)
        w.joint_start=np.asarray(q0);w.joint_goal=np.asarray(q1)
        return w
    right = [
        pose(1.5, pickup+[0,0,.070],FINGER_OPEN,"over the secateurs",pickup_R,0.),
        pose(1.2, pickup,FINGER_OPEN,"round the handles",pickup_R,0.),
        pose(.9,pickup,FINGER_CLOSED,"close for carrying",pickup_R,sa.CLOSE),
        pose(1.3,lift,FINGER_CLOSED,"lift closed",pickup_R,sa.CLOSE),
        transit(4.5,free,FINGER_CLOSED,"turn the closed secateurs",cut_R,sa.CLOSE,q_lift,q_free),
        pose(1.5,free,-.021,"recenter vertically",cut_R,.45),
        pose(.55,free,hold,"open beside the cane",cut_R,CARRY_PRECLOSE),
        pose(1.6,pre,hold,"beside the held cane",cut_R,CARRY_PRECLOSE),
        pose(1.2,mouth,hold,"follow the open hook throat",cut_R,CARRY_PRECLOSE),
        pose(.65,target,hold,"jaws round it, above the bud",cut_R,CARRY_PRECLOSE),
    ]
    t_ready = sum(w.duration for w in right)
    left: list[Waypoint] = []
    from yam_arena.constants import READY
    pre_left=np.r_[held[:2],left_z]+sa.BUSH_R@np.array([0.,-.070,0.])
    q_pre_left=endpoint(pre_left,left_R,'left',[-.49,1.38,1.37,-1.01,.29,.44])
    via_left=np.array([.18,.28,.98])
    q_via_left=endpoint(via_left,left_R,'left',[.42,1.10,1.24,-1.615,.60,1.40])
    left = [
        transit(1.5,via_left,LEFT_OPEN,"clear of the near neighbour",left_R,0.,READY,q_via_left),
        transit(1.5,pre_left,LEFT_OPEN,"from the robot side",left_R,0.,q_via_left,q_pre_left),
        Waypoint(1.5, (held[0], held[1], left_z), LEFT_OPEN, "round the cane near its tip", YAW_LEFT),
        Waypoint(0.8, (held[0], held[1], left_z), FINGER_CLOSED, "hold it", YAW_LEFT),
    ]
    left = [PoseWaypoint(w.duration,w.pos,w.finger,w.label,w.yaw,rotation=left_R) for w in left]
    t_held = sum(w.duration for w in left)
    t_cut0 = max(t_ready, t_held)
    if t_cut0>t_ready:
        right.append(pose(t_cut0-t_ready,target,hold,"wait for the holding hand",cut_R,CARRY_PRECLOSE))
    right += [pose(2.0,target,FINGER_CLOSED,"squeeze: cut",cut_R,sa.CLOSE),
              pose(.8,target,FINGER_CLOSED,"hold",cut_R,sa.CLOSE)]
    t_open = sum(w.duration for w in right)
    if mode in ("recut", "side_bud", "retrim", "repair"):
        later_s = {"recut": sa.BUD_S - 0.005, "side_bud": sa.BUD_S - 0.007, "retrim": sa.BUD_TOP_S + 0.003, "repair": sa.BUD_TOP_S + 0.006}[mode]
        later = _cane_point(later_s)
        later[0] -= .010 if mode == "side_bud" else 0.
        later_target = work_tcp(later)
        later_pre = later_target + entry_offset
        right += [pose(1.2,pre,FINGER_CLOSED,"back out of the first cut",cut_R,sa.CLOSE),
                  pose(1.,later_pre,FINGER_CLOSED,"beside the retained stub",cut_R,sa.CLOSE),
                  pose(.55,later_pre,hold,"open by the retained plant",cut_R,CARRY_PRECLOSE),
                  pose(1.2,later_target+mouth_offset,hold,"follow the retained hook throat",cut_R,CARRY_PRECLOSE),
                  pose(.65,later_target,hold,"jaws round the retained plant",cut_R,CARRY_PRECLOSE),
                  pose(.7,later_target,FINGER_CLOSED,"second cut",cut_R,sa.CLOSE),
                  pose(.6,later_target,FINGER_CLOSED,"hold",cut_R,sa.CLOSE)]
        target,pre = later_target,later_pre
    right += [
        pose(1.2,pre,FINGER_CLOSED,"back out closed",cut_R,sa.CLOSE),
        pose(1.5,free,FINGER_CLOSED,"to a free spot",cut_R,sa.CLOSE),
        transit(4.5,lift,FINGER_CLOSED,"turn back closed",pickup_R,sa.CLOSE,q_free,q_lift),
        pose(1.2,[down[0],down[1],TOP+.14],FINGER_CLOSED,"over the free table",pickup_R,sa.CLOSE),
        pose(1.3,[down[0],down[1],z_table+.002],FINGER_CLOSED,"lay them down",pickup_R,sa.CLOSE),
        pose(.6,[down[0],down[1],z_table+.002],FINGER_OPEN,"let go",pickup_R,0.),
        pose(.8,[down[0],down[1],TOP+.09],FINGER_OPEN,"rise",pickup_R,0.),
        pose(1.2,RIGHT_CLEAR,FINGER_OPEN,"clear",pickup_R,0.),
        pose(3.,RIGHT_CLEAR,FINGER_OPEN,"watch",pickup_R,0.),
    ]
    if mode == "failure":
        release=np.array([held[0],held[1],left_z])
        return {"left":left+[
            pose(max(t_open-t_held,.1),release,FINGER_CLOSED,"hold the cutting",left_R,0.),
            pose(.6,release,FINGER_OPEN,"release outside the bin",left_R,0.),
            pose(8.,release,FINGER_OPEN,"leave unsupported",left_R,0.),
        ],"right":right}
    left_down=grip_rotation(YAW_LEFT)
    q_hold=endpoint([held[0],held[1],left_z],left_R,'left',[-.53,1.06,1.16,-1.05,-.004,-.006])
    q_out=endpoint(OUT_OF_BUSH,left_down,'left',[-.02,1.71,1.74,-1.60,0.,1.55])
    if mode == "green":
        return {"left": left + [
            pose(max(t_open-t_held,.1),(held[0],held[1],left_z),FINGER_CLOSED,"hold the cane",left_R,0.),
            pose(.6,(held[0],held[1],left_z),LEFT_OPEN,"let go",left_R,0.),
            transit(2.,OUT_OF_BUSH,LEFT_OPEN,"back off past the tip",left_down,0.,q_hold,q_out),
            pose(1.2,LEFT_CLEAR,FINGER_OPEN,"clear",left_down,0.),
            pose(3.,LEFT_CLEAR,FINGER_OPEN,"watch",left_down,0.),
        ], "right": right}
    bx, by = sa.BIN_XY
    transported_direction=sa.BUSH_R @ sa.CANE_DIR
    centre_from_hand = (HOLD_AT - (CUT_AT + sa.TOP_S) / 2) * transported_direction
    over = (bx + centre_from_hand[0], by + centre_from_hand[1])
    from yam_arena.constants import READY
    clear_pos,clear_R=ik.tcp_pose(READY,station.BASES['left'])
    q_rise=endpoint((over[0],over[1],TOP+.20),left_R,'left',q_hold)
    retracted=np.asarray(READY).copy();retracted[0]=q_rise[0]
    retract_pos,retract_R=ik.tcp_pose(retracted,station.BASES['left'])
    left += [
        pose(max(t_open-t_held,.1),(held[0],held[1],left_z),FINGER_CLOSED,"hold the cutting",left_R,0.),
        pose(.8,np.r_[held[:2],left_z]+DRAW_OUT*(sa.BUSH_R@sa.CANE_DIR),FINGER_CLOSED,"draw it out",left_R,0.),
        pose(1.5,(.22,.25,1.0),FINGER_CLOSED,"out of the bush",left_R,0.),
        pose(1.5,(over[0],over[1],TOP+.20),FINGER_CLOSED,"over the bin",left_R,0.),
        pose(1.2,(over[0],over[1],TOP+.15),FINGER_CLOSED,"lower it in",left_R,0.),
        pose(.6,(over[0],over[1],TOP+.15),FINGER_OPEN,"let go",left_R,0.),
        pose(.8,(over[0],over[1],TOP+.20),FINGER_OPEN,"rise",left_R,0.),
        transit(1.5,retract_pos,FINGER_OPEN,"retract empty hand",retract_R,0.,q_rise,retracted),
        pose(1.5,retract_pos,FINGER_OPEN,"clear",retract_R,0.),
        pose(3.,retract_pos,FINGER_OPEN,"watch",retract_R,0.),
    ]
    left = [w if isinstance(w,PoseWaypoint) else PoseWaypoint(w.duration,w.pos,w.finger,w.label,w.yaw,
             rotation=left_R if w.label=='hold the cutting' else left_down) for w in left]
    return {"left": left, "right": right}


def _make_motion_controller(policy):
    from pruning_control import PruningController
    def snapshot():
        _S['motion_requested']=True
        return _S.get('motion')
    def diagnostics(detail):
        _S['demo_control']=detail
        folder=os.environ.get('TASK_SCORE_DIR')
        if folder:
            import json
            path=pathlib.Path(folder,'demo-control-final.json')
            temporary=path.with_suffix('.tmp')
            temporary.write_text(json.dumps(detail,indent=2)+'\n')
            temporary.replace(path)
    return PruningController(policy,snapshot,diagnostics,sa,
        {'grip_back':GRIP_BACK,
         'tcp_offset':TCP_OVER_TOOL+TOOL_SAG,'table_top':TOP,
         'cut_section':[ENTRY_CENTRE_X,CUT_RADIUS,sa.BLADE_Z]},os.environ.get('TASK_DEMO','success'))
