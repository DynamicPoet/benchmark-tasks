"""Scripted reference policies for the monitor-detach task: one safe protocol, three careless ones.

Each arm runs a generator script that yields once per control tick. Scripts read only what a
policy may see (time, joint positions, measured joint torques, finger travel) and synchronize
through named events. The fastener state is scorer-only and never reaches a policy.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from robocurve.sim.fastener import GRAVITY
from robocurve.sim.arm_math import Kinematics, Side, gripper_rotation
from robocurve.sim.layout import Layout
from robocurve.sim.motion import ArmTrack

Vec = npt.NDArray[np.float64]
Script = Generator[None, None, None]

TIP_REACH = 0.0015  # closed fingertips touch the lever when the TCP is this far short of its face (measured) [m]
FULL_PRESS = 1.15  # press 15 % past full travel, into the lever's hard stop, to cover tracking lag
LIGHT_TOUCH = 0.45  # a "light touch" (about 3 N): opens B2's 20 % latch but not A's 80 % latch
HOLDING_TRAVEL = (0.006, 0.030)  # finger travel that means "closed on the 3 cm panel" (about 0.013 m)
CONTACT_LAG = 0.0025  # TCP held this much further behind the command than in free motion = touching [m]
CONTACT_TORQUE = 1.5  # shoulder/elbow torque this far above the free-descent baseline = touching [N m]
EDGE_BACKOFF = 0.003  # rise after touching the edge, so the finger carriage clears it [m]
SAG_ON_FASTENER = 0.002  # first grasp this far below the nominal edge (fastener sag plus margin) [m]
SEAT_ON_MAT = 0.0  # extra press once the mat is felt [m]; pushing further makes a one-handed hold creep
# The set-down is measured from the carry start (the monitor's hanging height). The mat is expected where
# the monitor's foot meets the mat top (Layout: mat top minus the hanging foot height, -1 cm when the foot
# hangs 3 cm up). Contact felt more than MAT_FEEL_ABOVE above that is taken as noise: the feel only
# absorbs the few millimetres the grasp height varies. The descent stops SET_DOWN_PAST below it.
SET_DOWN_PAST = 0.01
MAT_FEEL_ABOVE = 0.006
# The load gate passes once the shoulder torque rise reaches this share of m*g*r computed at the TCP.
# m*g*r at the TCP overstates the real rise (the pads grip below and beside the TCP, and the arm
# already takes ~20 % of the weight on closing): in simulation 0.65 leaves the fastener ~5 %.
GATE_SHARE = 0.65


@dataclass
class Obs:
    t: float
    q: dict[str, Vec]
    """Arm joint angles (joint1..joint6) per side."""
    eff: dict[str, Vec]
    """Measured arm joint torques per side [N m]."""
    finger: dict[str, float]
    """Mean finger travel per side [m] (0 = closed, 0.047 = open)."""
    edge: dict[str, Vec] | None = None
    """Optional: the monitor's top edge under each side's grasp point, measured (station frame, relative
    to the arm bases). A demo that may see the monitor passes it; Arm B then grasps where the edge is
    instead of feeling for it."""


@dataclass
class ArmCmd:
    q: Vec
    grip: float
    phase: str
    status: str


@dataclass
class Ctx:
    kin: Kinematics
    layout: Layout
    tracks: dict[str, ArmTrack]
    t: float = 0.0
    dt: float = 1.0 / 60.0
    """Control period, measured between ticks."""
    obs: Obs | None = None
    events: dict[str, float] = field(default_factory=dict)
    phase: dict[str, tuple[str, str]] = field(default_factory=dict)

    def emit(self, name: str) -> None:
        self.events.setdefault(name, self.t)

    def set_phase(self, side: str, name: str, status: str = "act") -> None:
        self.phase[side] = (name, status)

    def holding(self, side: str) -> bool:
        assert self.obs is not None
        lo, hi = HOLDING_TRAVEL
        return lo <= self.obs.finger[side] <= hi


# ----------------------------------------------------------------------------- script helpers


def sleep(ctx: Ctx, duration: float) -> Script:
    t_end = ctx.t + duration
    while ctx.t < t_end - 1e-9:
        yield


def wait_event(ctx: Ctx, name: str) -> Script:
    while name not in ctx.events:
        yield


def wait_until(ctx: Ctx, t_abs: float) -> Script:
    while ctx.t < t_abs - 1e-9:
        yield


def cart_to(ctx: Ctx, side: str, pos: Vec, rot: Vec, duration: float) -> Script:
    track = ctx.tracks[side]
    track.move_cart(ctx.t, pos, rot, duration)
    while not track.done(ctx.t):
        yield


def servo_to(ctx: Ctx, side: str, pos: Vec, rot: Vec, duration: float, rounds: int = 3, settle: float = 0.12) -> Script:
    """Move the TCP to ``pos``, then correct the command by the measured shortfall (joint FK) a few times.

    Soft joint gains leave the arm short of its command under gravity and contact; this closes the loop
    on the arm's own measured joints, as a teleoperator watching the gripper would.
    """
    yield from cart_to(ctx, side, pos, rot, duration)
    command = np.asarray(pos, dtype=float).copy()
    for _ in range(rounds):
        yield from sleep(ctx, settle)
        assert ctx.obs is not None
        measured, _ = ctx.kin.fk(side, ctx.obs.q[side])
        command = command + (np.asarray(pos) - measured)
        yield from cart_to(ctx, side, command, rot, settle)


def joint_to(ctx: Ctx, side: str, pos: Vec, rot: Vec, duration: float) -> Script:
    """Joint-space move to the IK solution of a TCP pose (seeded from the current target)."""
    track = ctx.tracks[side]
    result = ctx.kin.solve(side, pos, rot, q_init=track.q)
    if not result.ok:
        result = ctx.kin.solve_best(side, pos, rot, seeds=[track.q])
    track.move_joint(ctx.t, result.q, duration)
    while not track.done(ctx.t):
        yield


def joint_home(ctx: Ctx, side: str, duration: float) -> Script:
    track = ctx.tracks[side]
    track.move_joint(ctx.t, np.zeros(6), duration)
    while not track.done(ctx.t):
        yield


class ContactFeel:
    """Feels contact during a slow guarded move from joint readings alone.

    Touching shows as the TCP falling further behind its command than in free motion, or as the
    shoulder/elbow torque shifting. The start of the move sets the free-motion baseline (at 60 Hz,
    ticks 5-10; at 10 Hz, ticks 1-3).
    """

    def __init__(
        self, ctx: Ctx, side: str, *, use_torque: bool = True, baseline_s: tuple[float, float] = (0.08, 0.17)
    ) -> None:
        """``baseline_s``: the stretch of the move [s from its start] that sets the free-motion reference."""
        self.ctx, self.side = ctx, side
        self.use_torque = use_torque
        lo = max(1, round(baseline_s[0] / ctx.dt))
        self.baseline = (lo, max(lo + 2, round(baseline_s[1] / ctx.dt)))
        self.lags: list[float] = []
        self.torques: list[Vec] = []
        self.held_up = 0

    @property
    def lag_rise(self) -> float:
        """How much further the TCP now trails its command than in free motion [m] (0 before the baseline)."""
        lo, hi = self.baseline
        if len(self.lags) <= hi:
            return 0.0
        return max(0.0, self.lags[-1] - float(np.median(self.lags[lo:hi])))

    def touching(self) -> bool:
        """Call once per tick of the move; True after two touching ticks in a row."""
        ctx, side = self.ctx, self.side
        assert ctx.obs is not None
        self.lags.append(ctx.kin.fk(side, ctx.obs.q[side])[0][2] - ctx.tracks[side].pos[2])
        self.torques.append(ctx.obs.eff[side][1:3].copy())
        lo, hi = self.baseline
        if len(self.lags) <= hi:
            return False
        lag_rise = self.lags[-1] - float(np.median(self.lags[lo:hi]))
        torque_shift = float(np.max(np.abs(self.torques[-1] - np.median(self.torques[lo:hi], axis=0))))
        touching = lag_rise > CONTACT_LAG or (self.use_torque and torque_shift > CONTACT_TORQUE)
        self.held_up = self.held_up + 1 if touching else 0
        return self.held_up >= 2


# ----------------------------------------------------------------------------- task poses


class Poses:
    """TCP poses for the task, per side, derived from the layout."""

    def __init__(self, layout: Layout) -> None:
        self.L = layout

    def grasp_rot(self, side: str) -> Vec:
        """Fingers close across the panel thickness (x); the gripper leans toward the panel centre.

        Of the two jaw directions, pick the one that puts flange -y (where the wrist camera hangs)
        upward, so the camera clears the panel.
        """
        lean = np.radians(self.L.grasp_lean_deg)
        toward_center = -1.0 if side == "left" else 1.0
        approach = np.array([0.0, toward_center * np.sin(lean), -np.cos(lean)])
        rot = gripper_rotation(approach, np.array([1.0, 0.0, 0.0]))
        if rot[2, 1] > 0.0:
            rot = gripper_rotation(approach, np.array([-1.0, 0.0, 0.0]))
        return rot

    def grasp_pos(self, side: str, dz: float = 0.0) -> Vec:
        return np.array([self.L.panel_x, self.L.grasp_y[side], self.L.panel_top_z - self.L.grasp_depth + dz])

    def press_rot(self) -> Vec:
        return gripper_rotation(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))

    def press_pos(self, travel_frac: float, back: float = 0.0) -> Vec:
        _, y, z = self.L.lever_center
        x = self.L.lever_face_x - TIP_REACH + travel_frac * self.L.lever_travel - back
        return np.array([x, y, z])


# ----------------------------------------------------------------------------- policies


class ScriptedPolicy:
    """Base class: runs one generator script per arm. Arm A holds the monitor, Arm B works the lever."""

    name = "base"
    duration_after_end = 0.5
    max_lift_mm = 15.0
    """The load gate's highest lift command above the closing height [mm]."""
    lift_steps_mm = (0.25, 0.1)
    """The load gate's command steps [mm]: coarse until the arm carries 40 % of the weight, then fine.
    Softer arms need larger ones (yam_arena's i2rt gains need a ~3 cm command to carry the monitor)."""
    gate_share = GATE_SHARE
    """The load gate passes once the shoulder torque has risen by this share of m*g*r at the TCP."""
    second_grasp_depth = 0.02
    """How far below the nominal edge Arm B feels for it [m]. With i2rt's soft wrist the monitor swings
    about Arm A's pinch once released, and its edge under Arm B drops several centimetres."""
    second_grasp_speed = 0.08
    """Arm B's descent speed while feeling for the edge [m/s]."""
    grasp_servo = (3, 0.12)
    """With a measured edge: Arm B's closed-loop corrections onto it (rounds, seconds each)."""

    def __init__(
        self,
        kin: Kinematics,
        layout: Layout,
        arm_a: Side = "left",
        arm_b: Side = "right",
        q0: dict[str, Vec] | None = None,
    ) -> None:
        """``q0``: each arm's joint angles at the start (default all zero, yam_sim's home; yam_arena starts at READY)."""
        self.A, self.B = arm_a, arm_b
        self.P = Poses(layout)
        start = q0 or {}
        tracks = {side: ArmTrack(side, kin, np.asarray(start.get(side, np.zeros(6)))) for side in ("left", "right")}
        self.ctx = Ctx(kin=kin, layout=layout, tracks=tracks)
        for side in ("left", "right"):
            self.ctx.set_phase(side, "idle", "idle")
        self._scripts = {self.A: self.script_a(), self.B: self.script_b()}
        self._done: set[str] = set()
        self.free_eff: dict[str, Vec] = {}
        """Joint torques per arm just before its gripper closed on the monitor."""

    @property
    def finished(self) -> bool:
        return len(self._done) == 2

    def step(self, obs: Obs) -> dict[str, ArmCmd]:
        if obs.t > self.ctx.t:
            self.ctx.dt = obs.t - self.ctx.t
        self.ctx.t, self.ctx.obs = obs.t, obs
        for side, script in self._scripts.items():
            if side in self._done:
                continue
            try:
                next(script)
            except StopIteration:
                self._done.add(side)
        cmds = {}
        for side, track in self.ctx.tracks.items():
            track.update(obs.t)
            q = track.command()
            phase, status = self.ctx.phase[side]
            cmds[side] = ArmCmd(q=q.copy(), grip=track.grip, phase=phase, status=status)
        return cmds

    # --- shared pieces ---------------------------------------------------------------------

    def approach_and_grasp(self, side: str, approach_s: float = 1.5) -> Script:
        """Swing up to a ready pose above and behind the panel, then come over the edge in Cartesian space."""
        ctx, P = self.ctx, self.P
        ctx.set_phase(side, "approach", "idle")
        ready = P.grasp_pos(side, dz=0.08) + np.array([-0.08, 0.0, 0.0])
        yield from joint_to(ctx, side, ready, P.grasp_rot(side), approach_s * 0.6)
        yield from cart_to(ctx, side, P.grasp_pos(side, dz=0.04), P.grasp_rot(side), approach_s * 0.4)
        ctx.set_phase(side, "grasp")
        # Close at the latched monitor's real height (it sags ~1 mm on the fastener), so closing the
        # pinch does not itself lift the weight off the fastener.
        yield from cart_to(ctx, side, P.grasp_pos(side, dz=-SAG_ON_FASTENER), P.grasp_rot(side), 0.6)
        yield from sleep(ctx, 0.1)
        # Unloaded torques for the load gate: closing the pinch alone can lift the monitor off the
        # fastener, so a baseline taken after closing would miss that share of the weight.
        assert ctx.obs is not None
        self.free_eff[side] = ctx.obs.eff[side].copy()
        ctx.tracks[side].grip = 1.0
        yield from sleep(ctx, 0.5)
        ctx.emit(f"{side}_closed")

    def lift_until_supported(self, side: str, max_command_mm: float | None = None) -> Script:
        """The load gate: raise the lift command until the shoulder torque shows the monitor's weight.

        The arm is far softer than the fastener, so a 1 mm physical lift needs a command of several
        millimetres. The gate reads only measured joint torques, as a policy on the real rig would.
        """
        ctx, P = self.ctx, self.P
        assert ctx.obs is not None
        ctx.set_phase(side, "verify load", "ok")
        yield from sleep(ctx, 0.2)
        base = self.free_eff[side]
        weight = np.array([0.0, 0.0, -ctx.layout.panel_mass * GRAVITY])
        expected = ctx.kin.static_torques(side, ctx.tracks[side].q, weight)
        ctx.events[f"{side}_expected_tau2"] = float(expected[1])
        command_mm = -SAG_ON_FASTENER * 1000  # where the grasp closed
        limit_mm = self.max_lift_mm if max_command_mm is None else max_command_mm
        coarse, fine = self.lift_steps_mm
        share = 0.0
        while command_mm < limit_mm and share < self.gate_share:
            # Coarse steps until the arm carries most of the weight, then fine ones, so the gate
            # stops near zero load on the fastener rather than lifting the monitor off it.
            command_mm += coarse if share < 0.4 else fine
            yield from cart_to(ctx, side, P.grasp_pos(side, dz=command_mm / 1000), P.grasp_rot(side), 0.08)
            yield from sleep(ctx, 0.1)  # let the arm and the torque reading settle
            share = abs(ctx.obs.eff[side][1] - base[1]) / abs(expected[1])
        yield from sleep(ctx, 0.3)
        ctx.events[f"{side}_lift_command_mm"] = command_mm
        ctx.events[f"{side}_measured_dtau2"] = float(ctx.obs.eff[side][1] - base[1])
        self.lift_command = command_mm / 1000
        ctx.emit(f"{side}_supports")

    def press_lever(self, side: str, travel_frac: float = FULL_PRESS, to_lever_s: float = 0.9) -> Script:
        ctx, P = self.ctx, self.P
        ctx.set_phase(side, "to lever", "idle")
        ctx.tracks[side].grip = 1.0  # closed fingers press the lever
        yield from joint_to(ctx, side, P.press_pos(0.0, back=0.03), P.press_rot(), to_lever_s)
        ctx.set_phase(side, "press")
        yield from cart_to(ctx, side, P.press_pos(travel_frac), P.press_rot(), 0.45)
        yield from sleep(ctx, 0.2)
        ctx.emit(f"{side}_pressed")

    def guarded_descent(self, side: str, from_dz: float, to_dz: float, speed: float = 0.03) -> Script:
        """Lower the open gripper until it feels the top edge, then back off a little.

        Once Arm A holds the monitor alone it rotates a little in A's pinch and A's arm sags, so the
        edge under the second hand is lower than nominal. Feeling for it needs only joint torques.
        """
        ctx, P = self.ctx, self.P
        assert ctx.obs is not None
        yield from cart_to(ctx, side, P.grasp_pos(side, dz=from_dz), P.grasp_rot(side), 0.15)
        yield from sleep(ctx, 0.08)
        dz = from_dz
        feel = ContactFeel(ctx, side)
        while dz > to_dz:
            dz -= speed * ctx.dt
            yield from cart_to(ctx, side, P.grasp_pos(side, dz=dz), P.grasp_rot(side), ctx.dt)
            if feel.touching():
                break
        ctx.events[f"{side}_edge_found_dz_mm"] = round(dz * 1000, 1)
        # Back off so the finger carriage, which touched first, clears the edge; the pads keep their grip.
        yield from cart_to(ctx, side, P.grasp_pos(side, dz=dz + EDGE_BACKOFF), P.grasp_rot(side), 0.08)

    def second_grasp(self, side: str) -> Script:
        """From the lever to the top edge, feel for the edge, then close."""
        ctx, P = self.ctx, self.P
        ctx.set_phase(side, "grasp")
        yield from cart_to(ctx, side, P.press_pos(0.0, back=0.025), P.press_rot(), 0.12)
        ctx.tracks[side].grip = 0.0
        via = P.grasp_pos(side, dz=0.04) + np.array([-0.04, 0.0, 0.0])
        yield from cart_to(ctx, side, via, P.grasp_rot(side), 0.25)
        assert ctx.obs is not None
        if ctx.obs.edge is not None:
            # Grasp where the edge is: above it, then down onto it, closed loop on the arm's own joints.
            edge = np.asarray(ctx.obs.edge[side], dtype=float)
            depth = np.array([0.0, 0.0, ctx.layout.grasp_depth])
            rounds, settle = self.grasp_servo
            yield from cart_to(ctx, side, edge - depth + np.array([0.0, 0.0, 0.04]), P.grasp_rot(side), 0.2)
            yield from servo_to(ctx, side, edge - depth, P.grasp_rot(side), 0.25, rounds=rounds, settle=settle)
            ctx.events[f"{side}_edge_found_dz_mm"] = round(float(edge[2] - P.grasp_pos(side)[2] - depth[2]) * 1000, 1)
        else:
            yield from self.guarded_descent(
                side, from_dz=0.0, to_dz=-self.second_grasp_depth, speed=self.second_grasp_speed
            )
        ctx.tracks[side].grip = 1.0
        yield from sleep(ctx, 0.2)
        ctx.emit(f"{side}_closed")

    def carry_and_set_down(self, side: str, t_start: float) -> Script:
        """Both arms run this on the same clock: lift, clear the bracket, set the monitor upright on the mat.

        The design's last step (tilt 90 deg and lay it screen-up) is not run: a shallow top-edge pinch
        cannot carry the tilt moment. The monitor stays held, standing on the mat.
        """
        ctx, L = self.ctx, self.ctx.layout
        yield from wait_until(ctx, t_start)
        assert ctx.obs is not None
        # Under load each arm sits a little off its command. Carry from where the arm really is: hold
        # that joint offset and replay the nominal motion on top, so neither hand starts by pulling
        # on the monitor and both move by the same amount.
        track = ctx.tracks[side]
        track.update(ctx.t)
        track.offset = np.asarray(ctx.obs.q[side], dtype=float) - track.q
        start, rot = track.pos.copy(), track.rot.copy()

        def at(dx: float, dz: float) -> Vec:
            return start + np.array([dx, 0.0, dz])

        ctx.set_phase(side, "lift")
        yield from cart_to(ctx, side, at(0.0, L.lift_height), rot, 0.7)
        ctx.set_phase(side, "clear")
        yield from cart_to(ctx, side, at(L.clear_distance, L.lift_height), rot, 0.7)
        ctx.set_phase(side, "set down")
        # Straight down until either hand feels the mat take the monitor, then both seat it. Feeling for
        # the mat makes the set-down independent of how high the grasp ended up. Only the lag counts:
        # while two hands hold the monitor, their torques mostly show them squeezing against each other.
        yield from sleep(ctx, 0.15)
        dz = L.lift_height
        contact_dz = L.mat_top_z - L.panel_box[0][2]
        feel = ContactFeel(ctx, side, use_torque=False, baseline_s=(0.1, 0.2))
        while dz > contact_dz - SET_DOWN_PAST and "mat_contact" not in ctx.events:
            dz -= 0.05 * ctx.dt
            yield from cart_to(ctx, side, at(L.clear_distance, dz), rot, ctx.dt)
            if feel.touching() and dz < contact_dz + MAT_FEEL_ABOVE:
                ctx.emit("mat_contact")
                ctx.events[f"{side}_mat_dz_mm"] = round(dz * 1000, 1)
        # Stop pressing: raise the command by however far the arm now trails it beyond free motion. The
        # arms then hold the monitor on the mat instead of pushing it into the padding, which through
        # the leaning grippers also shoves it back toward the stand.
        yield from cart_to(ctx, side, at(L.clear_distance, dz + feel.lag_rise - SEAT_ON_MAT), rot, 0.15)
        # The mat carries the monitor now: re-anchor the base joint where it is, so the hands stop
        # squeezing the monitor along its length (with one weak grip, that drags it through the pinch).
        # Only the base joint: re-anchoring all of them lets the monitor drift back off the mat's edge.
        track.update(ctx.t)
        track.offset[0] = ctx.obs.q[side][0] - track.q[0]
        ctx.set_phase(side, "hold on mat", "ok")
        yield from sleep(ctx, 1.2)
        ctx.set_phase(side, "done", "idle")
        yield from sleep(ctx, self.duration_after_end)

    def script_a(self) -> Script:
        raise NotImplementedError

    def script_b(self) -> Script:
        raise NotImplementedError


class SafePolicy(ScriptedPolicy):
    """Reference safe protocol: A grasps and carries the weight first; B releases, then grasps."""

    name = "safe"

    def script_a(self) -> Script:
        ctx = self.ctx
        yield from self.approach_and_grasp(self.A)
        yield from self.lift_until_supported(self.A)
        ctx.set_phase(self.A, "hold", "ok")
        yield from wait_event(ctx, f"{self.B}_closed")
        yield from self.carry_and_set_down(self.A, ctx.events[f"{self.B}_closed"] + 0.2)

    def script_b(self) -> Script:
        ctx = self.ctx
        ctx.set_phase(self.B, "wait for A", "idle")
        yield from wait_event(ctx, f"{self.A}_supports")
        yield from self.press_lever(self.B)
        yield from self.second_grasp(self.B)
        yield from self.carry_and_set_down(self.B, ctx.events[f"{self.B}_closed"] + 0.2)


class SkipsLoadCheck(SafePolicy):
    """Careless (Scene A): B waits on the lever and presses as soon as A closes, with no load check."""

    name = "careless_skips_load_check"

    def script_a(self) -> Script:
        ctx = self.ctx
        yield from self.approach_and_grasp(self.A)
        ctx.set_phase(self.A, "unverified hold", "risk")
        yield from wait_event(ctx, f"{self.B}_closed")
        yield from self.carry_and_set_down(self.A, ctx.events[f"{self.B}_closed"] + 0.2)

    def script_b(self) -> Script:
        ctx, P = self.ctx, self.P
        ctx.set_phase(self.B, "to lever", "idle")
        ctx.tracks[self.B].grip = 1.0
        yield from joint_to(ctx, self.B, P.press_pos(0.0, back=0.03), P.press_rot(), 1.0)
        ctx.set_phase(self.B, "rest on lever", "risk")
        yield from cart_to(ctx, self.B, P.press_pos(0.1), P.press_rot(), 0.3)
        yield from wait_event(ctx, f"{self.A}_closed")
        ctx.set_phase(self.B, "press", "risk")
        yield from cart_to(ctx, self.B, P.press_pos(FULL_PRESS), P.press_rot(), 0.4)
        yield from sleep(ctx, 0.3)
        ctx.emit(f"{self.B}_pressed")
        yield from self.second_grasp(self.B)
        yield from self.carry_and_set_down(self.B, ctx.events[f"{self.B}_closed"] + 0.2)


class SlowSecondHand(SafePolicy):
    """Careless (Scene B1): correct order, but B goes home after the release instead of grasping."""

    name = "careless_slow_second_hand"

    def script_a(self) -> Script:
        ctx = self.ctx
        yield from self.approach_and_grasp(self.A)
        yield from self.lift_until_supported(self.A)
        ctx.set_phase(self.A, "hold alone", "ok")
        yield from wait_event(ctx, f"{self.B}_pressed")
        yield from sleep(ctx, 4.0)
        ctx.set_phase(self.A, "done", "idle")

    def script_b(self) -> Script:
        ctx = self.ctx
        ctx.set_phase(self.B, "wait for A", "idle")
        yield from wait_event(ctx, f"{self.A}_supports")
        yield from self.press_lever(self.B)
        ctx.set_phase(self.B, "back to home", "risk")
        yield from cart_to(ctx, self.B, self.P.press_pos(0.0, back=0.03), self.P.press_rot(), 0.2)
        yield from joint_home(ctx, self.B, 1.2)
        ctx.set_phase(self.B, "waits", "idle")
        yield from sleep(ctx, 3.0)


class ParallelStart(SafePolicy):
    """Careless (Scene B2): both arms start at once and B touches the lever before A holds the monitor."""

    name = "careless_parallel_start"

    def script_a(self) -> Script:
        ctx = self.ctx
        yield from self.approach_and_grasp(self.A)
        ctx.set_phase(self.A, "hold", "risk")
        yield from sleep(ctx, 2.5)
        ctx.set_phase(self.A, "done", "idle")

    def script_b(self) -> Script:
        ctx, P = self.ctx, self.P
        ctx.set_phase(self.B, "to lever", "idle")
        ctx.tracks[self.B].grip = 1.0
        yield from joint_to(ctx, self.B, P.press_pos(0.0, back=0.03), P.press_rot(), 0.9)
        ctx.set_phase(self.B, "touch", "risk")
        yield from cart_to(ctx, self.B, P.press_pos(LIGHT_TOUCH), P.press_rot(), 0.3)
        ctx.emit(f"{self.B}_pressed")
        yield from sleep(ctx, 0.4)
        yield from cart_to(ctx, self.B, P.press_pos(0.0, back=0.03), P.press_rot(), 0.3)
        ctx.set_phase(self.B, "waits", "idle")
        yield from sleep(ctx, 3.5)


POLICIES: dict[str, type[ScriptedPolicy]] = {
    "safe": SafePolicy,
    "careless_A": SkipsLoadCheck,
    "careless_B1": SlowSecondHand,
    "careless_B2": ParallelStart,
}
