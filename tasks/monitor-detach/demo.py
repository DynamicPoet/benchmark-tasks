"""Scripted full-score demo for the monitor task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record] [--variant naive]

The script sees only what a model on yam_arena gets, the measured ``joint_pos``, plus its own commands:

1. Both arms fold from the ready pose. The left arm (Arm A) pinches the monitor's top edge over its centre of mass.
2. Arm A raises its command until its shoulder joint carries the monitor's weight. The joint's torque is read as
   i2rt's stiffness times how far the joint trails its command.
3. Only then does the right arm (Arm B) press the release lever.
4. Arm B grips the top edge 15 cm right of Arm A's grip. It places the edge from Arm A's measured pose: once A carries
   the weight, the monitor moves with A's hand.
5. Both arms lift the monitor, carry it forward and lower it until the mat stops it, then hold it upright.

``--variant naive`` starts both arms at once, so Arm B trips the latch before Arm A has arrived: the monitor falls and
the trial scores 0. Writes demo_top_cam.mp4, setup.png (the first top-camera frame) and result.json to the output folder.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
if (HERE / "src").is_dir():
    sys.path.insert(0, str(HERE / "src"))

CONTROL_HZ = 20
CAM_WIDTH, CAM_HEIGHT = 640, 360
RETREAT_S = 2.0  # fold from the ready pose (where the left gripper hangs beside the monitor's end) before the script
HOLD_S = 4.0  # keep stepping this long after the script ends, if the task has not succeeded by then
BASE_ABOVE_TABLE = 0.0025  # m: yam_arena's arm bases stand this far above the station's table top
VARIANTS = {"": "safe", "naive": "careless_B2"}


def policy_layout():
    """The task's layout as the arms see it: the scripted policies place targets relative to the arm base."""
    from robocurve.sim.layout import LAYOUT

    def down(box):
        (x0, y0, z0), (x1, y1, z1) = box
        return ((x0, y0, z0 - BASE_ABOVE_TABLE), (x1, y1, z1 - BASE_ABOVE_TABLE))

    names = ("panel_box", "plate_box", "neck_box", "bracket_box", "lever_box", "mat_box")
    # Arm A grips over the monitor's centre of mass: with i2rt's soft wrist (kp 10) a grip 7.5 cm off it lets the
    # released monitor swing ~11 degrees and drop its far corner onto the table. Arm B grips 15 cm to the right.
    centre = LAYOUT.panel_center[1]
    return replace(LAYOUT, grasp_y={"left": centre, "right": centre - 0.15}, **{n: down(getattr(LAYOUT, n)) for n in names})


def play(
    emb: Any,
    policy_name: str,
    instruction: str,
    hz: float = CONTROL_HZ,
    max_steps: int = 1200,
    on_frame: Callable[[Any], None] | None = None,
) -> dict:
    """Play a scripted policy on a yam_arena embodiment until the task succeeds, ``max_steps`` pass, or the script
    ends and HOLD_S more seconds pass. ``on_frame(observation)`` sees the first and every later observation."""
    from inspect_robots import Action, Scene
    from inspect_robots_yam.packing import STATE_KEY, pack, split
    from yam_arena.constants import FINGER_OPEN, SHOULDER, WRIST

    from robocurve.arena import monitor_detach as task
    from robocurve.arena.kinematics import ArenaKinematics
    from robocurve.sim.policies import POLICIES, Obs

    kin = ArenaKinematics()
    layout = policy_layout()
    kp = np.array([SHOULDER["stiffness"]] * 3 + [WRIST["stiffness"]] * 3)
    commanded: dict[str, np.ndarray] = {}  # each arm's last joint command
    carrying: dict[str, np.ndarray] = {}  # Arm A's tool point (from its measured joints) once it carries the monitor
    policy: Any = None

    def state(observation) -> tuple[dict, dict, dict]:
        """Both arms' joints, the torque each joint applies (stiffness x tracking error), and finger travel [m]."""
        left, right = split(observation.state[STATE_KEY])
        q = {"left": np.asarray(left[:6]), "right": np.asarray(right[:6])}
        eff = {s: kp * (commanded.get(s, q[s]) - q[s]) for s in q}
        finger = {"left": abs(FINGER_OPEN) * float(left[6]), "right": abs(FINGER_OPEN) * float(right[6])}
        return q, eff, finger

    def edges(q: dict) -> dict:
        """The top edge under each grasp point as Arm A's measured pose places it. Until the lever is pressed the latch
        holds the monitor where the layout puts it (A's pads may slide up the panel while A's command rises, so A's own
        pose says nothing about the edge then). Once A carries the weight, the monitor moves with A's hand and hangs
        from A's grip over its centre."""
        nominal = {s: np.array([layout.panel_x, y, layout.panel_top_z]) for s, y in layout.grasp_y.items()}
        tcp_a = kin.fk("left", q["left"])[0]
        if policy is not None and "left_supports" in policy.ctx.events:
            carrying.setdefault("left", tcp_a)
        moved = tcp_a - carrying["left"] if carrying else np.zeros(3)
        return {s: edge + moved for s, edge in nominal.items()}

    step = 0
    info: dict = {}
    success = False

    def advance(arms: list[np.ndarray], phase: dict, status: dict):
        nonlocal step, info, success
        commanded.update(left=np.asarray(arms[0][:6], float), right=np.asarray(arms[1][:6], float))
        result = emb.step(Action(data=pack(*arms)))
        step += 1
        info = result.info
        success = bool(result.terminated and result.termination_reason == "success")
        task.records()[-1].update(phase=phase, status=status)
        if on_frame is not None:
            on_frame(result.observation)
        return result.observation

    obs = emb.reset(Scene(id="demo", instruction=instruction), seed=0)
    if on_frame is not None:
        on_frame(obs)
    q, eff, finger = state(obs)
    start = {s: q[s].copy() for s in q}
    for i in range(1, max(1, round(RETREAT_S * hz)) + 1):
        w = i / max(1, round(RETREAT_S * hz))
        w = w * w * (3 - 2 * w)  # smoothstep
        obs = advance([np.array([*(start[s] * (1 - w)), 1.0]) for s in ("left", "right")],
                      {s: "fold to home" for s in start}, {s: "idle" for s in start})
    t_script = step / hz
    q, eff, finger = state(obs)
    policy = POLICIES[policy_name](kin, layout, "left", "right", q0=q)
    # i2rt's gains are soft: carrying the 2 kg monitor takes a lift command of about 3 cm, and the arm sags ~6 cm if the
    # latch opens while it carries only part of the weight, so the gate passes at 95 % of the expected torque rise.
    policy.max_lift_mm, policy.lift_steps_mm, policy.gate_share = 80.0, (3.0, 1.0), 0.95
    policy.second_grasp_depth, policy.second_grasp_speed = 0.07, 0.06
    policy.grasp_servo = (2, 0.08)  # quick: the released monitor swings in Arm A's single grip until Arm B has it
    ended = None
    while step < max_steps and not success:
        if policy.finished:
            ended = step if ended is None else ended
            if step - ended >= HOLD_S * hz:
                break
        cmds = policy.step(Obs(t=step / hz - t_script, q=q, eff=eff, finger=finger, edge=edges(q)))
        arms = [np.array([*cmds[s].q, 1.0 - cmds[s].grip]) for s in ("left", "right")]
        obs = advance(arms, {s: cmds[s].phase for s in cmds}, {s: cmds[s].status for s in cmds})
        q, eff, finger = state(obs)
    events = {k: (v + t_script if k.endswith(("_closed", "_supports", "_pressed")) else v)
              for k, v in policy.ctx.events.items()}
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    return {"task_score": score, "success": success, "steps": step, "seconds": step / hz,
            "events": {**task.events(), **events}}


def _write_mp4(frames: list[np.ndarray], path: pathlib.Path) -> None:
    import imageio

    # macro_block_size=1 keeps the frames' size (imageio would otherwise resize them to multiples of 16).
    imageio.mimsave(path, frames, fps=CONTROL_HZ, codec="libx264", macro_block_size=1,
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--variant", default="", choices=sorted(VARIANTS))
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROBOCURVE_TELEMETRY_DIR", str(out / "telemetry"))

    from yam_arena.inspect_embodiment import YamArenaEmbodiment

    emb = YamArenaEmbodiment(task=str(pathlib.Path(args.task).resolve()), control_hz=CONTROL_HZ, cam_width=CAM_WIDTH,
                             cam_height=CAM_HEIGHT, object_jitter=0.0)
    frames: list[np.ndarray] = []

    def on_frame(observation) -> None:
        if not args.no_record:
            frames.append(np.array(observation.images["top_cam"]))

    result = play(emb, VARIANTS[args.variant], emb.task.module.INSTRUCTION, max_steps=args.max_steps, on_frame=on_frame)
    if frames:
        import imageio

        _write_mp4(frames, out / "demo_top_cam.mp4")
        imageio.imwrite(out / "setup.png", frames[0])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    os._exit(0)  # SimulationApp.close() would end the process before the output is flushed


if __name__ == "__main__":
    main()
