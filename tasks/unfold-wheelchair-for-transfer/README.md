# Open a folded wheelchair with both arms and leave it in place

A standard lightweight folding wheelchair (18 in seat, 24 in rear wheels with handrims, 8 in casters) stands folded on
a vinyl floor in front of the two arms, facing them. The robot must open it fully, so that both seat rails rest in
their saddles, without letting it roll, slide or turn away, and leave it standing free.

The chair's two side frames are joined only by the X cross-brace under the seat. Opening it means spreading the X:
pushing the seat rails apart and down. Each side frame stands on two wheels in a line, so it slides sideways only
against the scrub of its tyres on the floor, and whichever side breaks loose first runs ahead: the chair slides or
turns towards it instead of opening in place. The arms must spread and press both seat rails at the same speed and
with matching force.

## Files

| File | What |
|---|---|
| `task.py` | The task file: the chair, the scorer (`score()`, `success()`) and the scripted demo (`demo()`) |
| `chair.py` | The wheelchair as one USD articulation (generated as `assets/wheelchair.usda` when the task loads), and the geometry the scorer measures |
| `wheelchair_floor_bay.py` | The embodiment `yam_floor_bay`: the station's arms and cameras, without the table |
| `scripted_demo.py` | The policy `wheelchair_scripted`, which plays the demo |
| `scenekit.py`, `usdkit.py`, `steptime.py` | Helpers: USD articulations in a yam_arena task file, plain-text USD meshes, step timing |
| `pyproject.toml` | Registers `yam_floor_bay` and `wheelchair_scripted` with Inspect Robots |

## The embodiment

A full-size wheelchair does not fit on the station's table, so the task declares its own embodiment, `yam_floor_bay`.
It is `yam_arena` (Isaac Lab-Arena 0.3.1, yam-isaac-lab-arena) with the table removed: the same two YAM arms at the
same mounts (0.61 m apart, 0.76 m above the floor) on steel pedestals, the same joint interface, gains, limits and
20 Hz control, the same calibrated top camera (on a floor mast) and wrist cameras. The floor is level vinyl. It
subclasses the harness's embodiment and changes only the room it builds; the harness is not modified.

Install it once, from this folder, into the yam_arena environment:

```bash
pip install -e .        # or uv pip install -e .: registers yam_floor_bay and wheelchair_scripted
inspect-robots list     # shows both
```

## Real-world setup and reset

- **Chair:** a standard lightweight folding manual wheelchair with an 18 in (457 mm) seat, aluminium frame, X cross-brace,
  seat rails that rest in saddles on the side frames, 24 in rear wheels with handrims, 8 in swivel casters and flip-back
  armrests, footrests removed; about 13 kg, 28 cm wide folded and 61 cm open over the handrims, seat 48 cm high.
- **Rig:** the two YAM arms 61 cm apart, each bolted on a steel pedestal (a 10 x 10 cm column on a 24 x 24 cm foot plate)
  with its mount 76 cm above a level vinyl floor; the station's top camera at its calibrated pose relative to the arms, on
  a floor mast behind them; the wrist cameras as on the station.
- **Floor marks (once):** tape the arms' base line (through both arm mounts), the centre line between the arms, a cross
  68 cm in front of the base line on the centre line, and a line 25 cm in front of the base line.
- **Before each trial:** footrests off, armrests down. Fold the chair by lifting both seat rails until the side frames meet
  (28 cm over the handrims). Stand it facing the robot with the midpoint between its rear-wheel hubs on the cross and the
  hub line parallel to the base line; turn both casters to trail behind their swivels, the swivels on the 25 cm line.
- **Scoring:** the chair's position and heading with a tape measure from the floor marks, the seat rails' spacing with a
  tape measure, the saddles and the clearances from a photo and the video.

## Run a trial

Dream-1:

```bash
inspect-robots run --embodiment yam_floor_bay --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 -P rows_per_request=8 \
  --instruction "Open the folded wheelchair fully, and leave it standing where it is." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir runs/dream
```

Scripted demo (scores 1.0): the same command with `--policy wheelchair_scripted`. The careless variant, in which the
left arm spreads and presses its side all the way while the right arm only grips its rail and holds it still:
`TASK_DEMO=failure` with the same command. The chair opens lopsided, is dragged and turns more than 20 degrees: 0.

`task_score` prints the trial's score: 1.0 once `success()` has held for 1 s, otherwise `score()` at the end.

## The scripted demo

Both grippers come down on the two raised seat rails, 30 cm behind their front ends, and close on them. Then both
spread and press their rail down along the cross-brace's own path, at the same speed, over 10 s, until the rails drop
into their saddles; they hold the chair seated for 1.5 s, ease off, open, rise 25 cm and back away. The side frames
first lean apart until the back upholstery is taut, then their wheels slide outward. The demo reaches 1.0 at step 424
(21.2 s); the chair ends 1.1 cm and 1.4 degrees from where it started.

The demo coordinates the two arms from their own joint angles (forward kinematics gives where each gripper is): each
arm presses until its gripper is as low as the plan says, and the two grippers' common sideways motion is held where
the plan puts it, so that if the chair starts to slide one way, both arms push it back together. It reads nothing but
the robot's joint positions and the clock.

## Scoring

The scorer (`_measure` and `score()` in `task.py`) runs after every control step and latches failures; only a
reset clears them. The chair's position is the midpoint between its rear-wheel hubs, its heading the direction of the
line between them, both against where they were at the first step.

- **0 at any moment, and it stays 0:** the chair tips over (any part other than its wheels comes within 1 cm of the
  floor), or its position moves more than 20 cm, or its heading turns more than 20 degrees.
- **Otherwise, by the state at the end:**
  - **0** if the seat rails opened less than half way from folded (13.0 cm between their centres) to fully open
    (45.7 cm). Doing nothing scores 0.
  - **0.25** if opened at least half way, but not both seat rails resting in their saddles. A rail rests in its
    saddles when, at both of its V-shaped saddles, its underside is at most 3 mm above its fully seated height and its
    centre within 12 mm of the cradle's centre line.
  - **0.5** if both rails are seated, but the chair moved more than 5 cm or turned more than 5 degrees.
  - **0.75** if both are seated and the chair is within 5 cm and 5 degrees, but a gripper fingertip is within 5 cm of
    any part of the chair, a wheel's lowest point is more than 1 cm above the floor, or the rear-hub midpoint or the
    middle of either seat rail moved 5 mm or more over the last 1 s.
  - **1.0** when none of that is so, held for 1 s.

`success()` holds when the 1.0 conditions do; the harness ends the trial once it has held for 1 s.

## Runtime and seeds

- Demo: about 424 control steps (21.2 s simulated) to success; about 3.2 minutes of wall time with 640 x 360 cameras on one L40S GPU,
  including start-up. The slowest 30 steps took 12.3 s.
- Dream-1: 1,200 steps (60 s simulated), about 8 minutes of wall time at 8 rows per request (one session, 146
  requests).
- `object_jitter=0` keeps the layout fixed, so no seed is needed. The chair, the floor and the demo are deterministic;
  two runs of the demo on the same machine gave identical trajectories.

## Sources

All meshes are generated in `chair.py` from primitives (tubes, tori, boxes); there are no third-party assets or
textures. Sizes follow manufacturers' published specifications for standard 18 in lightweight folding wheelchairs:
457 mm seat width, about 63 cm open and 30 cm folded width (this model: 61 and 28 cm over the handrims), 48-50 cm seat height, 24 in rear wheels with handrims,
8 in front casters, about 13 kg without footrests.
