# Sweep spilled cereal into a dustpan held flat

Twenty oat cereal rings are spilled over a 20 x 20 cm patch of the table. A hand dustpan lies flat beside the spill and a
hand brush lies behind it. The robot is asked to sweep the cereal into the dustpan. It has to do it the way people do: one
hand holds the dustpan by its handle, its rubber lip pressed flat on the table, while the other brushes the pieces
together and then over the lip into the pan with short strokes. If the pan lifts or tilts, the pieces slide under the
lip; if the brush flicks them, they scatter.

A task file for the `yam_arena` embodiment ([yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena),
Isaac Lab-Arena 0.3.1), on the station scene.

## Files

| File | What |
|---|---|
| `task.py` | The task file: objects, the failure tracker, `success()`, `score()` and the scripted `demo()` |
| `scene_assets.py` | Sizes, meshes, materials and colliders of the dustpan, the brush and the cereal rings, written as USD on load |
| `scenekit.py`, `usdkit.py` | Helpers: USD bodies and per-step callbacks for yam_arena task files; plain-text USD meshes and materials |
| `steptime.py` | Records the simulated time and the slowest control steps (only when `TASK_STEP_FILE` is set) |
| `textures/` | Noise textures (beech grain, bristles, plastic, toasted oat), made by `scene_assets.py` |
| `layout_template.py`, `layout_template.svg` | A 1:1 A4 template of the rings' start spots for real trials, written from `task.py`'s layout |

`scenekit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task runs
with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

From this folder, in a shell with the yam_arena environment active:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Sweep the spilled cereal into the dustpan." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir runs/dream
```

Scripted demo (scores 1.0): the same command with `--policy yam_scripted`. A scripted failure, `TASK_DEMO=failure` with the
same command, presses the dustpan's handle 2 cm down while the brush sweeps: the pan tips back on its heel, its lip
lifts, and the pieces pass under it.

The run prints `task_score` at the end: 1.0 once `success()` has held for 1 s, otherwise `score()` at the end.

## Real-world setup and reset

- **Objects:** an ordinary plastic hand dustpan about 25 cm wide with a soft rubber front lip and a short handle; a hand
  brush about 28 cm long with a wooden back and stiff polypropylene bristles; plain toasted oat cereal rings, a common
  breakfast cereal, about 12 mm across; a laminate table top. Sizes and masses as in `task.json`'s `setup`.
- **Marking the layout (once):** print `layout_template.svg` at 100 % and check its 100 mm bar. Lay it with its left edge
  5.5 cm to the right of the centre line between the arms, parallel to it, and the dashed patch's near edge 25 cm in front
  of the arms' base line (the line through both arm mounts). Prick the 20 ring centres onto the table with a pencil point
  and remove the sheet. Outline the dustpan (the middle of its lip's front edge 34 cm forward on the centre line, opening
  to the right) and the brush (bristles centred 35 cm forward, 30 cm right) with tape.
- **Before each trial:** lay one ring flat on each dot (a fresh ring for any broken one) and put the dustpan and the brush
  flat on their outlines.
- **Scoring:** count the rings in the pan from a photo from above at the end; the 5 cm and 2 cm clearances and the
  stillness come from the video.

## How the simulation scores

- **In the dustpan:** a ring's centre, in the dustpan's own frame, is at least 3 cm behind the lip's front edge (past the
  black rubber strip), inside the side walls and in front of the back wall, and between 3.5 mm (a ring lying flat on the
  2 mm floor has its centre 4.5 mm up) and 3 cm above the pan's bottom. A ring under the pan or on the lip never counts.
- **Falls off the table:** the dustpan's or the brush's origin drops more than 10 cm below the table top. Latched: it
  stays 0 until a reset.
- **Flat:** the dustpan is tilted less than 5 degrees and both ends of its lip are within 5 mm of the table top.
- **Let go:** every fingertip (the middle of each pad's tip) is at least 2 cm from every collider of both tools, and at
  least 5 cm from every ring.
- **Still:** over the last 21 samples (1 s at 20 Hz), no ring's position moved more than 5 mm between any two samples.

## Scripted demo

20 Hz, straight-down grips; the brush is held level.

1. The left arm grips the dustpan's handle 8 cm behind the pan (fingers across the handle) and lifts it 4 mm, tipping the
   pan forward onto its lip, as a hand does, so the lip stays pressed on the table while the brush works.
2. The right arm grips the brush's handle 3.3 cm from the bristle block.
3. Two gathering strokes, over the left and right halves of the spill, push the rings towards the dustpan and stop short
   of its lip.
4. Two strokes from behind the gathered rings carry them up the lip and 3 cm into the pan, 2 cm either side of the pan's
   centre so the brush stays inside its walls. These two strokes are aimed at where the held dustpan is (the demo reads
   its pose once, 8 s in: closing on the handle turns the pan about 2 degrees); the demo moves nothing in the simulation.
5. The left arm sets the pan flat and lets go; the brush is put back where it lay; both grippers open and rise.

## Runtime and seeds

- Demo: 819 steps (41 s simulated); about 5.5 minutes of wall time on an L40S at 640 x 360, including start-up.
- `object_jitter=0` keeps the layout fixed, so no seed is needed. The scripted demo is deterministic; apart from the one
  reading of the dustpan's pose in step 4 it is open loop.

## Physics notes

- Rigid bodies on GPU PhysX at 240 Hz, the station's settings. Each ring is the convex hull of its torus (60 points),
  0.3 g; the dustpan 0.20 kg and the brush 0.11 kg, their mass spread over their colliders.
- The bristles are approximated by one rigid block whose long bottom edges are bevelled (24 mm across the tips, 40 mm
  at 7 mm up), standing in for tips that lean and splay; it slides on the table with polypropylene friction.
- The 0.4 mm rubber lip is ten 25 mm wedges; the pan's floor, back wall and handle are box shapes.
- Friction (static/dynamic), averaged per contact as PhysX does by default: table 0.5/0.5 (PhysX's default material),
  cereal 0.45/0.40, polypropylene pan 0.35/0.30, rubber lip 0.8/0.7, beech 0.5/0.45, bristle tips 0.35/0.30.
- `TASK_DEBUG=1` with `TASK_SCORE_DIR` set writes every control step's tool, ring and fingertip positions to
  `TASK_SCORE_DIR/telemetry.jsonl`.

## Sources

All meshes are built in `scene_assets.py` from boxes, prisms, tubes and a torus; all textures are generated from noise in
the same file. No third-party assets. Sizes follow ordinary products: a plastic hand dustpan 25 cm wide (a common 10-inch
size) and 21.5 cm deep with a 12 cm handle, a 28 cm hand brush with an 18 x 4 cm bristle block, and ring-shaped oat cereal about 12 mm across.
