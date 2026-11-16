# Prune a dead rose stem above its bud without dropping the cutting

A task file for the `yam_arena` embodiment, using Isaac Lab-Arena 0.3.1.

A potted rose has five live canes and one dead cane, all leaning out over the pot's rim. The dead cane stands between two live canes that lean the same way, about 5 cm from it on either side at its bud. It is green near the soil, with a small red bud, and brown and leafless beyond. Bypass secateurs lie open on the table, and a garden tub stands beside the pot. The robot is told to cut out the dead stem and put it in the bin. The instruction does not say where to cut, or that the cut part must be held: a good cut is in the green wood, 3 to 15 mm beyond the bud, the blades must close on no live cane, and the thorny cutting must not drop. One arm works the secateurs by squeezing their handles; the other must already hold the cane beyond the cut when the blades close, then carry the cutting to the bin.

| File | What |
|---|---|
| `task.py` | The task file: objects, the cut, the failures that stay failed, `score()`, `success()`, `demo()` |
| `scene_assets.py` | The rose bush (its layout is the table `LIVE_CANES`), the secateurs and the tub, written as plain-text USD when the task loads |
| `rig_template.py`, `rig_template.svg` | The real-world layout from the same numbers: the cane table below, and a printable 1:1 template |
| `scenekit.py`, `usdkit.py` | Helpers: USD bodies and articulations on `yam_arena`; meshes, materials and colliders as USD text |
| `cut_checks.py` | Finite blade footprints, the blade's volume against live canes, and bud plane-section contact |
| `pruning_checks.py` | First-cut history, irreversible bud damage and final-stub credit |
| `motion_checks.py` | The one-second stillness check for both ends of the cutting |

## Run a trial

From this folder, in a shell with the `yam_arena` environment active.

Dream-1:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  -P rows_per_request=8 \
  --instruction "Cut out the dead stem on the rose bush and put it in the bin." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir runs/dream
```

Scripted demo:

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=360 \
  --instruction "Cut out the dead stem on the rose bush and put it in the bin." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir runs/demo
```

The run prints `task_score` at the end: 1.0 once `success()` has held for 1 s, otherwise `score()` at the end of the run.

Scorer checks use the same scripted command with `TASK_DEMO` set: `green` closes the jaws on the live cane beside the dead one; `failure` does not hold the cutting; `brown` places the first cut in dead wood; `recut` makes a good first cut and then cuts below its retained bud; `side_bud` closes across the bud; `retrim` makes a harmless second trim above the bud; `repair` makes a misplaced first cut and then corrects the stub. These select waypoint plans only; scoring is the same for all.

## Real-world setup and reset

### Parts

- **Pot:** a shallow terracotta pot (a bulb pan), 22 cm across the rim and 7.5 cm tall. Inside: wet floral foam (a standard soaked foam brick, cut to fit) up to 2.4 cm below the rim, then 1.5 cm of potting soil, firmed level, so the soil is 9 mm below the rim. About 3 kg in all.
- **Live canes (5):** fresh cut red roses from a florist, stems 8 to 10 mm thick, thorns left on, leaves left on above the height in the table. C keeps its open flower; B and D are roses in tight bud; A and E have the flower head cut off.
- **Dead cane (1 per trial):** a fresh rose stem about 10 mm thick, prepared as below.
- **Secateurs:** ordinary bypass secateurs, 19 cm long, about 0.2 kg, red plastic-covered handles, steel blade, black steel hook, with the usual spring that holds them open.
- **Bin:** an open black plastic garden tub, 26 cm across the top and 12 cm tall.
- **Also:** the printed template `rig_template.svg`, a wooden skewer, a tape measure, masking tape, matte brown water-based acrylic paint and a brush.

### Layout

Positions are in metres from the midpoint between the two arm bases (x forward, y to the robot's left); heights are above the table top. In the pot's frame, x and y are measured from the pot's centre in the same directions. "Leans toward" is the direction of the lean seen from above, in degrees counterclockwise from straight away from the robot (90 = the robot's left); the template's arrows show it. Each stem goes 6 cm into the pot, measured along the stem from the soil; "stem to cut" is its length above the soil plus those 6 cm.

| Cane | Base in the pot (x, y) | Base on the station (x, y) | Lean from upright | Leans toward | Above the soil | Stem to cut | Tip on the station (x, y, height) | Carries |
|---|---|---|---|---|---|---|---|---|
| A | -8.0, -1.0 cm | 0.290, 0.020 m | 50 deg | 108 deg | 26 cm | 32 cm | 0.228, 0.209, 0.233 m | leaves from 13.5 cm along; flower head cut off |
| dead | -4.0, -3.0 cm | 0.330, 0.000 m | 55 deg | 90 deg | 24 cm | 30 cm | 0.330, 0.197, 0.204 m | bud 5.5 cm along, facing the robot; brown from 7.9 cm along; leafless; square-cut tip |
| B | -0.3, -2.1 cm | 0.367, 0.009 m | 50 deg | 70 deg | 29 cm | 35 cm | 0.443, 0.218, 0.252 m | leaves from 13.0 cm along; flower in tight bud |
| C | +4.0, +3.0 cm | 0.410, 0.060 m | 40 deg | 40 deg | 30 cm | 36 cm | 0.558, 0.184, 0.296 m | leaves from 11.0 cm along; open bloom |
| D | +5.5, -1.2 cm | 0.425, 0.018 m | 35 deg | -5 deg | 30 cm | 36 cm | 0.596, 0.003, 0.312 m | leaves from 12.0 cm along; flower in tight bud |
| E | +2.5, -6.0 cm | 0.395, -0.030 m | 40 deg | -40 deg | 27 cm | 33 cm | 0.528, -0.142, 0.273 m | leaves from 12.0 cm along; flower head cut off |

At the bud's height, A is 5.3 cm and B 5.0 cm from the dead cane, centre to centre. The pot's centre is at (0.37, 0.03). The secateurs lie flat and open, the pivot at (0.22, -0.13), the jaws pointing to the robot's left. The tub stands at (0.20, 0.37).

### Build

1. Cut a floral foam brick to fit the pot, soak it until it sinks, press it in to 2.4 cm below the rim, and cover it with 1.5 cm of potting soil.
2. Print `rig_template.svg` at 100 % on A4; its bar must measure 100 mm. Cut out the disc and lay it on the soil, centred, its arrow pointing away from the robot. At each ring, push the skewer 6 cm in along its arrow at its lean angle (use the angle cards on the same sheet, held upright beside the skewer). Lift the disc off.
3. Cut each rose to its "stem to cut" length, measured from its bottom to the flower's base or the cut end. Strip its leaves below the height in the table, and cut off the flower head where the table says so. Push each into its hole, 6 cm along the stem, leaning along its arrow at its angle. Check each tip against the table with the tape measure: within 1 cm.
4. Put in a prepared dead cane (below).
5. Stand the pot with its centre at (0.37, 0.03) and the template's arrow direction pointing straight away from the robot. Mark its outline and its robot-side point on the table with tape. Mark the open secateurs and the tub in their places too.

**Preparing a dead cane** (a few minutes; prepare several ahead, the paint needs about 15 minutes to dry): take a fresh rose stem about 10 mm thick with a bud (a node, where a leaf grew). Cut its bottom 11.5 cm below the bud's middle and its top square, 30 cm from the bottom. Remove all its leaves, leaf stalks and any flower; leave the thorns. Paint it matte brown from 2.4 cm above the bud's middle (2 cm above the bud's top) to the tip, the cut end included, so that only the bottom part and the bud stay green. In the pot it stands 24 cm above the soil, the bud 5.5 cm along it.

### Reset after each trial

1. Pull out the used dead cane and any cut pieces; empty the tub.
2. Push a freshly prepared dead cane into its hole: 6 cm in, leaning 55 degrees toward the robot's left, the bud facing the robot.
3. Check each live cane's tip against the table. Replace any live cane that is cut, broken, bent or wilted with a fresh rose prepared as in step 3. Keep the foam wet.
4. Put the pot, the open secateurs and the tub back on their marks, and the arms at the ready pose.

## How the simulation judges each rule

- **The cut:** the jaws have no plant colliders. On the open-to-closed transition, the dead cane's material must intersect the current finite blade mesh within 0.8 to 5.5 cm of the pivot, using the cane's physical radius as a circle inside its oblique section. The recorded cut height is that intersection, fixed when the cut occurs. Earlier entry into the open jaws never authorizes a later cut. This is geometric contact, without a cutting-force model.
- **A live cane is cut** when, at the moment the jaws close fully, any part of it within its radius overlaps the closed blade's metal, whatever the tool's position or tilt. That scores 0 for good.
- **The dead cane parts there:** the stub's sleeves slide back into the soil to the cut, and the cut-off part's sleeves slide out to it. Until the cut, that part is held in place by a chain of six stiffly driven joints; the cut releases them. The retained stub remains cuttable: a later closure through it shortens its visible sleeves, without changing the first cutting. Later small trimmings are not free bodies.
- **Plant preservation:** any cut through or below the original bud, or across the bud itself (its actual mesh against the blade during any closing step), permanently scores 0. The stub's end and the bud's pose are read from their bodies every step; the bud must stay within 3 mm of its original attachment point, with the stub still extending past its top.
- **Good cut:** both the recorded first cut and the final stub end must be 3 to 15 mm beyond the original bud. A bad first cut followed by a correction stays at 0.25.
- **Held:** both finger pads of one gripper within 4 mm of the cutting's surface, with the gripper closed to at most the cane's width plus 6 mm.
- **Falls:** not held, and either end of the cutting 3 cm or more below its highest point since it was last held or over the bin, while its middle is not over the bin's opening.
- **In the bin:** not held, its middle over the bin's opening, its lowest end below the rim, and the bin upright within 30 degrees.
- **Still:** neither end of the cutting moves 5 mm or more between any two samples in the last 1 s (21 samples at 20 Hz). Every other full-credit condition must also hold in all those 21 samples.
- **Secateurs on the table outside the pot:** their middle within 2.2 cm of the table, flat within 20 degrees, and no sampled point (pivot, jaw tips, handles) over the pot.
- **Fingertips:** the middles and tips of both grippers' finger pads, measured to the surfaces of all canes and the cutting (their axes, less their radius), and to the secateurs (lines along each half's jaw and handle, less 8 mm).

## Runtime and seeds

- Scripted demo: full score after 356 steps (17.8 s simulated); 347 s of wall time at 640 x 360, start-up included, with the slowest 30 control steps taking 9.4 s. Wall time depends on GPU load.
- Dream-1: 1,200 steps (60 s simulated); about 12 minutes of wall time at 8 rows per request (one session), start-up included.
- The run limit is 1,200 steps (60 s at 20 Hz).
- `object_jitter=0` keeps the layout fixed, so no seed is needed. Scripted waypoints and timed waits are open loop.

## Scoring checks

`python -m pytest tests` checks the scorer without the simulator: closing the blades across each live cane at several heights and from eight directions, with the cane nearly in the blade's plane, or anywhere in the jaws; a careful cut on the dead cane from every side missing its neighbours; both-endpoint stillness; irreversible bud damage; first and final cut quality; finite blade contact; bud plane-sections; a withdrawn, partly closed tool; and the 21-sample hold. It needs NumPy and pytest in the task environment.

## Sources

All meshes are generated by `scene_assets.py`; there are no third-party assets. Sizes come from common products: a 22 cm shallow terracotta pot, florist's red roses (8 to 10 mm stems), 19 to 21 cm bypass secateurs, a 26 cm garden tub.
