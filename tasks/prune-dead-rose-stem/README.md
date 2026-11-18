# Prune a dead rose stem above its bud without dropping the cutting

A task file for the `yam_arena` embodiment, using Isaac Lab-Arena 0.3.1.

A potted rose has five live canes and one dead cane, all leaning out over the pot's rim. The dead cane stands between two live canes that lean the same way, about 5 cm from it on either side at its bud. It is green near the soil, with a small red bud, and brown and leafless beyond. Bypass secateurs lie open on the table, and a garden tub stands beside the pot. The robot is told to cut out the dead stem and put it in the bin. The instruction does not say where to cut, or that the cut piece must be held: a good cut is in the green wood, 3 to 15 mm beyond the bud; the blades must press on no live cane; and the cutting must already be held by a gripper when the blades part it, and must never drop. One arm works the secateurs by squeezing their handles while the other holds the cane beyond the cut, then carries the cutting to the bin.

| File | What |
|---|---|
| `task.py` | The task file: objects, the cut, the failures that stay failed, `score()`, `success()`, `demo()` |
| `scene_assets.py` | The rose bush (its layout is the table `LIVE_CANES`), the secateurs and the tub, written as plain-text USD when the task loads |
| `rig_template.py`, `rig_template.svg` | The real-world layout from the same numbers: the cane table below, and a printable 1:1 template |
| `scenekit.py`, `usdkit.py` | Helpers: USD bodies and articulations on `yam_arena`; meshes, materials and colliders as USD text |
| `cut_checks.py`, `blade_contact.py` | The blade's shape against each cane, the bud's contact with the blade, and the blade's measured load on a cane |
| `pruning_checks.py` | First-cut history, lasting bud damage and final-stub credit |
| `motion_checks.py` | Held, over the bin's opening, secateurs clear of the tub, and the one-second stillness check |
| `pruning_control.py`, `pruning_observation.py`, `pose_motion.py`, `supported_pitch.py`, `functional_pads.py`, `tool_mass.py`, `steptime.py` | The scripted demo's motion and its readings of the robot and the secateurs |

## Run a trial

From this folder, in a shell with the `yam_arena` environment active.

Dream-1:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  -P rows_per_request=6 \
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

The run prints `task_score` at the end: 1.0 once `success()` has held for 1 s, otherwise `score()` at the end of the run. The scripted demo scores 1.0 in 802 steps.

`TASK_DEMO=failure` runs the same two-hand cut, then opens the holding hand outside the bin, so the cutting drops: it scores 0. The scorer is the same in both modes. Scoring exploits and partial-credit cases are covered by the CPU tests below.

## Real-world setup and reset

### Parts

- **Pot:** a shallow terracotta pot (a bulb pan), 22 cm across the rim and 7.5 cm tall. Inside: wet floral foam (a standard soaked foam brick, cut to fit) up to 2.4 cm below the rim, then 1.5 cm of potting soil, firmed level, so the soil is 9 mm below the rim. About 3 kg in all.
- **Live canes (5):** fresh cut red roses from a florist, stems about 7 mm thick, all thorns and sharp bases stripped, leaves left on above the height in the table. C keeps its open flower; B and D are roses in tight bud; A and E have the flower head cut off.
- **Dead cane (1 per trial):** the intact upper part of a fresh rose stem, 4.3 to 4.7 mm thick (4.5 mm nominal), prepared as below.
- **Secateurs:** ordinary full-size bypass secateurs, 195 mm overall, 49 mm cutting edge and 160 g, red plastic-covered handles, steel blade, black steel hook, with the usual spring that holds them open. The ARS 120DX matches these figures; it is named for identification only.
- **Bin:** an open black plastic garden tub, 26 cm across the top and 12 cm tall.
- **Also:** the printed template `rig_template.svg`, a wooden skewer, a tape measure, masking tape, a matte brown permanent marker and clean white tissue.

### Layout

Positions are in metres from the midpoint between the two arm bases (x forward, y to the robot's left); heights are above the table top. In the pot's frame, x and y are measured from the pot's centre in the same directions. "Leans toward" is the direction of the lean seen from above, in degrees counterclockwise from straight away from the robot (90 = the robot's left); the template's arrows show it. Each stem goes 6 cm into the pot, measured along the stem from the soil; "stem to cut" is its length above the soil plus those 6 cm.

| Cane | Base in the pot (x, y) | Base on the station (x, y) | Lean from upright | Leans toward | Above the soil | Stem to cut | Tip on the station (x, y, height) | Carries |
|---|---|---|---|---|---|---|---|---|
| A | -5.9, -1.7 cm | 0.303, -0.007 m | 50 deg | 140 deg | 26 cm | 32 cm | 0.150, 0.121, 0.233 m | leaves from 13.5 cm along; flower head cut off |
| dead | -1.8, -4.7 cm | 0.345, -0.037 m | 55 deg | 122 deg | 31 cm | 37 cm | 0.209, 0.178, 0.244 m | bud 12.5 cm along, facing 212 degrees from forward; brown from 14.9 cm along; leafless; square-cut tip |
| B | -1.2, -1.8 cm | 0.351, -0.008 m | 50 deg | 102 deg | 29 cm | 35 cm | 0.304, 0.210, 0.252 m | leaves from 13.0 cm along; flower in tight bud |
| C | +1.8, +4.7 cm | 0.380, 0.057 m | 40 deg | 72 deg | 30 cm | 36 cm | 0.439, 0.240, 0.296 m | leaves from 11.0 cm along; open bloom |
| D | +5.3, +1.9 cm | 0.416, 0.029 m | 35 deg | 27 deg | 30 cm | 36 cm | 0.569, 0.108, 0.312 m | leaves from 12.0 cm along; flower in tight bud |
| E | +5.3, -3.7 cm | 0.416, -0.027 m | 40 deg | -8 deg | 27 cm | 33 cm | 0.588, -0.051, 0.273 m | leaves from 12.0 cm along; flower head cut off |

At the bud's height, A is 5.3 cm and B 5.0 cm from the dead cane, centre to centre. The pot's centre is at (0.363, 0.010). The secateurs lie flat and open, the pivot at (0.220, -0.130), the jaws pointing 27 degrees toward +y from forward (+x). The tub stands at (0.260, 0.370). The template gives the exact angles (for example 122.2 degrees for the dead cane's lean and 212.2 degrees for its bud).

### Build

1. Cut a floral foam brick to fit the pot, soak it until it sinks, press it in to 2.4 cm below the rim, and cover it with 1.5 cm of potting soil.
2. Print `rig_template.svg` at 100 % on A4; its bar must measure 100 mm. Cut out the disc and lay it on the soil, centred, its arrow pointing away from the robot. At each ring, push the skewer 6 cm in along its arrow at its lean angle (use the angle cards on the same sheet, held upright beside the skewer). Lift the disc off.
3. Cut each rose to its "stem to cut" length, measured from its bottom to the flower's base or the cut end. Strip its leaves below the height in the table, and cut off the flower head where the table says so. Push each into its hole, 6 cm along the stem, leaning along its arrow at its angle. Check each tip against the table with the tape measure: within 1 cm.
4. Put in a prepared dead cane (below).
5. Stand the pot with its centre at (0.363, 0.010) and the template's arrow pointing straight away from the robot. Mark its outline and its robot-side point on the table with tape. Mark the open secateurs and the tub in their places too.

**Preparing a dead cane** (a few minutes): take the intact upper part of a fresh rose stem, 4.3 to 4.7 mm thick, with a bud (a node, where a leaf grew). Keep stems with their cut ends in clean water at 2 to 5 °C, use them within 48 hours of purchase, and let them stand 30 minutes at room temperature first; never use a dried or brittle stem. Cut its bottom 18.5 cm below the bud's middle and its top square, 37 cm from the bottom. Remove all its leaves, leaf stalks and any flower; strip the thorns and sharp bases. Colour it matte brown with the permanent marker from 2.4 cm above the bud's middle (2 cm above the bud's top) to the tip, the cut end included, so that only the bottom part and the bud stay green. Let the ink dry for two minutes and check that white tissue picks up no colour. In the pot it stands 31 cm above the soil, the bud 12.5 cm along it.

### Reset after each trial

1. Pull out the used dead cane and any cut pieces; empty the tub.
2. Within five minutes, push a freshly prepared dead cane into its hole: 6 cm in, leaning 55 degrees along its arrow on the template, with its bud facing 212.2 degrees from forward.
3. Check each live cane's tip against the table. Replace any live cane that is cut, marked by a blade, broken, bent or wilted with a fresh rose prepared as in step 3. Keep the foam wet.
4. Put the pot, the open secateurs and the tub back on their marks, and the arms at the ready pose.

### Scoring a real trial

- **Where the cut is:** measure the cutting's length with the tape. A first cut 3 to 15 mm beyond the bud's top frees a cutting 16.6 to 17.8 cm long. Then measure the stub along its axis from the bud's top to its cut face: 3 to 15 mm is good.
- **The bud:** intact if no blade touched it and it is still on the stub where it started.
- **Held, falls, in the bin, fingertips clear, still:** from the video, with the tub's rim and the secateurs in view.

## How the simulation judges each rule

- **The cut:** the secateurs' blade and hook have colliders against every cane. The dead cane parts only when the blade presses it, between the jaws, with at least 142.25 N at two control steps in a row (50 ms apart). The load is read from the blade's contact with that cane, and the blade's stiff spring mounting on its handle must bend by a matching amount, so a push against the table, the pot or another cane cannot cut it. The gripper's fingers are limited to 40 N each.
- **A live cane is cut** when it lies between the jaws and the blade presses it with 10 N or more, read the same way. That scores 0 for good.
- **The dead cane parts there:** the stub's sleeves slide back into the soil to the cut, and the cut-off part's sleeves slide out to it. Until the cut, that part is held in place by a chain of six stiff joints; the cut releases them, and the cutting then moves freely. Right after the cut, the stub's collider stops just below its cut face, so the closed blades are not caught on it; it returns to the cut face once the pivot is 8 cm away. The stub can be cut again: a later cut shortens it without changing the first cutting. Later small trimmings are not free pieces.
- **Plant preservation:** any cut through or below the original bud, or a blade crossing the bud itself while it closes, scores 0 for good. The stub's end and the bud's position are read from their bodies every step; the bud must stay within 3 mm of its starting place, with the stub still reaching past its top.
- **Good cut:** both the first cut and the stub's final end must be 3 to 15 mm beyond the original bud. A bad first cut followed by a correction stays at 0.25.
- **Held:** one gripper closed to at most the cutting's width plus 6 mm, with both of its finger-pad middles within 4 mm of the cutting's surface.
- **Held as it parts:** at the control step the dead cane parts, a gripper must already hold the cutting. If none does, the cutting has fallen, and the score is 0 for good.
- **Falls:** later, not held, and either end of the cutting 3 cm or more below its highest point since it was last held or over the bin, while its middle is not over the bin's opening.
- **In the bin:** not held, both ends of the cutting over the bin's opening seen from above, its lowest end below the rim, and the bin upright within 30 degrees.
- **Still:** neither end of the cutting moves 5 mm or more between any two samples in the last 1 s (21 samples at 20 Hz). Every other full-credit condition must also hold in all those 21 samples.
- **Secateurs on the table:** their pivot (at mid-thickness) less than 2.15 cm above the table, flat within 20 degrees, and no sampled point (pivot, jaw tips, handles) over the pot, or over the tub's footprint (its axis from base to rim, widened by the rim's radius, so a tipped tub counts too).
- **Fingertips:** the middles and tips of both grippers' finger pads, measured to the surfaces of all canes and the cutting, and to the secateurs (lines along each half's jaw and handle, less 8 mm).

## Runtime and seeds

The run limit is 1,200 steps (60 seconds at 20 Hz); physics runs at 240 Hz. The fixed layout uses `object_jitter=0`; there is no randomness to seed. The scripted demo's 802 steps took about 5 minutes (308 s) of wall time on one GPU, including start-up; a full 1,200-step trial takes about 8 to 10 minutes of simulation, before any model response time. The Dream-1 run of the full 1,200 steps took about 12 minutes of wall time.

The scripted demo uses only ordinary joint-position commands at 20 Hz. It reads the robot's joints, the secateurs' pose and contacts to know when each move has arrived, and it never moves an object or changes a joint, force, gain or material. The right hand picks up the secateurs, turns them head down, opens them against their stop, slides the open jaws onto the dead cane, aimed 9 mm beyond the bud's top, and squeezes the handles over two seconds. The left hand already holds the cane near its tip. After the cut, the right hand lays the secateurs flat on the table; the left hand draws the cutting out along its axis, carries it over the tub, lowers it in and lets go; both hands move clear.

## Scoring checks

`python -m pytest tests` checks the scorer without the simulator: closing the blades on each live cane at several heights and from eight directions; a careful cut on the dead cane from every side missing its neighbours; a cut needing the blade's own load on the dead cane, from neither the table, the pot nor another cane; lasting bud damage; first and final cut quality; a cutting no gripper holds as it parts, even one that drops straight into the bin; a cutting lying across the rim; secateurs dropped into the tub; both-end stillness; and the 21-sample hold. It needs NumPy, SciPy and pytest in the task environment.

## Sources

All meshes are generated by `scene_assets.py`; there are no third-party assets. Sizes come from common products: a 22 cm shallow terracotta pot, florist's red roses, 195 mm / 49 mm / 160 g bypass secateurs, a 26 cm garden tub.

- **Secateurs:** the [ARS 120DX specification](https://www.arscorporation.jp/product/detail-spec.html?id=59) gives 195 mm overall, a 49 mm edge and 160 g; ARS is named for identification only. The simulated secateurs match those figures; their inside (a steel spine in plastic grips, a 3 mm steel neck) is estimated, not copied from the product. Their opening spring has 0.30 rad of preload and 1 N m/rad of stiffness.
- **Stem size and cutting load:** a [study of rose stem mechanics](https://pmc.ncbi.nlm.nih.gov/articles/PMC10011599/) gives a mean upper-stem diameter of 4.64 mm and a shear strength of 8.944 MPa (cultivar Dianhong, in a shear fixture, not with secateurs). The dead cane is 4.5 mm thick, the size of the upper stem where such a cut is made. Its shear strength times its 4.5 mm round section gives the 142.25 N cutting load. A real cut's peak also depends on the blade's sharpness, its slicing action, the cut's angle and the stem, so this is an engineering reference, not a measured secateur peak.
