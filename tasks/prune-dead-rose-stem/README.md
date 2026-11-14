# Prune a dead rose stem above its bud without dropping the cutting

A task file for the `yam_arena` embodiment, using Isaac Lab-Arena 0.3.1.

A potted rose has four green, leafy canes and one dead cane, which leans out over the pot's rim. The dead cane is green near the soil, with a small red bud, and brown beyond. Bypass secateurs lie open on the table, and a garden tub stands beside the pot. The robot is told to cut out the dead stem and put it in the bin. The instruction does not say where to cut, or that the cut part must be held: a good cut is in the green wood, 3 to 15 mm beyond the bud, and the thorny cutting must not drop. One arm works the secateurs by squeezing their handles; the other must already hold the cane beyond the cut when the blades close, then carry the cutting to the bin.

| File | What |
|---|---|
| `task.py` | The task file: objects, the cut, the failures that stay failed, `score()`, `success()`, `demo()` |
| `scene_assets.py` | The rose bush, the secateurs and the tub, written as plain-text USD when the task loads |
| `scenekit.py`, `usdkit.py` | Helpers: USD bodies and articulations on `yam_arena`; meshes, materials and colliders as USD text |
| `motion_checks.py` | The one-second stillness check for both ends of the cutting |
| `pruning_checks.py` | First-cut history, irreversible bud damage and final-stub credit |
| `cut_checks.py` | Finite blade footprints and complete bud plane-section contact |

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

Scorer checks use the same scripted command with `TASK_DEMO` set: `failure` does not hold the first cutting; `brown` places the first cut in dead wood; `recut` makes a good first cut and then cuts below its retained bud; `side_bud` closes across the side-facing bud; `retrim` makes a harmless second trim above the bud; `repair` makes a misplaced first cut and then corrects the stub. These select waypoint plans only; scoring always uses the same physical geometry and cut history.

The run prints `task_score` at the end: 1.0 once `success()` has held for 1 s, otherwise `score()` at the end of the run.

## How the simulation judges each rule

- **The cut:** the jaws have no plant colliders. On the open-to-closed transition, the material must intersect the current finite blade mesh within 0.8 to 5.5 cm of the pivot. A cane's physical radius gives an inscribed circular cross-section inside its oblique elliptical section; contact is conservative and never expands the cane. A bud is checked as a complete plane-section of its actual convex mesh, including crossing boundaries and containment. Real blade contact during any closing step irreversibly damages the bud, even if the blade has passed beyond it by full closure. A bypass blade may have passed the angular jaw gap while its metal still intersects the material. Earlier jaw entry cannot authorize a distant later cut. The recorded cut height is the current blade/material intersection, fixed when the cut occurs. This is geometric contact, without a cutting-force model.
- **The dead cane parts there:** the stub's sleeves slide back into the soil to the cut, and the cut-off part's sleeves slide out to it. Until the cut, that part is held in place by a chain of six stiffly driven joints; the cut releases them. The retained stub remains cuttable. Each later closure through it shortens its visible sleeves, without changing the first cutting. The original bud is also checked by intersecting the blade plane with its actual mesh, from any approach direction; a cut that severs it removes its visible mesh. Later small trimmings are not independent free bodies.
- **Plant preservation:** any cut through or below the original bud, or across the bud itself, permanently scores 0. Each step reads the actual stub sleeve endpoints and bud body pose. The bud must remain within 3 mm of its original attachment point, with the stub still extending past its top; putting a detached bud back does not clear the failure.
- **Good cut:** both the recorded first cut and the final stub end must be 3 to 15 mm beyond the original bud. A bad first cut followed by a correction stays at 0.25. A harmless second trim within the good range can still reach full credit.
- **Held:** both finger pads of one gripper within 4 mm of the cutting's surface, with the gripper closed to at most the cane's width plus 6 mm.
- **Falls:** not held, and either end of the cutting 3 cm or more below its highest point since it was last held or over the bin, while its middle is not over the bin's opening.
- **In the bin:** not held, its middle over the bin's opening, its lowest end below the rim, and the bin upright within 30 degrees.
- **Still:** neither end of the cutting moves 5 mm or more between any two samples in the last 1 s (21 samples at 20 Hz). Every other full-credit condition must also be true in all 21 control-step samples and in the current state. Score queries do not advance this window; the motion check covers the same second, without adding a second hold.
- **Secateurs on the table outside the pot:** their middle within 2.2 cm of the table, flat within 20 degrees, and no sampled point (pivot, jaw tips, handles) over the pot.
- **Fingertips:** the middles and tips of both grippers' finger pads, measured to the surfaces of the canes and the cutting (their axes, less their radius), and to the secateurs (lines along each half's jaw and handle, less 8 mm).

## Runtime and seeds

- Current scripted demo: 356 steps (17.80 s simulated), 156.65 s wall time at 640 x 360; slowest 30 control steps 11.24 s. Environment revision `e7a06b34d41d`, source `da4a980`. Wall time depends on GPU load.
- Dream-1: 1,200 steps (60 s simulated), about 10 minutes wall time.
- The run limit is 1,200 steps (60 s at 20 Hz).
- `object_jitter=0` keeps the layout fixed, so no seed is needed. Scripted waypoints and timed waits are open loop.

## Dream-1 run

The Dream-1 video is a complete 1,200-step episode on this revision (environment revision `e7a06b34d41d`), run with the command above at 320 x 176 and 8 action rows per request, in one model session from the first action to the end. Its scorer reported 0.0: the arms moved towards the plant and the secateurs, and the dead cane was not cut.

## Scripted controls

The following control episodes used environment revision `ed7f64ab4f90`, source `081cd51`:

| Control | Score | Recorded result |
|---|---:|---|
| Good first cut, then lower stub cut | 0.0 | Real blade contact removes the retained bud and shortens the stub |
| Side-facing bud contact | 0.0 | Actual bud contact at 14.45 s, before a further stub cut at 14.75 s |
| Harmless second trim | 1.0 | Original bud remains attached; both first cut and final stub are in range |
| Bad first cut, then corrected stub | 0.25 | A good final stub cannot improve the misplaced first cut |

The current source selects the actual 4.85 mm brown-sleeve radius, with 5 mm green/top sections. Every recorded cane contact in these controls has zero blade-footprint distance and remains within this tighter gate. This is an offline contact compatibility check; these controls were not rerun on source `da4a980`. The complete current normal demo above was run on that source.

## Scoring checks

`python -m pytest tests` checks both-endpoint stillness, irreversible bud damage, first/final cut quality, free-cutting geometry, finite blade contact, complete bud plane-sections, a withdrawn partly closed tool, legal bypass contact, exact material radii, and the 21-control-sample full-condition hold. Score and success queries must not advance the hold. These checks use NumPy and pytest in the task environment; they do not start the simulator.

## Sources

All meshes are generated by `scene_assets.py`; there are no third-party assets. Sizes come from common products: a patio rose in a 22 cm terracotta pot, 19 to 21 cm bypass secateurs, a 26 cm garden tub.
