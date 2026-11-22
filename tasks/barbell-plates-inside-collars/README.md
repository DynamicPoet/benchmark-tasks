# Add a plate to each end of a loaded curl bar, inside its locked collars

A task file for the `yam_arena` embodiment ([yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena), Isaac Lab-Arena 0.3.1).

A loaded E-Z curl bar lies level in two notched cradles. Each 1-inch sleeve carries a 1.25 lb plate held by a lever-lock collar. Two
2.5 lb plates stand in slot trays in front of the bar, between the arms. The instruction asks only for one plate on each end. The weight-room rule
is that plates go inside a locked collar: flip the collar's lever up, slide the collar off the sleeve end and set it in its rest,
thread the new plate on against the first one, slide the collar back against it and fold the lever down. A plate hung on the bare
sleeve outside the collar slides off when the bar tips.

## Run a trial

From this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Add one of the 2.5 lb plates to each end of the curl bar, then let go and move both hands away." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir runs/dream
```

Scripted demo (scores 1.0): the same command with `--policy yam_scripted`. Other scripted runs, set with `TASK_DEMO`:

| `TASK_DEMO` | What it does | Score |
|---|---|---|
| `success` (default) | each arm pushes its collar's lever up with closed fingertips, slides the collar off and sets it in its rest, threads its 2.5 lb plate on against the 1.25 lb plate, slides the collar back, pushes it home against the plate and presses the lever flat, then both hands move away | 1.0 |
| `failure` | each arm hangs its plate on the bare sleeve outside the locked collar | 0 |
| `unlocked` | as the demo, but the levers are left up | 0 |

`check_score.py` (run it from this folder with the harness's Python) feeds the scorer hand-built states: both ends done (1.0), one end
done (0.5), and these lazy routes and faults, all 0: nothing done, plates hung outside the locked collars, collars refitted with
levers up, collars left in their rests, collars locked 8 mm from the plates, new plates 15 mm from the first, the 2.5 lb plate pushed
inside the 1.25 lb plate, a clamped collar dragged along its sleeve, a collar slid off with its lever only 20 degrees up, a plate
dropped, a plate landing on its edge on the table, a collar dropped on the table. Both ends done with a hand still on a sleeve or a
plate still moving scores 0.5. Either 2.5 lb plate may go on either end (they are identical); the other side's plate hung
outside a locked collar scores 0.

The demo reads the plates' and collars' poses to line up holes with the sleeves (closed loop); it never moves an object itself.

## How the score is computed

`task.py` keeps a `Trial` that sees every control step (`on_step`). An end is done when a 2.5 lb plate's centre (either plate) is on the
sleeve's axis (within 4 mm: it hangs by its hole), outside the 1.25 lb plate with a gap of 1 cm or less to it, and its collar is on
the sleeve outside it with a gap of 5 mm or less, lever within 8 degrees of flat. Gaps are between the parts' nearest points,
counting each ring's lean on the sleeve. Any plate on a sleeve with no flat-levered collar outside it scores 0. Two failures latch
for the whole trial: a fall (a part's centre lower than it can be when standing in its tray, hanging on its sleeve or sitting in its
rest, so a plate lying flat or on its edge on the table, or a collar on the table or a tray floor), and a
collar slid more than 5 mm along its sleeve while its lever is under 30 degrees (the jaw still grips). One end done: 0.5; both
ends done: 0.5, and 1.0 once every fingertip of both hands is 4 cm or more from the sleeves, plates and collars and no plate or
collar moves more than 2 mm over the last 1 s, held for 1 s.

## What the simulation simplifies

- The collar's clamp is its lever's position: lever folded flat on the sleeve counts as clamped. The clamping force is not
  simulated, so a collar slid more than 5 mm along its sleeve while its lever is under 30 degrees is scored as a failure instead. On a real bench
  a 20 N spring-scale pull outward on each collar checks the clamp.
- The bar is fixed in place (a real 47 in E-Z bar weighs about 7 kg, an estimate, and lies in notched cradles).
- The plates' diameters and thicknesses, the collar's size and mass, and its lever (a hinge with light friction and a 45-degree
  stop) are engineering estimates for standard parts, not measurements; the maker publishes no lever force.

## Runtime and seeds

- Demo: 1,608 steps (80 s simulated), about 10 minutes including start-up; it ends once the full score has held for 1 s.
- Dream-1: 3,600 steps (180 s simulated), about 40 minutes.
- `object_jitter=0` keeps the layout fixed, so no seed is needed. The scripted demo is deterministic.

## Sources

All meshes are built in `task.py` from the parts' sizes: CAP Barbell 47 in standard E-Z curl bar (1-inch sleeves); CAP standard
1-inch-hole cast-iron plates, 1.25 lb and 2.5 lb (diameters and thicknesses estimated for standard plates); Body-Solid BSTLJ1
1-inch Lock-Jaw collar (size, mass and lever estimated). No external assets.

## Licence

MIT
