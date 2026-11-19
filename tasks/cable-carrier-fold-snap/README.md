# Fit a robot axis's side-mounted cable carrier: fold it round and click its free end into the carriage bracket

A linear axis's cable carrier (energy chain) lies on its side on the table, its fixed end clipped into the bed bracket, the
chain stretched out straight. Beside it, the axis's carriage sits at its mark on its rail, motor off, so it slides freely. The
free end has to be carried round in the plane of the table so the links roll into their 38 mm loop, then pushed into the
bracket under the carriage plate until the end link's upper pin clicks into the bracket's bore. The links bend one way only, so the hand
carrying the end turns with it through 180 degrees; the other hand holds the lower run still while the loop forms, then braces
the carriage while the end is pushed home.

## Run a trial

From this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 2400 \
  --instruction "Fold the cable carrier back on itself and snap its free end into the bracket on the carriage, keeping the carriage at its mark, then let go and move both hands clear."
```

The scripted demo: the same command with `--policy yam_scripted` and `TASK_DEMO=success`. `TASK_DEMO=failure` pushes the end
home without bracing the carriage. Offline scorer checks: `python check_score.py`.

The run prints `task_score` at the end. `score()` reads the end link's pose against its seat on the carriage, the bracket's upper cheek's lift,
the carriage's travel, every link's tilt and the fingertips' distance to the chain and bracket, tracked every control step.

## Runtime and seeds

- Scripted demo: 695 steps (34.8 s simulated), about 9 minutes on a shared GPU including start-up.
- Dream-1: 2,400 steps (2 minutes simulated), about 25 minutes with the whole machine, 6 rows per request.
- `object_jitter=0` keeps the layout fixed, so no seed is needed. The scripted demo reads the links' poses to place its grips.

## Check runs

| Run | Score | What happened |
|---|---|---|
| `TASK_DEMO=success` | 1.0 | the end link clicked in 0.3 mm from its seat; the braced carriage gave up to 19.9 mm under the push and ended 8.7 mm from its mark; both hands clear, still for 1 s |
| `TASK_DEMO=failure` (no brace) | 0.25 | the push shoved the carriage 55 mm along its rail; the end never clicked in |
| `TASK_DEMO=lazy_hands` (keeps hold after the click) | 0.25 | clicked in, but the hand never let go |
| `python check_score.py` | | hand-built states for every tier, including a link tipped mid-run (0) and the end let go at the bracket's mouth (0.25) |

## Sources

- igus E2 mini B15 delivery program, https://static.chipdip.ru/lib/113/DOC036113317.pdf (pitch, widths, heights, bend radius, chain length S/2 + K, mounting brackets 10...1 "with bore" and 10...2 "with pin", set 101-12P).
- igus, "Designing side-mounted" energy chains, https://www.igus.com/cable-carriers/resources/energy-chains-designing-sidemounted-com: side-mounted chains, turned 90 degrees, for a low installation height.
- Snap-fit forces for context: S.S. Abdul Manan and M.N. Osman Zahid, "Analyzing Insertion and Retention Forces in Cantilever Snap-Fits", International Journal of Automotive and Mechanical Engineering 22(3), 2025, https://doi.org/10.15282/ijame.22.3.2025.3.0960 (measured insertion forces 3.2 to 34 N).
- All meshes are built in task.py from boxes and convex hulls; no external assets.

## Building it for a real trial

- Cable carrier: igus E2 mini B15.015.038.0, 11 links (335.5 mm), with igus B15 polymer mounting brackets. Lay it on its side
  on the table, inner radius away from the robot, and screw the fixed-end bracket to the table.
- Axis: an MGN12 rail, 360 mm, screwed to the table; an MGN12H block on it, a 20 mm aluminium spacer and a 10 mm aluminium
  plate, 155 x 150 mm. Screw the moving-end bracket of the igus set 101-12P (polymer, "with bore") under the plate's end, its
  cheeks reaching back over where the end link will lie. Mark the carriage's start with tape beside the rail.
- Reset: unclip the end link from the bracket (lift the upper cheek with a thumbnail and slide it out), lay the chain straight
  on its tape line, slide the carriage back to its mark.

Brand names are for identification only.

## Licence

MIT
