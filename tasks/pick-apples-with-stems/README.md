# Pick two apples off a potted apple tree by lifting, stems on

Pick two weighted imitation apples from a repeatable practice tree and lay them in a padded bucket. In this rig, tilting combined with draw can detach the magnetic stem cap with the stem retained; a straight AXIAL pull instead removes the peg from the apple. This fixed failure ordering does not apply universally to real apples. This is a reusable physical practice rig, rather than a claim to reproduce every cultivar's living tissue.

## Run

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Pick both apples off the tree into the bucket." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir runs/demo
```

`TASK_DEMO=failure` selects the counterexample: pull each apple 10 cm along its stem without rolling, then place it in the same bucket. The intended outcomes are two clean seat releases for the normal plan and two peg pulls for the counterexample. Current-run results are recorded separately; these are not claims of measured hardware performance.

The normal plan approaches from the front with fingers above and below the fruit, rolls it upward about its own seat through approximately 60 degrees, then draws it away. The front-facing grasp does not change that roll into a sideways swing. Both arms' per-finger effort limits are set to 40 N and read back from the native articulation. The demo consists only of ordinary joint and gripper actions. Callbacks read the actual apples and bucket when each move begins; they do not set object states or select release events.

## Reproducible practice rig

- Use a 25 mm hardwood-dowel trunk, 500 mm above the table, in a 300 mm, 12 L black pot. Dry sand, bark mulch, pot, trunk and artificial leaves total 11.5 kg. Screw the two printed 25 mm spurs to the dowel at the positions in `task.json`. The pot is free to slide or tip.
- Print `rig/spur.scad` and `rig/socket.scad` in PLA or PETG. The seat holds a [K&J D84](https://www.kjmagnetics.com/d84-neodymium-disc-magnet), 12.7 x 6.35 mm N42 magnet. Its manufacturer's 10.88 lb (about 48 N) rating is for a thick steel plate, not a measured force for this assembly.
- Print `rig/stem.scad` in TPU 95A at 100% infill, bend plane parallel to the bed, with removable supports under the raised ends and span. The flexible length is 20 mm, diameter 6 mm, precurve 19 degrees. A 12.7 x 3 mm mild-steel disc sits in the cap's 12.9 x 3.1 mm pocket, cut after the polymer union. At the apple end, an 8.9 mm polymer spigot, 0.1 mm adhesive and 3 mm flat steel end make a 12 mm insertion; the disc is not recessed inside the smaller 12.6 mm spigot.
- Use a hollow 75 mm imitation apple, or make an internal cavity accessible through a 20 mm calyx port. Secure steel shot in epoxy around its centre so the complete handled apple/socket/peg/stem/cap assembly is 180 +- 5 g with centred COM (subtract every installed component from the ballast allowance); verify on a kitchen scale and by balancing on a pencil. A typical 170 g of added steel needs about 22 mL: the 20 mm hole is an access port, not the entire ballast cavity. Close the access port flush after securing the ballast.
- Make a **16 mm diameter, 30 mm deep open installation well** at the stem shoulder. Glue the 16 x 16.7 mm socket with its mouth 13 mm below the shoulder. The upper 13 mm stays open for the flexible stem; do not bury it in the imitation fruit. `rig/apple-installation-section.scad` shows the section. The 12.8 mm socket bore contains a [K&J D82](https://www.kjmagnetics.com/d82-neodymium-disc-magnet), 12.7 x 3.175 mm N42 magnet. Its thick-plate rating is 6.44 lb (about 29 N).
- Use a 290 mm diameter, 230 mm deep plastic bucket with a 25 mm polyurethane foam pad. Empty it and return pot and bucket to their tape marks between trials. Reseat caps and pulled-out pegs; no part is consumed.

### Calculations and proposed session checks

These are design calculations and acceptance checks. **No physical assembly or force calibration has been performed.** Manufacturer ratings do not establish the assembled release loads.

The circular stem has `I = pi*d^4/64 = 63.617 mm^4`. Selecting `E ~= 66 MPa` gives `EI/L ~= 0.21 N m/rad`. The manufacturer's printed [TPU 95A data sheet](https://um-support-files.ultimaker.com/materials/2.85mm/tds/TPU-95A/Ultimaker-TPU95A-TDS-v5.00.pdf) reports orientation-dependent tensile moduli around 56-67 MPa; printing and bending need the checks below. A 40 N seat tipping about its 6.35 mm edge gives 0.254 N m. The 19-degree precurve approximately balances the 180 g apple at a 44.5 mm seat-to-COM lever. Adding 100 g predicts roughly 11.9 degrees further droop.

Before a physical session, use a spring/luggage scale with 1 N resolution and a side-view phone photo:

| Check | Proposed acceptance |
|---|---|
| Pull stem cap straight off spur | 40 +- 3 N |
| Hold cap, pull apple off peg | 24 +- 2 N |
| Apple hanging level | Within 5 degrees |
| Add 100 g at apple centre | Further droop 11 +- 3 degrees |
| Pull downward at apple centre | Clean seat release at 5.5 +- 1.5 N |

Reduce excessive magnetic hold with thin paper shims, starting at 0.1 mm; replace a weak disc and reprint an out-of-range stem. These checks preserve the specified rig response. The estimated lift-and-draw load is about 17 N, or 14 N per rubber-padded finger; a 24 N straight pull needs about 20 N per finger, within the 40 N envelope.

### Biological evidence and limits

The 40 N seat, 24 N peg and calculated 0.254 N m edge-tip moment are fixed engineering targets for this deliberately stem-pull-prone attachment. They are not universal ripe-apple failure thresholds or measured upward-bending data. A native demonstration establishes this modeled rig's feasibility, not its physical calibration or transfer to every apple variety.

- [Design and Evaluation of End Effectors for a Vacuum-Based Robotic Apple Harvester](https://doi.org/10.13031/ja.14970) reports 95% thresholds of 28.3 N pulling and 0.257 N m **twisting**. That torque does not validate the rig's upward-bending threshold.
- [Green Fruit Removal Dynamics for Development of Robotic Green Fruit Thinning End-Effector](https://doi.org/10.13031/ja.14974) reports mean fruit-end and spur-end pulls of 24.78 and 19.91 N for **green-fruit thinning**, not ripe harvesting. These data do not establish the rig's failure ordering for ripe fruit.
- [Experimental and simulation analysis of optimum picking patterns for robotic apple harvesting](https://doi.org/10.1016/j.scienta.2019.108937), sections 3.1–3.3, reports harvesting-date Pink Lady fruit with strongly direction-dependent results: horizontal samples detached at the branch/stem; vertical pulling gave three fruit-end separations in 25 samples. Bending also failed to detach seven of 25 samples. Straight pulling therefore does not universally lose the stem. The rig selects one repeatable failure ordering rather than reproducing that biological variability.

## Simulation and scoring

Each apple is a free 180 g rigid body. `stemjoint.py` applies the seat spring wrench at 240 Hz and the equal/opposite wrench to the free tree. Translational stiffness is 20 kN/m, integrated implicitly; bending stiffness is 0.21 N m/rad with a gravity preload. Twist slips at the steel-cap friction limit (mu=0.25, about 0.04 N m without tension). The first overloaded connection releases from actual loads:

- Seat: positive stem tension plus bending moment / 6.35 mm reaches 40 N. The stem stays on the apple.
- Peg: stem tension reaches 24 N first. The stem stays on the spur; the apple shows its empty socket.

For the draw, the expert reads the native robot gains/limits and the current stem axis once, then ramps a feed-forward joint-position bias over 4 seconds (`drawplan.py`). Its 22 N intended TCP load is a planning reference, not an applied or measured force. It emits only the standard joint/gripper position actions; actual spring loads and physical release decide the outcome. The nominal grip reference is about 35.5 N per finger at the observed contact opening, under the unchanged 40 N native effort cap.

Stem bending is condensed to one hinge at the seat, not a distributed deformable beam. Magnetic force does not fade with gap before release. Apples, socket, trunk and pad are rigid; the pad does not predict bruising. Stem/cap visuals switch only after the corresponding physical release. Apple visual and separate annular colliders leave the recessed well open. Printed spurs collide; decorative leaves do not.

`task.json` gives the complete scoring definition. Per apple: 0.5 for a clean pick in the bucket without a drop, 0.25 for a stem pull in the bucket without a drop, 0 otherwise. Pot tilt over 20 degrees permanently makes the trial 0. Drops latch, including rim/table falls and high unheld release. Bucket occupancy requires support near the pad or other fruit. Full score also needs both apples untouched and still (seven material points each, maximum 5 mm movement over the last 1 second), followed by the harness's 1-second success hold. Otherwise the sum is capped at 0.75. Refusing scores 0. There is no required action trajectory in the scorer.

## Assets, validation and runtime

All meshes and textures are procedural in `scene_assets.py`; there are no third-party image assets. CAD solids and disc-pocket clearances have been rendered and checked. That verifies topology and declared clearance, not real magnetic or material performance. Physical calibration remains proposed.

The scripted plan is checked offline against six-axis kinematics and finite collision geometry. Native results and timings are listed below. The limit is 2,400 controls / 120 seconds at 20 Hz. `object_jitter=0` fixes the layout; no special seed is required.


### Native validation

The scripted trials use `object_jitter=0`, seed 0, 20 Hz control, 240 Hz physics and all three native cameras at 640 x 360.

| Run | Official score | Controls / simulated time | Total wall time | Slowest 30 controls |
|---|---:|---:|---:|---:|
| Upward roll and draw | 1.0 | 1,635 / 81.75 s | 566.06 s | 14.99 s |
| Straight axial pull | 0.5 | 1,281 / 64.05 s | 429.43 s | 14.59 s |
| Dream-1 | 0.0 | 2,400 / 120.00 s | 980.51 s | 16.20 s |

Both scripted trials completed without an error, drop or pot tip, with both apples untouched in the bucket. The normal plan produced two clean seat releases at actual fruit up-rolls of 58.7 and 58.4 degrees, with evaluated seat tensions of 18.52 and 18.40 N. The counterexample pulled both pegs at 24.08 and 24.05 N. These are simulator results for this modeled rig, not measured hardware forces. A 320 x 176 diagnostic reproduced the 1.0 and 0.5 outcomes. Timings were measured on an exclusive NVIDIA L40S with THP disabled and Python bytecode enabled.

Dream-1 used 320 x 176 cameras and eight rows per request. It completed the 2,400-control limit with score 0.0, 296 actual requests, one session and no expiry. CPU checks are `python tests/check_scoring.py`, `python tests/check_joint.py` and `python tests/check_drawplan.py` in the task environment. Assets and textures are generated on import; no external asset download is required.
