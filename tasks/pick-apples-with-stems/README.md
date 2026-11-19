# Pick two apples off a potted apple tree by lifting, stems on, and lay them in a picking bucket

Pick two weighted, rubber-coated imitation apples from a repeatable practice tree and lay them in a padded bucket. Each apple must come off its spur with its stem still on it. It must not be squeezed hard enough to bruise it (no finger above 24 N) or dropped.

Each apple hangs from a magnetic stem cap on its spur, and a second magnetic joint (a peg) holds the stem in the apple. The cap lets go under stem tension plus bending; the peg lets go under 24 N of stem tension. In the simulation, with an 18 N pinch:
- Rolling the apple up and drawing it away freed both caps with the stems on.
- A straight pull along the stem freed one cap at 20.6 N but pulled the other apple's stem out at 24.0 N.

On trees too, straight pulls sometimes pull the stalk out.

**No rig has been built or measured.** Its release loads are design targets. The section "How the rig compares with ripe apples" sets them against published measurements on ripe apples, including where they differ.

## Run

From this folder:

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=640 -E cam_height=360 \
  --instruction "Pick both apples off the tree into the bucket." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir runs/demo
```

Two variants use the same command:
- `TASK_DEMO=squeeze` (counterexample): the normal plan, but the first apple is held with the fingers fully closed, so each finger presses it with the 40 N effort cap. That apple is bruised and earns nothing, so the score is 0.5.
- `TASK_DEMO=straight` (counterexample): pull each apple 10 cm straight out along its stem without rolling, then place it in the same bucket. The upper apple's cap frees with the stem on; the lower apple's stem pulls out. The score is 0.75.

The results below come from the simulator, not from hardware.

The normal plan approaches from the front with fingers above and below the fruit, rolls it upward about its own seat through approximately 60 degrees, then draws it away. It holds each apple at about 18 N per finger, under the 24 N bruise limit. The front-facing grasp does not change that roll into a sideways swing. Both arms' per-finger effort limits are set to 40 N and read back from the native articulation. The demo consists only of ordinary joint and gripper actions. Callbacks read the actual apples and bucket when each move begins; they do not set object states or select release events.

## Reproducible practice rig

- Use a 25 mm hardwood-dowel trunk, 500 mm above the table, in a 300 mm, 12 L black pot. Dry sand, bark mulch, pot, trunk and artificial leaves total 11.5 kg. Screw the two printed 25 mm spurs to the dowel at the positions in `task.json`. The pot is free to slide or tip.
- Print `rig/spur.scad` and `rig/socket.scad` in PLA or PETG. The seat holds a [K&J D84](https://www.kjmagnetics.com/d84-neodymium-disc-magnet), 12.7 x 6.35 mm N42 magnet. Its manufacturer's 10.88 lb (about 48 N) rating is for a thick steel plate, not a measured force for this assembly.
- Print `rig/stem.scad` in TPU 95A at 100% infill, bend plane parallel to the bed, with removable supports under the raised ends and span. The flexible length is 20 mm, diameter 6 mm, precurve 19 degrees. A 12.7 x 3 mm mild-steel disc sits in the cap's 12.9 x 3.1 mm pocket, cut after the polymer union. At the apple end, an 8.9 mm polymer spigot, 0.1 mm adhesive and 3 mm flat steel end make a 12 mm insertion; the disc is not recessed inside the smaller 12.6 mm spigot.
- Use a hollow 75 mm imitation apple, or make an internal cavity accessible through a 20 mm calyx port. Give it two dipped or brushed coats of rubber coating (Plasti Dip or similar) and let it cure; the simulation gives the coated skin rubber-like friction (1.2 static, 1.0 dynamic), which grips better than real apple skin. Secure steel shot in epoxy around its centre so the complete handled apple/socket/peg/stem/cap assembly is 180 +- 5 g with centred COM (subtract every installed component from the ballast allowance); verify on a kitchen scale and by balancing on a pencil. A typical 170 g of added steel needs about 22 mL: the 20 mm hole is an access port, not the entire ballast cavity. Close the access port flush after securing the ballast.
- Make a **16 mm diameter, 30 mm deep open installation well** at the stem shoulder. Glue the 16 x 16.7 mm socket with its mouth 13 mm below the shoulder. The upper 13 mm stays open for the flexible stem; do not bury it in the imitation fruit. `rig/apple-installation-section.scad` shows the section. The 12.8 mm socket bore contains a [K&J D82](https://www.kjmagnetics.com/d82-neodymium-disc-magnet), 12.7 x 3.175 mm N42 magnet. Its thick-plate rating is 6.44 lb (about 29 N).
- Use a 290 mm diameter, 230 mm deep plastic bucket with a 25 mm polyurethane foam pad. Empty it and return pot and bucket to their tape marks between trials. Reseat caps and pulled-out pegs; no part is consumed.

### Calculations and proposed session checks

These are design calculations and acceptance checks. **No physical assembly or force calibration has been performed.** Manufacturer ratings do not establish the assembled release loads.

The circular stem has `I = pi*d^4/64 = 63.617 mm^4`. Selecting `E ~= 66 MPa` gives `EI/L ~= 0.21 N m/rad`. The manufacturer's printed [TPU 95A data sheet](https://um-support-files.ultimaker.com/materials/2.85mm/tds/TPU-95A/Ultimaker-TPU95A-TDS-v5.00.pdf) reports orientation-dependent tensile moduli around 56-67 MPa; printing and bending need the checks below. A 40 N seat tipping about its 6.35 mm edge gives 0.254 N m. The 19-degree precurve approximately balances the 180 g apple at a 44.5 mm seat-to-COM lever. Adding 100 g predicts roughly 11.9 degrees further droop.

How to check a built rig against these numbers is under "Calibrating a built rig" below.

### How the rig compares with ripe apples

The rig keeps the two places a picked apple can part:
- where the stem meets the spur: the cap on its magnet;
- where the stem meets the fruit: the peg in its socket.

The table compares the rig's loads with published measurements on ripe apples in the same load case. The rig's numbers come from the simulation: `rig/bench_model.py` for steady pulls at the apple's centre, and the native demo for the roll. They are design targets, not measurements on a built rig.

| Load case | This rig | Ripe apples on the tree |
|---|---|---|
| Straight pull along the stem | Held in an 18 N rubber pinch: the upper apple's cap freed at 20.6 N with its stem on; the lower apple's stem pulled out at 24.0 N. With the apple free to turn (gauge through its centre), the peg pulls out at 24.0 N. | Normal pulls of 23.9 to 26.1 N (Red Fuji, [doi:10.3390/agriculture14122280](https://doi.org/10.3390/agriculture14122280)). 95 % of apples come off at 28.3 N ([doi:10.13031/ja.14970](https://doi.org/10.13031/ja.14970)). Vertical pulls are "possibly leading to the pull-out of the stem" ([doi:10.1016/j.scienta.2019.108937](https://doi.org/10.1016/j.scienta.2019.108937)). |
| Lift and roll upward, then draw | The cap releases at 18.3 to 18.6 N of stem tension after a 58 to 59 degree roll; the stem stays on the apple. | No published measurement of this exact motion was found. Nearest: a median detachment force of 15 to 16 N for robot picks in an orchard ([arXiv:2408.06674](https://arxiv.org/abs/2408.06674)). Picking at about 58 degrees kept the stalk on 94.0 % of apples (abstract, [doi:10.3390/agriculture14122280](https://doi.org/10.3390/agriculture14122280)). |
| Pull at right angles to the stem, at the apple's centre | The cap releases at about 4.9 N on top of the apple's weight, about 6.7 N in all. | Tangential pulls of 12.3 to 14.3 N (Red Fuji, [doi:10.3390/agriculture14122280](https://doi.org/10.3390/agriculture14122280)). |
| Squeeze | Bruised above 24 N per finger. | Lowest damage force 24.33 N in compression (Red Fuji, [doi:10.3390/agriculture16020178](https://doi.org/10.3390/agriculture16020178)). |

The rig's direction-dependent release, predicted by `rig/bench_model.py`. The model uses the simulator's joint law and pulls at the fruit's centre with gravity acting. The first joint to release is the same at 240 and 480 Hz and at 2 or 4 N/s, and the forces agree within 0.02 N:

| Pull direction | Added force at first release | First to release |
|---|---:|---|
| Along the stem | 23.98 N | Peg: the stem stays on the spur |
| 15° above the stem | 24.40 N | Peg |
| 15° below the stem | 20.30 N | Cap: the stem stays on the apple |
| Straight down | 4.88 N | Cap |

Where the rig departs from ripe apples:
- On trees, direct picks lost the stalk about one time in four ([doi:10.3390/agriculture14122280](https://doi.org/10.3390/agriculture14122280)). On the rig, whether a straight pull loses the stem depends on how the pull lines up with it. In the simulation one of two straight pulls did, and neither rolled pick did. The rig is not calibrated to the tree's rate.
- Bending the stem sideways frees the rig's cap at about half the force a ripe apple needs. Rolling the apple off is therefore easier on the rig than on a tree, not harder.
- Rig apples are rigid; the bruise rule stands in for tissue damage. Their rubber coating grips better than real apple skin, so a gentle pinch carries more pull than it would on a real apple.

A published physical apple proxy sets precedent for standing in for apples with magnets and springs tuned to measured detachment forces. Its paper states that it "replicates the mechanics of a real world apple pick". Its proxy kept the main sensor signals of real picks, and a grasp classifier trained only on the proxy did as well on real trees as one trained on real picks ([doi:10.1109/IROS47612.2022.9981716](https://doi.org/10.1109/IROS47612.2022.9981716)). Its settings were 16 N detachment and 455 N/m branch stiffness, with 75 mm sand-filled apples of 220 g ([arXiv:2408.06674](https://arxiv.org/abs/2408.06674)). That proxy has one breakaway. This rig adds a second, at the fruit, so that the technique decides where the apple parts.

### Calibrating a built rig

Build both stations from `rig/`, then follow [`rig/release-checks.md`](rig/release-checks.md) with a luggage scale (1 N steps or better) and a side video showing the gauge line, the stem and which joint lets go. Make three pulls per check per station and record them in [`rig/directional-observations.csv`](rig/directional-observations.csv).

Use the rig only when every check is inside its band:

| Check | Band |
|---|---|
| Cap pulled straight off the spur | 40 ± 3 N |
| Apple pulled off its peg, cap held | 24 ± 2 N |
| Assembled straight pull along the stem | The peg pulls out first, at 24 ± 2 N |
| Downward pull at the apple's centre, in addition to its weight | The cap releases first, at 5.5 ± 1.5 N |
| Apple hanging at rest | Level within 5 degrees |
| 100 g added at the apple's centre | A further 11 ± 3 degrees of droop |

Lower a hold that is too strong with 0.1 mm paper shims; replace a weak magnet or disc, or reprint a stem that droops out of its band. If a directional check fails (for example, a straight pull frees the cap before the peg), adjust and repeat every check. A rig that cannot be brought inside the bands does not match this task's simulation.

### Bruising, and reading the grip force on the real YAM

An apple is bruised if a finger ever pressed on it with more than 24 N. The limit is set just below the lowest compression force reported to damage ripe Red Fuji apples, 24.33 N ([doi:10.3390/agriculture16020178](https://doi.org/10.3390/agriculture16020178)). A bruised apple earns nothing.
- **In the simulation**, each finger pad's contact force on each apple is read every control step, along the pad's own closing direction. Friction along the skin does not count.
- **On the real YAM**, the same quantity is the gripper's clamping force from the robot's log. The YAM's linear gripper reports its motor torque in every feedback frame. The i2rt driver converts torque to finger force with `linear_gripper_force_torque_map`: force = torque × motor stroke (rad) / finger stroke (m). It uses the same conversion for its gripper force limit. The largest force while the fingers are closed on an apple is its squeeze.

The scripted demo holds each apple at about 18 N per finger. The squeeze counterexample presses the first apple with the fingers' 40 N cap.

## Simulation and scoring

Each apple is a free 180 g rigid body. `stemjoint.py` applies the seat spring wrench at 240 Hz and the equal/opposite wrench to the free tree. Translational stiffness is 20 kN/m, integrated implicitly; bending stiffness is 0.21 N m/rad with a gravity preload. Twist slips at the steel-cap friction limit (mu=0.25, about 0.04 N m without tension). The joint selects the first threshold reached by interpolating the two utilization ratios between successive physics samples. An indistinguishable simultaneous crossing is scored as a peg pull:

- Seat: positive stem tension plus bending moment / 6.35 mm reaches 40 N. The stem stays on the apple.
- Peg: stem tension reaches 24 N first. The stem stays on the spur; the apple shows its empty socket.

For the draw, the expert reads the native robot gains/limits and the current stem axis once, then ramps a feed-forward joint-position bias over 4 seconds (`drawplan.py`). Its 22 N intended TCP load is a planning reference, not an applied or measured force. It emits only the standard joint/gripper position actions; actual spring loads and physical release decide the outcome. The grip reference is about 18 N per finger at the observed contact opening, under the unchanged 40 N native effort cap.

Stem bending is condensed to one hinge at the seat, not a distributed deformable beam. Magnetic force does not fade with gap before release. Apples, socket, trunk and pad are rigid; bruising is scored from finger-pad force, as described above. Stem/cap visuals switch only after the corresponding physical release. Apple visual and separate annular colliders leave the recessed well open. Printed spurs collide; decorative leaves do not.

`task.json` gives the complete scoring definition. Per apple: 0.5 for a clean pick in the bucket without a drop or bruise, 0.25 for a stem pull in the bucket without a drop or bruise, 0 otherwise. Pot tilt over 20 degrees permanently makes the trial 0. Drops latch, including rim/table falls and high unheld release. Bucket occupancy requires support near the pad or other fruit. Release height is recorded at physics rate and compared with the first supporting contact on the pad, fruit, pot, bucket rim or table. Pad and wall occupancy follows the bucket pose. Full score also needs both apples untouched and still (maximum 5 mm movement of any point on each 75 mm sphere across the last 1 second), followed by the harness's 1-second success hold. Otherwise the sum is capped at 0.75. Refusing scores 0. There is no required action trajectory in the scorer.

## Assets, validation and runtime

All meshes and textures are procedural in `scene_assets.py`; there are no third-party image assets. CAD solids and disc-pocket clearances have been rendered and checked. That verifies topology and declared clearance, not real magnetic or material performance. Physical calibration remains proposed.

The scripted plan is checked offline against six-axis kinematics and finite collision geometry. Native results and timings are listed below. The limit is 2,400 controls / 120 seconds at 20 Hz. `object_jitter=0` fixes the layout; no special seed is required.


### Native validation

All trials used seed 0, 640 × 360 native cameras, 20 Hz control and 240 Hz physics, with the rubber-coated apples and an 18 N grip.

| Scripted trial | Score | Controls | Simulated duration | Wall time | Slowest 30 controls |
|---|---:|---:|---:|---:|---:|
| Roll and draw (demo) | 1.0 | 1,633 | 81.65 s | 756 s | 31.0 s |
| Straight pull along the stem (`TASK_DEMO=straight`) | 0.75 | 1,281 | 64.05 s | 592 s | 20.7 s |
| First apple squeezed (`TASK_DEMO=squeeze`) | 0.5 | 1,833 | 91.65 s | 726 s | 15.9 s |
| Dream-1 (320 × 176 cameras, 5 rows per request) | 0.0 | 2,400 | 120.00 s | 1,759 s | 41.3 s |

Release events, read from the simulated joint loads (not a hardware sensor):
- **Roll and draw:** both caps freed with the stems on. The upper apple went at 21.05 s (stem tension 18.58 N, bending 0.137 N m, fruit rolled up 57.9°), the lower at 63.65 s (18.67 N, 0.135 N m, 58.2°). The largest finger-pad forces were 19.1 N and 20.3 N, under the 24 N bruise limit.
- **Straight pull:** the upper apple's cap freed at 20.58 N (bending 0.125 N m), with its stem on. The lower apple's stem pulled out at 24.02 N. Pad forces were 18.5 N and 20.1 N.
- **Squeeze:** the first apple's pad force reached 45.6 N, so it was bruised. The second, held at up to 20.3 N, came off cleanly.

Every trial ended with both apples in the bucket, with no drop and no pot tip. Every frame of every camera was checked: none is blank.

Wall times include the cameras and depend on the machine's load; the slowest 30 controls of the demo include other work on the same host. The straight-pull trial ran with other simulations on the same machine; the demo and squeeze trials had the GPUs to themselves.

Dream-1 ran the full 2,400 controls in one session (474 requests, no expiry) and scored 0.0. Its right hand reached toward the upper apple, but neither apple came off its spur.

Offline checks are `python tests/check_scoring.py`, `python tests/check_joint.py` and `python tests/check_drawplan.py` in the task environment. Run `python rig/bench_model.py` for the directional model predictions. Assets and textures are generated on import.
