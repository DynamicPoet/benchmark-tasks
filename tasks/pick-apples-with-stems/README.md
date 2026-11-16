# Pick two apples off a potted apple tree by lifting and twisting, stems on

A task file for the `yam_arena` embodiment ([yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena),
Isaac Lab-Arena 0.3.1). A columnar apple tree in a pot stands on the station table with two ripe apples on short fruiting
spurs, and a picking bucket with a foam pad stands in front of the arms. The arms must pick both apples the way a picker
does, lifting and twisting each one until its stem snaps where it joins the spur, and lay them in the bucket. Pulled
straight off, an apple's stem tears out of the fruit instead (a stem pull); an apple let go from high up is dropped.

## Run a trial

Dream-1:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Pick both apples off the tree into the bucket." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir runs/dream
```

Scripted demo (scores 1.0): the same command with `--policy yam_scripted`. The right arm grips each apple level from the
front, across its middle, turns it about 105 degrees about its stem while tilting it up 12 degrees (the stem snaps at the
spur on the way), backs off, carries it over the bucket, lowers it until it is 1 cm above the pad (or beside the first
apple) and lets go. The scripted failure (`TASK_DEMO=failure`, same command): the arm grips each apple and pulls it 10 cm
straight away from the trunk; each stem tears out of its apple (two stem pulls), and both are laid in the bucket: 0.5.

## How the simulation works

- **Apples on spurs** (`stemjoint.py`). Each apple is a free rigid body. Until it comes off, a joint spring holds its stem's
  end (the abscission joint, where the stem meets the spur) to the spur: every physics step (240 Hz) the task applies the
  spring's force and moment to the apple, and the opposite to the tree. The spring is stiff in travel (20 kN/m, integrated
  implicitly), and soft in bending and twisting, as a short apple stem is: it holds the apple level where it hangs, and
  its moment reaches the abscission joint's limit after about 90 degrees of lift or of twist. The spring's force and moment
  are the joint's loads. Every physics step they are checked against three limits, and the first one passed (the most
  overloaded) decides what gives way; the spring is then no longer applied:
  - the abscission joint snaps at 0.26 N m of bending and twisting moment together, or 33 N of tension or shear: a clean
    pick, stem on the apple, spur and its fruit bud on the tree;
  - the stem tears out of the apple at 25 N of tension along it: a stem pull; the stem stays on the spur and the apple
    shows its torn stem cavity;
  - the spur breaks off the trunk at 0.8 N m or 45 N at its base: it comes away on the stem, leaving a torn stub.
- **Loads, from published measurements**: about 0.26 N m of torque detached 95 % of hand-tested apples, with 28.3 N of
  pulling force (Dickinson, MS thesis, Michigan State University, 2022); the branch-stem joint of apples with an obtuse
  stem angle parted at 33.2 +- 7.5 N in a straight pull (Bu et al., Scientia Horticulturae 261, 2020, 108937); in
  straight pulls, stems tore out of the fruit at 24.8 +- 0.5 N (Hussain et al., Journal of the ASABE, 2022); bending and
  twisting need much less force than a straight pull (Bu et al. 2020; Li et al., Computers and Electronics in Agriculture
  127, 2016). The spur's limits are set above the others, so a spur breaks only when an apple is levered or pulled hard
  while its stem holds.
- The tree (pot, potting mix, trunk, spurs), the apples and the bucket are rigid bodies at real size and mass. The pot can
  slide and tip. The trunk is rigid. Leaves and spurs have no colliders. Apples, stems and the foam pad do not deform.

## How the simulation scores

`score()` and `success()` in `task.py` implement the scoring in `task.json` line by line:

- **Came off**: which joint gave way (above), recorded when it happens.
- **Touching**: a contact force above 0.02 N between an apple and either gripper or its fingers (a PhysX contact view),
  read every control step (20 Hz).
- **Dropped** (latched until a reset): at the first control step an apple that is off the tree is not touched, its centre's
  height above where it would come to rest if it fell straight down (on the pad, on the other apple if that one is off the
  tree, on the pot's soil or rim, on the bucket's rim, or on the table) is more than 5 cm; or, while not touched, its lowest
  point comes within 5 mm of the table top outside the bucket having been more than 5 cm higher since it was last touched
  or at rest (slower than 2 cm/s); or its lowest point is more than 10 cm below the table top (the floor).
- **In the bucket**: its centre inside the bucket's inner wall by at least half the apple's radius (19 mm), above the pad
  and below the rim, and within 1 cm of where it would rest if it fell straight down (on the pad or the other apple).
- **Tipped over** (latched): the pot's axis more than 20 degrees from vertical at any control step.
- **Still**: over the last 21 control steps (1 s), no point of seven on each apple (centre and six on its skin) moves more
  than 5 mm (largest distance between any two samples).
- Each apple: 0.5 in the bucket, stem on, spur whole, not dropped; 0.25 the same with a stem pull; otherwise 0. The sum
  is capped at 0.75 by `score()`; 1.0 comes from `success()` (both 0.5, neither touched, both still) held for 1 s (the
  harness's `SUCCESS_HOLD_S`). The pot tipped over makes it 0.

## Runtime and seeds

- Demo: 928 steps (46.4 s simulated) to success; about 6 minutes of wall time with 640 x 360 cameras on one L40S GPU,
  including start-up. The slowest 30 steps took 10.6 s.
- Dream-1: 2,400 steps (2 minutes simulated), about 24 minutes of wall time at 8 rows per request (one session), including start-up.
- `object_jitter=0` keeps the layout fixed, so no seed is needed. The scripted demo reads each apple's pose when it starts
  a move (what the cameras show) and the bucket's; it writes nothing into the simulation.

## Objects

- Apple: a ripe 'Gala'-type apple, 75 mm across and 68 mm tall, 180 g, with a 20 mm stem 3 mm thick, red-orange blush and
  stripes over a yellow ground.
- Tree: a columnar ('patio') apple tree in a 300 mm, 12 L plastic pot of moist potting mix, 11.5 kg in all; trunk 25 mm thick.
- Bucket: a 20 L plastic picking bucket, 290 mm across, 230 mm deep, with a 25 mm polyurethane foam pad on its floor.
- Real trials: the tree stands on tape marks with the two apples on the side facing the right arm, their centres 30 and
  38 cm above the table at the positions in `task.json`; the bucket stands on its own marks. Each trial picks two apples;
  the next trial uses two other ripe apples on spurs at the same places.
- Meshes and textures (apple skin, bark, potting mix, foam, leaf) are generated by `scene_assets.py`: procedural, no
  third-party images. `usdkit.py`, `scenekit.py` and `poseplay.py` are helper modules.
