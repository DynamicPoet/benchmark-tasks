# Buckle a practice doll into an infant carrier

A task file for the `yam_arena` embodiment, using Isaac Lab-Arena 0.3.1.

The open harness has 25 mm extra material per side above the final fitted upper-plus-lap reference. Closing the buckle and moving the chest clip to armpit level is only partial progress. Both shoulder straps must be correctly routed and untwisted, have no suspended or penetrating shoulder/chest sections, and pass the explicitly local fixed-endpoint material-fold assay; a real shoulder pinch-and-slide outcome is not established by it. The front adjustment tail must be physically pulled and released before the clip is raised to armpit level in the scripted solution.

## Real reference and starting adjustment

Use an intact Graco SnugRide 35 Lite LX five-point harness and carrier, instruction manual NWL0000993719A, section 4-C, with a 50 cm practice infant doll in thin clothing. Fit the shoulder slots at or just below the doll's shoulders, in the same row, flat and untwisted. Establish a snug reference on both sides with the manufacturer's shoulder pinch assessment, mark each strap at its shell exit, undo the hardware, then draw exactly 25 mm more material through each shoulder exit. Reset the front tail to its mark, put the tongues in the two hip pockets, and leave the chest clip open over the belly. Re-establish the reference before every trial; a fixed tail length is not a substitute for fitting a different doll.

The simulated doll measures 34.78 cm along the backrest, 23.00 cm across and 10.40 cm deep; its length is about 0.696 of the 50 cm real reference doll. The shell is 38.5 cm along the backrest and 27 cm wide. The simulation is a scaled training fixture, not a dimensional replica or a child-safety certification. Its surface assay has not been calibrated against real webbing, finger friction or a specific assessor.

## Files

- `task.py`: hardware latches, measured feed/shoulder joints, final fitting evidence, scorer and open-loop demo.
- `scene_assets.py`: carrier, doll, tongue/clip mechanisms, adjustment tail and ribbon surfaces.
- `body_distance.py`: conservative external-surface distance and whole-triangle certificates.
- `harness_fit.py`: independent full-width routing, twist, signed body gap, material-length and stillness checks.
- `pinch_assay.py`: full-width fixed-endpoint chord certificate and explicit conserved-length U/pad geometry candidates.
- `scenekit.py`, `usdkit.py`: scene and mesh helpers.
- `tests/test_harness_fit.py`: geometry counterexamples, including a suspended taut ribbon, body penetration, half twist and deletion of loose material.

## Run

From this folder with the `yam_arena` environment active:

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=360 \
  --instruction "Buckle the baby doll into the car seat." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir runs/demo
```

The `TASK_DEMO=loose` script omits the actual adjustment pull while completing the buckle and correctly positioned clip. `TASK_DEMO=failure` keeps the clip low. `TASK_DEMO=twisted` and `TASK_DEMO=misrouted` are diagnostic starting fixtures with a physical shoulder section turned or displaced; the geometric checker never reads those names.

For a model run, replace the policy by `dream`, add `-P rows_per_request=4`, and use 320 x 176 cameras. The recorded model run uses four control rows per request; its result is listed below.

## Scoring and observed quantities

The final score is 0 with no tongue, 0.25 with one tongue, 0.5 with two tongues and an open chest clip, and 0.75 with two tongues and the clip closed if any full-score condition fails. Full credit also requires the clip within 2 cm of the armpit line, both shoulder/lap routes correct, both shoulder material assays passed, hands at least 5 cm from the seat/doll and 3 cm from the adjustment tab, and clip range below 5 mm over all samples in the last second. Hold the full state for 1 s. Moving the seat over 2 cm, tilting it over 5 degrees or approaching the head within 1 cm invalidates the whole trial.

Every displayed shoulder/chest triangle and width edge is checked. Signed Euclidean distance to the external torso/shoulder/neck surface must lie between -2 mm and +4 mm. Analytic ellipsoid closest points and conservative solid-union bounds exclude buried internal surfaces. Centroid/radius bounds certify whole triangles by the signed-distance 1-Lipschitz property; uncertain faces are subdivided, and unresolved faces fail closed. A straight strap floating above the doll fails; pushing a ribbon into the doll also fails. Width edges must remain on their corresponding shoulder side; the signed width direction stays within 30 degrees of the backrest +v axis and does not reverse. The shoulder corridor is u 12 to 14.5 cm, and v 1.5 to 6 cm on its own side, including both width edges. Every chest width edge stays at least 9 mm from the midline on its own side. Each lap ribbon connects its own hip and actual tongue anchor within 3 mm, with every width edge at least 3 mm from the midline on that side. Backrest u runs up from the stated backrest origin, v points to the robot's left, and w points out of the cushion. Mesh validity also requires a cross-width chord of at least 17.6 mm (80% of the specified 22 mm webbing), and lap width edges retain increasing v order; these are shape guards, not manufacturer fitting thresholds.

The local assay clamps the two actual shoulder end sections at backrest u=0.140 and 0.120 m and prevents material entering from outside. The 20 mm is a projection in u; the normal fitted centre material length is about 21.77 mm. This imposed clamp is an added diagnostic boundary, absent from the manufacturer's hand test. The prescribed fitted route is not a proven shortest feasible path and is never used as the lower bound for an inability-to-fold claim.

For each longitudinal lane of the displayed triangle mesh, measure its material length L and its own endpoint chord D. A pair of parallel opposite legs of common overlap h, separated by a connector of length C, can be replaced by the line between their bases, of length g <= C. This removes at least 2h, while the remaining path is at least D: h <= (L-D)/2. Between width columns, the lane's length is an affine combination of its neighbours' lengths and its endpoint difference is affine. The implementation conservatively uses the larger neighbouring length and the minimum chord over that complete width interval, covering triangle interiors rather than only stored vertices. Ignoring body obstacles makes this impossibility bound conservative.

Only `no_fold_certified`, with every width-interval bound strictly below 2 mm, passes this clamped-window assay. Otherwise an explicit U-shaped candidate preserves each measured longitudinal lane and both complete end sections. Its 0.5 mm neutral-line bend radius gives 1 mm layer spacing; the sampled connector's actual length is included in the material equation. Two 6 by 4 mm pad faces, 2 mm thick, align with the actual opposed layers. At least 2 mm overlap, signed normal dot <= -0.99, pad contact within 0.25 mm, mutually disjoint pad solids with thickness extending away from the layer gap, and full candidate/pad distance within the benchmark's -2 mm doll tolerance are checked. This tolerance does not establish strict nonpenetration. A valid `fold_witness` fails; when neither certificate nor witness is established, `indeterminate` also fails. These are ideal jig dimensions and geometric candidates, without calibrated webbing thickness, force, woven transverse metrics, shear, bending resistance, self-contact dynamics or finger retention. Probe geometry is diagnostic only and is not rendered into the policy's observations.

The global upper-strap chord bound, local centre bound and reference-route residual are logged separately. The nominal global bound is about 10.45 mm, so it does not establish that a 2 mm whole-harness fold is impossible. Webbing could migrate from chest/lap outside the clamped window, and real thumb/forefinger friction is unvalidated. A local certificate therefore cannot be equated to the manufacturer's fingers-sliding-off test. A separate conservation guard rejects a mismatch above 0.5 mm between actual upper-plus-lap material and fixed starting budget minus measured feed. Moving a tongue into the buckle redistributes material rather than deleting it; grading does not read adjustment commands or scripted mode labels.

The front cam holds physically measured tail displacement. Its physical joint stop is updated from the material still available after reaching both actual tongues, so the open hip-pocket pose permits only about 9 mm feed and the fitted buckle pose permits 25 mm. Independently, feed exceeding either side's available budget by more than 1 mm permanently caps that trial at 0.75, preserving lower hardware-stage scores. Moving tongues later cannot erase this integrity failure; ordinary shoulder-fold tracking error does not trigger it. Shoulder section guides can move and rotate; checks are repeated at the end, so an earlier correct fit does not preserve full credit after displacement. Ribbon surfaces are constrained geometry; cloth friction/bending and finger grasp are not calibrated. Only tongues, clip, adjustment tab and shoulder patches collide. The adjustment tab starts at (0.060, 0.000, 0.080) m above the table, clear of the foot rim, and a visible tail joins it to the front exit at (0.100, 0.000, 0.080) m. Hip-pocket tops are at (0.200, +/-0.085, 0.102) m. This limitation must be resolved or independently validated before equating the assay to the manufacturer's real pinch test.

## Checks and runtime

`python -m pytest tests` passes 60 regression tests. The complete offline IK plan check passes without joint-limit or step warnings. The fixed layout uses `object_jitter=0` and 20 Hz control. Recorded trials use runtime revision `167efc2`:

| Trial | Score | Steps | Camera size | Wall time | Slowest 30 steps |
|---|---:|---:|---|---:|---:|
| Scripted fit | 1.0 | 1,095 | 640 x 360 | 646.019 s | 20.16 s |
| No-tightening control | 0.75 | 1,027 | 640 x 360 | 625.282 s | 24.78 s |
| Dream-1 | 0.0 | 1,600 | 320 x 176 | 1,157.715 s | 27.60 s |

Each is a complete official-harness trial with no evaluation error. The model uses `dreamlabs/dream-1-beta`, four control rows per request and one action session. Its 80-second episode ends at the step limit without a latched tongue. The local geometry proof and unresolved real finger-retention boundary remain as stated above. Wall time depends on shared hardware load.

## Sources

All visual meshes are procedural and have no third-party textures. The real fitting procedure is documented in [the manufacturer's manual, section 4-C](https://newellbrands.scene7.com/is/content/NewellRubbermaid/DASH/S7_int/2020/NWL0000993719A%20SnugRide35%20Lite%20LX%208BS01%20ENG.pdf) and [NHTSA's rear-facing-only infant-seat guidance](https://www.nhtsa.gov/how-install-rear-facing-only-infant-car-seat). The [NHTSA shoulder-pinch handout](https://www.trafficsafetymarketing.gov/sites/tsm.gov/files/2024-01/child-car-seat-booster-belt-harness-handout-en-p2023-13196-tag.pdf) describes a vertical thumb/forefinger assessment at the shoulder with no retained webbing. These sources specify flat, untwisted straps, a snug shoulder pinch assessment and armpit clip height; they do not prescribe this simulator's numerical fold or contact tolerances.
