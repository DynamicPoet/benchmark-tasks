# Directional release checks

Measure both installed apples. Label the stations A (lower) and B (upper). Record the mass of the complete fruit assembly, magnet and disc geometry, shim thickness, stem orientation, scale model, units, resolution and zero check. Film the gauge and the stem from the side. For a fast release, use a gauge that records the peak or a synchronized force trace. Keep each attempt with its setup ID, pull direction, loading rate, peak, release site and recording. Re-seat both attachments between attempts. Repeating each condition three times per station gives an initial check on variation.

## Measurement sequence

1. With the two ends tested separately, pull straight off the spur cap (design range 37–43 N), then pull the apple off the supported peg (22–26 N). Photograph the resting angle and the change with an added 100 g at the fruit centre (design range 8–14° of additional droop).
2. With both joints assembled, pull at the fruit centre. Mark the initial outward stem axis and the direction of gravity. Compare axial, 15° above axial, 15° below axial and vertically downward pulls. Record the first interface to release, or the highest load reached if both remain joined. For the downward trial, the proposed range is 4–7 N of additional force at the fruit centre with gravity already acting.
3. In a separate trial, roll the fruit upward about its seat, then draw it away. Record fruit angle and draw direction at release. If a fixture or grip also applies a couple, record that contribution. The force on the pull gauge, the force projected along the rotating stem, and a bending moment in N m are distinct quantities. A single pull gauge gives only its line tension.
4. Keep individual readings and all outcomes for each configuration. After changing a shim, magnet or stem, repeat the directional trials and compare both release force and release site. Report how often the stem remained on the fruit. Check that a repeated release and reseating leaves the two stations usable.

## Predictions from the fixed-tree model

Run `python rig/bench_model.py` from the task directory. NumPy is required. The script uses the same joint law as the simulation, applies a gradually increasing external force through the centre of the freely rotating fruit, and includes gravity. A positive angle points below the initial outward stem axis. The model applies no external torque about the fruit centre in these four tests.

| Pull direction | Additional gauge force at first release | First interface to release |
|---|---:|---|
| Along the initial stem | 23.98 N | Peg, stem stays on tree |
| 15° above the stem | 24.40 N | Peg |
| 15° below the stem | 20.30 N | Spur seat, stem stays on fruit |
| Vertically downward | 4.88 N | Spur seat |

Both stations gave these predictions at 240 Hz and 4 N/s. At 480 Hz, and separately at 2 N/s, the predicted interface was the same in all eight station/direction combinations. The largest force difference was 0.02 N. These are numerical checks. Hardware measurement fields remain open.

The implemented joint resolves cap and peg loads about the same rotating stem axis. In the physical design the spur magnet face is fixed to the spur. A sensitivity calculation using that fixed normal for the cap and the rotating axis for the peg predicted 6.68 N downward and 21.30 N at 15° below axial, with the same first-release sites. Both downward predictions fit the proposed 4–7 N range. Measurements through smaller angles and at documented loading rates are needed to distinguish these models.

## Ripe-fruit comparison

In three orchard trials per direction on mature Red Fuji fruit, peak pulls normal to the biological separation layer were 23.86–26.14 N and tangential pulls were 12.28–14.31 N. The same study found stalks on 75.7% of successful direct picks and 93.9% of successful picks along the optimized path. A direct pick can retain a stem. A 24 N predicted axial peg pull therefore cannot establish agreement just because its magnitude falls in a real-fruit range. The simulated downward draw includes gravity and cap tipping. The published tangential test uses a biological separation plane and a different load geometry. [Orchard experiments and retention outcomes](https://doi.org/10.3390/agriculture14122280)

For harvest-date Pink Lady apples, the reported vertical-pull series included three fruit-end separations in 25 trials, at 27.33–32.85 N. The other vertical pulls separated at the branch end. Document the stem-to-gravity angle before using a vertical pull as an axial comparator. [Primary harvesting study, sections 3.1–3.3](https://doi.org/10.1016/j.scienta.2019.108937)

The rig is intended as a repeatable, stem-pull-prone exercise. Its numerical predictions and the biological experiments describe different objects. Measurements of the assembled rig, including loads and first-release sites in matched geometries, are still needed to establish its performance.
