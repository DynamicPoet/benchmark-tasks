# Fish a stir bar out through a narrow flask neck on a magnetic retriever, then slide its magnet up to drop it in a beaker

A task file for the `yam_arena` embodiment ([yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena), Isaac Lab-Arena 0.3.1).

A 500 mL narrow-neck Erlenmeyer flask holds a red PTFE stir bar on its floor. The robot takes the magnetic stir-bar
positioner/retriever lying in front of it (a 305 mm polyethylene tube with a magnet rod sliding inside it and a ring
handle on top), turns it upright, steers the tube down through the neck to the bar, and the magnet picks the bar up
against the side of the tube (the maker's photo shows the bar clinging lengthwise along the tube). Tube and bar
together are about 24 mm across and the neck's bore about 29 mm, so the tool must come up straight and centred, or the
neck scrapes the bar off back into the flask. Over the beaker the tube is held still while the magnet rod is slid up
by its ring, and the bar drops into the beaker. In the demo the right hand holds the tube and the left pulls the ring;
the score does not ask which hand does what.

## Run a trial

From this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the stir bar out of the flask with the magnetic retriever and drop it into the beaker: hold the tube and slide its magnet up by the ring to let the bar go, then lift the retriever out of the beaker. Do not tip, pour or move the flask." \
  --scorer task_score --max-steps 3000 --store-frames --log-dir runs/dream
```

Scripted demo (scores 1.0): the same command with `--policy yam_scripted`. Variant: `TASK_DEMO=strip` draws the tube
straight up on its own axis instead of centring tube and bar in the neck, so the neck scrapes the bar off (0.5).

## How the simulation judges it

- **The magnet** is an approximation, not a simulated field. While the rod is down (slid up less than 8 mm) and the bar
  comes within 6 mm of the tube's lowest 40 mm, with no glass between them, the bar is placed lengthwise against the side
  of the tube facing it, its lower end at the tip (as a real magnet snaps it there), and held by a rigid fixed joint. It
  is lost as soon as any part of it (to the octagon's corners) reaches the neck's glass while it is level with the neck
  (the rim scrapes it off), or if something pushes it 3 mm off its place on the tube; it is let go when the rod is slid
  up 12 mm or more. The rod slides freely in the tube (no friction; light damping), so it also slides if the tube is
  tipped tip-up. A bar lost or let go falls freely and is picked up again only once it is 1 cm from the tube. The maker
  publishes no pull force; the model assumes the magnet holds the 4 g bar firmly through the moves.
- **Score.** 0 if the flask ever tilts more than 30 degrees or its base leaves its circle by more than 2 cm (latched).
  Otherwise the best reached: 0.25 the tube's tip inside the flask, below its rim; 0.5 the bar's lowest point 3 cm above
  the flask's floor; 0.75 the whole bar 1 cm above the rim; 1.0 the bar lying inside the beaker (its middle within 2 cm of
  the floor, still), let go by sliding the magnet rod up 12 mm or more while the bar was over the beaker's opening, no
  part of the tube inside the flask or the beaker (checked every 5 mm along its axis, with its radius), in each vessel's
  own frame, the tube at least 1 cm and every fingertip at least 3 cm from the bar (measured to the octagon's corners),
  flask and beaker within 10 degrees of upright, all held for 1 s. Doing nothing scores 0.

## Failures and lazy routes

| Run or check | What it does | Score |
|---|---|---|
| Scripted demo | picks the bar up, centres tube and bar in the neck, draws them out, slides the magnet up over the beaker | 1.0 |
| `TASK_DEMO=strip` | draws the tube straight up on its own axis, so the bar meets the neck and is scraped off | 0.5 |
| `check_score.py` | tips the flask and pours the bar out | 0 |
| `check_score.py` | scrapes the bar off on the beaker's rim instead of sliding the magnet up | 0.75 at most |
| `check_score.py` | slides the magnet up away from the beaker, then puts the bar in by hand | 0.75 at most |
| `check_score.py` | leaves the retriever in the beaker or lying across it, or against the bar; a fingertip by the bar | no 1.0 |
| `check_score.py` | knocks the beaker over, leaves the flask leaning, bar still rolling | no 1.0 |
| `check_score.py` | holds the tube outside the flask's wall next to the bar | no pick-up through the glass |

## How a real trial is judged

A phone films the trial from the side at table height, square to the flask and the beaker: the tube's tip entering the
neck, the red bar's height against the flask's 180 mm (3 cm off the floor; whole bar 1 cm above the rim), the ring
pulled up with the bar over the beaker before the bar falls, and the flask's and beaker's tilts. A photo at the end shows
the bar lying in the beaker; a ruler checks the 2 cm circle and the 1 cm and 3 cm clearances. In the station's top
camera the red bar shows faintly on the flask's floor through the glass, as a red mark at the tube's tip while carried,
and in the beaker at the end, once the retriever is lifted out and moved aside.

## Runtime and seeds

- Demo: about 1,270 steps (64 s simulated) at 20 Hz, about 8 minutes of wall time on one GPU; the run limit is 3,000
  steps.
`object_jitter=0` keeps the layout fixed. The scripted demo reads the tool's, flask's and beaker's poses to place the tube.

## Building it for a real trial

- Flask: DURAN narrow-neck Erlenmeyer flask, 500 mL (DWK 212164404), with 300 mL of tap water, on a 105 mm taped circle.
- Stir bar: SP Bel-Art Spinbar F37109-0010, red PTFE-coated, 25.4 x 8 mm (red so it shows on video), put back on the flask's floor at its centre, lying
  flat, between trials (lower it with the retriever and slide the magnet up); top the water back up to 300 mL.
- Retriever: SP Bel-Art stirring bar positioner/retriever H37780-0000 (or Chemglass CG-2019-01, the same design), lying
  flat with its magnet rod pushed fully down.
- Beaker: PYREX 1000-250, 250 mL, empty.
- Catch tray: a 40 x 30 x 2 cm polypropylene tray under flask and beaker, for spills and dropped glass.

## Sources

- SP Bel-Art positioner/retriever H37780-0000, 15.9 x 305 mm, "Retriever is housed inside a tube; hold it down to pick
  up a bar, then slide it up inside the tube to release": https://www.belart.com/positioner-retriever.html (makers named
  for identification only). The ring handle's size and the rod's travel were read off the maker's photograph.
- Chemglass CG-2019-01: https://chemglass.com/cg-2019-01
- SP Bel-Art colour octagon Spinbar F37109-0010 (red, 8 x 25.4 mm): https://belart.com/color-octagon-spinbar-magnetic-stirring-bars.html
- DURAN narrow-neck Erlenmeyer flask 500 mL: base 105 mm, height 180 mm, neck 34 mm outside (DWK listings; https://faust.ch/en/Erlenmeyer-flasks-DURAN-Super-Duty-narrow-neck/9141274). The bore is
  not published; 29 mm is used (a 2.5 mm wall), and a comparable 500 mL narrow-neck flask is listed with a 28 mm inner neck
  (Fisher Scientific S15470,
  https://www.fishersci.com/shop/products/erlenmeyer-flasks-narrow-neck-10/S15470).
- All meshes are built in `scene_assets.py`; no third-party assets.

## Licence

MIT
