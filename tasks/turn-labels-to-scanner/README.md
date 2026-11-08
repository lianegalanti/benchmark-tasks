# Turn each carton's label toward a fixed scanner as the cartons ride a moving line

A conveyor line crosses the table 34 cm in front of the arms and carries three small cardboard cartons (6 x 6 x 8 cm)
on grey carrier plates from the robot's right to its left. Each carton has a white shipping label with black barcode
bars on one side, under an orange band that folds over onto the carton's top (so the top camera can see which side the
label is on). Near the table's left end the line runs through a scanner gate whose scanner head, on the far side of the
line, reads the side of each carton that faces it. The labels start facing the wrong way: carton 1's back along the
line, carton 2's toward the robot, carton 3's along the line the way it moves. The robot must turn each carton on its
moving carrier so its label faces the scanner, leave it standing upright on its own carrier, and have let go of it
before it reaches the gate.

The line never stops: at 4 cm/s a carton passes the gate every 5.5 s. Turning a carton means taking it off a moving
carrier, turning it with the wrist while still following the carrier, and setting it back down on the carrier, square,
before the next one needs the same hand. Carton 2 needs a half turn; the scripted demo hands that off, a quarter turn
from each arm, as the carton moves from one arm's reach to the other's.

On parcel and warehouse lines a fixed scanner reads the barcode on one side of each carton; a carton whose label faces
elsewhere is a no-read and goes to manual handling, so people beside the line turn cartons label-out as they pass. The
board's other line tasks pick, sort or stack items off a belt; here the items stay on the line and only their heading
must change, under a deadline set by the line.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the line's motion, the scanner-read monitor, `success()`, `score()` and the scripted demo |
| `geometry.py` | Dimensions, layout and the scanner's read rule (`check()`), NumPy only |
| `motionkit.py` | Adds a `DrivenBox` object kind (a box whose pose and velocity follow a scripted path, set before every physics step; here the carriers, `belt_run`) and a free `UsdBody` (here the cartons), plus `on_step` / `on_reset` hooks for the per-step monitor |
| `make_assets.py` | Writes `assets/carton.usda` (needs usd-core): one box collider with the label, barcode bars and orange band as visual-only prims |
| `assets/carton.usda` | The carton |
| `test_task.py` | Offline checks (no Isaac Sim): the asset, the line's motion, the gate's clearances, the read rule at every score level, and that the demo is reachable, turns each carton the right way and lets go before the gate |

`motionkit.py` is copied from this repository's `stack-arriving-blocks` task (itself from `reach-between-sweeps`) and
extended with `UsdBody`, whose spawning follows the `scenekit.py` of this repository's `parcel-this-side-up` task. It
hooks `yam_arena.scene` and the embodiment's reset, simulation step and task scoring without modifying yam_arena, so the
task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score` scorer. The line's clock is
the episode clock (`observation.state_time`): the carriers stand still through the reset and start with the first
control step. Written against yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1).

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Before each carton reaches the scanner gate, turn it on its carrier so that its shipping label faces the scanner on the far side of the line. Leave each carton standing upright on its own carrier, and let go of it before it reaches the gate." \
  --scorer task_score --max-steps 700 --store-frames --log-dir ~/yam-eval/turn-labels-dream

# The scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Before each carton reaches the scanner gate, turn it on its carrier so that its shipping label faces the scanner on the far side of the line. Leave each carton standing upright on its own carrier, and let go of it before it reaches the gate." \
  --scorer task_score --max-steps 700 --store-frames --log-dir ~/yam-eval/turn-labels-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/turn-labels-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 once all three cartons have been read (held 1 s);
otherwise it scores `score()` at the end (`SCORE_OVER = "end"`), which is the tally of reads the monitor made at the gate.

## Scoring

The monitor (`update()`, run after every control step through `motionkit.on_step`) reads each carton once, at the first
control step at which its carrier's centre has reached the gate (y = 0.50 m). Using the carrier, not the carton, means
holding a carton back can't delay its read. A carton is **read** if at that moment:

- it stands on its own carrier: its centre within 3 cm of the carrier's centre in each of x and y, its underside
  within 1 cm of the carrier's top;
- it is upright: its top within 10 degrees of straight up (a carton laid on its side with the label facing the
  scanner, or stood upside down, is not read);
- both grippers' grasp points are 8 cm or more from its centre (let go, hands out of the gate);
- its label faces within 20 degrees of straight at the scanner (+x, seen from above);
- it has never touched the table during the trial (its lowest corner within 1 cm of the table top), even if put back.

| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | all three cartons read | a video from above the gate, or a photo of each carton in the gate with a protractor template on the carrier |
| 2/3 | two cartons read | same |
| 1/3 | one carton read | same |
| 0 | none read, including doing nothing (no carton starts within 20 degrees: each label is 90 or 180 degrees off) | |

## Runtime and seeds

- The scripted demo (`demo()`, plan in `TURNS`): for each turn the arm is over the line where the carton will be at
  its meeting time, then, one waypoint per control step following the carrier, goes down, closes, lifts the carton
  1.5 cm, turns it with the wrist (1.4 s), sets it back 3 mm above the carrier, lets go and goes up (4.3 s per turn).
  The right arm gives carton 2 a quarter turn (1.75–6.05 s) and turns carton 3 (7.75–12.05 s); the left arm turns
  carton 1 (7.25–11.55 s) and gives carton 2 its second quarter turn (13.25–17.55 s). Every carton is let go at least
  12 cm before the gate; the reads are at 15.75, 21.25 and 26.75 s, success holds to about 27.8 s (556 steps). Both arms wait at rest
  until 30 s, so that `yam_scripted` (which stops 3 s after the plan) doesn't end the trial early.
- Planned IK error under 0.1 mm and 0.02 degrees; the nearest joint limit is joint 4, 9 degrees away.
- Measured on an L40S: the scripted demo ended with success after 555 steps (27.8 s); Dream-1 scored 0 in a
  700-step trial.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start. The line always starts from rest with
  the trial.

## Fidelity

The carriers are driven exactly along their path, like a stiff servo-driven pallet conveyor: the arms can't push them
or stop them (a real line's drive would fault if blocked). They float 2 mm above the track bed and past the table's
left edge simply move straight down out of reach. Each carton is one rigid sharp-edged box with an even mass; its
label, barcode and orange band are visual only (0.5 mm proud, no collision), so the fingers grip the plain box. The
cartons have friction 0.9 static, 0.8 dynamic; the line's acceleration (under 0.15 m/s²) is far too low to slide them.
The scanner isn't simulated optically: the score applies its read rule (label within 20 degrees of facing it)
geometrically when each carton reaches the gate. The gate's posts and scanner head are fixed colliders; a carton pushed
3 cm or more off-centre toward the scanner and turned near 45 degrees can catch on its head, as on a real gate.
