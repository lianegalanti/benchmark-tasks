# Put lids on open tubs while they ride past on a conveyor

A conveyor line runs across the table, 34 cm in front of the arms, from the right toward the left arm. Three grey
carrier plates ride on it, 32 cm apart, each holding an open white tub (8 cm square, 5 cm tall). Three white lids lie
skirt-down on the table in a row in front of the left arm; each has a skirt round its edge that fits over a tub with
1 cm to spare on every side, and a blue grip fin on top. The line starts with the trial and never stops for the robot:
at 3 cm/s a tub passes the left arm every 10.7 s, and a red end line, taped across the table 46 cm left of the
midpoint, is where each tub is judged. The robot must put a lid on each tub while it moves, seated flat on its rim,
and let go of it before the tub reaches the line.

Placing an object on a static target is common in benchmarks; here the target moves and the placement is a fit (a
skirt over a rim with 1 cm of play), so the robot has to match the tub's motion while it lowers the lid, then let go
and be back with the next lid before the next tub passes. Lidding stations on food and packaging lines do this
continuously: an unlidded or skewed lid means a spill and a rejected pack.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the line's motion, the lidding monitor, `success()`, `score()` and the scripted demo |
| `geometry.py` | The tub's and lid's shapes |
| `make_lid.py` | Writes `assets/lid.usda` from `geometry.py` (plain text; no USD library needed) |
| `assets/lid.usda` | The lid: one rigid body of six boxes (plate, four skirt strips, grip fin) |
| `motionkit.py` | Adds `DrivenBox` (a box whose pose and velocity follow a scripted path, set before every physics step; here the carriers and the tubs' walls, `belt_run`) and `UsdBody` (a free rigid body from a USD file; here the lids) to yam_arena task files, and `on_step` / `on_reset` hooks with `robot_links()` / `arm_joints()` for a per-step monitor |
| `test_task.py` | Offline checks (no Isaac Sim): the line's motion, the tub and lid shapes, the lidding rule and score, and that the demo is reachable, on time and clear of the tubs and lids |

`motionkit.py` is copied from this repository's `stack-arriving-blocks` task; its `UsdBody` loader is adapted from
`scenekit.py` in this repository's `lock-before-loading` task. It hooks `yam_arena.scene` and the embodiment's
reset, simulation step and task scoring without modifying yam_arena, so the task runs with the standard `yam_arena`
embodiment, `yam_scripted` policy and `task_score` scorer. The line's clock is the episode clock
(`observation.state_time`): the carriers stand still through the reset and start with the first control step.
Written against yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1).

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put a lid on each of the three tubs as they go by on the conveyor, seated flat on the tub's rim, and let go of it before the tub reaches the red end line. The conveyor won't stop for you." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/lid-tubs-dream

# The scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put a lid on each of the three tubs as they go by on the conveyor, seated flat on the tub's rim, and let go of it before the tub reaches the red end line. The conveyor won't stop for you." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/lid-tubs-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/lid-tubs-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 when `success()` has held for 1 s (all three tubs
lidded at the end line, about 35.4 s in); otherwise it scores `score()` at the end of the trial (`SCORE_OVER = "end"`).

## Scoring

Each tub is judged once, at the control step its centre reaches the end line. It is **lidded** if then one of the
lids sits on it:

- the lid's centre within 1.5 cm (horizontally) of the tub's axis (a seated skirt allows at most 1 cm);
- the underside of the lid's plate within 5 mm of the tub's rim;
- the plate level within 5 degrees, right way up;
- the lid moving with the tub (velocity within 2 cm/s of the tub's);
- let go: both grippers' grasp points (between the fingers) 5 cm or more from the centre of the lid's grip fin.

| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | all three tubs lidded at the end line | a side video at the end line, ruler and a level on the lid in the frame each tub reaches it |
| 0.67 | two tubs lidded at the end line | same |
| 0.33 | one tub lidded at the end line | same |
| 0 | no tub lidded at the end line | |

A lid perched on the rim on its skirt (askew by more than 1 cm), tilted, upside down, inside the tub, or still held
when the tub reaches the line counts 0 for that tub. Lidding a tub after it has passed the line doesn't count, and
what happens to a lid after its tub has passed the line doesn't change the score. The monitor (`update()`, run after
every control step through `motionkit.on_step`) latches each tub's result and keeps both arms' grasp points.

## Runtime and seeds

- The scripted demo uses the left arm, fingers closing along the line, across the fin. For each tub it takes a lid
  by its grip fin (the far lid first), the pad tips 1.5 cm above the plate, and holds still for 1.4 s after the close
  command (the simulated gripper takes about 1 s to close on the fin). It lifts the lid straight up over 1.4 s, so
  its skirt is 3 cm above the rims, carries it over 2 s to the line 29 cm left of the midpoint (peak accelerations
  under 0.35 m/s², so the lid doesn't swing in the pinch), and waits there for the tub (0.85 to 0.95 s to spare).
  From the tub's arrival (7.4 s, 18.1 s, 28.8 s) it follows the tub with one waypoint per control step: it lowers
  the lid onto the rim over 1 s, holds 0.2 s, opens the fingers 3 mm above the seat and holds 1.3 s while they open,
  then rises 10 cm still following it. Each lid is seated 4.6 s, and let go 3.1 s, before its tub reaches the end
  line. The plan ends at 37 s (so that `yam_scripted`, which stops 3 s after the plan, doesn't end the trial early),
  the arm held back over the table; the last tub is judged at 34.4 s.
- A first Isaac Sim run of an earlier version (lift 0.5 s after the close command, a 3 cm tab) left two lids behind
  and let the third pivot 24 degrees in the pinch; this version holds longer, lifts and carries more gently, and
  grips a taller, longer fin.
- Measured on an L40S: this demo ended with success after 709 steps (35.5 s); Dream-1 scored 0 in a 1,200-step
  trial.
- `LID_DEBUG=1` in the environment prints each lid's pose every 0.5 s, the TCP relative to a lid's fin and the pad gap
  at every step a gripper is within 6 cm of a fin, and each tub's judgment as it is made.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start. The line always starts from rest
  with the trial.

## Fidelity

The carriers and the tubs' walls are driven exactly along their path, like a stiff servo-driven pallet conveyor with
the tubs held in nests: the arms can't push, stop or knock them over (a real line's drive would fault if blocked).
Each tub is four driven walls standing 1 mm above its carrier, which forms its floor; driven parts never touch each
other. The carriers float 2 mm above the track bed and past the table's left edge simply move straight down. Each lid
is one rigid body of six sharp-edged boxes with no snap bead: a seated lid rests on the rim under its own weight
(40 g), held in place by its skirt, with the station's rubber-like friction (1.2 static, 1.0 dynamic, more than
polypropylene); the line's acceleration (under 0.1 m/s²) is far too low to slide it. The tubs are empty.
