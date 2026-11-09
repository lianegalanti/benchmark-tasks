# Stand a fallen bottle back up on a moving conveyor before it reaches the guide

A conveyor line runs across the table, 32 cm in front of the arms, from the robot's left toward its right. Three grey
carrier plates ride on it, 24 cm apart, at 2 cm/s, each with one bottle: a white plastic cylinder 6 cm across and 13 cm
tall, its top 1.5 cm red (the cap). The two in front stand upright; the last one has fallen over and lies on its side
along the line, cap pointing downstream. Downstream, 18 cm right of the midpoint between the arms, the line enters a
pair of steel guide rails 9 cm apart, the neck that leads to the next machine's star wheel; red tape marks the guide's
mouth. The robot must stand the fallen bottle back up, cap up, on its own carrier before the carrier reaches the guide
and let go of it, without disturbing the upright bottles.

On bottling, filling and labelling lines a bottle that falls over jams the star wheel or the guide at the next machine
and stops the line; operators stand fallen bottles back up on the moving line. Doing it means taking a lying bottle off
a carrier that doesn't stop, turning it a quarter turn in the hand, and setting it down upright, centred on that same
moving carrier, with the guide coming. Taking the bottle off the line also prevents the jam but loses the bottle from
the batch, so it scores less.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the line's motion, the per-step monitor (disturbed bottles, the decision at the guide), `success()` and `score()` |
| `motionkit.py` | Adds a `DrivenBox` object kind (a box whose pose and velocity follow a scripted path, set before every physics step: here the carriers, `belt_run`), a `Bottle` object kind (a free cylinder that can start lying down, with a painted cap band), and `on_step` / `on_reset` hooks for the monitor |
| `demo.py` | The scripted demo (left arm), closed loop on the measured bottle pose |
| `test_task.py` | Offline checks (no Isaac Sim): the line's motion and layout, every scoring level and the penalty, and the demo played against a rigid-grip bottle: reachable, on time, clear of the line, guide and upright bottles, and scoring 1.0 |

`motionkit.py` is copied from this repository's `stack-arriving-blocks` task (itself from `reach-between-sweeps`), with
`belt_run` given a direction and `Bottle` added. The `Arm` helper and the run loop in `demo.py` are adapted from this
repository's `hang-bottle-on-drying-peg` task (its license applies). motionkit hooks `yam_arena.scene` and the
embodiment's reset, simulation step and task scoring without modifying yam_arena, so the task runs with the standard
`yam_arena` embodiment and `task_score` scorer. The line's clock is the episode clock (`observation.state_time`): the
carriers stand still through the reset and start with the first control step. Written against yam-isaac-lab-arena at
`bd62771` (Isaac Lab-Arena 0.3.1).

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "One bottle has fallen over on the conveyor. Stand it back up, cap up, on its own carrier before it reaches the guide rails, and let go of it, without disturbing the upright bottles." \
  --scorer task_score --max-steps 700 --store-frames --log-dir ~/yam-eval/fallen-bottle-dream
for log in ~/yam-eval/fallen-bottle-dream/*.json; do inspect-robots video "$log"; done

# The scripted demo (writes demo_*.mp4, setup.png and result.json)
python demo.py task.py demo_out

# Offline checks, in Arena's venv
PYTHONPATH=<yam-isaac-lab-arena> python -m pytest -q test_task.py
```

`yam_scripted` plays only straight-down grips, so `demo.py` drives the embodiment itself. The run prints `task_score`
at the end. The score is decided when the fallen bottle's carrier reaches the guide's mouth, 26.75 s into the trial;
the trial ends 1 s later on success, otherwise at the run limit with that decision (`SCORE_OVER = "end"`; before the
decision `score()` reports what the current state is worth).

## Scoring

Decided at the moment the fallen bottle's carrier (its centre) reaches the guide's mouth (the red tape), from the
fallen bottle's state then:

| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | upright (axis within 10 degrees of vertical, cap up) on its own carrier: underside within 1 cm of the carrier's top, centre within 1.5 cm of the carrier's centre across the line and 2.5 cm along it, moving with the carrier (relative speed under 5 cm/s), and let go (both grippers' grasp points 8 cm or more from its centre) | photo from above with a ruler as the carrier reaches the tape; video |
| 0.5 | upright on its own carrier (centre over the carrier's top face) but still held or not centred; or upright and let go, standing on the table off the line | photo |
| 0.25 | off the line but not upright: no part of it over the track bed, or its lowest point 2 cm or more above the carriers' tops (held up), and not off the table | photo |
| 0 | anything else: still lying on its carrier or the bed, standing on the bed between carriers, or off the table | |

Minus 0.5 for each upright bottle disturbed, not below 0. An upright bottle is disturbed if, at any time before its
own carrier reaches the guide's mouth, it tilts more than 20 degrees, its centre moves more than 3 cm (horizontally)
from its carrier's centre, or its underside rises more than 2 cm above its carrier's top. So taking the fallen bottle
off the line (0.25) or standing it on the table (0.5) scores less than standing it back on its carrier (1.0), and
knocking a bottle over to make room costs more than it can gain. Doing nothing scores 0.

## Runtime and seeds

- The decision comes 26.75 s into the trial; the run limit is 700 steps (35 s at 20 Hz).
- The scripted demo uses the left arm. It is over the line where the fallen bottle's carrier will be at 3.25 s (29 cm
  left of the midpoint), follows it down with the gripper pointing down and tilted 20 degrees toward the robot's left,
  fingers closing across the line, and grips the lying bottle across its body at 5.0 s, 1.5 cm toward the cap from its
  middle and 1 cm above its axis. Following the carrier all the while, it lifts the bottle 3 cm, rises 6 cm more, and
  turns the gripper a quarter turn about the finger axis (cap up, the gripper now pointing downstream and 20 degrees
  down). From the measured bottle pose it brings the axis upright 3 cm over the carrier's centre, lowers it to 3 mm
  above the carrier (nudging the target by the measured offset of the bottle from the carrier's centre), and lets go
  at 11.4 s, 15 s before the guide; then it rises straight up and parks at 14.7 s.
- Offline, against a rigid-grip bottle: IK error under 0.1 mm and 0.02 degrees; the arm's collision hulls stay 1.9 cm
  or more from the carriers and bed, 2.8 cm from the upright bottles, 5.5 cm from the guide; the open fingers 1.7 cm
  from the lying bottle.
- Measured on an L40S: the demo ended with success after 554 steps (27.7 s); Dream-1 scored 0 in a 700-step trial.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start. The line always starts from rest
  with the trial.

## Fidelity

The carriers are driven exactly along their path, like a stiff servo-driven pallet conveyor: the arms can't push or
stop them. They float 2 mm above the track bed, and past the table's right edge they move straight down out of reach.
The bottles are rigid plain cylinders (no neck or shoulder; the red cap is painted on) with the station's rubber-like
friction (1.2 static, 1.0 dynamic). The fallen bottle lies along the line, so the line's gentle start (under 0.1 m/s²)
pushes it along its own axis and can't roll it. The jam itself is not simulated: the score is decided at the guide's
mouth; the rails are there to show the neck the bottles must pass (1.5 cm clearance each side of an upright bottle).
