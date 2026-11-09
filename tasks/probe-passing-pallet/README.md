# Hold a test probe on each passing pallet's pad for 2 s before the end line

A conveyor line runs across the table, 32 cm in front of the arms, from the right toward the left arm. Two grey
aluminium pallets (12 x 12 x 3 cm) ride on it, 30 cm apart, each with a gold test pad (3 x 3 cm) in the middle of its
top. The line starts with the trial and never stops for the robot: at 4 cm/s a pallet comes past the left arm every
7.5 s, and a red end line, taped across the table 40 cm left of the midpoint, marks where it's too late. A test probe
(a black square handle, 2.6 x 2.6 x 10 cm, with a thin brass tip, 0.8 x 0.8 x 4 cm, under it) stands tip down in a
wooden stand between the line and the left arm. The robot must test each pallet: hold the probe's tip on its pad for
2 s without a break before the pallet reaches the end line.

On lines that run continuously, such as automotive final assembly, checks are made on the move: the operator walks
with the part and holds a gauge or test probe on a test point long enough for a reading. Here a pallet moves 8 cm in
those 2 s while the pad is 3 cm across, so the probe can't be held still and let the pad pass under it (that keeps
the tip on a pad for at most 0.75 s, a quarter of the credit): the arm has to match the line's speed and hold it,
in light contact, for the whole reading, then do it again for the next pallet. Unlike setting a part into a moving
tote or cup, where the motion has to match only at the moment of release, the match has to last.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the line's motion, the test monitor, `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds a `DrivenBox` object kind (a box whose pose and velocity follow a scripted path, set before every physics step; here the pallets and their pads, `belt_run` and `offset_path`), a `UsdBody` kind (a free rigid body from a USD file; here the probe), `clock()`, and `on_step` / `on_reset` hooks with `arm_joints()` for the monitor |
| `make_probe.py` | Writes `assets/probe.usda` (the probe: handle and tip, one rigid body) from the sizes in `task.py`; plain text, no USD library needed |
| `assets/probe.usda` | The probe |
| `test_task.py` | Offline checks (no Isaac Sim): the line's motion, the probe and its stand, every scoring rule and level, and that the demo is reachable, keeps the tip on each moving pad long enough before the end line (also with the arm lagging 0.2 s), and keeps its fingers clear |

`motionkit.py` is copied from this repository's `stack-arriving-blocks` task, with `offset_path()` from
`drop-into-moving-tote` and `UsdBody` adapted from `lock-before-loading`'s scenekit (rigid bodies only), and
`clock()` added. It hooks `yam_arena.scene` and the embodiment's reset, simulation step and task scoring without
modifying yam_arena, so the task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and
`task_score` scorer. The line's clock is the episode clock (`observation.state_time`): the pallets stand still
through the reset and start with the first control step. Written against yam-isaac-lab-arena at `bd62771` (Isaac
Lab-Arena 0.3.1).

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Test both pallets as they pass: take the probe from its stand and hold its tip on each pallet's gold test pad for 2 seconds without a break, before the pallet reaches the red end line. The line keeps moving and won't stop for you." \
  --scorer task_score --max-steps 800 --store-frames --log-dir ~/yam-eval/probe-pallet-dream

# The scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Test both pallets as they pass: take the probe from its stand and hold its tip on each pallet's gold test pad for 2 seconds without a break, before the pallet reaches the red end line. The line keeps moving and won't stop for you." \
  --scorer task_score --max-steps 800 --store-frames --log-dir ~/yam-eval/probe-pallet-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/probe-pallet-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 when `success()` has held for 1 s (both pallets
tested); otherwise it scores `score()` at the end of the trial. Both come from the monitor, which keeps each
pallet's longest unbroken contact over the whole trial, so the end score is the best reached.

## Scoring

The probe is **on a pad** at a control step when all hold:

- it is **held by its handle**: a gripper's grasp point (between the fingers) is within 3 cm of the grip point, on
  the handle's axis 3 cm below its top (so a probe set down on a pallet and let go, or held by its tip, doesn't count);
- it stands within 20 degrees of upright, tip down;
- the middle of the tip's end face is inside the pad's 3 x 3 cm outline seen from above, and no more than 4 mm above
  (a spring test pin's travel) or 5 mm below the pad's top surface (the sim lets pressed contacts sink a little);
- the pad's centre is short of the end line.

For each pallet the monitor (`update()`, run after every control step through `motionkit.on_step`) keeps its
longest unbroken time with the probe on its pad.

| Credit per pallet | Longest unbroken time on its pad | Check on a real table |
|---|---|---|
| 1.0 | 2 s or more | video from the side and above, timed; a ruler for the end line |
| 0.75 | 1.5 s | video |
| 0.5 | 1 s | video |
| 0.25 | any contact | video |
| 0 | none | |

The score is the mean of the two pallets' credits (0, 0.125, ..., 1.0); 1.0 when both were held for 2 s.

## Runtime and seeds

- The scripted demo uses the left arm, fingers closing across the line. It takes the probe from its stand by the
  handle (closed by 3.5 s), lifts it so the tip is 3 cm above pad height, and for each pallet is over the line where
  the pad will be at its meeting time (7.75 s and 15.25 s, at 10 cm left of the midpoint), follows it down in 0.6 s
  with one waypoint per control step, holds the tip on the pad (aimed 1 mm into it) for 2.8 s while following, and
  rises off it, still following, 14 cm before the end line. Then it moves the probe 10 cm back off the line and
  waits; the plan ends at 22.2 s.
- Measured on an L40S: the scripted demo ended with success after 373 steps (18.7 s); Dream-1 scored 0 in an
  800-step trial.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start. The line always starts from rest
  with the trial.

## Fidelity

The pallets and pads are driven exactly along their path, like a stiff servo-driven pallet conveyor: the arms can't
push or stop them (a real line's drive would fault if blocked). They float 2 mm above the bed and each pad 1 mm above
its pallet, so the driven boxes never touch (on a real pallet the pad is glued on). Past the table's left edge they
simply move straight down out of reach. The probe is one rigid body with the station's rubber-like friction; a real
test probe has a spring-loaded pin, which the 4 mm allowance above the pad stands in for, and contact is measured
from the probe's pose, not by an electrical reading.
