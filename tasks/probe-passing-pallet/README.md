# Hold a test probe on each passing pallet's pad for 2 s before the end line

A conveyor line runs across the table, 32 cm in front of the arms, from the right toward the left arm. Two grey
aluminium pallets (12 x 12 x 3 cm) ride on it, 30 cm apart, each with a gold test pad (3 x 3 cm) in the middle of its
top. The line starts with the trial and never stops for the robot: at 4 cm/s a pallet comes past the left arm every
7.5 s, and a red end line, taped across the table 40 cm left of the midpoint, marks where it's too late. A test probe
(a black square handle, 2.6 x 2.6 x 10 cm, with a thin brass tip, 0.8 x 0.8 x 4 cm, under it) stands tip down in a
wooden stand between the line and the left arm. The robot must test each pallet: hold the probe's tip on its pad for
2 s without a break before the pallet reaches the end line. Only real contact counts: the probe is rigid, and the
tip has to touch the pad for the whole 2 s (sensed at every physics step in sim; on a real table the pad lights an
LED only while the brass tip bridges it). Past the end line the pallets ride into a guarded end stop at the bed's
end and wait there, so nothing leaves the table.

On lines that run continuously, such as automotive final assembly, checks are made on the move: the operator walks
with the part and holds a gauge or test probe on a test point long enough for a reading. Here a pallet moves 8 cm in
those 2 s while the pad is 3 cm across, so the probe can't be held still and let the pad pass under it (that keeps
the tip on a pad for at most 0.75 s, a quarter of the credit): the arm has to match the line's speed and hold it,
in light contact, for the whole reading, then do it again for the next pallet. Unlike setting a part into a moving
tote or cup, where the motion has to match only at the moment of release, the match has to last.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the line's motion and end stop, the probe-pad contact sensor, the test monitor, `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds a `DrivenBox` object kind (a box whose pose and velocity follow a scripted path, set before every physics step; here the pallets and their pads, `belt_run` and `offset_path`), a `UsdBody` kind (a free rigid body from a USD file; here the probe), `clock()`, and `on_step` / `on_reset` hooks with `arm_joints()` for the monitor |
| `make_probe.py` | Writes `assets/probe.usda` (the probe: handle and tip, one rigid body with PhysX contact reporting) from the sizes in `task.py`; plain text, no USD library needed |
| `assets/probe.usda` | The probe |
| `test_task.py` | Offline checks (no Isaac Sim): the line's motion and the queue at the end stop, the probe and its stand, every scoring rule and level (hovering scores 0, also with PhysX reporting a contact point 0.4 mm away; one step without contact breaks a test; a stalled pallet earns nothing), and that the demo is reachable, presses the tip onto each moving pad long enough before the end line (also with the arm lagging 0.2 s), and keeps its fingers clear |

The probe-pad contact sensor in `task.py` is adapted from the `move-live-wire-with-stick` task in this repository
(another team's, under its license): a PhysX rigid-contact view polled after every physics step. `motionkit.py`
hooks `yam_arena.scene` and the embodiment's reset, simulation step and task scoring without
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

The probe is **on a pad** at a control step (1/20 s) when all hold:

- **the tip touches the pad throughout the step.** In sim, a PhysX rigid-contact view on the probe, filtered to the
  two pads, is read after every 240 Hz physics step; the probe touches a pad at a physics step when the view reports
  a probe-pad contact point with zero or negative separation (PhysX also reports points with a positive separation,
  a gap, before bodies touch; those don't count), and the control step counts only if it touched at all 12 of its
  physics steps. As a cross-check from the poses, the middle of the tip's end face must be no more than 0.2 mm above
  the pad's top surface (or 5 mm into it: the sim lets pressed contacts sink a little). A probe hovering over the
  pad, even 0.4 mm above it, never counts. On a real table the pad is the sensor: it is split into two interleaved
  gold combs wired to a coin cell and an LED on the pallet, and the brass tip lights the LED only while it touches
  the pad and bridges the combs. The gripper's finger pads are covered with insulating tape, so a finger can't
  light it;
- it is **held by its handle**: a gripper's grasp point (between the fingers) is within 3 cm of the grip point, on
  the handle's axis 3 cm below its top (so a probe set down on a pallet and let go, or held by its tip, doesn't count);
- it stands within 20 degrees of upright, tip down;
- the middle of the tip's end face is inside the pad's 3 x 3 cm outline seen from above;
- the pad's centre is short of the end line, and its pallet is moving along the line at 3.5 cm/s or more (a pallet
  held still, which on a real line the belt slips under, can't be tested; in sim the pallet's velocity is read).

For each pallet the monitor (`update()`, run after every control step through `motionkit.on_step`) keeps its
longest unbroken time with the probe on its pad over the whole trial.

| Credit per pallet | Longest unbroken time on its pad | Check on a real table |
|---|---|---|
| 1.0 | 2 s or more | on frame-timed video from the side and above, for 2 s without a break: the pallet's LED lit, the probe in a gripper by its handle, standing within 20 degrees of upright with its tip on the pad; the pallet passing the white tick marks at its normal pace (10 cm in under 2.9 s); the pad's centre short of the red tape |
| 0.75 | 1.5 s | the same, for 1.5 s |
| 0.5 | 1 s | the same, for 1 s |
| 0.25 | any contact | the same, for any moment |
| 0 | none | never, before the end line (an LED lit by a dropped probe lying on the pad, or by anything but the held probe's tip, doesn't count) |

The score is the mean of the two pallets' credits (0, 0.125, ..., 1.0); 1.0 when both were held for 2 s.

## Runtime and seeds

- The scripted demo uses the left arm, fingers closing across the line. It takes the probe from its stand by the
  handle (closed by 3.5 s), lifts it so the tip is 3 cm above pad height, and for each pallet is over the line where
  the pad will be at its meeting time (7.75 s and 15.25 s, at 10 cm left of the midpoint), follows it down in 0.6 s
  with one waypoint per control step, presses the tip onto the pad (aimed 2 mm into it) for 2.8 s while following, and
  rises off it, still following, 14 cm before the end line. Then it moves the probe 10 cm back off the line and
  waits; the plan ends at 22.2 s. The demo is open-loop: it is planned from the line's fixed, known schedule
  (`pallet_y()`, the same path that drives the pallets) and never reads the simulator's poses or contacts.
- Measured on an L40S: the scripted demo ended with success after 376 steps (18.8 s), contact sensed at every physics
  step; Dream-1 scored 0 in an 800-step trial.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start. The line always starts from rest
  with the trial.

## Safety on a real table

Nothing leaves the line. The pallets ride on a flat belt between two low side guide rails (5 mm wide, 1 cm tall,
5 mm clear of each side of a pallet), and the bed ends in a steel end stop (16 cm across, 2 cm deep, 5 cm above the
bed, with a rubber bumper face) bolted across its last 2 cm, 3 cm inside the table's left edge. The belt slips under
a pallet that reaches it (an accumulating pallet line, as on standard pallet transfer systems), so the first pallet
waits against the bumper (centre 54 cm left of the midpoint, from 18.7 s) and the second against the first (centre
42 cm, from 23.2 s), both past the end line and out of the test. The pallets arrive at 4 cm/s. The same slip means
an arm that holds a pallet back only stops it (no drive fault, no pile-up force beyond the belt's friction), and a
stopped pallet can't be tested (credit needs it moving at 3.5 cm/s or more). In sim the rails and end stop are fixed
boxes and each pallet's path ends at its place in the queue (2 mm short of the bumper or the pallet ahead).

## Fidelity

The pallets and pads are driven exactly along their path: the arms can't push or stop them in sim, while a real
pallet held back stops as the belt slips under it. Since credit needs the pallet moving at 3.5 cm/s or more, holding
a pallet still earns nothing either way. In sim the pallets float 2 mm above the bed (a real belt carries them) and
each pad 1 mm above its pallet (on a real pallet it is glued on), so the driven boxes never touch; at the end stop
they stop dead (the real bumper gives a little). The probe is one rigid body with the station's rubber-like friction,
as is the real probe (no spring-loaded pin). Contact is sensed in sim from PhysX contact reports between the probe
and the pads, and on a real table by the split pad's LED; the sim's pad is one solid box (the real comb's 0.3 mm gaps
are far narrower than the 8 mm tip).
